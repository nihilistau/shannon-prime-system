/* sp_model_bridge.c — sp_model_to_qwen3 (declared in sp/sp_model.h).
 *
 * Reconstructs a runnable qwen3_model from a loaded .sp-model handle, so the L1
 * session ABI (which takes const sp_model*) can drive the relocated reference
 * forward. Re-derived from PPT-LAT-SP-MODEL-v0 + PPT-LAT-Systems Appendix B — NOT
 * copied from any engine/legacy reconstruction body.
 *
 * Output shape (matches the .sp-model adapter path documented in sp/model.h):
 *   - matmul weights (attn q/k/v/o, ffn gate/up/down, token_embd) -> a packed arena
 *     (sp_arena_from_packed) keyed by GGUF tensor name; the .sp-model OK_Q8 codes +
 *     paired <name>.scale (per-row Frobenius max-abs) map directly onto the per-row
 *     Q8 sp_frob_packed_tensor layout (dequant = code * scale / 127).
 *   - norms (attn/ffn/qk/output) -> owned f32 copies in norm_buf, with norm_src
 *     pointer-identity-matched to the layer's synthetic gguf_tensor.
 *   - gguf = NULL, released = 1: the forward reads only the arena + owned norms, so
 *     gguf_tensor_data is never reached.
 * Freed entirely by qwen3_free.
 *
 * Note (frozen-format gap, see SESSION offload): sp_arch_info carries neither n_ff
 * nor rms_eps. n_ff is recovered from the ffn_gate tensor's out dimension; rms_eps
 * defaults to 1e-6 (the parity gate is bit-exact on the same reconstruction, so the
 * default does not affect it — but a real transcode needs a format-v1 field). */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/sp_model.h"
#include "sp/sp_l1.h"
#include "sp/model.h"
#include "sp/arena.h"
#include "sp/frobenius_lift.h"
#include "sp/gguf.h"
#include "sp/sp_error.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Build a per-row-Q8 packed tensor aliasing codes + row_scale directly from the
 * mmap (zero copy). alias_mask=0x3 prevents sp_frob_packed_free from freeing them.
 * Only row_prec and row_off are heap-allocated (tiny: rows * 5 bytes). */
static int build_packed_q8(const sp_model *sm, const char *name, sp_frob_packed_tensor *out) {
    memset(out, 0, sizeof *out);
    const sp_tensor_entry *e = sp_model_find_tensor(sm, name);
    if (!e || e->dtype_id != (uint32_t)SP_DT_OK_Q8 || e->n_dims < 2) return 1;
    /* rank-3 expert tensors [cols, rows, n_expert] pack as (rows*n_expert) rows. */
    uint64_t cols = e->dims[0], rows = e->dims[1] * (e->n_dims >= 3 ? e->dims[2] : 1u);
    if (rows == 0 || cols == 0 || e->size_bytes != rows * cols) return 1;
    const uint8_t *codes = (const uint8_t *)sp_model_tensor_data(sm, e);
    if (!codes) return 1;

    char sn[96];
    snprintf(sn, sizeof sn, "%s.scale", name);
    const sp_tensor_entry *se = sp_model_find_tensor(sm, sn);
    if (!se || se->dtype_id != (uint32_t)SP_DT_FROBENIUS_SCALE_FP32 ||
        se->dims[0] != rows || se->size_bytes != rows * 4u) return 1;
    const float *scale = (const float *)sp_model_tensor_data(sm, se);
    if (!scale) return 1;

    out->rows = (int)rows; out->cols = (int)cols; out->codes_bytes = rows * cols;
    out->row_prec = (uint8_t *)malloc((size_t)rows);
    out->row_off  = (size_t  *)malloc((size_t)rows * sizeof(size_t));
    if (!out->row_prec || !out->row_off) {
        sp_frob_packed_free(out); return 1;  /* alias_mask=0, codes/scale=NULL -> safe */
    }
    out->alias_mask = 0x3;
    out->codes     = (uint8_t *)(uintptr_t)codes;   /* mmap PAGE_READONLY/PROT_READ */
    out->row_scale = (float   *)(uintptr_t)scale;
    for (uint64_t r = 0; r < rows; r++) {
        out->row_prec[r] = 8u;
        out->row_off[r]  = (size_t)(r * cols);
    }
    return 0;
}

/* Build a per-row-Q4 packed tensor aliasing codes + row_scale from the mmap.
 * On-disk layout: rows * ceil(cols/2) nibble bytes (low nibble = even col index). */
static int build_packed_q4(const sp_model *sm, const char *name, sp_frob_packed_tensor *out) {
    memset(out, 0, sizeof *out);
    const sp_tensor_entry *e = sp_model_find_tensor(sm, name);
    if (!e || e->dtype_id != (uint32_t)SP_DT_OK_Q4 || e->n_dims < 2) return 1;
    /* rank-3 expert tensors [cols, rows, n_expert] pack as (rows*n_expert) rows. */
    uint64_t cols = e->dims[0], rows = e->dims[1] * (e->n_dims >= 3 ? e->dims[2] : 1u);
    if (rows == 0 || cols == 0) return 1;
    uint64_t nib_cols = (cols + 1u) / 2u;
    if (e->size_bytes != rows * nib_cols) return 1;
    const uint8_t *codes = (const uint8_t *)sp_model_tensor_data(sm, e);
    if (!codes) return 1;

    char sn[96];
    snprintf(sn, sizeof sn, "%s.scale", name);
    const sp_tensor_entry *se = sp_model_find_tensor(sm, sn);
    if (!se || se->dtype_id != (uint32_t)SP_DT_FROBENIUS_SCALE_FP32 ||
        se->dims[0] != rows || se->size_bytes != rows * 4u) return 1;
    const float *scale = (const float *)sp_model_tensor_data(sm, se);
    if (!scale) return 1;

    out->rows = (int)rows; out->cols = (int)cols; out->codes_bytes = (size_t)(rows * nib_cols);
    out->row_prec = (uint8_t *)malloc((size_t)rows);
    out->row_off  = (size_t  *)malloc((size_t)rows * sizeof(size_t));
    if (!out->row_prec || !out->row_off) {
        sp_frob_packed_free(out); return 1;
    }
    out->alias_mask = 0x3;
    out->codes     = (uint8_t *)(uintptr_t)codes;
    out->row_scale = (float   *)(uintptr_t)scale;
    for (uint64_t r = 0; r < rows; r++) {
        out->row_prec[r] = 4u;
        out->row_off[r]  = (size_t)(r * nib_cols);
    }
    return 0;
}

/* Build an OK_Q4B packed tensor (SPEC OK_Q4B, arena layout v2): nibble codes as
 * per-row Q4 PLUS per-32-block f16 scales aliased from the ".bscale" sibling.
 * row_scale stays NULL — the block scale IS the step size (store-then-derive). */
static int build_packed_q4b(const sp_model *sm, const char *name, sp_frob_packed_tensor *out) {
    memset(out, 0, sizeof *out);
    const sp_tensor_entry *e = sp_model_find_tensor(sm, name);
    if (!e || e->dtype_id != (uint32_t)SP_DT_OK_Q4B || e->n_dims < 2) return 1;
    uint64_t cols = e->dims[0], rows = e->dims[1] * (e->n_dims >= 3 ? e->dims[2] : 1u);
    if (rows == 0 || cols == 0) return 1;
    uint64_t nib_cols = (cols + 1u) / 2u;
    uint64_t nblk     = (cols + 31u) / 32u;
    if (e->size_bytes != rows * nib_cols) return 1;
    const uint8_t *codes = (const uint8_t *)sp_model_tensor_data(sm, e);
    if (!codes) return 1;

    char sn[96];
    snprintf(sn, sizeof sn, "%s.bscale", name);
    const sp_tensor_entry *se = sp_model_find_tensor(sm, sn);
    if (!se || se->dtype_id != (uint32_t)SP_DT_BLOCK_SCALE_FP16 ||
        se->size_bytes != rows * nblk * 2u) return 1;
    const uint16_t *bs = (const uint16_t *)sp_model_tensor_data(sm, se);
    if (!bs) return 1;

    out->rows = (int)rows; out->cols = (int)cols; out->codes_bytes = (size_t)(rows * nib_cols);
    out->row_prec = (uint8_t *)malloc((size_t)rows);
    out->row_off  = (size_t  *)malloc((size_t)rows * sizeof(size_t));
    if (!out->row_prec || !out->row_off) { sp_frob_packed_free(out); return 1; }
    out->alias_mask = 0x7;                          /* codes + (absent) row_scale + bscale aliased */
    out->codes     = (uint8_t *)(uintptr_t)codes;
    out->row_scale = NULL;
    out->bscale    = bs;
    out->bs_nblk   = (int)nblk;
    for (uint64_t r = 0; r < rows; r++) {
        out->row_prec[r] = 4u;
        out->row_off[r]  = (size_t)(r * nib_cols);
    }
    return 0;
}

/* Dispatch to Q8 / Q4 / Q4B builder depending on the tensor's on-disk dtype. */
static int build_packed(const sp_model *sm, const char *name, sp_frob_packed_tensor *out) {
    const sp_tensor_entry *e = sp_model_find_tensor(sm, name);
    if (!e || e->n_dims < 2) { memset(out, 0, sizeof *out); return 1; }
    if (e->dtype_id == (uint32_t)SP_DT_OK_Q4)  return build_packed_q4(sm, name, out);
    if (e->dtype_id == (uint32_t)SP_DT_OK_Q4B) return build_packed_q4b(sm, name, out);
    return build_packed_q8(sm, name, out);
}

/* Copy an F32 norm tensor `name` into a fresh owned buffer. Returns 0 on success. */
static int copy_norm(const sp_model *sm, const char *name, float **out_buf) {
    *out_buf = NULL;
    const sp_tensor_entry *e = sp_model_find_tensor(sm, name);
    if (!e || e->dtype_id != (uint32_t)SP_DT_F32 || e->size_bytes == 0 || (e->size_bytes % 4u) != 0u) return 1;
    const void *src = sp_model_tensor_data(sm, e);
    if (!src) return 1;
    float *b = (float *)malloc((size_t)e->size_bytes);
    if (!b) return 1;
    memcpy(b, src, (size_t)e->size_bytes);
    *out_buf = b;
    return 0;
}

struct qwen3_model *sp_model_to_gemma3(const sp_model *sm) {
    gguf_tensor       *synth  = NULL;
    qwen3_layer       *layers = NULL;
    sp_arena_tensor   *ats    = NULL;
    const gguf_tensor **nsrc  = NULL;
    float            **nbuf   = NULL;
    sp_arena          *arena  = NULL;
    qwen3_model       *qm     = NULL;
    int si = 0, ari = 0, ni = 0, embd_syn = 0, onorm_syn = 0;
    char nm[96];

    if (!sm) { sp_set_error("sp_model_to_gemma3: null handle"); return NULL; }
    sp_arch_info ai;
    if (sp_model_arch(sm, &ai) != SP_OK) { sp_set_error("sp_model_to_gemma3: arch query failed"); return NULL; }
    if (ai.arch_id != (uint32_t)SP_ARCH_ID_GEMMA3) { sp_set_error("sp_model_to_gemma3: model is not Gemma3"); return NULL; }
    const uint32_t NL = ai.n_layers;
    if (NL == 0) { sp_set_error("sp_model_to_gemma3: zero layers"); return NULL; }

    uint32_t n_ff = ai.n_ff;
    if (n_ff == 0) {
        const sp_tensor_entry *fg = sp_model_find_tensor(sm, "blk.0.ffn_gate.weight");
        if (!fg || fg->n_dims < 2) { sp_set_error("sp_model_to_gemma3: n_ff unknown"); return NULL; }
        n_ff = (uint32_t)fg->dims[1];
    }

    const int has_output_w = (sp_model_find_tensor(sm, "output.weight") != NULL) ? 1 : 0;

    /* 13 synth tensors per layer: 7 matmul + 6 norms (attn_norm, attn_q_norm, attn_k_norm,
     * post_attn_norm, ffn_norm, post_ffw_norm). NARENA = 7/layer + embd + optional output.
     * NNORM = 6/layer + output_norm. */
    const int NSYN   = 2 + has_output_w + 13 * (int)NL;
    const int NARENA = 1 + has_output_w + 7  * (int)NL;
    const int NNORM  = 1 + 6 * (int)NL;

    synth  = (gguf_tensor *)calloc((size_t)NSYN, sizeof(gguf_tensor));
    layers = (qwen3_layer *)calloc((size_t)NL, sizeof(qwen3_layer));
    ats    = (sp_arena_tensor *)calloc((size_t)NARENA, sizeof(sp_arena_tensor));
    nsrc   = (const gguf_tensor **)calloc((size_t)NNORM, sizeof(*nsrc));
    nbuf   = (float **)calloc((size_t)NNORM, sizeof(*nbuf));
    if (!synth || !layers || !ats || !nsrc || !nbuf) { sp_set_error("sp_model_to_gemma3: out of memory"); goto fail; }

    /* Set the synth tensor's name AND copy its dims from the .sp-model entry, so
     * gemma4_forward can recover per-layer geometry (per-layer n_ff via
     * ffn_gate->dims[1] — Gemma4 elastic FFN). Harmless for the other arches. */
    #define SYNTH_NAME(idx, str) do { \
        snprintf(synth[(idx)].name, sizeof synth[(idx)].name, "%s", (str)); \
        const sp_tensor_entry *e_ = sp_model_find_tensor(sm, (str)); \
        if (e_) { synth[(idx)].n_dims = e_->n_dims; \
            for (uint32_t d_ = 0; d_ < e_->n_dims && d_ < 8u; d_++) synth[(idx)].dims[d_] = e_->dims[d_]; } \
    } while (0)
    #define ARENA(wname) do { \
        if (build_packed(sm, (wname), &ats[ari].pt)) { \
            char eb_[128]; snprintf(eb_, sizeof eb_, "sp_model_to_gemma3: bad weight %s", (wname)); \
            sp_set_error(eb_); goto fail; } \
        snprintf(ats[ari].name, sizeof ats[ari].name, "%s", (wname)); ari++; \
    } while (0)

    embd_syn = si;
    SYNTH_NAME(si, "token_embd.weight"); si++;
    ARENA("token_embd.weight");

    int out_syn = embd_syn;
    if (has_output_w) {
        out_syn = si;
        SYNTH_NAME(si, "output.weight"); si++;
        ARENA("output.weight");
    }

    for (uint32_t L = 0; L < NL; L++) {
        qwen3_layer *ly = &layers[L];
        #define MM(field, suffix) do { \
            snprintf(nm, sizeof nm, "blk.%u." suffix, L); \
            SYNTH_NAME(si, nm); ly->field = &synth[si]; si++; \
            ARENA(nm); \
        } while (0)
        #define NORM(field, suffix) do { \
            snprintf(nm, sizeof nm, "blk.%u." suffix, L); \
            SYNTH_NAME(si, nm); ly->field = &synth[si]; \
            if (copy_norm(sm, nm, &nbuf[ni])) { sp_set_error("sp_model_to_gemma3: bad norm " suffix); goto fail; } \
            nsrc[ni] = &synth[si]; si++; ni++; \
        } while (0)

        NORM(attn_norm,      "attn_norm.weight");
        MM  (attn_q,         "attn_q.weight");
        MM  (attn_k,         "attn_k.weight");
        MM  (attn_v,         "attn_v.weight");
        MM  (attn_output,    "attn_output.weight");
        NORM(attn_q_norm,    "attn_q_norm.weight");
        NORM(attn_k_norm,    "attn_k_norm.weight");
        NORM(post_attn_norm, "post_attention_norm.weight");
        NORM(ffn_norm,       "ffn_norm.weight");
        MM  (ffn_gate,       "ffn_gate.weight");
        MM  (ffn_up,         "ffn_up.weight");
        MM  (ffn_down,       "ffn_down.weight");
        NORM(post_ffw_norm,  "post_ffw_norm.weight");
        #undef MM
        #undef NORM
    }

    onorm_syn = si;
    SYNTH_NAME(si, "output_norm.weight"); si++;
    if (copy_norm(sm, "output_norm.weight", &nbuf[ni])) { sp_set_error("sp_model_to_gemma3: bad output_norm"); goto fail; }
    nsrc[ni] = &synth[onorm_syn]; ni++;
    #undef SYNTH_NAME
    #undef ARENA

    int arena_prec = (ari > 0 && ats[0].pt.rows > 0 && ats[0].pt.row_prec[0] == 4u) ? 4 : 8;
    arena = sp_arena_from_packed(ats, NARENA, arena_prec);
    if (!arena) { sp_set_error("sp_model_to_gemma3: arena adoption failed"); goto fail; }
    free(ats); ats = NULL; ari = 0;

    qm = (qwen3_model *)calloc(1, sizeof(*qm));
    if (!qm) { sp_set_error("sp_model_to_gemma3: out of memory (model)"); sp_arena_free(arena); arena = NULL; goto fail; }

    qwen3_config *c = &qm->cfg;
    c->arch           = SP_ARCH_GEMMA3;
    c->n_layers       = NL;
    c->n_embd         = ai.hidden_dim;
    c->n_ff           = n_ff;
    c->n_head         = ai.n_heads;
    c->n_head_kv      = ai.n_kv_heads;
    c->head_dim       = ai.head_dim;
    c->n_vocab        = ai.vocab_size;
    c->context_length = ai.max_context;
    c->sliding_window = ai.swa_window;
    c->rope_freq_base = ai.rope_freq_base;
    c->rms_eps        = (ai.rms_eps != 0.0f) ? ai.rms_eps : 1e-6f;
    c->has_qk_norm    = 1;
    c->tied_embedding = has_output_w ? 0 : 1;

    qm->gguf          = NULL;
    qm->layers        = layers;
    qm->synth_tensors = synth;
    qm->arena         = arena;
    qm->released      = 1;
    qm->norm_src      = nsrc;
    qm->norm_buf      = nbuf;
    qm->n_norm        = ni;
    qm->token_embd    = &synth[embd_syn];
    qm->output_norm   = &synth[onorm_syn];
    qm->output        = has_output_w ? &synth[out_syn] : qm->token_embd;
    return qm;

fail:
    if (ats) { for (int i = 0; i < ari; i++) sp_frob_packed_free(&ats[i].pt); }
    for (int i = 0; i < ni; i++) free(nbuf[i]);
    free(ats); free(nsrc); free(nbuf); free(layers); free(synth);
    return NULL;
}

struct qwen3_model *sp_model_to_qwen3(const sp_model *sm) {
    /* all cleanup-relevant state declared + zeroed up front (goto-safe) */
    gguf_tensor       *synth  = NULL;
    qwen3_layer       *layers = NULL;
    sp_arena_tensor   *ats    = NULL;
    const gguf_tensor **nsrc  = NULL;
    float            **nbuf   = NULL;
    sp_arena          *arena  = NULL;
    qwen3_model       *qm     = NULL;
    int si = 0, ari = 0, ni = 0, embd_syn = 0, onorm_syn = 0;
    char nm[96];

    if (!sm) { sp_set_error("sp_model_to_qwen3: null handle"); return NULL; }
    sp_arch_info ai;
    if (sp_model_arch(sm, &ai) != SP_OK) { sp_set_error("sp_model_to_qwen3: arch query failed"); return NULL; }
    if (ai.arch_id != (uint32_t)SP_ARCH_ID_QWEN3) { sp_set_error("sp_model_to_qwen3: model is not Qwen3"); return NULL; }
    const uint32_t NL = ai.n_layers;
    if (NL == 0) { sp_set_error("sp_model_to_qwen3: zero layers"); return NULL; }

    /* n_ff (2-L1.FP16): prefer arch_struct.n_ff; fall back to the ffn_gate out-dim for
     * old .sp-model files that predate the field (n_ff == 0). */
    uint32_t n_ff = ai.n_ff;
    if (n_ff == 0) {
        const sp_tensor_entry *fg = sp_model_find_tensor(sm, "blk.0.ffn_gate.weight");
        if (!fg || fg->n_dims < 2) {
            sp_set_error("sp_model_to_qwen3: arch_struct.n_ff==0 and blk.0.ffn_gate.weight missing");
            return NULL;
        }
        n_ff = (uint32_t)fg->dims[1];
    }

    /* Untied LM head: present in the .sp-model as "output.weight". Detect by tensor
     * presence rather than arch_struct.tied_embeddings for robustness. */
    const int has_output_w = (sp_model_find_tensor(sm, "output.weight") != NULL) ? 1 : 0;

    const int NSYN   = 2 + has_output_w + 11 * (int)NL;
    const int NARENA = 1 + has_output_w + 7  * (int)NL;
    const int NNORM  = 1 + 4 * (int)NL;

    synth  = (gguf_tensor *)calloc((size_t)NSYN, sizeof(gguf_tensor));
    layers = (qwen3_layer *)calloc((size_t)NL, sizeof(qwen3_layer));
    ats    = (sp_arena_tensor *)calloc((size_t)NARENA, sizeof(sp_arena_tensor));
    nsrc   = (const gguf_tensor **)calloc((size_t)NNORM, sizeof(*nsrc));
    nbuf   = (float **)calloc((size_t)NNORM, sizeof(*nbuf));
    if (!synth || !layers || !ats || !nsrc || !nbuf) { sp_set_error("sp_model_to_qwen3: out of memory"); goto fail; }

    /* Set the synth tensor's name AND copy its dims from the .sp-model entry, so
     * gemma4_forward can recover per-layer geometry (per-layer n_ff via
     * ffn_gate->dims[1] — Gemma4 elastic FFN). Harmless for the other arches. */
    #define SYNTH_NAME(idx, str) do { \
        snprintf(synth[(idx)].name, sizeof synth[(idx)].name, "%s", (str)); \
        const sp_tensor_entry *e_ = sp_model_find_tensor(sm, (str)); \
        if (e_) { synth[(idx)].n_dims = e_->n_dims; \
            for (uint32_t d_ = 0; d_ < e_->n_dims && d_ < 8u; d_++) synth[(idx)].dims[d_] = e_->dims[d_]; } \
    } while (0)
    #define ARENA(wname) do { \
        if (build_packed(sm, (wname), &ats[ari].pt)) { \
            char eb_[128]; snprintf(eb_, sizeof eb_, "sp_model_to_qwen3: bad weight %s", (wname)); \
            sp_set_error(eb_); goto fail; } \
        snprintf(ats[ari].name, sizeof ats[ari].name, "%s", (wname)); ari++; \
    } while (0)

    /* token_embd (embedding; LM head tied or separate below) */
    embd_syn = si;
    SYNTH_NAME(si, "token_embd.weight"); si++;
    ARENA("token_embd.weight");

    /* untied LM head: pack output.weight as a separate arena entry */
    int out_syn = embd_syn;
    if (has_output_w) {
        out_syn = si;
        SYNTH_NAME(si, "output.weight"); si++;
        ARENA("output.weight");
    }

    for (uint32_t L = 0; L < NL; L++) {
        qwen3_layer *ly = &layers[L];
        #define MM(field, suffix) do { \
            snprintf(nm, sizeof nm, "blk.%u." suffix, L); \
            SYNTH_NAME(si, nm); ly->field = &synth[si]; si++; \
            ARENA(nm); \
        } while (0)
        #define NORM(field, suffix) do { \
            snprintf(nm, sizeof nm, "blk.%u." suffix, L); \
            SYNTH_NAME(si, nm); ly->field = &synth[si]; \
            if (copy_norm(sm, nm, &nbuf[ni])) { sp_set_error("sp_model_to_qwen3: bad norm " suffix); goto fail; } \
            nsrc[ni] = &synth[si]; si++; ni++; \
        } while (0)

        NORM(attn_norm,   "attn_norm.weight");
        MM  (attn_q,      "attn_q.weight");
        MM  (attn_k,      "attn_k.weight");
        MM  (attn_v,      "attn_v.weight");
        MM  (attn_output, "attn_output.weight");
        NORM(attn_q_norm, "attn_q_norm.weight");
        NORM(attn_k_norm, "attn_k_norm.weight");
        NORM(ffn_norm,    "ffn_norm.weight");
        MM  (ffn_gate,    "ffn_gate.weight");
        MM  (ffn_up,      "ffn_up.weight");
        MM  (ffn_down,    "ffn_down.weight");
        #undef MM
        #undef NORM
    }

    /* output_norm */
    onorm_syn = si;
    SYNTH_NAME(si, "output_norm.weight"); si++;
    if (copy_norm(sm, "output_norm.weight", &nbuf[ni])) { sp_set_error("sp_model_to_qwen3: bad output_norm"); goto fail; }
    nsrc[ni] = &synth[onorm_syn]; ni++;
    #undef SYNTH_NAME
    #undef ARENA

    /* adopt the packed matmul weights (arena takes ownership of the pt buffers).
     * Precision is derived from the first arena tensor's row_prec (4 for Q4, 8 for Q8). */
    int arena_prec = (ari > 0 && ats[0].pt.rows > 0 && ats[0].pt.row_prec[0] == 4u) ? 4 : 8;
    arena = sp_arena_from_packed(ats, NARENA, arena_prec);
    if (!arena) { sp_set_error("sp_model_to_qwen3: arena adoption failed"); goto fail; }
    free(ats); ats = NULL; ari = 0;   /* pt buffers now owned by the arena */

    qm = (qwen3_model *)calloc(1, sizeof(*qm));
    if (!qm) { sp_set_error("sp_model_to_qwen3: out of memory (model)"); sp_arena_free(arena); arena = NULL; goto fail; }

    qwen3_config *c = &qm->cfg;
    c->arch           = SP_ARCH_QWEN3;
    c->n_layers       = NL;
    c->n_embd         = ai.hidden_dim;
    c->n_ff           = n_ff;
    c->n_head         = ai.n_heads;
    c->n_head_kv      = ai.n_kv_heads;
    c->head_dim       = ai.head_dim;
    c->n_vocab        = ai.vocab_size;
    c->context_length = ai.max_context;
    c->sliding_window = ai.swa_window;
    c->rope_freq_base = ai.rope_freq_base;
    if (ai.rms_eps != 0.0f) {                  /* 2-L1.FP16: prefer arch_struct.rms_eps */
        c->rms_eps = ai.rms_eps;
    } else {                                   /* old .sp-model (field absent) -> default + warn */
        c->rms_eps = 1e-6f;
        fprintf(stderr, "[sp_model_to_qwen3] warning: arch_struct.rms_eps == 0; defaulting to 1e-6 "
                        "(model predates the rms_eps field)\n");
    }
    c->has_qk_norm    = ai.has_qk_norm ? 1 : 0;
    c->tied_embedding = has_output_w ? 0 : 1;

    qm->gguf          = NULL;
    qm->layers        = layers;
    qm->synth_tensors = synth;
    qm->arena         = arena;
    qm->released      = 1;
    qm->norm_src      = nsrc;
    qm->norm_buf      = nbuf;
    qm->n_norm        = ni;
    qm->token_embd    = &synth[embd_syn];
    qm->output_norm   = &synth[onorm_syn];
    qm->output        = has_output_w ? &synth[out_syn] : qm->token_embd;
    return qm;

fail:
    if (ats) { for (int i = 0; i < ari; i++) sp_frob_packed_free(&ats[i].pt); }
    for (int i = 0; i < ni; i++) free(nbuf[i]);
    free(ats); free(nsrc); free(nbuf); free(layers); free(synth);
    return NULL;
}

struct qwen3_model *sp_model_to_qwen25(const sp_model *sm) {
    gguf_tensor       *synth  = NULL;
    qwen3_layer       *layers = NULL;
    sp_arena_tensor   *ats    = NULL;
    const gguf_tensor **nsrc  = NULL;
    float            **nbuf   = NULL;
    sp_arena          *arena  = NULL;
    qwen3_model       *qm     = NULL;
    int si = 0, ari = 0, ni = 0, embd_syn = 0, onorm_syn = 0;
    char nm[96];

    if (!sm) { sp_set_error("sp_model_to_qwen25: null handle"); return NULL; }
    sp_arch_info ai;
    if (sp_model_arch(sm, &ai) != SP_OK) { sp_set_error("sp_model_to_qwen25: arch query failed"); return NULL; }
    if (ai.arch_id != (uint32_t)SP_ARCH_ID_QWEN25) { sp_set_error("sp_model_to_qwen25: model is not Qwen2.5"); return NULL; }
    const uint32_t NL = ai.n_layers;
    if (NL == 0) { sp_set_error("sp_model_to_qwen25: zero layers"); return NULL; }

    uint32_t n_ff = ai.n_ff;
    if (n_ff == 0) {
        const sp_tensor_entry *fg = sp_model_find_tensor(sm, "blk.0.ffn_gate.weight");
        if (!fg || fg->n_dims < 2) { sp_set_error("sp_model_to_qwen25: n_ff unknown"); return NULL; }
        n_ff = (uint32_t)fg->dims[1];
    }

    const int has_output_w = (sp_model_find_tensor(sm, "output.weight") != NULL) ? 1 : 0;

    /* 12 synth tensors per layer: 7 matmul (attn_q/k/v/output, ffn_gate/up/down)
     * + 5 F32 (attn_norm, attn_q.bias, attn_k.bias, attn_v.bias, ffn_norm).
     * NARENA = 7/layer + embd + optional output.
     * NNORM = 5/layer + output_norm. */
    const int NSYN   = 2 + has_output_w + 12 * (int)NL;
    const int NARENA = 1 + has_output_w + 7  * (int)NL;
    const int NNORM  = 1 + 5 * (int)NL;

    synth  = (gguf_tensor *)calloc((size_t)NSYN, sizeof(gguf_tensor));
    layers = (qwen3_layer *)calloc((size_t)NL, sizeof(qwen3_layer));
    ats    = (sp_arena_tensor *)calloc((size_t)NARENA, sizeof(sp_arena_tensor));
    nsrc   = (const gguf_tensor **)calloc((size_t)NNORM, sizeof(*nsrc));
    nbuf   = (float **)calloc((size_t)NNORM, sizeof(*nbuf));
    if (!synth || !layers || !ats || !nsrc || !nbuf) { sp_set_error("sp_model_to_qwen25: out of memory"); goto fail25; }

    /* Set the synth tensor's name AND copy its dims from the .sp-model entry, so
     * gemma4_forward can recover per-layer geometry (per-layer n_ff via
     * ffn_gate->dims[1] — Gemma4 elastic FFN). Harmless for the other arches. */
    #define SYNTH_NAME(idx, str) do { \
        snprintf(synth[(idx)].name, sizeof synth[(idx)].name, "%s", (str)); \
        const sp_tensor_entry *e_ = sp_model_find_tensor(sm, (str)); \
        if (e_) { synth[(idx)].n_dims = e_->n_dims; \
            for (uint32_t d_ = 0; d_ < e_->n_dims && d_ < 8u; d_++) synth[(idx)].dims[d_] = e_->dims[d_]; } \
    } while (0)
    #define ARENA(wname) do { \
        if (build_packed(sm, (wname), &ats[ari].pt)) { \
            char eb_[128]; snprintf(eb_, sizeof eb_, "sp_model_to_qwen25: bad weight %s", (wname)); \
            sp_set_error(eb_); goto fail25; } \
        snprintf(ats[ari].name, sizeof ats[ari].name, "%s", (wname)); ari++; \
    } while (0)
    #define NORMF(field, str_nm) do { \
        SYNTH_NAME(si, (str_nm)); ly->field = &synth[si]; \
        if (copy_norm(sm, (str_nm), &nbuf[ni])) { sp_set_error("sp_model_to_qwen25: bad norm " #field); goto fail25; } \
        nsrc[ni] = &synth[si]; si++; ni++; \
    } while (0)

    embd_syn = si;
    SYNTH_NAME(si, "token_embd.weight"); si++;
    ARENA("token_embd.weight");

    int out_syn = embd_syn;
    if (has_output_w) {
        out_syn = si;
        SYNTH_NAME(si, "output.weight"); si++;
        ARENA("output.weight");
    }

    for (uint32_t L = 0; L < NL; L++) {
        qwen3_layer *ly = &layers[L];
        #define MM(field, suffix) do { \
            snprintf(nm, sizeof nm, "blk.%u." suffix, L); \
            SYNTH_NAME(si, nm); ly->field = &synth[si]; si++; \
            ARENA(nm); \
        } while (0)

        snprintf(nm, sizeof nm, "blk.%u.attn_norm.weight", L);    NORMF(attn_norm,   nm);
        MM  (attn_q,      "attn_q.weight");
        MM  (attn_k,      "attn_k.weight");
        MM  (attn_v,      "attn_v.weight");
        MM  (attn_output, "attn_output.weight");
        snprintf(nm, sizeof nm, "blk.%u.attn_q.bias", L);         NORMF(attn_q_bias, nm);
        snprintf(nm, sizeof nm, "blk.%u.attn_k.bias", L);         NORMF(attn_k_bias, nm);
        snprintf(nm, sizeof nm, "blk.%u.attn_v.bias", L);         NORMF(attn_v_bias, nm);
        snprintf(nm, sizeof nm, "blk.%u.ffn_norm.weight", L);     NORMF(ffn_norm,    nm);
        MM  (ffn_gate,    "ffn_gate.weight");
        MM  (ffn_up,      "ffn_up.weight");
        MM  (ffn_down,    "ffn_down.weight");
        #undef MM
    }

    onorm_syn = si;
    SYNTH_NAME(si, "output_norm.weight"); si++;
    if (copy_norm(sm, "output_norm.weight", &nbuf[ni])) { sp_set_error("sp_model_to_qwen25: bad output_norm"); goto fail25; }
    nsrc[ni] = &synth[onorm_syn]; ni++;
    #undef SYNTH_NAME
    #undef ARENA
    #undef NORMF

    int arena_prec = (ari > 0 && ats[0].pt.rows > 0 && ats[0].pt.row_prec[0] == 4u) ? 4 : 8;
    arena = sp_arena_from_packed(ats, NARENA, arena_prec);
    if (!arena) { sp_set_error("sp_model_to_qwen25: arena adoption failed"); goto fail25; }
    free(ats); ats = NULL; ari = 0;

    qm = (qwen3_model *)calloc(1, sizeof(*qm));
    if (!qm) { sp_set_error("sp_model_to_qwen25: out of memory (model)"); sp_arena_free(arena); arena = NULL; goto fail25; }

    qwen3_config *c = &qm->cfg;
    c->arch           = SP_ARCH_QWEN25;
    c->n_layers       = NL;
    c->n_embd         = ai.hidden_dim;
    c->n_ff           = n_ff;
    c->n_head         = ai.n_heads;
    c->n_head_kv      = ai.n_kv_heads;
    c->head_dim       = ai.head_dim;
    c->n_vocab        = ai.vocab_size;
    c->context_length = ai.max_context;
    c->sliding_window = 0;
    c->rope_freq_base = (ai.rope_freq_base != 0.0f) ? ai.rope_freq_base : 1e6f;
    c->rms_eps        = (ai.rms_eps != 0.0f) ? ai.rms_eps : 1e-6f;
    c->has_qk_norm    = 0;
    c->tied_embedding = has_output_w ? 0 : 1;

    qm->gguf          = NULL;
    qm->layers        = layers;
    qm->synth_tensors = synth;
    qm->arena         = arena;
    qm->released      = 1;
    qm->norm_src      = nsrc;
    qm->norm_buf      = nbuf;
    qm->n_norm        = ni;
    qm->token_embd    = &synth[embd_syn];
    qm->output_norm   = &synth[onorm_syn];
    qm->output        = has_output_w ? &synth[out_syn] : qm->token_embd;
    return qm;

fail25:
    if (ats) { for (int i = 0; i < ari; i++) sp_frob_packed_free(&ats[i].pt); }
    for (int i = 0; i < ni; i++) free(nbuf[i]);
    free(ats); free(nsrc); free(nbuf); free(layers); free(synth);
    return NULL;
}

struct qwen3_model *sp_model_to_gemma4(const sp_model *sm) {
    gguf_tensor       *synth  = NULL;
    qwen3_layer       *layers = NULL;
    sp_arena_tensor   *ats    = NULL;
    const gguf_tensor **nsrc  = NULL;
    float            **nbuf   = NULL;
    sp_arena          *arena  = NULL;
    qwen3_model       *qm     = NULL;
    int si = 0, ari = 0, ni = 0, embd_syn = 0, onorm_syn = 0;
    int ple_syn = 0, plm_syn = 0, pln_syn = 0, rf_syn = 0;
    char nm[96];

    if (!sm) { sp_set_error("sp_model_to_gemma4: null handle"); return NULL; }
    sp_arch_info ai;
    if (sp_model_arch(sm, &ai) != SP_OK) { sp_set_error("sp_model_to_gemma4: arch query failed"); return NULL; }
    if (ai.arch_id != (uint32_t)SP_ARCH_ID_GEMMA4) { sp_set_error("sp_model_to_gemma4: model is not Gemma4"); return NULL; }
    const uint32_t NL = ai.n_layers;
    if (NL == 0) { sp_set_error("sp_model_to_gemma4: zero layers"); return NULL; }

    uint32_t n_ff = ai.n_ff;
    if (n_ff == 0) {
        const sp_tensor_entry *fg = sp_model_find_tensor(sm, "blk.0.ffn_gate.weight");
        if (!fg || fg->n_dims < 2) { sp_set_error("sp_model_to_gemma4: n_ff unknown"); return NULL; }
        n_ff = (uint32_t)fg->dims[1];
    }

    const int has_output_w = (sp_model_find_tensor(sm, "output.weight") != NULL) ? 1 : 0;
    const int has_ple = (ai.g4_n_embd_per_layer != 0u &&
                         sp_model_find_tensor(sm, "per_layer_token_embd.weight") != NULL) ? 1 : 0;
    /* DENSE gemma-4 (e.g. 12B): NO AltUp/PLE (PL=0), but it still carries
     * layer_output_scale per layer AND the global-layer rope_freqs table —
     * both are INDEPENDENT of the E-series MatFormer machinery. Key on the
     * tensors, not on has_ple (the oracle applies them whenever present). */
    const int has_oscale = (sp_model_find_tensor(sm, "blk.0.layer_output_scale.weight") != NULL) ? 1 : 0;
    const int has_rfreq  = (sp_model_find_tensor(sm, "rope_freqs.weight") != NULL) ? 1 : 0;

    /* Per layer: 9 matmul (attn q/k/v/output, ffn gate/up/down, per_layer inp_gate/proj)
     * + 8 norms (attn_norm, attn_q_norm, attn_k_norm, post_attention_norm, ffn_norm,
     * post_ffw_norm, post_norm=per_layer_post_norm, layer_output_scale). Globals:
     * token_embd (+output) + per_layer_token_embd + per_layer_model_proj (arena);
     * per_layer_proj_norm + rope_freqs + output_norm (norms). */
    const int PLM = has_ple ? 2 : 0;                 /* arena: per_layer_token_embd + per_layer_model_proj */
    const int PLN = (has_ple ? 1 : 0) + has_rfreq;   /* norms: per_layer_proj_norm + rope_freqs            */
    const int MMPL = has_ple ? 9 : 7;                /* matmul per layer (incl per_layer inp_gate/proj)    */
    const int NMPL = (has_ple ? 7 : 6) + has_oscale; /* norms/layer (incl per_layer_post_norm + out_scale) */
    const int NSYN   = 2 + has_output_w + PLM + PLN + (MMPL + NMPL) * (int)NL;
    const int NARENA = 1 + has_output_w + PLM + MMPL * (int)NL;
    const int NNORM  = 1 + PLN + NMPL * (int)NL;

    synth  = (gguf_tensor *)calloc((size_t)NSYN, sizeof(gguf_tensor));
    layers = (qwen3_layer *)calloc((size_t)NL, sizeof(qwen3_layer));
    ats    = (sp_arena_tensor *)calloc((size_t)NARENA, sizeof(sp_arena_tensor));
    nsrc   = (const gguf_tensor **)calloc((size_t)NNORM, sizeof(*nsrc));
    nbuf   = (float **)calloc((size_t)NNORM, sizeof(*nbuf));
    if (!synth || !layers || !ats || !nsrc || !nbuf) { sp_set_error("sp_model_to_gemma4: out of memory"); goto fail4; }

    /* Set the synth tensor's name AND copy its dims from the .sp-model entry, so
     * gemma4_forward can recover per-layer geometry (per-layer n_ff via
     * ffn_gate->dims[1] — Gemma4 elastic FFN). Harmless for the other arches. */
    #define SYNTH_NAME(idx, str) do { \
        snprintf(synth[(idx)].name, sizeof synth[(idx)].name, "%s", (str)); \
        const sp_tensor_entry *e_ = sp_model_find_tensor(sm, (str)); \
        if (e_) { synth[(idx)].n_dims = e_->n_dims; \
            for (uint32_t d_ = 0; d_ < e_->n_dims && d_ < 8u; d_++) synth[(idx)].dims[d_] = e_->dims[d_]; } \
    } while (0)
    #define ARENA(wname) do { \
        if (build_packed(sm, (wname), &ats[ari].pt)) { \
            char eb_[128]; snprintf(eb_, sizeof eb_, "sp_model_to_gemma4: bad weight %s", (wname)); \
            sp_set_error(eb_); goto fail4; } \
        snprintf(ats[ari].name, sizeof ats[ari].name, "%s", (wname)); ari++; \
    } while (0)
    #define NORMN(field, str_nm) do { \
        SYNTH_NAME(si, (str_nm)); ly->field = &synth[si]; \
        if (copy_norm(sm, (str_nm), &nbuf[ni])) { sp_set_error("sp_model_to_gemma4: bad norm " #field); goto fail4; } \
        nsrc[ni] = &synth[si]; si++; ni++; \
    } while (0)

    embd_syn = si; SYNTH_NAME(si, "token_embd.weight"); si++; ARENA("token_embd.weight");

    int out_syn = embd_syn;
    if (has_output_w) { out_syn = si; SYNTH_NAME(si, "output.weight"); si++; ARENA("output.weight"); }

    if (has_ple) {
        ple_syn = si; SYNTH_NAME(si, "per_layer_token_embd.weight"); si++; ARENA("per_layer_token_embd.weight");
        plm_syn = si; SYNTH_NAME(si, "per_layer_model_proj.weight"); si++; ARENA("per_layer_model_proj.weight");
        pln_syn = si; SYNTH_NAME(si, "per_layer_proj_norm.weight"); si++;
        if (copy_norm(sm, "per_layer_proj_norm.weight", &nbuf[ni])) { sp_set_error("sp_model_to_gemma4: bad per_layer_proj_norm"); goto fail4; }
        nsrc[ni] = &synth[pln_syn]; ni++;
    }
    if (has_rfreq) {   /* global-layer proportional RoPE factors — dense 12B has them too */
        rf_syn = si; SYNTH_NAME(si, "rope_freqs.weight"); si++;
        if (copy_norm(sm, "rope_freqs.weight", &nbuf[ni])) { sp_set_error("sp_model_to_gemma4: bad rope_freqs"); goto fail4; }
        nsrc[ni] = &synth[rf_syn]; ni++;
    }

    for (uint32_t L = 0; L < NL; L++) {
        qwen3_layer *ly = &layers[L];
        #define MM(field, suffix) do { \
            snprintf(nm, sizeof nm, "blk.%u." suffix, L); \
            SYNTH_NAME(si, nm); ly->field = &synth[si]; si++; ARENA(nm); \
        } while (0)
        #define NORM(field, suffix) do { \
            snprintf(nm, sizeof nm, "blk.%u." suffix, L); \
            SYNTH_NAME(si, nm); ly->field = &synth[si]; \
            if (copy_norm(sm, nm, &nbuf[ni])) { sp_set_error("sp_model_to_gemma4: bad norm " suffix); goto fail4; } \
            nsrc[ni] = &synth[si]; si++; ni++; \
        } while (0)

        NORM(attn_norm,      "attn_norm.weight");
        MM  (attn_q,         "attn_q.weight");
        MM  (attn_k,         "attn_k.weight");
        /* V-less layers (dense gemma-4 globals): attn_v may be ABSENT — the
         * forward then uses the raw K projection as V (llama.cpp gemma4-iswa
         * "use_alternative_attention"). ly->attn_v stays NULL. */
        snprintf(nm, sizeof nm, "blk.%u.attn_v.weight", L);
        if (sp_model_find_tensor(sm, nm)) MM(attn_v, "attn_v.weight");
        MM  (attn_output,    "attn_output.weight");
        NORM(attn_q_norm,    "attn_q_norm.weight");
        NORM(attn_k_norm,    "attn_k_norm.weight");
        NORM(post_attn_norm, "post_attention_norm.weight");
        NORM(ffn_norm,       "ffn_norm.weight");
        MM  (ffn_gate,       "ffn_gate.weight");
        MM  (ffn_up,         "ffn_up.weight");
        MM  (ffn_down,       "ffn_down.weight");
        NORM(post_ffw_norm,  "post_ffw_norm.weight");
        if (has_ple) {
            MM  (per_layer_inp_gate, "inp_gate.weight");
            MM  (per_layer_proj,     "proj.weight");
            NORM(per_layer_post_norm,"post_norm.weight");
        }
        if (has_oscale) NORM(out_scale, "layer_output_scale.weight");
        #undef MM
        #undef NORM
    }

    onorm_syn = si; SYNTH_NAME(si, "output_norm.weight"); si++;
    if (copy_norm(sm, "output_norm.weight", &nbuf[ni])) { sp_set_error("sp_model_to_gemma4: bad output_norm"); goto fail4; }
    nsrc[ni] = &synth[onorm_syn]; ni++;
    #undef SYNTH_NAME
    #undef ARENA
    #undef NORMN

    int arena_prec = (ari > 0 && ats[0].pt.rows > 0 && ats[0].pt.row_prec[0] == 4u) ? 4 : 8;
    arena = sp_arena_from_packed(ats, NARENA, arena_prec);
    if (!arena) { sp_set_error("sp_model_to_gemma4: arena adoption failed"); goto fail4; }
    free(ats); ats = NULL; ari = 0;

    qm = (qwen3_model *)calloc(1, sizeof(*qm));
    if (!qm) { sp_set_error("sp_model_to_gemma4: out of memory (model)"); sp_arena_free(arena); arena = NULL; goto fail4; }

    qwen3_config *c = &qm->cfg;
    c->arch           = SP_ARCH_GEMMA4;
    c->n_layers       = NL;
    c->n_embd         = ai.hidden_dim;
    c->n_ff           = n_ff;
    c->n_head         = ai.n_heads;      /* GLOBAL geometry */
    c->n_head_kv      = ai.n_kv_heads;
    c->head_dim       = ai.head_dim;
    c->n_vocab        = ai.vocab_size;
    c->context_length = ai.max_context;
    c->sliding_window = ai.swa_window;
    c->rope_freq_base = ai.rope_freq_base;
    c->rms_eps        = (ai.rms_eps != 0.0f) ? ai.rms_eps : 1e-6f;
    c->has_qk_norm    = 1;
    c->tied_embedding = has_output_w ? 0 : 1;
    /* Gemma4 extras */
    c->g4_hd_swa           = ai.g4_hd_swa;
    c->g4_nh_swa           = ai.g4_nh_swa;
    c->g4_nkv_swa          = ai.g4_nkv_swa;
    c->g4_rope_base_swa    = ai.g4_rope_base_swa;
    c->g4_n_embd_per_layer = has_ple ? ai.g4_n_embd_per_layer : 0u;
    c->g4_n_kv_from_start  = ai.g4_n_kv_from_start ? ai.g4_n_kv_from_start : NL;
    c->g4_logit_softcap    = ai.g4_logit_softcap;
    c->g4_swa_period       = ai.g4_swa_period ? ai.g4_swa_period : 6u;

    qm->gguf          = NULL;
    qm->layers        = layers;
    qm->synth_tensors = synth;
    qm->arena         = arena;
    qm->released      = 1;
    qm->norm_src      = nsrc;
    qm->norm_buf      = nbuf;
    qm->n_norm        = ni;
    qm->token_embd    = &synth[embd_syn];
    qm->output_norm   = &synth[onorm_syn];
    qm->output        = has_output_w ? &synth[out_syn] : qm->token_embd;
    if (has_ple) {
        qm->per_layer_token_embd = &synth[ple_syn];
        qm->per_layer_model_proj = &synth[plm_syn];
        qm->per_layer_proj_norm  = &synth[pln_syn];
    }
    if (has_rfreq) qm->rope_freqs = &synth[rf_syn];
    return qm;

fail4:
    if (ats) { for (int i = 0; i < ari; i++) sp_frob_packed_free(&ats[i].pt); }
    for (int i = 0; i < ni; i++) free(nbuf[i]);
    free(ats); free(nsrc); free(nbuf); free(layers); free(synth);
    return NULL;
}

/* qwen35moe (Qwen3.6) bridge: const sp_model* -> runnable q36 qwen3_model.
 * Per-layer bifurcation (full-attn iff (L+1)%interval==0, else GDN); MoE FFN on every
 * layer. Q-packed (arena): GDN qkv/gate/alpha/beta/out OR attn q/k/v/output; ffn
 * {gate,up,down}_exps (rank-3) + {gate,up,down}_shexp. Owned-f32 ("norm" copy, served
 * by sp_as_f32): attn_norm, post_attn_norm, ssm_norm/conv1d/a/dt, attn_q/k_norm, AND the
 * F32 router ffn_gate_inp + shared-gate ffn_gate_inp_shexp (the qwen36 forward reads the
 * router via sp_as_f32, not sp_matmul, since there is no GGUF on this path). Rank-3 expert
 * synth dims (n_dims=3) preserved so the forward + arena slice expert e. */
struct qwen3_model *sp_model_to_qwen36(const sp_model *sm) {
    gguf_tensor       *synth  = NULL;
    qwen3_layer       *layers = NULL;
    sp_arena_tensor   *ats    = NULL;
    const gguf_tensor **nsrc  = NULL;
    float            **nbuf   = NULL;
    sp_arena          *arena  = NULL;
    qwen3_model       *qm     = NULL;
    int si = 0, ari = 0, ni = 0, embd_syn = 0, onorm_syn = 0, out_syn = 0;
    char nm[96];

    if (!sm) { sp_set_error("sp_model_to_qwen36: null handle"); return NULL; }
    sp_arch_info ai;
    if (sp_model_arch(sm, &ai) != SP_OK) { sp_set_error("sp_model_to_qwen36: arch query failed"); return NULL; }
    if (ai.arch_id != (uint32_t)SP_ARCH_ID_QWEN36) { sp_set_error("sp_model_to_qwen36: model is not qwen35moe"); return NULL; }
    const uint32_t NL = ai.n_layers;
    if (NL == 0) { sp_set_error("sp_model_to_qwen36: zero layers"); return NULL; }
    const uint32_t interval = ai.q36_full_attn_interval ? ai.q36_full_attn_interval : 4u;

    const int has_output_w = (sp_model_find_tensor(sm, "output.weight") != NULL) ? 1 : 0;

    /* generous upper bounds (GDN layer: 11 arena + 6 layer-norms + 2 moe-f32; full-attn:
     * 10 arena + 4 + 2). Over-allocate; unused calloc entries are harmless. */
    const int NSYN   = 3 + has_output_w + 20 * (int)NL;
    const int NARENA = 2 + has_output_w + 12 * (int)NL;
    const int NNORM  = 2 + 9 * (int)NL;
    synth  = (gguf_tensor *)calloc((size_t)NSYN, sizeof(gguf_tensor));
    layers = (qwen3_layer *)calloc((size_t)NL, sizeof(qwen3_layer));
    ats    = (sp_arena_tensor *)calloc((size_t)NARENA, sizeof(sp_arena_tensor));
    nsrc   = (const gguf_tensor **)calloc((size_t)NNORM, sizeof(*nsrc));
    nbuf   = (float **)calloc((size_t)NNORM, sizeof(*nbuf));
    if (!synth || !layers || !ats || !nsrc || !nbuf) { sp_set_error("sp_model_to_qwen36: out of memory"); goto fail6; }

    #define SYNTH_NAME(idx, str) do { \
        snprintf(synth[(idx)].name, sizeof synth[(idx)].name, "%s", (str)); \
        const sp_tensor_entry *e_ = sp_model_find_tensor(sm, (str)); \
        if (e_) { synth[(idx)].n_dims = e_->n_dims; \
            for (uint32_t d_ = 0; d_ < e_->n_dims && d_ < 8u; d_++) synth[(idx)].dims[d_] = e_->dims[d_]; } \
    } while (0)
    #define ARENA(wname) do { \
        if (build_packed(sm, (wname), &ats[ari].pt)) { \
            char eb_[128]; snprintf(eb_, sizeof eb_, "sp_model_to_qwen36: bad weight %s", (wname)); \
            sp_set_error(eb_); goto fail6; } \
        snprintf(ats[ari].name, sizeof ats[ari].name, "%s", (wname)); ari++; \
    } while (0)

    embd_syn = si; SYNTH_NAME(si, "token_embd.weight"); si++; ARENA("token_embd.weight");
    out_syn = embd_syn;
    if (has_output_w) { out_syn = si; SYNTH_NAME(si, "output.weight"); si++; ARENA("output.weight"); }

    for (uint32_t L = 0; L < NL; L++) {
        qwen3_layer *ly = &layers[L];
        int full_attn = (((L + 1u) % interval) == 0u);
        ly->q36_is_recurrent = !full_attn;
        #define MM(field, suffix) do { \
            snprintf(nm, sizeof nm, "blk.%u." suffix, L); \
            SYNTH_NAME(si, nm); ly->field = &synth[si]; si++; ARENA(nm); \
        } while (0)
        #define NORM(field, suffix) do { \
            snprintf(nm, sizeof nm, "blk.%u." suffix, L); \
            SYNTH_NAME(si, nm); ly->field = &synth[si]; \
            if (copy_norm(sm, nm, &nbuf[ni])) { sp_set_error("sp_model_to_qwen36: bad f32 " suffix); goto fail6; } \
            nsrc[ni] = &synth[si]; si++; ni++; \
        } while (0)

        NORM(attn_norm,      "attn_norm.weight");
        NORM(post_attn_norm, "post_attention_norm.weight");
        if (full_attn) {
            MM  (attn_q,      "attn_q.weight");
            MM  (attn_k,      "attn_k.weight");
            MM  (attn_v,      "attn_v.weight");
            MM  (attn_output, "attn_output.weight");
            NORM(attn_q_norm, "attn_q_norm.weight");
            NORM(attn_k_norm, "attn_k_norm.weight");
        } else {
            MM  (gdn_qkv,     "attn_qkv.weight");
            MM  (gdn_gate,    "attn_gate.weight");
            MM  (gdn_alpha,   "ssm_alpha.weight");
            MM  (gdn_beta,    "ssm_beta.weight");
            MM  (gdn_out,     "ssm_out.weight");
            NORM(gdn_norm,    "ssm_norm.weight");
            NORM(gdn_conv1d,  "ssm_conv1d.weight");
            NORM(gdn_a,       "ssm_a");
            NORM(gdn_dt_bias, "ssm_dt.bias");
        }
        /* MoE FFN — every layer. Router + shared-gate are F32 (owned, read via sp_as_f32). */
        NORM(ffn_gate_inp,       "ffn_gate_inp.weight");
        NORM(ffn_gate_inp_shexp, "ffn_gate_inp_shexp.weight");
        MM  (ffn_gate_exps,      "ffn_gate_exps.weight");
        MM  (ffn_up_exps,        "ffn_up_exps.weight");
        MM  (ffn_down_exps,      "ffn_down_exps.weight");
        MM  (ffn_gate_shexp,     "ffn_gate_shexp.weight");
        MM  (ffn_up_shexp,       "ffn_up_shexp.weight");
        MM  (ffn_down_shexp,     "ffn_down_shexp.weight");
        #undef MM
        #undef NORM
    }

    onorm_syn = si; SYNTH_NAME(si, "output_norm.weight"); si++;
    if (copy_norm(sm, "output_norm.weight", &nbuf[ni])) { sp_set_error("sp_model_to_qwen36: bad output_norm"); goto fail6; }
    nsrc[ni] = &synth[onorm_syn]; ni++;
    #undef SYNTH_NAME
    #undef ARENA

    {
        int arena_prec = (ari > 0 && ats[0].pt.rows > 0 && ats[0].pt.row_prec[0] == 4u) ? 4 : 8;
        arena = sp_arena_from_packed(ats, ari, arena_prec);
    }
    if (!arena) { sp_set_error("sp_model_to_qwen36: arena adoption failed"); goto fail6; }
    free(ats); ats = NULL; ari = 0;

    qm = (qwen3_model *)calloc(1, sizeof(*qm));
    if (!qm) { sp_set_error("sp_model_to_qwen36: out of memory (model)"); sp_arena_free(arena); arena = NULL; goto fail6; }

    qwen3_config *c = &qm->cfg;
    c->arch           = SP_ARCH_QWEN36;
    c->n_layers       = NL;
    c->n_embd         = ai.hidden_dim;
    c->n_head         = ai.n_heads;
    c->n_head_kv      = ai.n_kv_heads;
    c->head_dim       = ai.head_dim;
    c->n_vocab        = ai.vocab_size;
    c->context_length = ai.max_context;
    c->rope_freq_base = ai.rope_freq_base;
    c->rms_eps        = (ai.rms_eps != 0.0f) ? ai.rms_eps : 1e-6f;
    c->has_qk_norm    = 1;
    c->tied_embedding = has_output_w ? 0 : 1;
    c->n_ff           = ai.n_ff;
    c->q36_full_attn_interval  = interval;
    c->q36_n_expert            = ai.q36_n_expert;
    c->q36_n_expert_used       = ai.q36_n_expert_used;
    c->q36_n_ff_exp            = ai.q36_n_ff_exp;
    c->q36_n_ff_shexp          = ai.q36_n_ff_shexp;
    c->q36_expert_weights_scale= ai.q36_expert_weights_scale != 0.0f ? ai.q36_expert_weights_scale : 1.0f;
    c->q36_gdn_conv_k          = ai.q36_gdn_conv_k;
    c->q36_gdn_state           = ai.q36_gdn_state;
    c->q36_gdn_n_k_heads       = ai.q36_gdn_n_k_heads;
    c->q36_gdn_n_v_heads       = ai.q36_gdn_n_v_heads;
    c->q36_gdn_inner           = ai.q36_gdn_inner;
    for (int s = 0; s < 4; s++) c->q36_rope_sections[s] = ai.q36_rope_sections[s];
    c->q36_rope_dim            = ai.q36_rope_dim ? ai.q36_rope_dim : 64u;
    c->q36_rope_base           = ai.q36_rope_base != 0.0f ? ai.q36_rope_base : ai.rope_freq_base;
    c->q36_nextn_predict_layers= ai.q36_nextn_predict_layers;

    qm->gguf          = NULL;
    qm->layers        = layers;
    qm->synth_tensors = synth;
    qm->arena         = arena;
    qm->released      = 1;
    qm->norm_src      = nsrc;
    qm->norm_buf      = nbuf;
    qm->n_norm        = ni;
    qm->token_embd    = &synth[embd_syn];
    qm->output_norm   = &synth[onorm_syn];
    qm->output        = has_output_w ? &synth[out_syn] : qm->token_embd;
    return qm;

fail6:
    if (ats) { for (int i = 0; i < ari; i++) sp_frob_packed_free(&ats[i].pt); }
    for (int i = 0; i < ni; i++) free(nbuf[i]);
    free(ats); free(nsrc); free(nbuf); free(layers); free(synth);
    return NULL;
}

/* DiffusionGemma (arch_id == DIFFUSION_GEMMA, SP_ARCH_DIFFUSION_GEMMA) bridge:
 * const sp_model* -> runnable q36-style MoE qwen3_model. The backbone is the dense
 * gemma4 attention path (per-layer SWA/global geometry via g4_*, V-less global
 * layers omit attn_v, sandwich norms, layer_output_scale, rope_freqs); the FFN is a
 * DENSE shared MLP (ffn_gate/up/down) PLUS a 128-expert MoE (router ffn_gate_inp +
 * FUSED ffn_gate_up_exps rank-3 + ffn_down_exps rank-3, with F32 scale sidecars) and
 * the MoE sandwich norms (pre_ffw_norm_2, post_ffw_norm_1, post_ffw_norm_2). Model-
 * global diffusion surface: the self-conditioning block (self_cond_pre_norm/gate/up/
 * down) + the per-layer encoder output scalar (enc_layer_output_scale). NO AltUp/PLE
 * (g4_n_embd_per_layer == 0). N1a LOADER ONLY: maps every tensor with no missing/shape
 * error; the forward is N1b (deferred, blocked on a CUDA MoE backbone). */
struct qwen3_model *sp_model_to_diffusion_gemma(const sp_model *sm) {
    gguf_tensor       *synth  = NULL;
    qwen3_layer       *layers = NULL;
    sp_arena_tensor   *ats    = NULL;
    const gguf_tensor **nsrc  = NULL;
    float            **nbuf   = NULL;
    sp_arena          *arena  = NULL;
    qwen3_model       *qm     = NULL;
    int si = 0, ari = 0, ni = 0, embd_syn = 0, onorm_syn = 0, out_syn = 0;
    int scp_syn = 0, scg_syn = 0, scu_syn = 0, scd_syn = 0, rf_syn = 0;
    char nm[96];

    if (!sm) { sp_set_error("sp_model_to_diffusion_gemma: null handle"); return NULL; }
    sp_arch_info ai;
    if (sp_model_arch(sm, &ai) != SP_OK) { sp_set_error("sp_model_to_diffusion_gemma: arch query failed"); return NULL; }
    if (ai.arch_id != (uint32_t)SP_ARCH_ID_DIFFUSION_GEMMA) { sp_set_error("sp_model_to_diffusion_gemma: model is not DiffusionGemma"); return NULL; }
    const uint32_t NL = ai.n_layers;
    if (NL == 0) { sp_set_error("sp_model_to_diffusion_gemma: zero layers"); return NULL; }

    uint32_t n_ff = ai.n_ff;
    if (n_ff == 0) {
        const sp_tensor_entry *fg = sp_model_find_tensor(sm, "blk.0.ffn_gate.weight");
        if (!fg || fg->n_dims < 2) { sp_set_error("sp_model_to_diffusion_gemma: n_ff unknown"); return NULL; }
        n_ff = (uint32_t)fg->dims[1];
    }

    const int has_output_w = (sp_model_find_tensor(sm, "output.weight") != NULL) ? 1 : 0;
    const int has_rfreq    = (sp_model_find_tensor(sm, "rope_freqs.weight") != NULL) ? 1 : 0;
    const int has_selfcond = (sp_model_find_tensor(sm, "self_cond_pre_norm.weight") != NULL) ? 1 : 0;

    /* Generous upper bounds (calloc-zeroed over-allocation is harmless; the actual
     * ari/ni counts drive the arena adoption). Per layer: <= 9 arena (4 attn incl
     * optional V + 3 dense MLP + 2 MoE expert) + <= 14 norms (6 backbone + 3 MoE-F32
     * [router + 2 scale sidecars] + 3 sandwich + 2 scalars). Globals: 4 arena
     * (token_embd + 3 self_cond) + 3 norms (output_norm + rope_freqs + self_cond_pre_norm). */
    const int NSYN   = 4 + has_output_w + 4 + 24 * (int)NL;
    const int NARENA = 4 + has_output_w + 9  * (int)NL;
    const int NNORM  = 3 + 14 * (int)NL;
    synth  = (gguf_tensor *)calloc((size_t)NSYN, sizeof(gguf_tensor));
    layers = (qwen3_layer *)calloc((size_t)NL, sizeof(qwen3_layer));
    ats    = (sp_arena_tensor *)calloc((size_t)NARENA, sizeof(sp_arena_tensor));
    nsrc   = (const gguf_tensor **)calloc((size_t)NNORM, sizeof(*nsrc));
    nbuf   = (float **)calloc((size_t)NNORM, sizeof(*nbuf));
    if (!synth || !layers || !ats || !nsrc || !nbuf) { sp_set_error("sp_model_to_diffusion_gemma: out of memory"); goto faildg; }

    #define SYNTH_NAME(idx, str) do { \
        snprintf(synth[(idx)].name, sizeof synth[(idx)].name, "%s", (str)); \
        const sp_tensor_entry *e_ = sp_model_find_tensor(sm, (str)); \
        if (e_) { synth[(idx)].n_dims = e_->n_dims; \
            for (uint32_t d_ = 0; d_ < e_->n_dims && d_ < 8u; d_++) synth[(idx)].dims[d_] = e_->dims[d_]; } \
    } while (0)
    #define ARENA(wname) do { \
        if (build_packed(sm, (wname), &ats[ari].pt)) { \
            char eb_[128]; snprintf(eb_, sizeof eb_, "sp_model_to_diffusion_gemma: bad weight %s", (wname)); \
            sp_set_error(eb_); goto faildg; } \
        snprintf(ats[ari].name, sizeof ats[ari].name, "%s", (wname)); ari++; \
    } while (0)

    embd_syn = si; SYNTH_NAME(si, "token_embd.weight"); si++; ARENA("token_embd.weight");
    out_syn = embd_syn;
    if (has_output_w) { out_syn = si; SYNTH_NAME(si, "output.weight"); si++; ARENA("output.weight"); }

    /* model-global self-conditioning block (arena gate/up/down + an F32 pre-norm) */
    if (has_selfcond) {
        scp_syn = si; SYNTH_NAME(si, "self_cond_pre_norm.weight"); si++;
        if (copy_norm(sm, "self_cond_pre_norm.weight", &nbuf[ni])) { sp_set_error("sp_model_to_diffusion_gemma: bad self_cond_pre_norm"); goto faildg; }
        nsrc[ni] = &synth[scp_syn]; ni++;
        scg_syn = si; SYNTH_NAME(si, "self_cond_gate.weight"); si++; ARENA("self_cond_gate.weight");
        scu_syn = si; SYNTH_NAME(si, "self_cond_up.weight");   si++; ARENA("self_cond_up.weight");
        scd_syn = si; SYNTH_NAME(si, "self_cond_down.weight"); si++; ARENA("self_cond_down.weight");
    }
    if (has_rfreq) {
        rf_syn = si; SYNTH_NAME(si, "rope_freqs.weight"); si++;
        if (copy_norm(sm, "rope_freqs.weight", &nbuf[ni])) { sp_set_error("sp_model_to_diffusion_gemma: bad rope_freqs"); goto faildg; }
        nsrc[ni] = &synth[rf_syn]; ni++;
    }

    for (uint32_t L = 0; L < NL; L++) {
        qwen3_layer *ly = &layers[L];
        #define MM(field, suffix) do { \
            snprintf(nm, sizeof nm, "blk.%u." suffix, L); \
            SYNTH_NAME(si, nm); ly->field = &synth[si]; si++; ARENA(nm); \
        } while (0)
        #define NORM(field, suffix) do { \
            snprintf(nm, sizeof nm, "blk.%u." suffix, L); \
            SYNTH_NAME(si, nm); ly->field = &synth[si]; \
            if (copy_norm(sm, nm, &nbuf[ni])) { sp_set_error("sp_model_to_diffusion_gemma: bad norm " suffix); goto faildg; } \
            nsrc[ni] = &synth[si]; si++; ni++; \
        } while (0)

        /* attention (gemma4 backbone; V-less global layers omit attn_v) */
        NORM(attn_norm,      "attn_norm.weight");
        MM  (attn_q,         "attn_q.weight");
        MM  (attn_k,         "attn_k.weight");
        snprintf(nm, sizeof nm, "blk.%u.attn_v.weight", L);
        if (sp_model_find_tensor(sm, nm)) MM(attn_v, "attn_v.weight");   /* else ly->attn_v stays NULL */
        MM  (attn_output,    "attn_output.weight");
        NORM(attn_q_norm,    "attn_q_norm.weight");
        NORM(attn_k_norm,    "attn_k_norm.weight");
        NORM(post_attn_norm, "post_attention_norm.weight");

        /* dense shared MLP */
        NORM(ffn_norm,       "ffn_norm.weight");
        MM  (ffn_gate,       "ffn_gate.weight");
        MM  (ffn_up,         "ffn_up.weight");
        MM  (ffn_down,       "ffn_down.weight");
        NORM(post_ffw_norm,  "post_ffw_norm.weight");

        /* MoE FFN: F32 router + scale sidecars (read via sp_as_f32) + FUSED rank-3
         * gate|up experts + rank-3 down experts + per-expert down scale */
        NORM(ffn_gate_inp,        "ffn_gate_inp.weight");
        NORM(ffn_gate_inp_scale,  "ffn_gate_inp.scale");
        MM  (ffn_gate_up_exps,    "ffn_gate_up_exps.weight");
        MM  (ffn_down_exps,       "ffn_down_exps.weight");
        NORM(ffn_down_exps_scale, "ffn_down_exps.scale");

        /* MoE sandwich norms + the per-layer encoder output scalar */
        NORM(pre_ffw_norm_2,  "pre_ffw_norm_2.weight");
        NORM(post_ffw_norm_1, "post_ffw_norm_1.weight");
        NORM(post_ffw_norm_2, "post_ffw_norm_2.weight");
        NORM(out_scale,       "layer_output_scale.weight");
        NORM(enc_out_scale,   "enc_layer_output_scale.weight");
        #undef MM
        #undef NORM
    }

    onorm_syn = si; SYNTH_NAME(si, "output_norm.weight"); si++;
    if (copy_norm(sm, "output_norm.weight", &nbuf[ni])) { sp_set_error("sp_model_to_diffusion_gemma: bad output_norm"); goto faildg; }
    nsrc[ni] = &synth[onorm_syn]; ni++;
    #undef SYNTH_NAME
    #undef ARENA

    {
        int arena_prec = (ari > 0 && ats[0].pt.rows > 0 && ats[0].pt.row_prec[0] == 4u) ? 4 : 8;
        arena = sp_arena_from_packed(ats, ari, arena_prec);
    }
    if (!arena) { sp_set_error("sp_model_to_diffusion_gemma: arena adoption failed"); goto faildg; }
    free(ats); ats = NULL; ari = 0;

    qm = (qwen3_model *)calloc(1, sizeof(*qm));
    if (!qm) { sp_set_error("sp_model_to_diffusion_gemma: out of memory (model)"); sp_arena_free(arena); arena = NULL; goto faildg; }

    qwen3_config *c = &qm->cfg;
    c->arch           = SP_ARCH_DIFFUSION_GEMMA;
    c->n_layers       = NL;
    c->n_embd         = ai.hidden_dim;
    c->n_ff           = n_ff;
    c->n_head         = ai.n_heads;      /* GLOBAL geometry */
    c->n_head_kv      = ai.n_kv_heads;
    c->head_dim       = ai.head_dim;
    c->n_vocab        = ai.vocab_size;
    c->context_length = ai.max_context;
    c->sliding_window = ai.swa_window;
    c->rope_freq_base = ai.rope_freq_base;
    c->rms_eps        = (ai.rms_eps != 0.0f) ? ai.rms_eps : 1e-6f;
    c->has_qk_norm    = 1;
    c->tied_embedding = has_output_w ? 0 : 1;
    /* gemma4 backbone extras */
    c->g4_hd_swa           = ai.g4_hd_swa;
    c->g4_nh_swa           = ai.g4_nh_swa;
    c->g4_nkv_swa          = ai.g4_nkv_swa;
    c->g4_rope_base_swa    = ai.g4_rope_base_swa;
    c->g4_n_embd_per_layer = 0u;         /* DiffusionGemma: NO AltUp/PLE */
    c->g4_n_kv_from_start  = ai.g4_n_kv_from_start ? ai.g4_n_kv_from_start : NL;
    c->g4_logit_softcap    = ai.g4_logit_softcap;
    c->g4_swa_period       = ai.g4_swa_period ? ai.g4_swa_period : 6u;
    /* MoE expert counts (reused q36_* fields) + the diffusion canvas split */
    c->q36_n_expert      = ai.q36_n_expert;
    c->q36_n_expert_used = ai.q36_n_expert_used;
    c->q36_n_ff_exp      = ai.q36_n_ff_exp;
    c->dg_canvas_length  = ai.dg_canvas_length;

    qm->gguf          = NULL;
    qm->layers        = layers;
    qm->synth_tensors = synth;
    qm->arena         = arena;
    qm->released      = 1;
    qm->norm_src      = nsrc;
    qm->norm_buf      = nbuf;
    qm->n_norm        = ni;
    qm->token_embd    = &synth[embd_syn];
    qm->output_norm   = &synth[onorm_syn];
    qm->output        = has_output_w ? &synth[out_syn] : qm->token_embd;
    if (has_rfreq)    qm->rope_freqs = &synth[rf_syn];
    if (has_selfcond) {
        qm->self_cond_pre_norm = &synth[scp_syn];
        qm->self_cond_gate     = &synth[scg_syn];
        qm->self_cond_up       = &synth[scu_syn];
        qm->self_cond_down     = &synth[scd_syn];
    }
    return qm;

faildg:
    if (ats) { for (int i = 0; i < ari; i++) sp_frob_packed_free(&ats[i].pt); }
    for (int i = 0; i < ni; i++) free(nbuf[i]);
    free(ats); free(nsrc); free(nbuf); free(layers); free(synth);
    return NULL;
}
