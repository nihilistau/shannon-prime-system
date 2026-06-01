/* model.c — Qwen3/Gemma3 model representation lifecycle: open a GGUF, read the
 * architecture config, and bind every weight tensor (the load/free/release half of
 * sp/model.h). Relocated verbatim (logic) out of the engine's src/forward/model.c.
 *
 * Two deliberate changes vs the engine original: (1) the F16<->F32 / per-row dequant
 * helpers that headed the engine file now live in sp/weight_dtype.h and are included,
 * not re-defined; (2) the engine's qwen3_free called backend device-memory release
 * hooks (sp_cuda/vulkan/hexagon_model_release) under #ifdef SP_ENGINE_WITH_*. Backend
 * lifecycle is an engine/L2 session concern, not the math core's — those hooks have no
 * place here, so the math-core qwen3_free frees only the model's host memory. When the
 * engine consumes this library it must drive backend release from its own teardown.
 *
 * The forward pass and generation declared in sp/model.h are a separate (later)
 * increment; nothing here references them, so the binding builds and links standalone. */
#define _CRT_SECURE_NO_WARNINGS   /* getenv (SP_ARENA) / snprintf are fine here (MSVC C4996) */
#include "sp/model.h"
#include "sp/arena.h"
#include "sp/weight_dtype.h"      /* sp_dequant_row (release-path norm copies) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const gguf_tensor *want(const gguf_ctx *g, const char *name) {
    return gguf_find_tensor(g, name);
}

/* Read a u64 scalar, OR if the key is an integer array (e.g. gemma4's
 * per-layer feed_forward_length arr[i32,n_layer]), read element 0. Gemma4
 * stores some shape metadata as per-layer arrays even though all entries are
 * equal. Returns 1 on success. */
static int get_u64_or_arr0(const gguf_ctx *g, const char *key, uint64_t *out) {
    if (gguf_get_u64(g, key, out)) return 1;
    const gguf_kv *kv = gguf_find_kv(g, key);
    if (!kv || kv->type != GGUF_T_ARRAY || kv->arr_len == 0 || !kv->arr_data) return 0;
    switch (kv->arr_type) {
        case GGUF_T_INT32:  *out = (uint64_t)(uint32_t)((const int32_t  *)kv->arr_data)[0]; return 1;
        case GGUF_T_UINT32: *out = (uint64_t)((const uint32_t *)kv->arr_data)[0]; return 1;
        case GGUF_T_INT64:  *out = (uint64_t)((const int64_t  *)kv->arr_data)[0]; return 1;
        case GGUF_T_UINT64: *out = ((const uint64_t *)kv->arr_data)[0]; return 1;
        case GGUF_T_INT16:  *out = (uint64_t)(uint16_t)((const int16_t *)kv->arr_data)[0]; return 1;
        case GGUF_T_UINT16: *out = (uint64_t)((const uint16_t *)kv->arr_data)[0]; return 1;
        default: return 0;
    }
}

qwen3_model *qwen3_load(const char *path) {
    gguf_ctx *g = gguf_open(path);
    if (!g) return NULL;

    const char *arch = gguf_get_str(g, "general.architecture");
    if (!arch || (strcmp(arch, "qwen3") != 0 && strcmp(arch, "gemma3") != 0 &&
                  strcmp(arch, "gemma4") != 0 && strcmp(arch, "qwen2") != 0)) {
        gguf_close(g); return NULL;
    }

    qwen3_model *m = (qwen3_model *)calloc(1, sizeof(*m));
    if (!m) { gguf_close(g); return NULL; }
    m->gguf = g;
    qwen3_config *c = &m->cfg;
    c->arch = (strcmp(arch, "gemma4") == 0) ? SP_ARCH_GEMMA4 :
              (strcmp(arch, "gemma3") == 0) ? SP_ARCH_GEMMA3 :
              (strcmp(arch, "qwen2")  == 0) ? SP_ARCH_QWEN25 : SP_ARCH_QWEN3;

    /* metadata keys are namespaced by architecture (e.g. "qwen3.block_count" /
     * "gemma3.block_count"); build the prefix once. */
    const char *p = arch;   /* == "qwen3" | "gemma3" */
    char key[96];
    #define K(suffix) (snprintf(key, sizeof key, "%s." suffix, p), key)

    uint64_t v;
    int ok = 1;
    ok &= gguf_get_u64(g, K("block_count"), &v);             c->n_layers  = (uint32_t)v;
    ok &= gguf_get_u64(g, K("embedding_length"), &v);        c->n_embd    = (uint32_t)v;
    ok &= get_u64_or_arr0(g, K("feed_forward_length"), &v);  c->n_ff      = (uint32_t)v;
    ok &= gguf_get_u64(g, K("attention.head_count"), &v);    c->n_head    = (uint32_t)v;
    ok &= gguf_get_u64(g, K("attention.head_count_kv"), &v); c->n_head_kv = (uint32_t)v;
    ok &= gguf_get_u64(g, K("attention.key_length"), &v);    c->head_dim  = (uint32_t)v;
    if (gguf_get_u64(g, K("context_length"), &v)) c->context_length = (uint32_t)v;
    if (gguf_get_u64(g, K("attention.sliding_window"), &v)) c->sliding_window = (uint32_t)v;
    if (!gguf_get_f32(g, K("rope.freq_base"), &c->rope_freq_base)) c->rope_freq_base = 1e6f;
    if (!gguf_get_f32(g, K("attention.layer_norm_rms_epsilon"), &c->rms_eps)) c->rms_eps = 1e-6f;

    /* ── Gemma4 deltas: per-layer SWA/global geometry + AltUp + shared-KV ──
     * The GGUF head_count/head_count_kv/key_length keys hold the GLOBAL geometry
     * (8/1/512); SWA geometry (g4_*) is derived so the Q/K/V projection widths
     * QD/KVD stay constant across layer types. */
    if (c->arch == SP_ARCH_GEMMA4) {
        uint64_t gv;
        /* Real Gemma4 (E2B-Q8_0, llama.cpp @5dcb711) geometry: head_count and
         * head_count_kv are CONSTANT across layer types (8 / 1); only the per-head
         * dim varies — global key_length=512, SWA key_length_swa=256. So the Q/K/V
         * PROJECTION widths differ between layer types (QD_swa=2048 vs QD_global=4096;
         * KVD_swa=256 vs KVD_global=512). This corrects the Stage-1 spec assumption
         * that the projection widths are constant — verified bit-faithfully against
         * the GGUF tensor dims (blk.0.attn_q=[1536,2048] vs blk.4.attn_q=[1536,4096]). */
        const uint32_t g_hd = c->head_dim;
        uint32_t hd_swa = 0;
        if (gguf_get_u64(g, K("attention.key_length_swa"), &gv)) hd_swa = (uint32_t)gv;
        if (hd_swa == 0) hd_swa = g_hd;   /* fallback: same as global */
        c->g4_hd_swa  = hd_swa;
        c->g4_nh_swa  = c->n_head;     /* n_head constant across layer types */
        c->g4_nkv_swa = c->n_head_kv;  /* n_head_kv constant across layer types */
        if (!gguf_get_f32(g, K("rope.freq_base_swa"), &c->g4_rope_base_swa)) c->g4_rope_base_swa = 1e4f;
        if (gguf_get_u64(g, K("embedding_length_per_layer_input"), &gv)) c->g4_n_embd_per_layer = (uint32_t)gv;
        if (!gguf_get_f32(g, K("final_logit_softcapping"), &c->g4_logit_softcap)) c->g4_logit_softcap = 0.0f;
        /* n_kv_from_start = n_layer - shared_kv_layers */
        if (gguf_get_u64(g, K("attention.shared_kv_layers"), &gv))
            c->g4_n_kv_from_start = (c->n_layers > (uint32_t)gv) ? (c->n_layers - (uint32_t)gv) : c->n_layers;
        else
            c->g4_n_kv_from_start = c->n_layers;
        /* swa_period from the sliding_window_pattern bool array (global where
         * pattern[L]==false). Confirm strict periodicity (global at L%period==
         * period-1) before trusting a scalar period; else fail loudly (the
         * forward only models the periodic case). */
        {
            const gguf_kv *kv = gguf_find_kv(g, K("attention.sliding_window_pattern"));
            uint32_t period = 0;
            if (kv && kv->type == GGUF_T_ARRAY && kv->arr_type == GGUF_T_BOOL &&
                kv->arr_len == c->n_layers && kv->arr_data) {
                const uint8_t *pat = (const uint8_t *)kv->arr_data;  /* 1=SWA(true), 0=global(false) */
                /* find first global index */
                int first_global = -1;
                for (uint32_t L = 0; L < c->n_layers; L++)
                    if (!pat[L]) { first_global = (int)L; break; }
                if (first_global >= 0) {
                    period = (uint32_t)first_global + 1u;   /* global at L%period==period-1 */
                    /* verify the whole array matches the periodic model */
                    for (uint32_t L = 0; L < c->n_layers; L++) {
                        int model_global = ((L % period) == period - 1u);
                        int real_global  = !pat[L];
                        if (model_global != real_global) { period = 0; break; }
                    }
                }
            }
            if (period == 0) { qwen3_free(m); return NULL; }  /* non-periodic SWA pattern: surface upstream */
            c->g4_swa_period = period;
        }
    }
    #undef K
    if (!ok) { qwen3_free(m); return NULL; }

    m->token_embd  = want(g, "token_embd.weight");
    m->output_norm = want(g, "output_norm.weight");
    m->output      = want(g, "output.weight");
    if (!m->token_embd || !m->output_norm) { qwen3_free(m); return NULL; }
    if (!m->output) { m->output = m->token_embd; c->tied_embedding = 1; }

    /* vocab from token_embd: dims = [n_embd, n_vocab] (ne0=embd, ne1=vocab) */
    if (m->token_embd->n_dims < 2 || m->token_embd->dims[0] != c->n_embd) { qwen3_free(m); return NULL; }
    c->n_vocab = (uint32_t)m->token_embd->dims[1];

    m->layers = (qwen3_layer *)calloc(c->n_layers, sizeof(qwen3_layer));
    if (!m->layers) { qwen3_free(m); return NULL; }

    char nm[96];
    for (uint32_t i = 0; i < c->n_layers; i++) {
        qwen3_layer *L = &m->layers[i];
        #define BIND(field, suffix) \
            do { snprintf(nm, sizeof nm, "blk.%u." suffix, i); L->field = want(g, nm); } while (0)
        BIND(attn_norm,   "attn_norm.weight");
        BIND(attn_q,      "attn_q.weight");
        BIND(attn_k,      "attn_k.weight");
        BIND(attn_v,      "attn_v.weight");
        BIND(attn_output, "attn_output.weight");
        BIND(attn_q_norm, "attn_q_norm.weight");   /* may be NULL */
        BIND(attn_k_norm, "attn_k_norm.weight");   /* may be NULL */
        BIND(ffn_norm,    "ffn_norm.weight");
        BIND(ffn_gate,    "ffn_gate.weight");
        BIND(ffn_up,      "ffn_up.weight");
        BIND(ffn_down,    "ffn_down.weight");
        if (c->arch == SP_ARCH_GEMMA3) {
            BIND(post_attn_norm, "post_attention_norm.weight");  /* sandwich norms */
            BIND(post_ffw_norm,  "post_ffw_norm.weight");
        }
        if (c->arch == SP_ARCH_GEMMA4) {
            BIND(post_attn_norm, "post_attention_norm.weight");  /* sandwich norms */
            BIND(post_ffw_norm,  "post_ffw_norm.weight");
            BIND(per_layer_inp_gate, "inp_gate.weight");         /* AltUp per-layer-input */
            BIND(per_layer_proj,     "proj.weight");
            BIND(per_layer_post_norm,"post_norm.weight");
            BIND(out_scale,          "layer_output_scale.weight");
        }
        if (c->arch == SP_ARCH_QWEN25) {
            BIND(attn_q_bias, "attn_q.bias");
            BIND(attn_k_bias, "attn_k.bias");
            BIND(attn_v_bias, "attn_v.bias");
        }
        #undef BIND
        if (!L->attn_norm || !L->attn_q || !L->attn_k || !L->attn_v || !L->attn_output ||
            !L->ffn_norm  || !L->ffn_gate || !L->ffn_up || !L->ffn_down) {
            qwen3_free(m); return NULL;
        }
        if (c->arch == SP_ARCH_GEMMA3 &&
            (!L->attn_q_norm || !L->attn_k_norm || !L->post_attn_norm || !L->post_ffw_norm)) {
            qwen3_free(m); return NULL;   /* gemma3 requires sandwich + QK norms */
        }
        if (c->arch == SP_ARCH_GEMMA4 &&
            (!L->attn_q_norm || !L->attn_k_norm || !L->post_attn_norm || !L->post_ffw_norm ||
             !L->per_layer_inp_gate || !L->per_layer_proj || !L->per_layer_post_norm)) {
            qwen3_free(m); return NULL;   /* gemma4 requires sandwich + QK norms + AltUp blocks */
        }
        if (c->arch == SP_ARCH_QWEN25 &&
            (!L->attn_q_bias || !L->attn_k_bias || !L->attn_v_bias)) {
            qwen3_free(m); return NULL;   /* qwen2.5 requires QKV biases */
        }
    }
    /* QK-norm is present iff layer 0 carries it (uniform across layers). */
    m->cfg.has_qk_norm = (m->layers[0].attn_q_norm != NULL && m->layers[0].attn_k_norm != NULL);

    /* Gemma4 model-level globals: AltUp per-layer-input + the shared rope_freqs
     * proportional freq-factor table (global layers). */
    if (c->arch == SP_ARCH_GEMMA4) {
        m->per_layer_token_embd = want(g, "per_layer_token_embd.weight");
        m->per_layer_model_proj = want(g, "per_layer_model_proj.weight");
        m->per_layer_proj_norm  = want(g, "per_layer_proj_norm.weight");
        m->rope_freqs           = want(g, "rope_freqs.weight");
        if (!m->per_layer_token_embd || !m->per_layer_model_proj ||
            !m->per_layer_proj_norm  || !m->rope_freqs) { qwen3_free(m); return NULL; }
    }

    /* Packed-weight arena (roadmap §4.8), env-gated: SP_ARENA=q8|q4. Quantizes
     * the matmul weights once; the forward then lifts inline from the arena.
     * Default unset => NULL => the dequant-on-demand reference path (E_CPU_2). */
    {
        const char *e = getenv("SP_ARENA");
        if (e && (strcmp(e, "q8") == 0 || strcmp(e, "q4") == 0)) {
            int prec = (e[1] == '8') ? 8 : 4;
            const char *pe = getenv("SP_Q4_PROMOTE");
            float promote = pe ? (float)atof(pe) : 0.25f;
            const char *re = getenv("SP_ARENA_RELEASE");
            const char *ee = getenv("SP_ARENA_EMBED");
            int release = (re && re[0] == '1');
            int embed   = release || (ee && ee[0] == '1');   /* release implies embed */
            m->arena = sp_arena_build(m, prec, promote, embed);
            if (!m->arena) { qwen3_free(m); return NULL; }   /* fail loudly, no silent fallback */
            if (release && qwen3_release_source(m)) { qwen3_free(m); return NULL; }
        }
    }

    return m;
}

/* Phase 1b: copy norms to owned f32, then unmap the GGUF data. See sp/model.h. */
int qwen3_release_source(qwen3_model *m) {
    if (!m) return 1;
    if (m->released) return 0;
    if (!m->arena || !m->token_embd) return 1;                 /* need an arena to release into */
    /* the embedding must be packed (else the forward still needs the mapping) */
    if (!sp_arena_find(m->arena, m->token_embd->name)) return 1;

    int cap = (int)m->cfg.n_layers * 7 + 1;
    m->norm_src = (const gguf_tensor **)malloc((size_t)cap * sizeof(*m->norm_src));
    m->norm_buf = (float **)malloc((size_t)cap * sizeof(*m->norm_buf));
    if (!m->norm_src || !m->norm_buf) return 1;
    int k = 0, rc = 0;
    #define COPY_NORM(T) do { \
        const gguf_tensor *t_ = (T); \
        if (t_ && rc == 0) { \
            int len = (int)t_->n_elements; \
            float *b = (float *)malloc((size_t)len * sizeof(float)); \
            if (!b || sp_dequant_row(gguf_tensor_data(m->gguf, t_), t_->type, len, b)) { free(b); rc = 1; } \
            else { m->norm_src[k] = t_; m->norm_buf[k] = b; k++; } \
        } } while (0)
    for (uint32_t i = 0; i < m->cfg.n_layers && rc == 0; i++) {
        const qwen3_layer *L = &m->layers[i];
        COPY_NORM(L->attn_norm); COPY_NORM(L->ffn_norm);
        COPY_NORM(L->attn_q_norm); COPY_NORM(L->attn_k_norm);
        COPY_NORM(L->attn_q_bias); COPY_NORM(L->attn_k_bias); COPY_NORM(L->attn_v_bias);
    }
    COPY_NORM(m->output_norm);
    #undef COPY_NORM
    m->n_norm = k;
    if (rc) return 1;

    gguf_release_data(m->gguf);   /* drop the F16 source; structs/names kept */
    m->released = 1;
    return 0;
}

void qwen3_free(qwen3_model *m) {
    if (!m) return;
    /* Backend device-memory release (the engine's CUDA/Vulkan/Hexagon model caches)
     * is intentionally NOT done here — that lifecycle belongs to the engine/L2 layer
     * that owns the backends; the math-core model frees only its own host memory. */
    for (int i = 0; i < m->n_norm; i++) free(m->norm_buf[i]);
    free(m->norm_buf); free((void *)m->norm_src);
    sp_arena_free(m->arena);
    free(m->layers);
    free(m->synth_tensors);   /* .sp-model adapter synthetic tensors (NULL otherwise) */
    if (m->gguf) gguf_close(m->gguf);
    free(m);
}
