/* sp_session.c — the L1 session ABI (PPT-LAT-L1-ABI-v0 §1/§3/§4, declared in
 * sp/sp_l1.h). The session is the single-thread mutable inference state over an
 * immutable model: it reconstructs the runnable qwen3_model from the const sp_model
 * handle (sp_model_to_qwen3 — NOT a veneer over a GGUF-loaded model, per §8.7.4
 * spec discipline) and drives the relocated reference forward.
 *
 *  - sp_prefill_chunk re-runs the reference forward (qwen3_forward) over the
 *    accumulated history and writes the last position's logits — BIT-EXACT to the
 *    reference forward in deterministic mode (the §8.7.4 first-parity gate).
 *  - sp_decode_step advances a session-owned persistent f32 KV cache by one token
 *    (O(1) incremental; the prompt KV is filled lazily from history on the first
 *    step). Its per-token math is the gate-off f32 path of qwen3_generate_kv, so the
 *    greedy argmax trajectory matches qwen3_generate_kv exactly (the §8.7.4 full gate).
 *    This is the f32 reference path; the Spinor-block KV overlay (SP_KV_SPINOR) is a
 *    production/engine concern, not the L1 session reference.
 *  - sp_session_clone deep-copies history + KV into an independent session (the
 *    spec-decode fork); sp_session_rewind rolls position back (stale KV is refilled).
 *  - Cancellation: the L2-owned atomic is read (relaxed) at the call boundary; a
 *    tripped flag returns SP_ECANCEL before any state mutation (a consistent boundary). */
#include "sp/sp_l1.h"
#include "sp/sp_model.h"
#include "sp/model.h"
#include "sp/forward_dispatch.h"   /* sp_matmul / sp_embed_row / sp_as_f32 */
#include "sp/forward_kernels.h"    /* sp_rmsnorm / sp_rmsnorm_head / sp_rope_neox */
#include "sp/spinor_block.h"       /* 2-L1.PARITY: VHT2+Mobius Spinor KV codec */
#include "sp/sp_error.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SP_SESSION_CTX_FALLBACK 4096u   /* when neither cfg nor arch caps the context */

struct sp_session {
    const sp_model    *model;       /* borrowed: immutable source, for clone re-derivation */
    qwen3_model       *qm;          /* owned: reconstructed runnable model */
    sp_session_config  cfg;         /* immutable for the session's lifetime */
    volatile int      *cancel;      /* borrowed L2-owned atomic; NULL = no cancellation */
    int32_t           *hist;        /* token history [0, pos) */
    size_t             hist_cap;    /* == effective max_context (== kv_cap) */
    size_t             pos;         /* current sequence position */
    uint32_t           n_vocab;
    uint32_t           resolved_precision; /* sp_precision (2-L1.FP16); fixed at create */
    /* persistent decode KV (lazily allocated on first sp_decode_step).
     * kv_mode (2-L1.PARITY): 0 = f32 cache (default); 1 = Spinor block cache (SP_KV_SPINOR);
     * 2 = f32 cache + in-place Spinor roundtrip (SP_KV_SPINOR_REF parity reference). */
    int                kv_mode;
    int                nblk;        /* sp_spinor_blocks_for(head_dim); blocks per head (mode 1) */
    float             *kc, *vc;     /* [n_layers * hist_cap * KVD] (modes 0/2) */
    sp_spinor_block_t *kcb, *vcb;   /* [n_layers * hist_cap * NKV * nblk] (mode 1, compressed) */
    float             *kdec, *vdec; /* [hist_cap * KVD] per-layer window decode scratch (mode 1) */
    size_t             kv_filled;   /* positions [0, kv_filled) hold valid KV */
    /* per-step scratch */
    float *sx, *snx, *sq, *sknew, *svnew, *sao, *sap, *sgg, *sup, *sdn, *ssc;
    /* ── Sprint NTT.5b: optional compute backend (FastRPC dispatcher or other).
     * NULL fields = host path. Set via sp_session_register_compute_backend.
     * Caller owns `compute_backend_handle`; we only re-emit it to the dispatch
     * fn pointers (we never dereference it ourselves). */
    void                       *compute_backend_handle;
    sp_compute_ntt_dispatch_fn  compute_backend_forward;
    sp_compute_ntt_dispatch_fn  compute_backend_inverse;
    /* ── Sprint WIRE-HEX: optional full-forward backend (engine's per-backend
     * gemma3_forward_hexagon / _cuda / _vulkan). NULL = host reference path.
     * Set via sp_session_register_forward_backend. Caller owns the handle. */
    void                       *forward_backend_handle;
    sp_forward_dispatch_fn      forward_backend_fn;
};

static int cancelled(const sp_session *s) { return (s->cancel && *s->cancel) ? 1 : 0; }

/* ── lifecycle ── */
sp_status sp_session_create(const sp_model *m, const sp_session_config *cfg,
                            volatile int *cancel_flag, sp_session **out_session) {
    if (!m || !out_session) { sp_set_error("sp_session_create: null arg"); return SP_EBADARG; }
    *out_session = NULL;

    sp_arch_info ai; memset(&ai, 0, sizeof ai);
    (void)sp_model_arch(m, &ai);

    /* Borrow the model-level hoisted qwen3_model; build + hoist on first create.
     * NOT thread-safe on the first call per model handle — L2 must serialize. */
    qwen3_model *qm = sp_model_borrow_qm(m);
    if (!qm) {
        if (ai.arch_id == (uint32_t)SP_ARCH_ID_GEMMA3)
            qm = sp_model_to_gemma3(m);
        else if (ai.arch_id == (uint32_t)SP_ARCH_ID_QWEN25)
            qm = sp_model_to_qwen25(m);
        else
            qm = sp_model_to_qwen3(m);
        if (!qm) return SP_EBADFORMAT;
        sp_model_store_qm((sp_model *)m, qm, qwen3_free);  /* one-shot hoist; model owns it */
    }

    sp_session *s = (sp_session *)calloc(1, sizeof *s);
    if (!s) { sp_set_error("sp_session_create: OOM"); return SP_ENOMEM; }
    s->model = m; s->qm = qm; s->cancel = cancel_flag; s->n_vocab = qm->cfg.n_vocab;
    if (cfg) s->cfg = *cfg;

    /* 2-L1.FP16 working-precision resolution: override > arch preference > f32. Fixed at
     * create; backend dispatch reads it via sp_session_precision. (math-core stays f32.) */
    {
        uint32_t ovr  = cfg ? cfg->precision_override : 0u;
        uint32_t pref = ai.preferred_precision;
        s->resolved_precision = ovr ? ovr : (pref ? pref : (uint32_t)SP_PRECISION_F32);
    }

    /* 2-L1.PARITY: KV codec mode from cfg->flags (default 0 = persistent f32). */
    {
        uint32_t kvf = cfg ? cfg->flags : 0u;
        s->kv_mode = (kvf & SP_KV_SPINOR) ? 1 : (kvf & SP_KV_SPINOR_REF) ? 2 : 0;
    }

    size_t cap = s->cfg.max_context;
    if (cap == 0) cap = qm->cfg.context_length;
    if (cap == 0) cap = SP_SESSION_CTX_FALLBACK;
    s->hist_cap = cap;
    s->hist = (int32_t *)malloc(cap * sizeof(int32_t));
    if (!s->hist) { free(s); sp_set_error("sp_session_create: OOM (history)"); return SP_ENOMEM; }

    *out_session = s;
    return SP_OK;
}

void sp_session_destroy(sp_session *s) {
    if (!s) return;
    /* s->qm is borrowed from the model handle — freed by sp_model_unload, not here */
    free(s->hist);
    free(s->kc); free(s->vc);
    free(s->kcb); free(s->vcb); free(s->kdec); free(s->vdec);
    free(s->sx); free(s->snx); free(s->sq); free(s->sknew); free(s->svnew);
    free(s->sao); free(s->sap); free(s->sgg); free(s->sup); free(s->sdn); free(s->ssc);
    free(s);
}

sp_status sp_session_position(const sp_session *s, size_t *pos_out) {
    if (!s || !pos_out) { sp_set_error("sp_session_position: null arg"); return SP_EBADARG; }
    *pos_out = s->pos;
    return SP_OK;
}

uint32_t sp_session_precision(const sp_session *s) {
    return s ? s->resolved_precision : (uint32_t)SP_PRECISION_UNSPECIFIED;
}

/* ── prefill (bit-exact reference forward over the accumulated history) ── */
sp_status sp_prefill_chunk(sp_session *s, const int32_t *tokens, size_t n_tokens,
                           float *logits_last, size_t logits_capacity) {
    if (!s || !tokens || !logits_last) { sp_set_error("sp_prefill_chunk: null arg"); return SP_EBADARG; }
    if (n_tokens == 0) { sp_set_error("sp_prefill_chunk: n_tokens == 0"); return SP_EBADARG; }
    if (logits_capacity < s->n_vocab) { sp_set_error("sp_prefill_chunk: logits_capacity < vocab_size"); return SP_EBADARG; }
    if (n_tokens > s->hist_cap - s->pos) { sp_set_error("sp_prefill_chunk: context full"); return SP_ECONTEXT_FULL; }
    if (cancelled(s)) return SP_ECANCEL;

    memcpy(s->hist + s->pos, tokens, n_tokens * sizeof(int32_t));
    const size_t new_len = s->pos + n_tokens;

    float *tmp = (float *)malloc(new_len * (size_t)s->n_vocab * sizeof(float));
    if (!tmp) { sp_set_error("sp_prefill_chunk: OOM (logits scratch)"); return SP_ENOMEM; }
    /* NTT.5c: extract the session's registered compute backend (NTT.5b
     * sp_session_register_compute_backend) and pass it as an explicit
     * triple to the forward's _ex2 entry point. This avoids a circular
     * dependency between the forward and session CMake modules — the
     * forward TU stays ignorant of struct sp_session entirely.
     *
     * The `_ex2` variants thread the triple into sp_pr_bluestein_set_backend
     * inside the NTT-attention overlay when SP_ENGINE_NTT_ATTN=1 + HD ∈
     * {2..256} ∖ {512} + at least one backend pointer is non-NULL. All
     * three NULL = host-only path. */
    void *bh                            = s->compute_backend_handle;
    sp_compute_ntt_dispatch_fn bfwd     = s->compute_backend_forward;
    sp_compute_ntt_dispatch_fn binv     = s->compute_backend_inverse;
    sp_arch_t _arch = s->qm->cfg.arch;
    /* Sprint WIRE-HEX: if a full-forward backend is registered (e.g. the
     * engine's gemma3_forward_hexagon via the daemon's android trampoline),
     * dispatch through IT instead of the math-core reference forward. The
     * backend re-runs the full forward over the accumulated history (same
     * shape as the reference) and writes the full [new_len * n_vocab] logits
     * into `tmp`; we extract the last position the same way either path. */
    int frc;
    if (s->forward_backend_fn) {
        frc = s->forward_backend_fn(s->forward_backend_handle,
                                    (const void *)s->qm,
                                    s->hist, (int)new_len, tmp);
    } else {
        frc = (_arch == SP_ARCH_GEMMA3) ? gemma3_forward_ex2(s->qm, s->hist, (int)new_len, tmp,    bh, bfwd, binv)
            : (_arch == SP_ARCH_QWEN25)  ? qwen25_forward_ex2(s->qm, s->hist, (int)new_len, tmp,    bh, bfwd, binv)
            :                              qwen3_forward_ex2 (s->qm, s->hist, (int)new_len, tmp, NULL, bh, bfwd, binv);
    }
    if (frc != 0) {
        /* Preserve any inner sp_set_error detail from the dispatched fn
         * (e.g. sp_hex_host.c's "hexagon: ..." strings) — only overwrite
         * with a generic message if no detail is present yet. */
        const char *prev = sp_last_error();
        if (!prev || prev[0] == '\0') sp_set_error("sp_prefill_chunk: forward failed");
        free(tmp);
        return SP_EBADSTATE;
    }
    memcpy(logits_last, tmp + (new_len - 1) * (size_t)s->n_vocab, (size_t)s->n_vocab * sizeof(float));
    free(tmp);

    s->pos = new_len;   /* commit only after a successful forward */
    return SP_OK;
}

/* ── decode (persistent-KV, one token) ── */
#define SP_KV_MAX_BLOCKS 16   /* head_dim up to 16*55 = 880 */
/* In-place Spinor encode->decode roundtrip of a length-d head vector (mode-2 parity
 * reference): produces exactly what the block cache decodes back, so mode 1 == mode 2. */
static void spinor_roundtrip(float *vec, int d) {
    sp_spinor_block_t blks[SP_KV_MAX_BLOCKS];
    if (sp_spinor_blocks_for(d) > SP_KV_MAX_BLOCKS) return;
    sp_spinor_encode_vec(vec, d, blks);
    (void)sp_spinor_decode_vec(blks, d, vec);
}

static int ensure_decode_bufs(sp_session *s) {
    if (s->kc || s->kcb) return 0;
    const qwen3_config *c = &s->qm->cfg;
    const int HD = (int)c->head_dim, NKV = (int)c->n_head_kv;
    const size_t KVD = (size_t)NKV * HD;
    const size_t QD  = (size_t)c->n_head * HD;
    const size_t P = s->hist_cap;
    s->nblk = sp_spinor_blocks_for(HD);
    if (s->kv_mode == 1) {   /* compressed Spinor block cache + per-layer window decode scratch */
        const size_t nb = (size_t)c->n_layers * P * (size_t)NKV * (size_t)s->nblk;
        s->kcb  = (sp_spinor_block_t *)malloc(nb * sizeof(sp_spinor_block_t));
        s->vcb  = (sp_spinor_block_t *)malloc(nb * sizeof(sp_spinor_block_t));
        s->kdec = (float *)malloc(P * KVD * sizeof(float));
        s->vdec = (float *)malloc(P * KVD * sizeof(float));
        if (!s->kcb || !s->vcb || !s->kdec || !s->vdec) return 1;
    } else {                 /* f32 cache (mode 0 pristine, mode 2 in-place roundtrip ref) */
        const size_t kvsz = (size_t)c->n_layers * P * KVD;
        s->kc = (float *)malloc(kvsz * sizeof(float));
        s->vc = (float *)malloc(kvsz * sizeof(float));
        if (!s->kc || !s->vc) return 1;
    }
    s->sx  = (float *)malloc((size_t)c->n_embd * sizeof(float));
    s->snx = (float *)malloc((size_t)c->n_embd * sizeof(float));
    s->sq  = (float *)malloc(QD * sizeof(float));
    s->sknew = (float *)malloc(KVD * sizeof(float));
    s->svnew = (float *)malloc(KVD * sizeof(float));
    s->sao = (float *)malloc(QD * sizeof(float));
    s->sap = (float *)malloc((size_t)c->n_embd * sizeof(float));
    s->sgg = (float *)malloc((size_t)c->n_ff * sizeof(float));
    s->sup = (float *)malloc((size_t)c->n_ff * sizeof(float));
    s->sdn = (float *)malloc((size_t)c->n_embd * sizeof(float));
    s->ssc = (float *)malloc(s->hist_cap * sizeof(float));
    if (!s->sx || !s->snx || !s->sq || !s->sknew || !s->svnew ||
        !s->sao || !s->sap || !s->sgg || !s->sup || !s->sdn || !s->ssc) return 1;
    return 0;
}

/* One persistent-KV token step at sequence position `pos` (gate-off f32 path,
 * identical math to qwen3_generate_kv). Writes K/V[pos] into the cache and, if
 * out_logits != NULL, that token's logits. Returns 0 on success. */
static int kv_step(sp_session *s, int32_t tok, int pos, float *out_logits) {
    const qwen3_model *m = s->qm;
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv;
    const int QD = NH * HD, KVD = NKV * HD, group = NH / NKV, V = (int)c->n_vocab;
    const float eps = c->rms_eps, base = c->rope_freq_base, ascale = 1.0f / sqrtf((float)HD);
    const int P = (int)s->hist_cap;
    float *x = s->sx, *nx = s->snx, *q = s->sq, *knew = s->sknew, *vnew = s->svnew;
    float *ao = s->sao, *ap = s->sap, *gg = s->sgg, *up = s->sup, *dn = s->sdn, *sc = s->ssc;

    if (sp_embed_row(m, tok, E, x)) return 1;
    for (uint32_t L = 0; L < c->n_layers; L++) {
        const qwen3_layer *ly = &m->layers[L];
        sp_rmsnorm(x, sp_as_f32(m, ly->attn_norm), E, eps, nx);
        if (sp_matmul(m, ly->attn_q, nx, 1, E, QD, q))    return 1;
        if (sp_matmul(m, ly->attn_k, nx, 1, E, KVD, knew)) return 1;
        if (sp_matmul(m, ly->attn_v, nx, 1, E, KVD, vnew)) return 1;
        const float *qn = sp_as_f32(m, ly->attn_q_norm), *kn = sp_as_f32(m, ly->attn_k_norm);
        for (int h = 0; h < NH;  h++) { float *qh = q    + (size_t)h * HD; sp_rmsnorm_head(qh, qn, HD, eps); sp_rope_neox(qh, HD, pos, base); }
        for (int h = 0; h < NKV; h++) { float *kh = knew + (size_t)h * HD; sp_rmsnorm_head(kh, kn, HD, eps); sp_rope_neox(kh, HD, pos, base); }

        /* 2-L1.PARITY KV codec: store K/V[pos] and expose KC/VC over the window [0,pos]. */
        const float *KC, *VC;
        if (s->kv_mode == 1) {            /* compressed Spinor block cache (the §4.9 layout) */
            const int NB = s->nblk;
            sp_spinor_block_t *kb = s->kcb + ((size_t)L * P + pos) * NKV * NB;
            sp_spinor_block_t *vb = s->vcb + ((size_t)L * P + pos) * NKV * NB;
            for (int h = 0; h < NKV; h++) {
                sp_spinor_encode_vec(knew + (size_t)h * HD, HD, kb + (size_t)h * NB);
                sp_spinor_encode_vec(vnew + (size_t)h * HD, HD, vb + (size_t)h * NB);
            }
            for (int sp_ = 0; sp_ <= pos; sp_++) {   /* decode the live window into f32 scratch */
                const sp_spinor_block_t *ks = s->kcb + ((size_t)L * P + sp_) * NKV * NB;
                const sp_spinor_block_t *vs = s->vcb + ((size_t)L * P + sp_) * NKV * NB;
                for (int h = 0; h < NKV; h++) {
                    (void)sp_spinor_decode_vec(ks + (size_t)h * NB, HD, s->kdec + (size_t)sp_ * KVD + (size_t)h * HD);
                    (void)sp_spinor_decode_vec(vs + (size_t)h * NB, HD, s->vdec + (size_t)sp_ * KVD + (size_t)h * HD);
                }
            }
            KC = s->kdec; VC = s->vdec;
        } else {
            if (s->kv_mode == 2)          /* parity reference: in-place lossy Spinor roundtrip */
                for (int h = 0; h < NKV; h++) {
                    spinor_roundtrip(knew + (size_t)h * HD, HD);
                    spinor_roundtrip(vnew + (size_t)h * HD, HD);
                }
            float *kcl = s->kc + (size_t)L * P * KVD, *vcl = s->vc + (size_t)L * P * KVD;
            memcpy(kcl + (size_t)pos * KVD, knew, (size_t)KVD * sizeof(float));
            memcpy(vcl + (size_t)pos * KVD, vnew, (size_t)KVD * sizeof(float));
            KC = kcl; VC = vcl;
        }

        for (int h = 0; h < NH; h++) {
            int kvh = h / group;
            const float *qh = q + (size_t)h * HD;
            float maxs = -INFINITY;
            for (int sp_ = 0; sp_ <= pos; sp_++) {
                const float *kh = KC + (size_t)sp_ * KVD + (size_t)kvh * HD;
                float acc = 0.0f;
                for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
                float d = acc * ascale; sc[sp_] = d; if (d > maxs) maxs = d;
            }
            float sum = 0.0f;
            for (int sp_ = 0; sp_ <= pos; sp_++) { sc[sp_] = expf(sc[sp_] - maxs); sum += sc[sp_]; }
            float inv = 1.0f / sum;
            float *out = ao + (size_t)h * HD;
            for (int i = 0; i < HD; i++) out[i] = 0.0f;
            for (int sp_ = 0; sp_ <= pos; sp_++) {
                float w = sc[sp_] * inv;
                const float *vh = VC + (size_t)sp_ * KVD + (size_t)kvh * HD;
                for (int i = 0; i < HD; i++) out[i] += w * vh[i];
            }
        }
        if (sp_matmul(m, ly->attn_output, ao, 1, QD, E, ap)) return 1;
        for (int i = 0; i < E; i++) x[i] += ap[i];

        sp_rmsnorm(x, sp_as_f32(m, ly->ffn_norm), E, eps, nx);
        if (sp_matmul(m, ly->ffn_gate, nx, 1, E, FF, gg)) return 1;
        if (sp_matmul(m, ly->ffn_up,   nx, 1, E, FF, up)) return 1;
        for (int i = 0; i < FF; i++) { float gv = gg[i]; gg[i] = gv / (1.0f + expf(-gv)) * up[i]; }
        if (sp_matmul(m, ly->ffn_down, gg, 1, FF, E, dn)) return 1;
        for (int i = 0; i < E; i++) x[i] += dn[i];
    }
    if (out_logits) {
        sp_rmsnorm(x, sp_as_f32(m, m->output_norm), E, eps, nx);
        if (sp_matmul(m, m->output, nx, 1, E, V, out_logits)) return 1;
    }
    return 0;
}

static float gelu_tanh(float x) {
    const float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

/* Gemma3 persistent-KV step: sandwich norms, GeGLU FFN, dual RoPE, SWA via sp_attn_head. */
static int kv_step_gemma3(sp_session *s, int32_t tok, int pos, float *out_logits) {
    const qwen3_model *m = s->qm;
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv;
    const int QD = NH * HD, KVD = NKV * HD, group = NH / NKV, V = (int)c->n_vocab;
    const float eps = c->rms_eps, gbase = c->rope_freq_base, ascale = 1.0f / sqrtf((float)HD);
    const int P = (int)s->hist_cap;
    float *x = s->sx, *nx = s->snx, *q = s->sq, *knew = s->sknew, *vnew = s->svnew;
    float *ao = s->sao, *ap = s->sap, *gg = s->sgg, *up = s->sup, *dn = s->sdn, *sc = s->ssc;

    if (sp_embed_row(m, tok, E, x)) return 1;
    { const float es = sqrtf((float)E); for (int i = 0; i < E; i++) x[i] *= es; }

    for (uint32_t L = 0; L < c->n_layers; L++) {
        const qwen3_layer *ly = &m->layers[L];
        const int global = ((L % 6) == 5);
        const float rbase = global ? gbase : 10000.0f;
        const int win = global ? -1 : (int)c->sliding_window;

        sp_rmsnorm(x, sp_as_f32(m, ly->attn_norm), E, eps, nx);
        if (sp_matmul(m, ly->attn_q, nx, 1, E, QD, q))     return 1;
        if (sp_matmul(m, ly->attn_k, nx, 1, E, KVD, knew))  return 1;
        if (sp_matmul(m, ly->attn_v, nx, 1, E, KVD, vnew))  return 1;

        const float *qn = sp_as_f32(m, ly->attn_q_norm), *kn = sp_as_f32(m, ly->attn_k_norm);
        for (int h = 0; h < NH;  h++) { float *qh = q    + (size_t)h * HD; sp_rmsnorm_head(qh, qn, HD, eps); sp_rope_neox(qh, HD, pos, rbase); }
        for (int h = 0; h < NKV; h++) { float *kh = knew + (size_t)h * HD; sp_rmsnorm_head(kh, kn, HD, eps); sp_rope_neox(kh, HD, pos, rbase); }

        const float *KC, *VC;
        if (s->kv_mode == 1) {
            const int NB = s->nblk;
            sp_spinor_block_t *kb = s->kcb + ((size_t)L * P + pos) * NKV * NB;
            sp_spinor_block_t *vb = s->vcb + ((size_t)L * P + pos) * NKV * NB;
            for (int h = 0; h < NKV; h++) {
                sp_spinor_encode_vec(knew + (size_t)h * HD, HD, kb + (size_t)h * NB);
                sp_spinor_encode_vec(vnew + (size_t)h * HD, HD, vb + (size_t)h * NB);
            }
            for (int sp_ = 0; sp_ <= pos; sp_++) {
                const sp_spinor_block_t *ks = s->kcb + ((size_t)L * P + sp_) * NKV * NB;
                const sp_spinor_block_t *vs = s->vcb + ((size_t)L * P + sp_) * NKV * NB;
                for (int h = 0; h < NKV; h++) {
                    (void)sp_spinor_decode_vec(ks + (size_t)h * NB, HD, s->kdec + (size_t)sp_ * KVD + (size_t)h * HD);
                    (void)sp_spinor_decode_vec(vs + (size_t)h * NB, HD, s->vdec + (size_t)sp_ * KVD + (size_t)h * HD);
                }
            }
            KC = s->kdec; VC = s->vdec;
        } else {
            if (s->kv_mode == 2)
                for (int h = 0; h < NKV; h++) {
                    spinor_roundtrip(knew + (size_t)h * HD, HD);
                    spinor_roundtrip(vnew + (size_t)h * HD, HD);
                }
            float *kcl = s->kc + (size_t)L * P * KVD, *vcl = s->vc + (size_t)L * P * KVD;
            memcpy(kcl + (size_t)pos * KVD, knew, (size_t)KVD * sizeof(float));
            memcpy(vcl + (size_t)pos * KVD, vnew, (size_t)KVD * sizeof(float));
            KC = kcl; VC = vcl;
        }

        /* GQA with SWA window: sp_attn_head handles the sliding-window mask */
        for (int h = 0; h < NH; h++)
            sp_attn_head(q + (size_t)h * HD, KC, VC, pos, KVD, h / group, HD, ascale, win, sc,
                         ao + (size_t)h * HD);

        if (sp_matmul(m, ly->attn_output, ao, 1, QD, E, ap)) return 1;
        /* sandwich: x += post_attn_norm(attn_proj) */
        sp_rmsnorm(ap, sp_as_f32(m, ly->post_attn_norm), E, eps, nx);
        for (int i = 0; i < E; i++) x[i] += nx[i];

        /* FFN (GeGLU) */
        sp_rmsnorm(x, sp_as_f32(m, ly->ffn_norm), E, eps, nx);
        if (sp_matmul(m, ly->ffn_gate, nx, 1, E, FF, gg)) return 1;
        if (sp_matmul(m, ly->ffn_up,   nx, 1, E, FF, up)) return 1;
        for (int i = 0; i < FF; i++) gg[i] = gelu_tanh(gg[i]) * up[i];
        if (sp_matmul(m, ly->ffn_down, gg, 1, FF, E, dn)) return 1;
        /* sandwich: x += post_ffw_norm(ffn_proj) */
        sp_rmsnorm(dn, sp_as_f32(m, ly->post_ffw_norm), E, eps, nx);
        for (int i = 0; i < E; i++) x[i] += nx[i];
    }
    if (out_logits) {
        sp_rmsnorm(x, sp_as_f32(m, m->output_norm), E, eps, nx);
        if (sp_matmul(m, m->output, nx, 1, E, V, out_logits)) return 1;
    }
    return 0;
}

/* Qwen2.5 persistent-KV step: no QK norms, QKV biases, SwiGLU, simple residual. */
static int kv_step_qwen25(sp_session *s, int32_t tok, int pos, float *out_logits) {
    const qwen3_model *m = s->qm;
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv;
    const int QD = NH * HD, KVD = NKV * HD, group = NH / NKV, V = (int)c->n_vocab;
    const float eps = c->rms_eps, base = c->rope_freq_base, ascale = 1.0f / sqrtf((float)HD);
    const int P = (int)s->hist_cap;
    float *x = s->sx, *nx = s->snx, *q = s->sq, *knew = s->sknew, *vnew = s->svnew;
    float *ao = s->sao, *ap = s->sap, *gg = s->sgg, *up = s->sup, *dn = s->sdn, *sc = s->ssc;

    if (sp_embed_row(m, tok, E, x)) return 1;
    for (uint32_t L = 0; L < c->n_layers; L++) {
        const qwen3_layer *ly = &m->layers[L];
        sp_rmsnorm(x, sp_as_f32(m, ly->attn_norm), E, eps, nx);
        if (sp_matmul(m, ly->attn_q, nx, 1, E, QD, q))     return 1;
        if (sp_matmul(m, ly->attn_k, nx, 1, E, KVD, knew))  return 1;
        if (sp_matmul(m, ly->attn_v, nx, 1, E, KVD, vnew))  return 1;

        /* add QKV biases */
        const float *qb = sp_as_f32(m, ly->attn_q_bias);
        const float *kb = sp_as_f32(m, ly->attn_k_bias);
        const float *vb = sp_as_f32(m, ly->attn_v_bias);
        for (int i = 0; i < QD;  i++) q[i]    += qb[i];
        for (int i = 0; i < KVD; i++) knew[i]  += kb[i];
        for (int i = 0; i < KVD; i++) vnew[i]  += vb[i];

        /* NEOX RoPE (no QK norm) */
        for (int h = 0; h < NH;  h++) { float *qh = q    + (size_t)h * HD; sp_rope_neox(qh, HD, pos, base); }
        for (int h = 0; h < NKV; h++) { float *kh = knew + (size_t)h * HD; sp_rope_neox(kh, HD, pos, base); }

        const float *KC, *VC;
        if (s->kv_mode == 1) {
            const int NB = s->nblk;
            sp_spinor_block_t *kb2 = s->kcb + ((size_t)L * P + pos) * NKV * NB;
            sp_spinor_block_t *vb2 = s->vcb + ((size_t)L * P + pos) * NKV * NB;
            for (int h = 0; h < NKV; h++) {
                sp_spinor_encode_vec(knew + (size_t)h * HD, HD, kb2 + (size_t)h * NB);
                sp_spinor_encode_vec(vnew + (size_t)h * HD, HD, vb2 + (size_t)h * NB);
            }
            for (int sp_ = 0; sp_ <= pos; sp_++) {
                const sp_spinor_block_t *ks = s->kcb + ((size_t)L * P + sp_) * NKV * NB;
                const sp_spinor_block_t *vs = s->vcb + ((size_t)L * P + sp_) * NKV * NB;
                for (int h = 0; h < NKV; h++) {
                    (void)sp_spinor_decode_vec(ks + (size_t)h * NB, HD, s->kdec + (size_t)sp_ * KVD + (size_t)h * HD);
                    (void)sp_spinor_decode_vec(vs + (size_t)h * NB, HD, s->vdec + (size_t)sp_ * KVD + (size_t)h * HD);
                }
            }
            KC = s->kdec; VC = s->vdec;
        } else {
            if (s->kv_mode == 2)
                for (int h = 0; h < NKV; h++) {
                    spinor_roundtrip(knew + (size_t)h * HD, HD);
                    spinor_roundtrip(vnew + (size_t)h * HD, HD);
                }
            float *kcl = s->kc + (size_t)L * P * KVD, *vcl = s->vc + (size_t)L * P * KVD;
            memcpy(kcl + (size_t)pos * KVD, knew, (size_t)KVD * sizeof(float));
            memcpy(vcl + (size_t)pos * KVD, vnew, (size_t)KVD * sizeof(float));
            KC = kcl; VC = vcl;
        }

        for (int h = 0; h < NH; h++) {
            int kvh = h / group;
            const float *qh = q + (size_t)h * HD;
            float maxs = -INFINITY;
            for (int sp_ = 0; sp_ <= pos; sp_++) {
                const float *kh = KC + (size_t)sp_ * KVD + (size_t)kvh * HD;
                float acc = 0.0f;
                for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
                float d = acc * ascale; sc[sp_] = d; if (d > maxs) maxs = d;
            }
            float sum = 0.0f;
            for (int sp_ = 0; sp_ <= pos; sp_++) { sc[sp_] = expf(sc[sp_] - maxs); sum += sc[sp_]; }
            float inv = 1.0f / sum;
            float *out = ao + (size_t)h * HD;
            for (int i = 0; i < HD; i++) out[i] = 0.0f;
            for (int sp_ = 0; sp_ <= pos; sp_++) {
                float w = sc[sp_] * inv;
                const float *vh = VC + (size_t)sp_ * KVD + (size_t)kvh * HD;
                for (int i = 0; i < HD; i++) out[i] += w * vh[i];
            }
        }
        if (sp_matmul(m, ly->attn_output, ao, 1, QD, E, ap)) return 1;
        for (int i = 0; i < E; i++) x[i] += ap[i];

        sp_rmsnorm(x, sp_as_f32(m, ly->ffn_norm), E, eps, nx);
        if (sp_matmul(m, ly->ffn_gate, nx, 1, E, FF, gg)) return 1;
        if (sp_matmul(m, ly->ffn_up,   nx, 1, E, FF, up)) return 1;
        for (int i = 0; i < FF; i++) { float gv = gg[i]; gg[i] = gv / (1.0f + expf(-gv)) * up[i]; }
        if (sp_matmul(m, ly->ffn_down, gg, 1, FF, E, dn)) return 1;
        for (int i = 0; i < E; i++) x[i] += dn[i];
    }
    if (out_logits) {
        sp_rmsnorm(x, sp_as_f32(m, m->output_norm), E, eps, nx);
        if (sp_matmul(m, m->output, nx, 1, E, V, out_logits)) return 1;
    }
    return 0;
}

sp_status sp_decode_step(sp_session *s, int32_t token, float *logits, size_t logits_capacity) {
    if (!s || !logits) { sp_set_error("sp_decode_step: null arg"); return SP_EBADARG; }
    if (logits_capacity < s->n_vocab) { sp_set_error("sp_decode_step: logits_capacity < vocab_size"); return SP_EBADARG; }
    if (s->pos >= s->hist_cap) { sp_set_error("sp_decode_step: context full"); return SP_ECONTEXT_FULL; }
    if (cancelled(s)) return SP_ECANCEL;
    if (ensure_decode_bufs(s)) { sp_set_error("sp_decode_step: OOM (KV)"); return SP_ENOMEM; }

    const int p = (int)s->pos;
    const sp_arch_t _arch = s->qm->cfg.arch;
    #define STEP(tok, pos, lo) \
        (_arch == SP_ARCH_GEMMA3 ? kv_step_gemma3(s, tok, pos, lo) : \
         _arch == SP_ARCH_QWEN25 ? kv_step_qwen25(s, tok, pos, lo) : \
                                   kv_step(s, tok, pos, lo))
    /* lazily fill KV for any prior positions not yet cached (e.g. a re-forward prefill) */
    for (size_t f = s->kv_filled; f < s->pos; f++) {
        if (STEP(s->hist[f], (int)f, NULL)) { sp_set_error("sp_decode_step: KV backfill failed"); return SP_EBADSTATE; }
    }

    s->hist[p] = token;
    int drc = STEP(token, p, logits);
    #undef STEP
    if (drc) { sp_set_error("sp_decode_step: forward failed"); return SP_EBADSTATE; }

    s->kv_filled = (size_t)p + 1;
    s->pos = (size_t)p + 1;
    return SP_OK;
}

/* ── session manipulation ── */
sp_status sp_session_clone(const sp_session *s, volatile int *cancel_flag, sp_session **out) {
    if (!s || !out) { sp_set_error("sp_session_clone: null arg"); return SP_EBADARG; }
    *out = NULL;
    sp_status st = sp_session_create(s->model, &s->cfg, cancel_flag, out);
    if (st != SP_OK) return st;
    sp_session *c = *out;
    memcpy(c->hist, s->hist, s->pos * sizeof(int32_t));
    c->pos = s->pos;
    /* Sprint WIRE-HEX + NTT.5b: propagate registered backends to the clone so
     * the fork hits the same accelerator dispatch path as the parent. Both
     * handle lifetimes are caller-owned (sp_l1.h:§5,§6); clone borrows the
     * same handle pointers (no deep copy of opaque state). The parent's
     * SpSession-side lifetime guard (AppState Arc) keeps them alive past the
     * clone's last use. */
    c->compute_backend_handle  = s->compute_backend_handle;
    c->compute_backend_forward = s->compute_backend_forward;
    c->compute_backend_inverse = s->compute_backend_inverse;
    c->forward_backend_handle  = s->forward_backend_handle;
    c->forward_backend_fn      = s->forward_backend_fn;
    if (s->kc || s->kcb) {   /* deep-copy the live KV so the fork doesn't re-prefill */
        if (ensure_decode_bufs(c)) { sp_session_destroy(c); *out = NULL; sp_set_error("sp_session_clone: OOM (KV)"); return SP_ENOMEM; }
        const qwen3_config *cf = &c->qm->cfg;
        const size_t P = c->hist_cap;
        if (s->kv_mode == 1) {
            const size_t nb = (size_t)cf->n_layers * P * (size_t)cf->n_head_kv * (size_t)c->nblk;
            memcpy(c->kcb, s->kcb, nb * sizeof(sp_spinor_block_t));
            memcpy(c->vcb, s->vcb, nb * sizeof(sp_spinor_block_t));
        } else {
            const size_t kvsz = (size_t)cf->n_layers * P * (size_t)cf->n_head_kv * cf->head_dim;
            memcpy(c->kc, s->kc, kvsz * sizeof(float));
            memcpy(c->vc, s->vc, kvsz * sizeof(float));
        }
        c->kv_filled = s->kv_filled;
    }
    return SP_OK;
}

sp_status sp_session_rewind(sp_session *s, size_t n_tokens) {
    if (!s) { sp_set_error("sp_session_rewind: null handle"); return SP_EBADARG; }
    if (n_tokens > s->pos) { sp_set_error("sp_session_rewind: n_tokens > position"); return SP_EBADARG; }
    s->pos -= n_tokens;
    if (s->kv_filled > s->pos) s->kv_filled = s->pos;   /* KV beyond the new position is stale */
    return SP_OK;
}

/* ── §5 Sprint NTT.5b: compute-backend registration + readback ───────────────
 *
 * The fields are calloc-zeroed at sp_session_create, so a session that never
 * calls the registration fn is implicitly "host path only" — every consumer
 * that reads via the getters sees NULL and falls back. The register fn
 * stores (handle, forward, inverse) on the session; the getters are how
 * math-core (currently sp_pr_bluestein_*) opts in to dispatch. The handle
 * lifetime contract is in sp_l1.h.
 *
 * Validation: NULL session → SP_EBADARG. All-NULL handle+forward+inverse is
 * the explicit "unregister" call (zeroes the fields). A partially-NULL
 * combination (e.g. handle non-NULL but forward NULL) is accepted as written —
 * downstream consumers MUST check the specific fn pointer they need before
 * dispatching, and fall back when it's NULL. This matches the per-direction
 * fallback the math-core wrappers implement. */
sp_status sp_session_register_compute_backend(
    sp_session *s,
    void *handle,
    sp_compute_ntt_dispatch_fn forward,
    sp_compute_ntt_dispatch_fn inverse) {
    if (!s) { sp_set_error("sp_session_register_compute_backend: null session"); return SP_EBADARG; }
    s->compute_backend_handle  = handle;
    s->compute_backend_forward = forward;
    s->compute_backend_inverse = inverse;
    return SP_OK;
}

void *sp_session_compute_backend_handle(const sp_session *s) {
    return s ? s->compute_backend_handle : NULL;
}

sp_compute_ntt_dispatch_fn sp_session_compute_backend_forward(const sp_session *s) {
    return s ? s->compute_backend_forward : NULL;
}

sp_compute_ntt_dispatch_fn sp_session_compute_backend_inverse(const sp_session *s) {
    return s ? s->compute_backend_inverse : NULL;
}

/* ── §6 Sprint WIRE-HEX: forward-backend registration + readback ───────────
 *
 * Plumbs the engine's per-backend full-forward entry points (currently
 * gemma3_forward_hexagon; future _cuda / _vulkan) into sp_prefill_chunk so
 * the daemon's chat path actually exercises accelerator silicon instead of
 * the math-core reference. Decode (persistent KV) stays on the reference
 * path — backend forwards don't expose a persistent-KV API. See sp_l1.h §6
 * for the full ABI contract + lifetime rules.
 *
 * Validation: NULL session → SP_EBADARG. NULL handle + NULL fn is the
 * "unregister" call (zeroes the fields). Both fields are calloc-zeroed at
 * session create, so a session that never calls the registration fn keeps
 * the existing math-core dispatch unchanged — backward-compatible with
 * every existing daemon + smoke harness. */
sp_status sp_session_register_forward_backend(
    sp_session *s,
    void *handle,
    sp_forward_dispatch_fn fn) {
    if (!s) { sp_set_error("sp_session_register_forward_backend: null session"); return SP_EBADARG; }
    s->forward_backend_handle = handle;
    s->forward_backend_fn     = fn;
    return SP_OK;
}

void *sp_session_forward_backend_handle(const sp_session *s) {
    return s ? s->forward_backend_handle : NULL;
}

sp_forward_dispatch_fn sp_session_forward_backend_fn(const sp_session *s) {
    return s ? s->forward_backend_fn : NULL;
}

/* Borrowed accessor for the session's internal qwen3_model pointer. Returned
 * as const void * to keep sp_l1.h free of qwen3_model's definition (which
 * lives in sp/model.h, the engine-bridge header); the daemon-side trampoline
 * casts back to const qwen3_model * before handing off to the backend. */
const void *sp_session_qwen3_model(const sp_session *s) {
    return s ? (const void *)s->qm : NULL;
}
