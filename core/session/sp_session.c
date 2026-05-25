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
    /* persistent decode KV (lazily allocated on first sp_decode_step) */
    float             *kc, *vc;     /* [n_layers * hist_cap * KVD] each */
    size_t             kv_filled;   /* positions [0, kv_filled) hold valid KV */
    /* per-step scratch (allocated with kc/vc) */
    float *sx, *snx, *sq, *sknew, *svnew, *sao, *sap, *sgg, *sup, *sdn, *ssc;
};

static int cancelled(const sp_session *s) { return (s->cancel && *s->cancel) ? 1 : 0; }

/* ── lifecycle ── */
sp_status sp_session_create(const sp_model *m, const sp_session_config *cfg,
                            volatile int *cancel_flag, sp_session **out_session) {
    if (!m || !out_session) { sp_set_error("sp_session_create: null arg"); return SP_EBADARG; }
    *out_session = NULL;

    qwen3_model *qm = sp_model_to_qwen3(m);
    if (!qm) return SP_EBADFORMAT;   /* sp_model_to_qwen3 set the detail */

    sp_session *s = (sp_session *)calloc(1, sizeof *s);
    if (!s) { qwen3_free(qm); sp_set_error("sp_session_create: OOM"); return SP_ENOMEM; }
    s->model = m; s->qm = qm; s->cancel = cancel_flag; s->n_vocab = qm->cfg.n_vocab;
    if (cfg) s->cfg = *cfg;

    size_t cap = s->cfg.max_context;
    if (cap == 0) cap = qm->cfg.context_length;
    if (cap == 0) cap = SP_SESSION_CTX_FALLBACK;
    s->hist_cap = cap;
    s->hist = (int32_t *)malloc(cap * sizeof(int32_t));
    if (!s->hist) { qwen3_free(qm); free(s); sp_set_error("sp_session_create: OOM (history)"); return SP_ENOMEM; }

    *out_session = s;
    return SP_OK;
}

void sp_session_destroy(sp_session *s) {
    if (!s) return;
    qwen3_free(s->qm);
    free(s->hist);
    free(s->kc); free(s->vc);
    free(s->sx); free(s->snx); free(s->sq); free(s->sknew); free(s->svnew);
    free(s->sao); free(s->sap); free(s->sgg); free(s->sup); free(s->sdn); free(s->ssc);
    free(s);
}

sp_status sp_session_position(const sp_session *s, size_t *pos_out) {
    if (!s || !pos_out) { sp_set_error("sp_session_position: null arg"); return SP_EBADARG; }
    *pos_out = s->pos;
    return SP_OK;
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
    if (qwen3_forward(s->qm, s->hist, (int)new_len, tmp) != 0) {
        free(tmp); sp_set_error("sp_prefill_chunk: forward failed"); return SP_EBADSTATE;
    }
    memcpy(logits_last, tmp + (new_len - 1) * (size_t)s->n_vocab, (size_t)s->n_vocab * sizeof(float));
    free(tmp);

    s->pos = new_len;   /* commit only after a successful forward */
    return SP_OK;
}

/* ── decode (persistent-KV, one token) ── */
static int ensure_decode_bufs(sp_session *s) {
    if (s->kc) return 0;
    const qwen3_config *c = &s->qm->cfg;
    const size_t KVD = (size_t)c->n_head_kv * c->head_dim;
    const size_t QD  = (size_t)c->n_head * c->head_dim;
    const size_t kvsz = (size_t)c->n_layers * s->hist_cap * KVD;
    s->kc = (float *)malloc(kvsz * sizeof(float));
    s->vc = (float *)malloc(kvsz * sizeof(float));
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
    if (!s->kc || !s->vc || !s->sx || !s->snx || !s->sq || !s->sknew || !s->svnew ||
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

        float *KC = s->kc + (size_t)L * P * KVD;
        float *VC = s->vc + (size_t)L * P * KVD;
        memcpy(KC + (size_t)pos * KVD, knew, (size_t)KVD * sizeof(float));
        memcpy(VC + (size_t)pos * KVD, vnew, (size_t)KVD * sizeof(float));

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
    /* lazily fill KV for any prior positions not yet cached (e.g. a re-forward prefill) */
    for (size_t f = s->kv_filled; f < s->pos; f++)
        if (kv_step(s, s->hist[f], (int)f, NULL)) { sp_set_error("sp_decode_step: KV backfill failed"); return SP_EBADSTATE; }

    s->hist[p] = token;
    if (kv_step(s, token, p, logits)) { sp_set_error("sp_decode_step: forward failed"); return SP_EBADSTATE; }

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
    if (s->kc) {   /* deep-copy the live KV so the fork doesn't re-prefill */
        if (ensure_decode_bufs(c)) { sp_session_destroy(c); *out = NULL; sp_set_error("sp_session_clone: OOM (KV)"); return SP_ENOMEM; }
        const qwen3_config *cf = &c->qm->cfg;
        size_t kvsz = (size_t)cf->n_layers * c->hist_cap * (size_t)cf->n_head_kv * cf->head_dim;
        memcpy(c->kc, s->kc, kvsz * sizeof(float));
        memcpy(c->vc, s->vc, kvsz * sizeof(float));
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
