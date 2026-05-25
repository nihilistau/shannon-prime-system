/* sp_session.c — the L1 session ABI (PPT-LAT-L1-ABI-v0 §1/§3, declared in
 * sp/sp_l1.h). The session is the single-thread mutable inference state over an
 * immutable model: it reconstructs the runnable qwen3_model from the const sp_model
 * handle (sp_model_to_qwen3 — NOT a veneer over a GGUF-loaded model, per §8.7.4
 * spec discipline) and drives the relocated reference forward.
 *
 * Series 1 (this file, prefill + parity): sp_session_create / destroy / position /
 * sp_prefill_chunk. Prefill re-runs the reference forward (qwen3_forward) over the
 * accumulated token history and writes the last position's logits — bit-exact to the
 * reference forward by construction (same reconstruction, deterministic scalar path).
 * The persistent-KV decode (sp_decode_step) + clone/rewind land in series 2. */
#include "sp/sp_l1.h"
#include "sp/sp_model.h"
#include "sp/model.h"
#include "sp/sp_error.h"

#include <stdlib.h>
#include <string.h>

#define SP_SESSION_CTX_FALLBACK 4096u   /* when neither cfg nor arch caps the context */

struct sp_session {
    const sp_model    *model;       /* borrowed: immutable source, for clone re-derivation */
    qwen3_model       *qm;          /* owned: reconstructed runnable model */
    sp_session_config  cfg;         /* immutable for the session's lifetime */
    volatile int      *cancel;      /* borrowed L2-owned atomic; NULL = no cancellation */
    int32_t           *hist;        /* token history [0, pos) */
    size_t             hist_cap;    /* == effective max_context */
    size_t             pos;         /* current sequence position */
    uint32_t           n_vocab;
};

static int cancelled(const sp_session *s) {
    return (s->cancel && *s->cancel) ? 1 : 0;
}

sp_status sp_session_create(const sp_model *m, const sp_session_config *cfg,
                            volatile int *cancel_flag, sp_session **out_session) {
    if (!m || !out_session) { sp_set_error("sp_session_create: null arg"); return SP_EBADARG; }
    *out_session = NULL;

    qwen3_model *qm = sp_model_to_qwen3(m);
    if (!qm) return SP_EBADFORMAT;   /* sp_model_to_qwen3 set the detail */

    sp_session *s = (sp_session *)calloc(1, sizeof *s);
    if (!s) { qwen3_free(qm); sp_set_error("sp_session_create: OOM"); return SP_ENOMEM; }

    s->model  = m;
    s->qm     = qm;
    s->cancel = cancel_flag;
    s->n_vocab = qm->cfg.n_vocab;
    if (cfg) s->cfg = *cfg;

    size_t cap = s->cfg.max_context;
    if (cap == 0) cap = qm->cfg.context_length;
    if (cap == 0) cap = SP_SESSION_CTX_FALLBACK;
    s->hist_cap = cap;
    s->hist = (int32_t *)malloc(cap * sizeof(int32_t));
    if (!s->hist) { qwen3_free(qm); free(s); sp_set_error("sp_session_create: OOM (history)"); return SP_ENOMEM; }
    s->pos = 0;

    *out_session = s;
    return SP_OK;
}

void sp_session_destroy(sp_session *s) {
    if (!s) return;
    qwen3_free(s->qm);
    free(s->hist);
    free(s);
}

sp_status sp_session_position(const sp_session *s, size_t *pos_out) {
    if (!s || !pos_out) { sp_set_error("sp_session_position: null arg"); return SP_EBADARG; }
    *pos_out = s->pos;
    return SP_OK;
}

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
