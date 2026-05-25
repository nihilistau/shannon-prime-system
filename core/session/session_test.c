/* session_test.c — T_SESSION: the L1 session ABI over a synthetic qwen3-shaped
 * .sp-model (qwen3_fixture.c). Series 1: the bridge (sp_model_to_qwen3) + the
 * prefill parity gate — sp_prefill_chunk's last-position logits are BIT-EXACT to
 * the relocated reference forward (qwen3_forward) in deterministic mode. The bridge
 * is exercised by exactly this gate (its correctness == prefill parity, so it gets
 * no separate tag). All status codes are the frozen sp_l1.h / sp_status.h codes. */
#include "sp/sp_test.h"
#include "sp/sp_l1.h"
#include "sp/sp_model.h"
#include "sp/model.h"
#include "qwen3_fixture.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

static const int32_t TOKS[5] = { 1, 7, 3, 42, 13 };
#define NTOK 5u

static int load_fixture(sp_qwen3_fixture_info *info, sp_model **out) {
    uint8_t *mb = NULL, *tb = NULL;
    if (sp_qwen3_fixture_build(&mb, &tb, info)) return 1;
    int rc = sp_qwen3_fixture_write("fx_q3.spm", mb, info->model_len)
           | sp_qwen3_fixture_write("fx_q3.spt", tb, info->tok_len);
    free(mb); free(tb);
    if (rc) return 1;
    return (sp_model_load("fx_q3.spm", "fx_q3.spt", out) == SP_OK) ? 0 : 1;
}
static void cleanup_files(void) { remove("fx_q3.spm"); remove("fx_q3.spt"); }

static void T_SESSION_BRIDGE(void) {
    sp_qwen3_fixture_info info; sp_model *m = NULL;
    SP_CHECK(load_fixture(&info, &m) == 0, "fixture load");
    if (!m) { cleanup_files(); return; }

    qwen3_model *qm = sp_model_to_qwen3(m);
    SP_CHECK(qm != NULL, "sp_model_to_qwen3 -> non-NULL");
    if (qm) {
        SP_CHECK_EQ_I64(qm->cfg.n_layers, info.n_layers, "cfg.n_layers");
        SP_CHECK_EQ_I64(qm->cfg.n_embd,   info.n_embd,   "cfg.n_embd");
        SP_CHECK_EQ_I64(qm->cfg.n_ff,     info.n_ff,     "cfg.n_ff (derived from ffn_gate)");
        SP_CHECK_EQ_I64(qm->cfg.n_vocab,  info.n_vocab,  "cfg.n_vocab");
        SP_CHECK(qm->gguf == NULL && qm->released == 1, "released arena-only reconstruction");

        size_t N = (size_t)NTOK * info.n_vocab;
        float *lg = (float *)malloc(N * sizeof(float));
        SP_CHECK(qwen3_forward(qm, TOKS, (int)NTOK, lg) == 0, "qwen3_forward over reconstruction");
        int finite = 1;
        for (size_t i = 0; i < N; i++) if (!isfinite(lg[i])) finite = 0;
        SP_CHECK(finite, "all logits finite");
        free(lg);
        qwen3_free(qm);
    }
    sp_model_unload(m); cleanup_files();
}

static void T_SESSION_PREFILL_PARITY(void) {
    sp_qwen3_fixture_info info; sp_model *m = NULL;
    SP_CHECK(load_fixture(&info, &m) == 0, "fixture load");
    if (!m) { cleanup_files(); return; }
    const uint32_t V = info.n_vocab;

    /* reference: full forward over the chunk; take the last position's logits */
    qwen3_model *ref = sp_model_to_qwen3(m);
    SP_CHECK(ref != NULL, "reference reconstruction");
    float *ref_all = (float *)malloc((size_t)NTOK * V * sizeof(float));
    SP_CHECK(ref && qwen3_forward(ref, TOKS, (int)NTOK, ref_all) == 0, "reference forward");

    /* session: prefill the same chunk on a fresh deterministic session */
    sp_session *s = NULL;
    sp_session_config cfg; memset(&cfg, 0, sizeof cfg); cfg.deterministic = 1;
    SP_CHECK_EQ_I64(sp_session_create(m, &cfg, NULL, &s), SP_OK, "sp_session_create -> SP_OK");
    float *sess = (float *)malloc((size_t)V * sizeof(float));
    SP_CHECK_EQ_I64(sp_prefill_chunk(s, TOKS, NTOK, sess, V), SP_OK, "sp_prefill_chunk -> SP_OK");

    if (ref && sess) {
        const float *ref_last = ref_all + (size_t)(NTOK - 1u) * V;
        SP_CHECK(memcmp(sess, ref_last, (size_t)V * sizeof(float)) == 0,
                 "prefill last-position logits BIT-EXACT vs qwen3_forward");
    }
    size_t pos = 0; sp_session_position(s, &pos);
    SP_CHECK_EQ_I64(pos, NTOK, "position == n_tokens after prefill");

    free(sess); free(ref_all);
    if (ref) qwen3_free(ref);
    sp_session_destroy(s);
    sp_model_unload(m); cleanup_files();
}

static void T_SESSION_GUARDS(void) {
    sp_qwen3_fixture_info info; sp_model *m = NULL;
    SP_CHECK(load_fixture(&info, &m) == 0, "fixture load");
    if (!m) { cleanup_files(); return; }

    sp_session *s = NULL;
    SP_CHECK_EQ_I64(sp_session_create(NULL, NULL, NULL, &s), SP_EBADARG, "create(NULL model) -> EBADARG");
    SP_CHECK_EQ_I64(sp_session_create(m, NULL, NULL, NULL), SP_EBADARG, "create(NULL out) -> EBADARG");
    SP_CHECK_EQ_I64(sp_session_create(m, NULL, NULL, &s), SP_OK, "create(NULL cfg) -> SP_OK (defaults)");

    size_t pos = 99;
    SP_CHECK_EQ_I64(sp_session_position(NULL, &pos), SP_EBADARG, "position(NULL) -> EBADARG");

    float *lg = (float *)malloc((size_t)info.n_vocab * sizeof(float));
    SP_CHECK_EQ_I64(sp_prefill_chunk(s, TOKS, NTOK, lg, 1), SP_EBADARG, "prefill capacity<vocab -> EBADARG");
    SP_CHECK_EQ_I64(sp_prefill_chunk(s, TOKS, 0, lg, info.n_vocab), SP_EBADARG, "prefill n_tokens=0 -> EBADARG");
    free(lg);

    if (s) sp_session_destroy(s);
    sp_model_unload(m); cleanup_files();
}

int main(void) {
    SP_RUN(T_SESSION_BRIDGE);
    SP_RUN(T_SESSION_PREFILL_PARITY);
    SP_RUN(T_SESSION_GUARDS);
    return SP_DONE();
}
