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
#include "sp/sp_hash.h"
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
    static int sha_printed = 0;
    if (!sha_printed) {
        uint8_t dg[32]; sp_sha256(mb, info->model_len, dg);
        fprintf(stderr, "    [fixture] qwen3 .sp-model SHA-256 (%zu bytes): ", info->model_len);
        for (unsigned i = 0; i < 32u; i++) fprintf(stderr, "%02x", (unsigned)dg[i]);
        fprintf(stderr, "\n");
        sha_printed = 1;
    }
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

#define NPROMPT 5u    /* == sizeof(TOKS) */
#define NGEN    100u

static int argmax_f(const float *v, uint32_t n) {
    int a = 0;
    for (uint32_t j = 1; j < n; j++) if (v[j] > v[(uint32_t)a]) a = (int)j;
    return a;
}

static void T_SESSION_DECODE_TRAJECTORY(void) {
    sp_qwen3_fixture_info info; sp_model *m = NULL;
    SP_CHECK(load_fixture(&info, &m) == 0, "fixture load");
    if (!m) { cleanup_files(); return; }
    const uint32_t V = info.n_vocab;

    /* reference: greedy qwen3_generate_kv (persistent-KV) over NGEN tokens */
    qwen3_model *ref = sp_model_to_qwen3(m);
    int32_t *seq = (int32_t *)malloc((size_t)(NPROMPT + NGEN) * sizeof(int32_t));
    for (uint32_t i = 0; i < NPROMPT; i++) seq[i] = TOKS[i];
    int total = ref ? qwen3_generate_kv(ref, seq, (int)NPROMPT, (int)NGEN, -1) : -1;
    SP_CHECK_EQ_I64(total, (int)(NPROMPT + NGEN), "qwen3_generate_kv produced NGEN tokens");

    /* session: prefill the prompt, then greedily decode, taking our own argmax */
    sp_session *s = NULL; sp_session_config cfg; memset(&cfg, 0, sizeof cfg); cfg.deterministic = 1;
    SP_CHECK_EQ_I64(sp_session_create(m, &cfg, NULL, &s), SP_OK, "create");
    float *lg = (float *)malloc((size_t)V * sizeof(float));
    SP_CHECK_EQ_I64(sp_prefill_chunk(s, TOKS, NPROMPT, lg, V), SP_OK, "prefill");

    int t = argmax_f(lg, V), matched = 1; uint32_t first_div = NGEN;
    for (uint32_t k = 0; k < NGEN; k++) {
        if (t != seq[NPROMPT + k]) { matched = 0; first_div = k; break; }
        if (k + 1 < NGEN) {
            if (sp_decode_step(s, t, lg, V) != SP_OK) { matched = 0; first_div = k; break; }
            t = argmax_f(lg, V);
        }
    }
    SP_CHECK(matched, "session greedy argmax trajectory == qwen3_generate_kv over 100 steps");
    if (!matched) fprintf(stderr, "    [traj] first divergence at step %u\n", first_div);
    size_t pos = 0; sp_session_position(s, &pos);
    SP_CHECK_EQ_I64(pos, NPROMPT + (NGEN - 1u), "position == prompt + 99 decode steps");

    free(lg); free(seq); if (ref) qwen3_free(ref);
    sp_session_destroy(s); sp_model_unload(m); cleanup_files();
}

static void T_SESSION_CLONE_REWIND(void) {
    sp_qwen3_fixture_info info; sp_model *m = NULL;
    SP_CHECK(load_fixture(&info, &m) == 0, "fixture load");
    if (!m) { cleanup_files(); return; }
    const uint32_t V = info.n_vocab;

    sp_session *s = NULL;
    SP_CHECK_EQ_I64(sp_session_create(m, NULL, NULL, &s), SP_OK, "create");
    float *lg = (float *)malloc((size_t)V * sizeof(float));
    sp_prefill_chunk(s, TOKS, NPROMPT, lg, V);
    int t = argmax_f(lg, V);
    for (int k = 0; k < 3; k++) { sp_decode_step(s, t, lg, V); t = argmax_f(lg, V); }   /* pos = 5+3 = 8 */

    /* clone: independent session with identical state */
    sp_session *cl = NULL;
    SP_CHECK_EQ_I64(sp_session_clone(s, NULL, &cl), SP_OK, "clone -> SP_OK");
    size_t ps = 0, pc = 0; sp_session_position(s, &ps); sp_session_position(cl, &pc);
    SP_CHECK_EQ_I64(ps, 8, "original position");
    SP_CHECK_EQ_I64(pc, ps, "clone position matches original");

    /* decoding the same token on both yields bit-identical logits (independent state) */
    float *a = (float *)malloc((size_t)V * sizeof(float));
    float *b = (float *)malloc((size_t)V * sizeof(float));
    SP_CHECK_EQ_I64(sp_decode_step(s,  t, a, V), SP_OK, "orig decode");
    SP_CHECK_EQ_I64(sp_decode_step(cl, t, b, V), SP_OK, "clone decode");
    SP_CHECK(memcmp(a, b, (size_t)V * sizeof(float)) == 0, "clone decode logits BIT-EXACT vs original");

    /* rewind */
    SP_CHECK_EQ_I64(sp_session_rewind(s, 2), SP_OK, "rewind 2 -> SP_OK");
    sp_session_position(s, &ps);
    SP_CHECK_EQ_I64(ps, 7, "position after rewind (9 -> 7)");
    SP_CHECK_EQ_I64(sp_session_rewind(s, 9999), SP_EBADARG, "rewind > position -> EBADARG");

    free(lg); free(a); free(b);
    sp_session_destroy(cl); sp_session_destroy(s); sp_model_unload(m); cleanup_files();
}

static void T_SESSION_CANCEL(void) {
    sp_qwen3_fixture_info info; sp_model *m = NULL;
    SP_CHECK(load_fixture(&info, &m) == 0, "fixture load");
    if (!m) { cleanup_files(); return; }
    const uint32_t V = info.n_vocab;

    volatile int flag = 0;
    sp_session *s = NULL;
    SP_CHECK_EQ_I64(sp_session_create(m, NULL, &flag, &s), SP_OK, "create with cancel flag");
    float *lg = (float *)malloc((size_t)V * sizeof(float));
    SP_CHECK_EQ_I64(sp_prefill_chunk(s, TOKS, NPROMPT, lg, V), SP_OK, "prefill (flag clear)");
    int t = argmax_f(lg, V);

    flag = 1;
    SP_CHECK_EQ_I64(sp_decode_step(s, t, lg, V), SP_ECANCEL, "decode with cancel -> SP_ECANCEL");
    size_t pos = 0; sp_session_position(s, &pos);
    SP_CHECK_EQ_I64(pos, NPROMPT, "position unchanged after cancel");
    SP_CHECK_EQ_I64(sp_prefill_chunk(s, TOKS, 1, lg, V), SP_ECANCEL, "prefill with cancel -> SP_ECANCEL");

    flag = 0;
    SP_CHECK_EQ_I64(sp_decode_step(s, t, lg, V), SP_OK, "decode resumes after clearing cancel");

    free(lg);
    sp_session_destroy(s); sp_model_unload(m); cleanup_files();
}

int main(void) {
    SP_RUN(T_SESSION_BRIDGE);
    SP_RUN(T_SESSION_PREFILL_PARITY);
    SP_RUN(T_SESSION_GUARDS);
    SP_RUN(T_SESSION_DECODE_TRAJECTORY);
    SP_RUN(T_SESSION_CLONE_REWIND);
    SP_RUN(T_SESSION_CANCEL);
    return SP_DONE();
}
