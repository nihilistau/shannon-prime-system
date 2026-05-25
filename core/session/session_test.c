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
#include "sp/spinor_block.h"
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

/* ── 2-L1.FP16 ABI extension: arch_struct growth + precision precedence ── */
#include <stddef.h>   /* offsetof */

static int load_fixture_opts(const sp_qwen3_fixture_opts *opts,
                             sp_qwen3_fixture_info *info, sp_model **out) {
    uint8_t *mb = NULL, *tb = NULL;
    if (sp_qwen3_fixture_build_ex(&mb, &tb, info, opts)) return 1;
    int rc = sp_qwen3_fixture_write("fx_q3.spm", mb, info->model_len)
           | sp_qwen3_fixture_write("fx_q3.spt", tb, info->tok_len);
    free(mb); free(tb);
    if (rc) return 1;
    return (sp_model_load("fx_q3.spm", "fx_q3.spt", out) == SP_OK) ? 0 : 1;
}

static void T_ARCH_GROWTH_OLD(void) {
    /* arch_struct_size written = the pre-FP16 size (offsetof n_ff): the appended fields
     * are TRUNCATED by the size-limited memcpy, so they read 0 and the bridge defaults. */
    sp_qwen3_fixture_info info; sp_model *m = NULL;
    sp_qwen3_fixture_opts opts; memset(&opts, 0, sizeof opts);
    opts.arch_struct_size    = (uint32_t)offsetof(sp_arch_info, n_ff);
    opts.preferred_precision = SP_PRECISION_FP16;   /* present in payload but truncated away */
    opts.rms_eps_field       = 2.5e-5f;
    opts.n_ff_field          = 999u;                /* bogus; must be ignored (truncated) */
    SP_CHECK(load_fixture_opts(&opts, &info, &m) == 0, "old-size .sp-model loads (no SP_EBADFORMAT)");
    if (!m) { cleanup_files(); return; }

    sp_arch_info ai; memset(&ai, 0, sizeof ai);
    SP_CHECK_EQ_I64(sp_model_arch(m, &ai), SP_OK, "arch query");
    SP_CHECK_EQ_I64(ai.n_ff, 0, "old file: appended n_ff reads 0");
    SP_CHECK(ai.rms_eps == 0.0f, "old file: appended rms_eps reads 0");
    SP_CHECK_EQ_I64(ai.preferred_precision, SP_PRECISION_UNSPECIFIED, "old file: preferred_precision 0");

    qwen3_model *qm = sp_model_to_qwen3(m);
    SP_CHECK(qm != NULL, "bridge reconstructs");
    if (qm) {
        SP_CHECK_EQ_I64(qm->cfg.n_ff, info.n_ff, "n_ff derived from ffn_gate (not the bogus 999)");
        SP_CHECK(qm->cfg.rms_eps == 1e-6f, "rms_eps defaulted to 1e-6");
        qwen3_free(qm);
    }
    sp_session *s = NULL;
    SP_CHECK_EQ_I64(sp_session_create(m, NULL, NULL, &s), SP_OK, "create");
    SP_CHECK_EQ_I64(sp_session_precision(s), SP_PRECISION_F32, "old file -> resolved precision F32");
    sp_session_destroy(s);
    sp_model_unload(m); cleanup_files();
}

static void T_ARCH_GROWTH_NEW(void) {
    /* full arch_struct_size: the appended fields are populated and read back. */
    sp_qwen3_fixture_info info; sp_model *m = NULL;
    sp_qwen3_fixture_opts opts; memset(&opts, 0, sizeof opts);
    opts.arch_struct_size    = (uint32_t)sizeof(sp_arch_info);
    opts.preferred_precision = SP_PRECISION_FP16;
    opts.rms_eps_field       = 2.5e-5f;
    opts.n_ff_field          = 12345u;   /* arbitrary; proves the field is read (not derived) */
    SP_CHECK(load_fixture_opts(&opts, &info, &m) == 0, "new-size .sp-model loads");
    if (!m) { cleanup_files(); return; }

    sp_arch_info ai; memset(&ai, 0, sizeof ai);
    SP_CHECK_EQ_I64(sp_model_arch(m, &ai), SP_OK, "arch query");
    SP_CHECK_EQ_I64(ai.n_ff, 12345, "new file: n_ff read from arch_struct");
    SP_CHECK(ai.rms_eps == 2.5e-5f, "new file: rms_eps read from arch_struct");
    SP_CHECK_EQ_I64(ai.preferred_precision, SP_PRECISION_FP16, "new file: preferred_precision read");

    qwen3_model *qm = sp_model_to_qwen3(m);   /* note: do NOT forward (n_ff=12345 != real ffn_gate) */
    SP_CHECK(qm != NULL, "bridge reconstructs");
    if (qm) {
        SP_CHECK_EQ_I64(qm->cfg.n_ff, 12345, "bridge uses arch_struct.n_ff");
        SP_CHECK(qm->cfg.rms_eps == 2.5e-5f, "bridge uses arch_struct.rms_eps");
        qwen3_free(qm);
    }
    sp_model_unload(m); cleanup_files();
}

static void T_SESSION_PRECISION_PRECEDENCE(void) {
    sp_qwen3_fixture_info info; sp_model *m = NULL;

    /* model prefers FP16 */
    sp_qwen3_fixture_opts opts; memset(&opts, 0, sizeof opts);
    opts.arch_struct_size = (uint32_t)sizeof(sp_arch_info);
    opts.preferred_precision = SP_PRECISION_FP16;
    SP_CHECK(load_fixture_opts(&opts, &info, &m) == 0, "load (model prefers FP16)");
    if (m) {
        sp_session *s = NULL;
        sp_session_config cfg; memset(&cfg, 0, sizeof cfg);
        cfg.precision_override = SP_PRECISION_QF32;
        SP_CHECK_EQ_I64(sp_session_create(m, &cfg, NULL, &s), SP_OK, "create (override QF32)");
        SP_CHECK_EQ_I64(sp_session_precision(s), SP_PRECISION_QF32, "override > arch preference");
        sp_session_destroy(s);

        s = NULL;
        SP_CHECK_EQ_I64(sp_session_create(m, NULL, NULL, &s), SP_OK, "create (no override)");
        SP_CHECK_EQ_I64(sp_session_precision(s), SP_PRECISION_FP16, "arch preference > default");
        sp_session_destroy(s);
        sp_model_unload(m);
    }
    cleanup_files();

    /* model unspecified -> f32 default */
    m = NULL;
    sp_qwen3_fixture_opts opts2; memset(&opts2, 0, sizeof opts2);
    opts2.arch_struct_size = (uint32_t)sizeof(sp_arch_info);   /* preferred_precision left 0 */
    SP_CHECK(load_fixture_opts(&opts2, &info, &m) == 0, "load (model unspecified)");
    if (m) {
        sp_session *s = NULL;
        SP_CHECK_EQ_I64(sp_session_create(m, NULL, NULL, &s), SP_OK, "create (no override, no preference)");
        SP_CHECK_EQ_I64(sp_session_precision(s), SP_PRECISION_F32, "default -> F32");
        sp_session_destroy(s);
        sp_model_unload(m);
    }
    cleanup_files();
}

static void T_PARITY_KV_SPINOR(void) {
    /* E_PARITY_1: the session's Spinor-block KV cache (SP_KV_SPINOR) must produce
     * decode logits BIT-IDENTICAL to the f32 + in-place-Spinor-roundtrip reference
     * (SP_KV_SPINOR_REF) — decode-from-block === in-place-roundtrip by codec identity
     * (the E_CPU_8 / GEN_KV_SPINOR gate, verifiable without a real model). */
    sp_qwen3_fixture_info info; sp_model *m = NULL;
    SP_CHECK(load_fixture(&info, &m) == 0, "fixture load");
    if (!m) { cleanup_files(); return; }
    const uint32_t V = info.n_vocab;

    sp_session *sb = NULL, *sr = NULL;
    sp_session_config cb; memset(&cb, 0, sizeof cb); cb.deterministic = 1; cb.flags = SP_KV_SPINOR;
    sp_session_config cr; memset(&cr, 0, sizeof cr); cr.deterministic = 1; cr.flags = SP_KV_SPINOR_REF;
    SP_CHECK_EQ_I64(sp_session_create(m, &cb, NULL, &sb), SP_OK, "create (SP_KV_SPINOR block cache)");
    SP_CHECK_EQ_I64(sp_session_create(m, &cr, NULL, &sr), SP_OK, "create (SP_KV_SPINOR_REF f32 roundtrip)");

    float *lb = (float *)malloc((size_t)V * sizeof(float));
    float *lr = (float *)malloc((size_t)V * sizeof(float));
    SP_CHECK_EQ_I64(sp_prefill_chunk(sb, TOKS, NPROMPT, lb, V), SP_OK, "prefill block");
    SP_CHECK_EQ_I64(sp_prefill_chunk(sr, TOKS, NPROMPT, lr, V), SP_OK, "prefill ref");
    int tb = argmax_f(lb, V), tr = argmax_f(lr, V);
    int identical = 1;
    for (int k = 0; k < 8; k++) {
        if (sp_decode_step(sb, tb, lb, V) != SP_OK || sp_decode_step(sr, tr, lr, V) != SP_OK) { identical = 0; break; }
        if (memcmp(lb, lr, (size_t)V * sizeof(float)) != 0) { identical = 0; break; }
        tb = argmax_f(lb, V); tr = argmax_f(lr, V);
    }
    SP_CHECK(identical, "Spinor block-KV decode logits BIT-IDENTICAL to f32-roundtrip ref (E_PARITY_1)");

    /* structural footprint: per (layer,position,kv-head) the compressed cache stores
     * sp_spinor_blocks_for(head_dim) 63-byte blocks vs head_dim*4 f32 bytes. */
    int nblk = sp_spinor_blocks_for((int)info.head_dim);
    size_t blk_bytes = (size_t)nblk * sizeof(sp_spinor_block_t);
    size_t f32_bytes = (size_t)info.head_dim * sizeof(float);
    fprintf(stderr, "    [E_PARITY_1] per-head KV: spinor %zuB (%d blk) vs f32 %zuB  (head_dim=%u; "
                    "real head_dim=128 -> 3*63=189B vs 512B = 2.71x)\n",
            blk_bytes, nblk, f32_bytes, info.head_dim);
    SP_CHECK(nblk == sp_spinor_blocks_for((int)info.head_dim), "blocks-per-head deterministic");

    free(lb); free(lr);
    sp_session_destroy(sb); sp_session_destroy(sr);
    sp_model_unload(m); cleanup_files();
}

int main(void) {
    SP_RUN(T_PARITY_KV_SPINOR);
    SP_RUN(T_SESSION_BRIDGE);
    SP_RUN(T_SESSION_PREFILL_PARITY);
    SP_RUN(T_SESSION_GUARDS);
    SP_RUN(T_SESSION_DECODE_TRAJECTORY);
    SP_RUN(T_SESSION_CLONE_REWIND);
    SP_RUN(T_SESSION_CANCEL);
    SP_RUN(T_ARCH_GROWTH_OLD);
    SP_RUN(T_ARCH_GROWTH_NEW);
    SP_RUN(T_SESSION_PRECISION_PRECEDENCE);
    return SP_DONE();
}
