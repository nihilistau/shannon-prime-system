/* arm_test.c — T_ARM: gates for the ARM two-ring core (sp/arm.h).
 *
 * The recall router was proven oracle-perfect (8/8 planted needles, cos 1.0)
 * in the engine's C2 gate before promotion; these gates pin the promoted
 * module to the same semantics:
 *   T_ARM_R_FROZEN          — R is deterministic from the frozen seed, ±1, balanced.
 *   T_ARM_PROJECT_EXACT     — sp_arm_project == naive R·v (same arithmetic order).
 *   T_ARM_SELECT_PARITY     — B=0 / within-budget => identity [0,pos] (the
 *                             bit-exact-when-off guarantee the forward relies on).
 *   T_ARM_SELECT_ORACLE     — planted needle outside the window is recalled;
 *                             sinks pinned; window kept; m == B.
 *   T_ARM_TOPK_SET          — quickselect top-k SET == full-sort top-k set.
 *   T_ARM_R1SLOT            — sinks pinned, window modulo, identity when off.
 *   T_ARM_RING2_STDIO       — portable backend round-trips blocks byte-exact.
 */
#include "sp/sp_test.h"
#include "sp/arm.h"

#include <stdlib.h>
#include <string.h>

/* deterministic vector filler (LCG; test-local, NOT the frozen proj seed) */
static uint32_t lcg_s = 0x13572468u;
static float lcg_unit(void) {
    lcg_s = lcg_s * 1664525u + 1013904223u;
    return ((float)(lcg_s >> 8) / 16777216.0f) * 2.0f - 1.0f;   /* [-1,1) */
}

static void T_ARM_R_FROZEN(void) {
    enum { R = 16, HD = 32 };
    signed char A[R * HD], B[R * HD];
    sp_arm_build_R(A, R, HD);
    sp_arm_build_R(B, R, HD);
    SP_CHECK(memcmp(A, B, sizeof(A)) == 0, "R deterministic from frozen seed");
    int plus = 0, ok = 1;
    for (int i = 0; i < R * HD; i++) {
        if (A[i] != 1 && A[i] != -1) ok = 0;
        if (A[i] == 1) plus++;
    }
    SP_CHECK(ok, "R entries are exactly +/-1");
    SP_CHECK(plus > (R * HD) / 4 && plus < (3 * R * HD) / 4,
             "R is roughly balanced (not degenerate)");
}

static void T_ARM_PROJECT_EXACT(void) {
    enum { R = 8, HD = 16 };
    signed char Rm[R * HD];
    float v[HD], got[R];
    sp_arm_build_R(Rm, R, HD);
    for (int i = 0; i < HD; i++) v[i] = lcg_unit();
    sp_arm_project(Rm, R, HD, v, got);
    for (int p = 0; p < R; p++) {
        float want = 0.0f;
        for (int d = 0; d < HD; d++) want += (float)Rm[(size_t)p * HD + d] * v[d];
        SP_CHECK(got[p] == want, "projection row == naive dot (exact)");
    }
}

static void T_ARM_SELECT_PARITY(void) {
    enum { POS = 19 };
    int ri[POS + 1];
    sp_arm_sidx cand[POS + 1];
    /* B=0: full identity */
    int m = sp_arm_select(NULL, 16, 8, NULL, NULL, 0, POS + 1, 1, 0,
                          /*B=*/0, 4, 2, POS, cand, ri);
    SP_CHECK_EQ_I64(m, POS + 1, "B=0 returns full [0,pos]");
    int id = 1; for (int s = 0; s <= POS; s++) if (ri[s] != s) id = 0;
    SP_CHECK(id, "B=0 set is the identity");
    /* within budget: full identity too */
    m = sp_arm_select(NULL, 16, 8, NULL, NULL, 0, POS + 1, 1, 0,
                      /*B=*/POS + 1, 4, 2, POS, cand, ri);
    SP_CHECK_EQ_I64(m, POS + 1, "pos+1<=B returns full [0,pos]");
}

static void T_ARM_SELECT_ORACLE(void) {
    enum { R = 16, HD = 16, P = 64, NKV = 1, NEEDLE = 10,
           SINK = 2, W = 4, B = 8 };           /* topk = B-W-SINK = 2 */
    signed char Rm[R * HD];
    sp_arm_build_R(Rm, R, HD);

    /* keys: small noise everywhere; a large needle at NEEDLE (deep history). */
    static float keys[P][HD];
    for (int s = 0; s < P; s++)
        for (int d = 0; d < HD; d++) keys[s][d] = 0.05f * lcg_unit();
    float q[HD];
    for (int d = 0; d < HD; d++) { q[d] = lcg_unit(); keys[NEEDLE][d] = 4.0f * q[d]; }

    /* stored projection sidecar, exactly as the forward lays it out */
    static float projk[(size_t)P * NKV * R];
    for (int s = 0; s < P; s++)
        sp_arm_project(Rm, R, HD, keys[s], projk + ((size_t)s * NKV + 0) * R);

    int ri[P]; sp_arm_sidx cand[P];
    int pos = P - 1;
    int m = sp_arm_select(Rm, R, HD, q, projk, /*L=*/0, P, NKV, /*kvh=*/0,
                          B, W, SINK, pos, cand, ri);
    SP_CHECK_EQ_I64(m, B, "selection size == budget B");
    int has_needle = 0, has_sink0 = 0, has_sink1 = 0, has_last = 0;
    for (int i = 0; i < m; i++) {
        if (ri[i] == NEEDLE)  has_needle = 1;
        if (ri[i] == 0)       has_sink0 = 1;
        if (ri[i] == 1)       has_sink1 = 1;
        if (ri[i] == pos)     has_last = 1;
    }
    SP_CHECK(has_needle, "planted needle outside the window is recalled (router oracle)");
    SP_CHECK(has_sink0 && has_sink1, "sink anchors pinned in the selection");
    SP_CHECK(has_last, "recent window kept in the selection");
}

static void T_ARM_TOPK_SET(void) {
    enum { N = 257, K = 31, P = 64, NKV = 1, R = 16, HD = 16 };
    /* drive qsel through sp_arm_select against a brute-force reference set */
    signed char Rm[R * HD];
    sp_arm_build_R(Rm, R, HD);
    static float keys[N][HD];
    for (int s = 0; s < N; s++)
        for (int d = 0; d < HD; d++) keys[s][d] = lcg_unit();
    float q[HD]; for (int d = 0; d < HD; d++) q[d] = lcg_unit();
    static float projk[(size_t)N * R];
    for (int s = 0; s < N; s++) sp_arm_project(Rm, R, HD, keys[s], projk + (size_t)s * R);

    enum { SINK = 0, W = 0 };
    int B = K;                                  /* topk = K, no sinks/window */
    int ri[N]; sp_arm_sidx cand[N];
    int m = sp_arm_select(Rm, R, HD, q, projk, 0, N, 1, 0, B, W, SINK, N - 1, cand, ri);
    SP_CHECK_EQ_I64(m, K, "topk-only selection returns K entries");

    /* brute force: exact projected scores, full sort by score */
    float pq[SP_ARM_R_MAX];
    sp_arm_project(Rm, R, HD, q, pq);
    static float score[N];
    for (int s = 0; s < N; s++) {
        float a = 0.0f;
        for (int p = 0; p < R; p++) a += pq[p] * projk[(size_t)s * R + p];
        score[s] = a;
    }
    /* threshold = K-th largest (selection sort over a copy, K passes) */
    static float cp[N]; memcpy(cp, score, sizeof(cp));
    for (int k = 0; k < K; k++) {
        int mx = k;
        for (int j = k + 1; j < N; j++) if (cp[j] > cp[mx]) mx = j;
        float t = cp[k]; cp[k] = cp[mx]; cp[mx] = t;
    }
    float thresh = cp[K - 1];
    int ok = 1;
    for (int i = 0; i < m; i++) if (score[ri[i]] < thresh) ok = 0;
    SP_CHECK(ok, "quickselect top-K set == sort top-K set (>= K-th score)");
}

static void T_ARM_R1SLOT(void) {
    enum { SINK = 4, W = 8 };
    SP_CHECK_EQ_I64(sp_arm_r1slot(2, 1, SINK, W), 2, "sink position pinned (s<sink -> s)");
    SP_CHECK_EQ_I64(sp_arm_r1slot(4, 1, SINK, W), 4, "first window slot");
    SP_CHECK_EQ_I64(sp_arm_r1slot(12, 1, SINK, W), 4, "window wraps modulo W");
    SP_CHECK_EQ_I64(sp_arm_r1slot(17, 1, SINK, W), 4 + (17 - 4) % 8, "arbitrary wrap");
    SP_CHECK_EQ_I64(sp_arm_r1slot(123, 0, SINK, W), 123, "identity when not offloading");
}

static void T_ARM_RING2_STDIO(void) {
    enum { BLK = 256, NB = 3 };
    sp_arm_ring2_backend be;
    SP_CHECK_EQ_I64(sp_arm_ring2_stdio_open(".", &be), 0, "stdio backend opens");
    unsigned char w[NB][BLK], r[BLK];
    for (int b = 0; b < NB; b++)
        for (int i = 0; i < BLK; i++) w[b][i] = (unsigned char)(b * 37 + i);
    /* interleave K/V streams at non-monotonic offsets */
    SP_CHECK_EQ_I64(be.write_block(be.handle, 0, 2 * BLK, w[0], BLK), 0, "write K @2");
    SP_CHECK_EQ_I64(be.write_block(be.handle, 1, 0,       w[1], BLK), 0, "write V @0");
    SP_CHECK_EQ_I64(be.write_block(be.handle, 0, 0,       w[2], BLK), 0, "write K @0");
    SP_CHECK_EQ_I64(be.read_block(be.handle, 0, 2 * BLK, r, BLK), 0, "read K @2");
    SP_CHECK(memcmp(r, w[0], BLK) == 0, "K @2 round-trips byte-exact");
    SP_CHECK_EQ_I64(be.read_block(be.handle, 1, 0, r, BLK), 0, "read V @0");
    SP_CHECK(memcmp(r, w[1], BLK) == 0, "V @0 round-trips byte-exact");
    SP_CHECK_EQ_I64(be.read_block(be.handle, 0, 0, r, BLK), 0, "read K @0");
    SP_CHECK(memcmp(r, w[2], BLK) == 0, "K @0 round-trips byte-exact");
    SP_CHECK(be.read_batch == NULL, "stdio reference declares no batch path (serial)");
    be.close(be.handle);
}


static int dummy_wr(void *h, int w, uint64_t o, const void *s, size_t l) { (void)h;(void)w;(void)o;(void)s;(void)l; return 0; }
static int dummy_rd(void *h, int w, uint64_t o, void *d, size_t l)       { (void)h;(void)w;(void)o;(void)d;(void)l; return 0; }

static void T_ARM_REGISTER(void) {
    sp_arm_ring2_backend got;
    SP_CHECK_EQ_I64(sp_arm_ring2_registered(&got), 0, "no backend registered initially");
    sp_arm_ring2_backend be; memset(&be, 0, sizeof(be));
    be.handle = (void *)0x5150; be.write_block = dummy_wr; be.read_block = dummy_rd;
    sp_arm_ring2_register(&be);
    SP_CHECK_EQ_I64(sp_arm_ring2_registered(&got), 1, "registered backend is visible");
    SP_CHECK(got.handle == (void *)0x5150 && got.write_block == dummy_wr,
             "registered backend round-trips intact");
    sp_arm_ring2_register(NULL);
    SP_CHECK_EQ_I64(sp_arm_ring2_registered(&got), 0, "unregister clears the hook");
}

/* T_ARM_SIG — the bit-packed popcount router (SimHash overlay):
 *   (a) sig bits == signs of the exact float projections (same arithmetic);
 *   (b) B=0 / within-budget identity parity (same off-guarantee);
 *   (c) planted-needle recall through the 1-bit sidecar (the lossy estimator
 *       still finds a strongly-aligned needle — the structural sanity bound);
 *   (d) top-k Hamming property: max selected-candidate distance <= min
 *       unselected distance (tie-tolerant quickselect contract);
 *   (e) sinks + window pinned, m == B. */
static void T_ARM_SIG(void) {
    enum { R = 32, HD = 16, P = 64, NKV = 1, NEEDLE = 10,
           SINK = 2, W = 4, B = 8 };
    signed char Rm[R * HD];
    sp_arm_build_R(Rm, R, HD);

    /* (a) sign agreement with the float projection */
    float v[HD], proj[R];
    for (int d = 0; d < HD; d++) v[d] = lcg_unit();
    sp_arm_project(Rm, R, HD, v, proj);
    uint64_t sig = sp_arm_project_sig(Rm, R, HD, v);
    int sign_ok = 1;
    for (int p = 0; p < R; p++)
        if (((sig >> p) & 1ULL) != (proj[p] >= 0.0f ? 1ULL : 0ULL)) sign_ok = 0;
    SP_CHECK(sign_ok, "sig bits == float projection signs (exact)");

    /* (b) identity parity when off / within budget */
    int ri[P]; sp_arm_sidx cand[P];
    int m = sp_arm_select_sig(NULL, R, HD, NULL, NULL, 0, P, 1, 0,
                              /*B=*/0, W, SINK, P - 1, cand, ri);
    SP_CHECK_EQ_I64(m, P, "sig: B=0 returns full [0,pos]");
    int id = 1; for (int s = 0; s < P; s++) if (ri[s] != s) id = 0;
    SP_CHECK(id, "sig: B=0 set is the identity");

    /* (c)+(d)+(e) needle recall + Hamming top-k property */
    static float keys[P][HD];
    for (int s = 0; s < P; s++)
        for (int d = 0; d < HD; d++) keys[s][d] = 0.05f * lcg_unit();
    float q[HD];
    for (int d = 0; d < HD; d++) { q[d] = lcg_unit(); keys[NEEDLE][d] = 4.0f * q[d]; }
    static uint64_t sigk[(size_t)P * NKV];
    for (int s = 0; s < P; s++)
        sigk[(size_t)s * NKV + 0] = sp_arm_project_sig(Rm, R, HD, keys[s]);

    int pos = P - 1;
    m = sp_arm_select_sig(Rm, R, HD, q, sigk, /*L=*/0, P, NKV, /*kvh=*/0,
                          B, W, SINK, pos, cand, ri);
    SP_CHECK_EQ_I64(m, B, "sig: selection size == budget B");
    int has_needle = 0, has_sink0 = 0, has_last = 0;
    char in_sel[P]; memset(in_sel, 0, sizeof(in_sel));
    for (int i = 0; i < m; i++) {
        in_sel[ri[i]] = 1;
        if (ri[i] == NEEDLE) has_needle = 1;
        if (ri[i] == 0)      has_sink0 = 1;
        if (ri[i] == pos)    has_last = 1;
    }
    SP_CHECK(has_needle, "sig: aligned needle recalled through the 1-bit sidecar");
    SP_CHECK(has_sink0 && has_last, "sig: sinks + window pinned");
    /* Hamming top-k property over the candidate range [SINK, P-W) */
    uint64_t qsig = sp_arm_project_sig(Rm, R, HD, q);
    int max_sel = -1, min_unsel = 999;
    for (int s = SINK; s < P - W; s++) {
        uint64_t x = qsig ^ sigk[(size_t)s * NKV];
        int ham = 0; while (x) { ham += (int)(x & 1ULL); x >>= 1; }
        if (in_sel[s]) { if (ham > max_sel) max_sel = ham; }
        else           { if (ham < min_unsel) min_unsel = ham; }
    }
    SP_CHECK(max_sel <= min_unsel,
             "sig: top-k Hamming property (selected <= unselected, tie-tolerant)");
}

int main(void) {
    SP_RUN(T_ARM_R_FROZEN);
    SP_RUN(T_ARM_PROJECT_EXACT);
    SP_RUN(T_ARM_SELECT_PARITY);
    SP_RUN(T_ARM_SELECT_ORACLE);
    SP_RUN(T_ARM_TOPK_SET);
    SP_RUN(T_ARM_R1SLOT);
    SP_RUN(T_ARM_RING2_STDIO);
    SP_RUN(T_ARM_REGISTER);
    SP_RUN(T_ARM_SIG);
    return SP_DONE();
}
