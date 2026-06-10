/* arm_geom_test.c — T_ARM_GEOM: per-layer-class geometry router gate.
 *
 * The SUBSTRATE half of G-P3-GEOM (CONTRACT-XBAR-C1-lite §3b): the recall
 * router must carry per-layer-class {nkv, head_dim} for the gemma-4 port
 * (GLOBAL: n_kv=1/head_dim=512; SWA: n_kv=2/head_dim=256; class period 6,
 * global at L % 6 == 5 — core/forward/gemma4.c:148-157). Synthetic
 * heterogeneous fixture in exactly that shape, 12 layers (two full periods).
 *
 *   T_GEOM_LAYOUT        — sp_arm_geom_layout offsets are cumulative blocks;
 *                          uniform case reproduces the legacy L*P*NKV*r /
 *                          L*NKV*P offsets exactly.
 *   T_GEOM_STRIDE_GUARD  — (d) R sized at hd_max covers both classes; a
 *                          256-dim projection reads R rows at STRIDE hd_max
 *                          (first 256 columns), bit-equal to the naive dot;
 *                          the legacy call with hd-as-stride provably reads
 *                          DIFFERENT R elements (the audit's corruption mode);
 *                          hd == hd_max is bit-identical to sp_arm_project.
 *   T_GEOM_REPROJECT     — (a) per-class projection determinism: re-projecting
 *                          the stored K (fresh R from the frozen seed)
 *                          reproduces the heterogeneous geom index
 *                          bit-identically (the C1L.0a property, per-class).
 *   T_GEOM_SELECT_ORACLE — (b) selection vs a brute-force f32 reference scan
 *                          per layer/kv-head: m == B, sinks + window pinned,
 *                          content top-k SET == sort top-k set (threshold
 *                          check, the C2.0.6 oracle style), planted per-class
 *                          needle recalled from deep history.
 *   T_GEOM_UNIFORM_NULL  — (c) both classes configured identical => geom
 *                          selections (f32 + sig) are bit-identical to the
 *                          legacy sp_arm_select / sp_arm_select_sig outputs.
 *   T_GEOM_SIG           — sig-geom sign bits == f32 geom projection signs
 *                          per class; heterogeneous select_sig_geom satisfies
 *                          the Hamming top-k property per layer.
 */
#include "sp/sp_test.h"
#include "sp/arm.h"

#include <stdlib.h>
#include <string.h>

/* gemma4-shaped heterogeneous fixture: two classes, period 6, global at
 * L % 6 == 5 — class A (GLOBAL): nkv=1, hd=512; class B (SWA): nkv=2, hd=256. */
enum { R_ = 16, NL = 12, PERIOD = 6, P = 32, POS = P - 1,
       HD_G = 512, NKV_G = 1, HD_S = 256, NKV_S = 2, HD_MAX = HD_G,
       SINK = 2, W = 4, B = 12 };          /* topk = B-W-SINK = 6, cand_hi = 28 */

static int is_global(int L) { return (L % PERIOD) == PERIOD - 1; }

/* deterministic vector filler (LCG; test-local, NOT the frozen proj seed) */
static uint32_t lcg_s = 0x2468ACE1u;
static float lcg_unit(void) {
    lcg_s = lcg_s * 1664525u + 1013904223u;
    return ((float)(lcg_s >> 8) / 16777216.0f) * 2.0f - 1.0f;   /* [-1,1) */
}

static void fill_geom(sp_arm_geom *g) {
    for (int L = 0; L < NL; L++) {
        g[L].nkv = is_global(L) ? NKV_G : NKV_S;
        g[L].hd  = is_global(L) ? HD_G  : HD_S;
    }
}

/* per-layer K block is nkv*P*hd floats — 16384 for BOTH classes (1*32*512 ==
 * 2*32*256; the same constant-KVD coincidence as the episode store). */
static float g_keys[NL][16384];

static void fill_keys(void) {
    for (int L = 0; L < NL; L++)
        for (int i = 0; i < 16384; i++) g_keys[L][i] = 0.05f * lcg_unit();
}
static float *key_at(sp_arm_geom *g, int L, int s, int kvh) {
    return g_keys[L] + ((size_t)s * (size_t)g[L].nkv + (size_t)kvh) * (size_t)g[L].hd;
}

static void T_GEOM_LAYOUT(void) {
    sp_arm_geom g[NL];
    fill_geom(g);
    /* f32 kind: cumulative P*nkv*r blocks */
    size_t tot = sp_arm_geom_layout(g, NL, P, R_, 0);
    size_t want = 0;
    int ok = 1;
    for (int L = 0; L < NL; L++) {
        if (g[L].off != want) ok = 0;
        want += (size_t)P * (size_t)g[L].nkv * (size_t)R_;
    }
    SP_CHECK(ok, "f32 offsets are cumulative per-layer blocks");
    SP_CHECK_EQ_I64((int64_t)tot, (int64_t)want, "f32 total == sum of blocks");
    /* sig kind: cumulative nkv*P blocks */
    tot = sp_arm_geom_layout(g, NL, P, R_, 1);
    want = 0; ok = 1;
    for (int L = 0; L < NL; L++) {
        if (g[L].off != want) ok = 0;
        want += (size_t)g[L].nkv * (size_t)P;
    }
    SP_CHECK(ok, "sig offsets are cumulative per-layer blocks");
    SP_CHECK_EQ_I64((int64_t)tot, (int64_t)want, "sig total == sum of blocks");
    /* uniform case == the legacy layout constants */
    sp_arm_geom u[NL];
    for (int L = 0; L < NL; L++) { u[L].nkv = NKV_S; u[L].hd = HD_S; }
    (void)sp_arm_geom_layout(u, NL, P, R_, 0);
    ok = 1;
    for (int L = 0; L < NL; L++)
        if (u[L].off != (size_t)L * P * NKV_S * R_) ok = 0;
    SP_CHECK(ok, "uniform f32 offsets == legacy L*P*NKV*r");
    (void)sp_arm_geom_layout(u, NL, P, R_, 1);
    ok = 1;
    for (int L = 0; L < NL; L++)
        if (u[L].off != (size_t)L * NKV_S * P) ok = 0;
    SP_CHECK(ok, "uniform sig offsets == legacy L*NKV*P");
}

static void T_GEOM_STRIDE_GUARD(void) {
    /* ONE R at hd_max = 512 serves both classes. */
    static signed char Rm[(size_t)R_ * HD_MAX];
    sp_arm_build_R(Rm, R_, HD_MAX);

    static float v[HD_S], got[R_];
    for (int d = 0; d < HD_S; d++) v[d] = lcg_unit();
    sp_arm_project_geom(Rm, R_, HD_MAX, HD_S, v, got);
    /* reference: first HD_S columns of each row, ROW STRIDE HD_MAX */
    int exact = 1, wrong_differs = 0;
    for (int p = 0; p < R_; p++) {
        float want = 0.0f, corrupt = 0.0f;
        for (int d = 0; d < HD_S; d++) {
            want    += (float)Rm[(size_t)p * HD_MAX + (size_t)d] * v[d];   /* right stride */
            corrupt += (float)Rm[(size_t)p * HD_S  + (size_t)d] * v[d];    /* hd-as-stride */
        }
        if (got[p] != want) exact = 0;
        if (got[p] != corrupt) wrong_differs = 1;   /* p>=1 rows read different R */
    }
    SP_CHECK(exact, "256-dim projection through a 512-built R: row stride hd_max (exact)");
    SP_CHECK(wrong_differs, "hd-as-stride read provably differs (the audit corruption mode is detectable)");

    /* hd == hd_max: geom projection bit-identical to the legacy projection */
    static float v2[HD_MAX], a[R_], b[R_];
    for (int d = 0; d < HD_MAX; d++) v2[d] = lcg_unit();
    sp_arm_project(Rm, R_, HD_MAX, v2, a);
    sp_arm_project_geom(Rm, R_, HD_MAX, HD_MAX, v2, b);
    SP_CHECK(memcmp(a, b, sizeof(a)) == 0, "hd==hd_max geom projection bit-identical to legacy");
    /* same for the sign-bit projection */
    SP_CHECK(sp_arm_project_sig(Rm, R_, HD_MAX, v2)
             == sp_arm_project_sig_geom(Rm, R_, HD_MAX, HD_MAX, v2),
             "hd==hd_max sig projection bit-identical to legacy");
}

static void T_GEOM_REPROJECT(void) {
    sp_arm_geom g[NL];
    fill_geom(g);
    size_t tot = sp_arm_geom_layout(g, NL, P, R_, 0);
    static signed char Rm[(size_t)R_ * HD_MAX];
    sp_arm_build_R(Rm, R_, HD_MAX);
    fill_keys();

    float *projk = (float *)malloc(tot * sizeof(float));
    float *projk2 = (float *)malloc(tot * sizeof(float));
    SP_CHECK(projk && projk2, "sidecar allocations");
    if (!projk || !projk2) { free(projk); free(projk2); return; }

    for (int L = 0; L < NL; L++)
        for (int s = 0; s < P; s++)
            for (int kvh = 0; kvh < g[L].nkv; kvh++)
                sp_arm_project_geom(Rm, R_, HD_MAX, g[L].hd, key_at(g, L, s, kvh),
                                    projk + g[L].off + ((size_t)s * (size_t)g[L].nkv + (size_t)kvh) * R_);

    /* re-projection: fresh R from the frozen seed + a second pass over the
     * STORED K must reproduce the heterogeneous index bit-identically
     * (C1L.0a, now per-class — the episode-replay rebuild property). */
    static signed char Rm2[(size_t)R_ * HD_MAX];
    sp_arm_build_R(Rm2, R_, HD_MAX);
    SP_CHECK(memcmp(Rm, Rm2, sizeof(Rm)) == 0, "R rebuilt from the frozen seed is identical");
    for (int L = 0; L < NL; L++)
        for (int s = 0; s < P; s++)
            for (int kvh = 0; kvh < g[L].nkv; kvh++)
                sp_arm_project_geom(Rm2, R_, HD_MAX, g[L].hd, key_at(g, L, s, kvh),
                                    projk2 + g[L].off + ((size_t)s * (size_t)g[L].nkv + (size_t)kvh) * R_);
    SP_CHECK(memcmp(projk, projk2, tot * sizeof(float)) == 0,
             "re-projection of stored K reproduces the geom index bit-identically");
    free(projk); free(projk2);
}

static void T_GEOM_SELECT_ORACLE(void) {
    sp_arm_geom g[NL];
    fill_geom(g);
    size_t tot = sp_arm_geom_layout(g, NL, P, R_, 0);
    static signed char Rm[(size_t)R_ * HD_MAX];
    sp_arm_build_R(Rm, R_, HD_MAX);
    fill_keys();

    float *projk = (float *)malloc(tot * sizeof(float));
    SP_CHECK(projk != NULL, "sidecar allocation");
    if (!projk) return;

    enum { NEEDLE = 9 };                       /* deep history: < cand_hi, > SINK */
    static float q[HD_MAX];
    int all_ok = 1, all_needle = 1;
    for (int L = 0; L < NL; L++) {
        for (int kvh = 0; kvh < g[L].nkv; kvh++) {
            const int hd = g[L].hd;
            for (int d = 0; d < hd; d++) q[d] = lcg_unit();
            /* plant a per-class needle for THIS head at NEEDLE */
            float *nk = key_at(g, L, NEEDLE, kvh);
            for (int d = 0; d < hd; d++) nk[d] = 4.0f * q[d];
            /* (re)project this layer's keys */
            for (int s = 0; s < P; s++)
                for (int h2 = 0; h2 < g[L].nkv; h2++)
                    sp_arm_project_geom(Rm, R_, HD_MAX, hd, key_at(g, L, s, h2),
                                        projk + g[L].off + ((size_t)s * (size_t)g[L].nkv + (size_t)h2) * R_);

            int ri[P]; sp_arm_sidx cand[P];
            int m = sp_arm_select_geom(Rm, R_, HD_MAX, &g[L], q, projk, P, kvh,
                                       B, W, SINK, POS, cand, ri);
            if (m != B) all_ok = 0;

            /* brute-force f32 reference scan over THIS layer's class dims */
            float pq[SP_ARM_R_MAX];
            sp_arm_project_geom(Rm, R_, HD_MAX, hd, q, pq);
            const int cand_hi = POS + 1 - W, topk = B - W - SINK;
            float score[P];
            for (int s = SINK; s < cand_hi; s++) {
                const float *pk = projk + g[L].off + ((size_t)s * (size_t)g[L].nkv + (size_t)kvh) * R_;
                float a = 0.0f;
                for (int p = 0; p < R_; p++) a += pq[p] * pk[p];
                score[s] = a;
            }
            /* threshold = topk-th largest (selection sort over a copy) */
            float cp[P]; int ncand = cand_hi - SINK;
            memcpy(cp, score + SINK, (size_t)ncand * sizeof(float));
            for (int k = 0; k < topk; k++) {
                int mx = k;
                for (int j = k + 1; j < ncand; j++) if (cp[j] > cp[mx]) mx = j;
                float t = cp[k]; cp[k] = cp[mx]; cp[mx] = t;
            }
            const float thresh = cp[topk - 1];
            /* check structure: sinks + window all present; every content pick
             * >= threshold (set equality vs the sort reference, tie-free data);
             * exactly topk content picks; needle recalled. */
            int nsink = 0, nwin = 0, ncontent = 0, needle = 0, sel_ok = 1;
            for (int i = 0; i < m; i++) {
                int s = ri[i];
                if (s < SINK) nsink++;
                else if (s >= cand_hi) nwin++;
                else {
                    ncontent++;
                    if (score[s] < thresh) sel_ok = 0;
                    if (s == NEEDLE) needle = 1;
                }
            }
            if (nsink != SINK || nwin != W || ncontent != topk || !sel_ok) all_ok = 0;
            if (!needle) all_needle = 0;
        }
    }
    SP_CHECK(all_ok, "per-layer geom selection == brute-force f32 top-B reference (all layers x kv-heads)");
    SP_CHECK(all_needle, "per-class planted needle recalled from deep history (both classes)");
    free(projk);
}

static void T_GEOM_UNIFORM_NULL(void) {
    /* both classes configured IDENTICAL (the qwen3 degenerate case):
     * geom selections must be bit-identical to the legacy path. */
    enum { UNKV = 2, UHD = 256 };
    sp_arm_geom g[NL];
    for (int L = 0; L < NL; L++) { g[L].nkv = UNKV; g[L].hd = UHD; }
    size_t tot = sp_arm_geom_layout(g, NL, P, R_, 0);
    SP_CHECK_EQ_I64((int64_t)tot, (int64_t)((size_t)NL * P * UNKV * R_),
                    "uniform f32 total == legacy allocation size");
    static signed char Rm[(size_t)R_ * UHD];
    sp_arm_build_R(Rm, R_, UHD);
    fill_keys();   /* reuse the key pool; uniform addressing below */

    float *projk = (float *)malloc(tot * sizeof(float));
    uint64_t *sigk = (uint64_t *)malloc((size_t)NL * UNKV * P * sizeof(uint64_t));
    SP_CHECK(projk && sigk, "sidecar allocations");
    if (!projk || !sigk) { free(projk); free(sigk); return; }

    /* fill BOTH sidecars via the LEGACY entry points + legacy layouts */
    for (int L = 0; L < NL; L++)
        for (int s = 0; s < P; s++)
            for (int kvh = 0; kvh < UNKV; kvh++) {
                const float *k = g_keys[L] + ((size_t)s * UNKV + (size_t)kvh) * UHD;
                sp_arm_project(Rm, R_, UHD, k,
                               projk + (((size_t)L * P + (size_t)s) * UNKV + (size_t)kvh) * R_);
                sigk[((size_t)L * UNKV + (size_t)kvh) * P + (size_t)s] =
                    sp_arm_project_sig(Rm, R_, UHD, k);
            }

    static float q[UHD];
    int f32_ok = 1, sig_ok = 1;
    for (int L = 0; L < NL; L++)
        for (int kvh = 0; kvh < UNKV; kvh++) {
            for (int d = 0; d < UHD; d++) q[d] = lcg_unit();
            int ri_a[P], ri_b[P]; sp_arm_sidx cand[P];
            int ma = sp_arm_select(Rm, R_, UHD, q, projk, (size_t)L, P, UNKV, kvh,
                                   B, W, SINK, POS, cand, ri_a);
            int mb = sp_arm_select_geom(Rm, R_, UHD, &g[L], q, projk, P, kvh,
                                        B, W, SINK, POS, cand, ri_b);
            if (ma != mb || memcmp(ri_a, ri_b, (size_t)ma * sizeof(int)) != 0) f32_ok = 0;
        }
    SP_CHECK(f32_ok, "uniform-null: f32 geom selections bit-identical to legacy sp_arm_select");

    sp_arm_geom gs[NL];
    for (int L = 0; L < NL; L++) { gs[L].nkv = UNKV; gs[L].hd = UHD; }
    (void)sp_arm_geom_layout(gs, NL, P, R_, 1);
    for (int L = 0; L < NL; L++)
        for (int kvh = 0; kvh < UNKV; kvh++) {
            for (int d = 0; d < UHD; d++) q[d] = lcg_unit();
            int ri_a[P], ri_b[P]; sp_arm_sidx cand[P];
            int ma = sp_arm_select_sig(Rm, R_, UHD, q, sigk, (size_t)L, P, UNKV, kvh,
                                       B, W, SINK, POS, cand, ri_a);
            int mb = sp_arm_select_sig_geom(Rm, R_, UHD, &gs[L], q, sigk, P, kvh,
                                            B, W, SINK, POS, cand, ri_b);
            if (ma != mb || memcmp(ri_a, ri_b, (size_t)ma * sizeof(int)) != 0) sig_ok = 0;
        }
    SP_CHECK(sig_ok, "uniform-null: sig geom selections bit-identical to legacy sp_arm_select_sig");

    /* identity parity carries over: B=0 => full [0,pos] on the geom path too */
    int ri[P]; sp_arm_sidx cand[P];
    int m = sp_arm_select_geom(NULL, R_, UHD, &g[0], NULL, NULL, P, 0,
                               0, W, SINK, POS, cand, ri);
    SP_CHECK_EQ_I64(m, P, "geom: B=0 returns full [0,pos]");
    int id = 1; for (int s = 0; s < P; s++) if (ri[s] != s) id = 0;
    SP_CHECK(id, "geom: B=0 set is the identity");
    free(projk); free(sigk);
}

static void T_GEOM_SIG(void) {
    sp_arm_geom g[NL];
    fill_geom(g);
    size_t tot = sp_arm_geom_layout(g, NL, P, R_, 1);
    static signed char Rm[(size_t)R_ * HD_MAX];
    sp_arm_build_R(Rm, R_, HD_MAX);
    fill_keys();

    /* sign agreement per class: sig bits == f32 geom projection signs */
    static float v[HD_S], proj[R_];
    for (int d = 0; d < HD_S; d++) v[d] = lcg_unit();
    sp_arm_project_geom(Rm, R_, HD_MAX, HD_S, v, proj);
    uint64_t sig = sp_arm_project_sig_geom(Rm, R_, HD_MAX, HD_S, v);
    int sign_ok = 1;
    for (int p = 0; p < R_; p++)
        if (((sig >> p) & 1ULL) != (proj[p] >= 0.0f ? 1ULL : 0ULL)) sign_ok = 0;
    SP_CHECK(sign_ok, "sig-geom bits == f32 geom projection signs (256-dim class through 512-built R)");

    uint64_t *sigk = (uint64_t *)malloc(tot * sizeof(uint64_t));
    SP_CHECK(sigk != NULL, "sig sidecar allocation");
    if (!sigk) return;
    for (int L = 0; L < NL; L++)
        for (int s = 0; s < P; s++)
            for (int kvh = 0; kvh < g[L].nkv; kvh++)
                sigk[g[L].off + (size_t)kvh * P + (size_t)s] =
                    sp_arm_project_sig_geom(Rm, R_, HD_MAX, g[L].hd, key_at(g, L, s, kvh));

    /* Hamming top-k property per layer/kv-head (tie-tolerant) */
    static float q[HD_MAX];
    int ham_ok = 1, struct_ok = 1;
    for (int L = 0; L < NL; L++)
        for (int kvh = 0; kvh < g[L].nkv; kvh++) {
            const int hd = g[L].hd;
            for (int d = 0; d < hd; d++) q[d] = lcg_unit();
            int ri[P]; sp_arm_sidx cand[P];
            int m = sp_arm_select_sig_geom(Rm, R_, HD_MAX, &g[L], q, sigk, P, kvh,
                                           B, W, SINK, POS, cand, ri);
            if (m != B) struct_ok = 0;
            char in_sel[P]; memset(in_sel, 0, sizeof(in_sel));
            for (int i = 0; i < m; i++) in_sel[ri[i]] = 1;
            if (!in_sel[0] || !in_sel[POS]) struct_ok = 0;
            uint64_t qsig = sp_arm_project_sig_geom(Rm, R_, HD_MAX, hd, q);
            const int cand_hi = POS + 1 - W;
            int max_sel = -1, min_unsel = 999;
            for (int s = SINK; s < cand_hi; s++) {
                uint64_t x = qsig ^ sigk[g[L].off + (size_t)kvh * P + (size_t)s];
                int ham = 0; while (x) { ham += (int)(x & 1ULL); x >>= 1; }
                if (in_sel[s]) { if (ham > max_sel) max_sel = ham; }
                else           { if (ham < min_unsel) min_unsel = ham; }
            }
            if (max_sel > min_unsel) ham_ok = 0;
        }
    SP_CHECK(struct_ok, "sig-geom: m == B, sinks + window pinned (all layers x kv-heads)");
    SP_CHECK(ham_ok, "sig-geom: top-k Hamming property per layer (selected <= unselected)");
    free(sigk);
}

int main(void) {
    SP_RUN(T_GEOM_LAYOUT);
    SP_RUN(T_GEOM_STRIDE_GUARD);
    SP_RUN(T_GEOM_REPROJECT);
    SP_RUN(T_GEOM_SELECT_ORACLE);
    SP_RUN(T_GEOM_UNIFORM_NULL);
    SP_RUN(T_GEOM_SIG);
    return SP_DONE();
}
