/* ring3_gate.c — T_RING3_NATIVE: the native-C Ring-3 VSA + NIGHTSHIFT gate.
 *
 * Gates the core/ring3 port (ring3.c) against the host-Python reference
 * (tools/ring3/ok_bind.py + g_r3_nightshift.py), bit-exact where the math is
 * integer-exact. Pre-registered thresholds (set BEFORE the asserts):
 *
 *   (a) ARITHMETIC BIT-EXACTNESS — for the fixed seed set in the fixture, the C
 *       sp_r3_bind / sp_r3_unbind / superpose M-coefficient vectors are
 *       BIT-IDENTICAL (int64, exact) to the Python ok_bind reference (which
 *       routes through the SAME native sp_pr_mul). Threshold: ZERO differing
 *       coefficients across all cases. (Exact by construction — both go through
 *       sp_pr_mul + the canonical splitmix64 ±1 generator.)
 *
 *   (b) NIGHTSHIFT VERDICT REPRODUCED — D=1024 CAP=32 over the same 40-episode
 *       seed batch: vectors seal at sizes [32, 8], every consolidated episode
 *       recalls@1 (GREEN); D=128: the shadow-gate fires BEFORE the cap (max
 *       vector size < 32). Thresholds: sizes==[32,8] & all_recall_ok &
 *       cap_seal (D=1024); all_recall_ok & gate_seal & max<32 (D=128).
 *
 *   (c) REDUCTION-ORDER IMMUNITY — the superposition M is BYTE-IDENTICAL across
 *       >=2 episode-bind orderings (inherited from integer arithmetic).
 *       Threshold: memcmp == 0.
 *
 * Fixture: T_RING3_NATIVE_fixture.bin (emitted by gen_ring3_fixture.py).
 * Reproduce:
 *   cmake -S core/ring3 -B core/ring3/build -G Ninja
 *   cmake --build core/ring3/build --target test_ring3
 *   (cd core/ring3 && build/test_ring3.exe)
 */
#include "sp/sp_test.h"
#include "sp/ring3.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Located relative to the test's CWD; the gate is run from core/ring3/. */
#define FIXTURE_PATH "T_RING3_NATIVE_fixture.bin"
#define FIXTURE_MAGIC 0x52334E54u

static uint64_t  g_seeds[64];
static int       g_nseeds = 0;

typedef struct { int D; uint64_t a, b; int64_t *bind, *unbind, *super2; } r3_case;
static r3_case g_cases[8];
static int     g_ncases = 0;

static int rd_u32(FILE *f, uint32_t *v) { return fread(v, 4, 1, f) == 1; }
static int rd_i64(FILE *f, int64_t  *v) { return fread(v, 8, 1, f) == 1; }

static int load_fixture(void) {
    FILE *f = fopen(FIXTURE_PATH, "rb");
    if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", FIXTURE_PATH); return 1; }
    uint32_t magic = 0, ver = 0, n = 0, nc = 0;
    if (!rd_u32(f, &magic) || magic != FIXTURE_MAGIC) { fclose(f); fprintf(stderr, "FATAL: bad magic\n"); return 1; }
    rd_u32(f, &ver);
    rd_u32(f, &n);
    if (n > 64) { fclose(f); return 1; }
    g_nseeds = (int)n;
    for (int i = 0; i < g_nseeds; i++) { int64_t s; rd_i64(f, &s); g_seeds[i] = (uint64_t)s; }
    rd_u32(f, &nc);
    if (nc > 8) { fclose(f); return 1; }
    g_ncases = (int)nc;
    for (int c = 0; c < g_ncases; c++) {
        uint32_t D = 0; int64_t a = 0, b = 0;
        rd_u32(f, &D); rd_i64(f, &a); rd_i64(f, &b);
        r3_case *cc = &g_cases[c];
        cc->D = (int)D; cc->a = (uint64_t)a; cc->b = (uint64_t)b;
        cc->bind   = malloc((size_t)D * sizeof(int64_t));
        cc->unbind = malloc((size_t)D * sizeof(int64_t));
        cc->super2 = malloc((size_t)D * sizeof(int64_t));
        if (fread(cc->bind,   8, D, f) != D) { fclose(f); return 1; }
        if (fread(cc->unbind, 8, D, f) != D) { fclose(f); return 1; }
        if (fread(cc->super2, 8, D, f) != D) { fclose(f); return 1; }
    }
    fclose(f);
    fprintf(stderr, "    [fixture] %d seeds, %d bind/unbind cases loaded\n", g_nseeds, g_ncases);
    return 0;
}

/* (a) arithmetic bit-exactness vs the Python ok_bind reference. */
static void T_RING3_BIND_BITEXACT(void) {
    for (int c = 0; c < g_ncases; c++) {
        r3_case *cc = &g_cases[c];
        int D = cc->D;
        int8_t  *addr = malloc((size_t)D), *idv = malloc((size_t)D);
        int8_t  *addrB = malloc((size_t)D), *idvB = malloc((size_t)D);
        int64_t *bind = malloc((size_t)D * sizeof(int64_t));
        int64_t *unb  = malloc((size_t)D * sizeof(int64_t));
        int64_t *sup  = malloc((size_t)D * sizeof(int64_t));

        sp_r3_carrier(cc->a, D, addr);  sp_r3_idvec(cc->a, D, idv);
        sp_r3_carrier(cc->b, D, addrB); sp_r3_idvec(cc->b, D, idvB);

        SP_CHECK_EQ_I64(sp_r3_bind(addr, idv, D, bind), 0, "sp_r3_bind ok");
        SP_CHECK(memcmp(bind, cc->bind, (size_t)D * sizeof(int64_t)) == 0,
                 "sp_r3_bind == ok_bind.bind reference (bit-identical int64)");

        SP_CHECK_EQ_I64(sp_r3_unbind(bind, addr, D, unb), 0, "sp_r3_unbind ok");
        SP_CHECK(memcmp(unb, cc->unbind, (size_t)D * sizeof(int64_t)) == 0,
                 "sp_r3_unbind == ok_bind.unbind reference (bit-identical int64)");

        memset(sup, 0, (size_t)D * sizeof(int64_t));
        SP_CHECK_EQ_I64(sp_r3_superpose(sup, addr,  idv,  D), 0, "superpose A ok");
        SP_CHECK_EQ_I64(sp_r3_superpose(sup, addrB, idvB, D), 0, "superpose B ok");
        SP_CHECK(memcmp(sup, cc->super2, (size_t)D * sizeof(int64_t)) == 0,
                 "superpose(A)+superpose(B) == ok reference (bit-identical int64)");

        free(addr); free(idv); free(addrB); free(idvB);
        free(bind); free(unb); free(sup);
    }
}

/* (c) reduction-order immunity: M byte-identical across two bind orderings. */
static void T_RING3_ORDER_IMMUNE(void) {
    for (int c = 0; c < g_ncases; c++) {
        r3_case *cc = &g_cases[c];
        int D = cc->D;
        int8_t *aA = malloc((size_t)D), *iA = malloc((size_t)D);
        int8_t *aB = malloc((size_t)D), *iB = malloc((size_t)D);
        sp_r3_carrier(cc->a, D, aA); sp_r3_idvec(cc->a, D, iA);
        sp_r3_carrier(cc->b, D, aB); sp_r3_idvec(cc->b, D, iB);

        int64_t *fwd = calloc((size_t)D, sizeof(int64_t));
        int64_t *rev = calloc((size_t)D, sizeof(int64_t));
        sp_r3_superpose(fwd, aA, iA, D); sp_r3_superpose(fwd, aB, iB, D);  /* A then B */
        sp_r3_superpose(rev, aB, iB, D); sp_r3_superpose(rev, aA, iA, D);  /* B then A */
        SP_CHECK(memcmp(fwd, rev, (size_t)D * sizeof(int64_t)) == 0,
                 "superposition M is byte-identical across bind orderings (order-immune)");
        free(aA); free(iA); free(aB); free(iB); free(fwd); free(rev);
    }
}

/* (b) the NIGHTSHIFT verdict reproduced natively. */
static void T_RING3_NIGHTSHIFT(void) {
    SP_CHECK(g_nseeds == 40, "fixture carries the canonical 40-episode batch");

    sp_r3_nightshift_result rA;
    SP_CHECK_EQ_I64(sp_r3_nightshift(1024, g_seeds, g_nseeds, &rA), 0,
                    "nightshift D=1024 runs");
    fprintf(stderr, "    [ns D=1024] vectors=%d sizes=[", rA.n_vectors);
    for (int i = 0; i < rA.n_vectors; i++)
        fprintf(stderr, "%s%d", i ? "," : "", rA.sizes[i]);
    fprintf(stderr, "] recall_ok=%d cap_seal=%d\n", rA.all_recall_ok, rA.cap_seal);

    SP_CHECK(rA.n_vectors == 2 && rA.sizes[0] == 32 && rA.sizes[1] == 8,
             "D=1024 vectors seal at sizes [32, 8] (matches Python verdict)");
    SP_CHECK(rA.all_recall_ok,
             "D=1024 every consolidated episode recalls@1 (GREEN)");
    SP_CHECK(rA.cap_seal,
             "D=1024 non-last vectors sealed at CAP=32 (production seal)");
    SP_CHECK(rA.consolidated == 40, "D=1024 all 40 episodes consolidated");

    sp_r3_nightshift_result rB;
    SP_CHECK_EQ_I64(sp_r3_nightshift(128, g_seeds, g_nseeds, &rB), 0,
                    "nightshift D=128 runs");
    int maxB = 0;
    for (int i = 0; i < rB.n_vectors; i++) if (rB.sizes[i] > maxB) maxB = rB.sizes[i];
    fprintf(stderr, "    [ns D=128 ] vectors=%d sizes=[", rB.n_vectors);
    for (int i = 0; i < rB.n_vectors; i++)
        fprintf(stderr, "%s%d", i ? "," : "", rB.sizes[i]);
    fprintf(stderr, "] max=%d recall_ok=%d gate_seal=%d\n", maxB, rB.all_recall_ok, rB.gate_seal);

    SP_CHECK(rB.all_recall_ok,
             "D=128 every consolidated episode recalls@1 (GREEN)");
    SP_CHECK(rB.gate_seal && maxB < SP_R3_CAP,
             "D=128 shadow-gate fires BEFORE the cap (max vector < 32) — seal is the math");
    SP_CHECK(rB.consolidated == 40, "D=128 all 40 episodes consolidated");
}

int main(void) {
    if (load_fixture()) return 1;
    SP_RUN(T_RING3_BIND_BITEXACT);
    SP_RUN(T_RING3_ORDER_IMMUNE);
    SP_RUN(T_RING3_NIGHTSHIFT);
    for (int c = 0; c < g_ncases; c++) {
        free(g_cases[c].bind); free(g_cases[c].unbind); free(g_cases[c].super2);
    }
    return SP_DONE();
}
