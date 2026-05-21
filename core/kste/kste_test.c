/* kste_test.c — contract tests for the Knight-Spinor Tree Encoder (Phase 1F).
 *
 * One T_KSTE_n per roadmap item, driven by sp_test.h. All randomness uses a
 * fixed-seed xorshift64* RNG so every run (and every platform) is identical.
 */
#include "sp/sp_test.h"
#include "sp/kste.h"

#include <stdint.h>
#include <string.h>

/* ---- deterministic RNG (xorshift64*, fixed seed) ------------------------- */

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static void rng_seed(uint64_t s) { rng_state = s ? s : 0x9E3779B97F4A7C15ull; }

static uint64_t rng_next(void) {
    uint64_t x = rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1Dull;
}

/* uniform int32 in [-range, range] */
static int32_t rng_i32(int32_t range) {
    uint64_t span = (uint64_t)(2 * (int64_t)range + 1);
    return (int32_t)((int64_t)(rng_next() % span) - range);
}

/* ---- local helpers ------------------------------------------------------- */

static int tree_eq(const sp_kste_tree_t *a, const sp_kste_tree_t *b) {
    return memcmp(a->bytes, b->bytes, 64) == 0;
}

/* compose a relation r = compare(a,b) with s = compare(b,c) for transitivity:
 * if a<=b and b<=c then a<=c. We test the "<=" facet (DOMINATED or EQUIVALENT)
 * and the ">=" facet (DOMINATES or EQUIVALENT) separately. */
static int leq(sp_dom_t r) { return r == SP_DOMINATED || r == SP_EQUIVALENT; }
static int geq(sp_dom_t r) { return r == SP_DOMINATES || r == SP_EQUIVALENT; }

/* ---- T_KSTE_1: encoder determinism over 1,000,000 trials ----------------- */

static void T_KSTE_1(void) {
    rng_seed(0x1111111111111111ull);

    /* (a) the SAME fixed vector encodes identically 1,000,000 times */
    int32_t fixed[16] = { 7, -3, 19, 0, 4, -100, 55, 2,
                          8, 8, -8, 13, 21, -34, 1, 999 };
    sp_kste_tree_t base;
    sp_kste_encode(fixed, 16, &base);

    int stable = 1;
    for (int i = 0; i < 1000000; i++) {
        sp_kste_tree_t t;
        sp_kste_encode(fixed, 16, &t);
        if (!tree_eq(&t, &base)) { stable = 0; break; }
    }
    SP_CHECK(stable, "T_KSTE_1 same vector -> identical bytes (1e6 trials)");

    /* (b) varied vectors: each re-encode of a given vector matches its first */
    int varied_stable = 1;
    rng_seed(0x2222222222222222ull);
    for (int i = 0; i < 5000 && varied_stable; i++) {
        int32_t v[12];
        for (int j = 0; j < 12; j++) v[j] = rng_i32(50000);
        sp_kste_tree_t a, b;
        sp_kste_encode(v, 12, &a);
        sp_kste_encode(v, 12, &b);
        if (!tree_eq(&a, &b)) varied_stable = 0;
    }
    SP_CHECK(varied_stable, "T_KSTE_1 varied vectors re-encode identically");

    /* (c) header fields are exactly the frozen constants */
    SP_CHECK_EQ_I64(base.bytes[SP_KSTE_OFF_VERSION], SP_KSTE_LAYOUT_VERSION,
                    "T_KSTE_1 version byte");
    SP_CHECK_EQ_I64(base.bytes[SP_KSTE_OFF_BRANCH], SP_KSTE_BRANCHING,
                    "T_KSTE_1 branching byte");
    SP_CHECK_EQ_I64(base.bytes[SP_KSTE_OFF_DEPTH], SP_KSTE_DEPTH,
                    "T_KSTE_1 depth byte");
}

/* ---- T_KSTE_2: Tier-0 partial-order axioms ------------------------------- */

static void T_KSTE_2(void) {
    rng_seed(0x3333333333333333ull);

    enum { N = 400 };
    static sp_kste_tree_t trees[N];
    for (int i = 0; i < N; i++) {
        int k = (int)(rng_next() % 24) + 1;       /* 1..24 components */
        int32_t v[24];
        for (int j = 0; j < k; j++) v[j] = rng_i32(2000);
        sp_kste_encode(v, k, &trees[i]);
    }

    /* reflexivity: a vs a is EQUIVALENT */
    int refl = 1;
    for (int i = 0; i < N; i++)
        if (sp_kste_tier0(&trees[i], &trees[i]) != SP_EQUIVALENT) refl = 0;
    SP_CHECK(refl, "T_KSTE_2 Tier-0 reflexive (a vs a == EQUIVALENT)");

    /* antisymmetry up to equivalence: a<=b and b<=a  =>  EQUIVALENT;
     * and the relation is consistent (tier0(a,b) is the dual of tier0(b,a)) */
    int antisym = 1, dual = 1, saw_incomp = 0, saw_dom = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            sp_dom_t rij = sp_kste_tier0(&trees[i], &trees[j]);
            sp_dom_t rji = sp_kste_tier0(&trees[j], &trees[i]);

            if (rij == SP_INCOMPARABLE) saw_incomp = 1;
            if (rij == SP_DOMINATES || rij == SP_DOMINATED) saw_dom = 1;

            /* duality: DOMINATES <-> DOMINATED, EQUIVALENT/INCOMPARABLE self */
            sp_dom_t expect;
            switch (rij) {
                case SP_DOMINATES:    expect = SP_DOMINATED; break;
                case SP_DOMINATED:    expect = SP_DOMINATES; break;
                default:              expect = rij;          break;
            }
            if (rji != expect) dual = 0;

            /* a<=b and b<=a => equivalent */
            if (leq(rij) && geq(rij) && rij != SP_EQUIVALENT) antisym = 0;
            if (leq(rij) && leq(rji) && rij != SP_EQUIVALENT) antisym = 0;
        }
    }
    SP_CHECK(antisym, "T_KSTE_2 Tier-0 antisymmetric up to equivalence");
    SP_CHECK(dual, "T_KSTE_2 Tier-0 relation is its own dual under swap");
    SP_CHECK(saw_incomp, "T_KSTE_2 Tier-0 produced INCOMPARABLE pairs");
    SP_CHECK(saw_dom, "T_KSTE_2 Tier-0 produced strict dominance pairs");

    /* transitivity on the <= facet: a<=b and b<=c => a<=c */
    int trans = 1;
    for (int i = 0; i < N && trans; i++)
        for (int j = 0; j < N && trans; j++) {
            if (!leq(sp_kste_tier0(&trees[i], &trees[j]))) continue;
            for (int m = 0; m < N; m++) {
                if (!leq(sp_kste_tier0(&trees[j], &trees[m]))) continue;
                if (!leq(sp_kste_tier0(&trees[i], &trees[m]))) { trans = 0; break; }
            }
        }
    SP_CHECK(trans, "T_KSTE_2 Tier-0 transitive on <= facet");
}

/* ---- T_KSTE_3: Tier-1 confirmation rule ---------------------------------- */

static void T_KSTE_3(void) {
    /* Match case: identical vectors -> identical first-level child region,
     * so Tier-1 reports EQUIVALENT. */
    int32_t v1[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    sp_kste_tree_t a, b;
    sp_kste_encode(v1, 8, &a);
    sp_kste_encode(v1, 8, &b);
    SP_CHECK(sp_kste_tier0(&a, &b) == SP_EQUIVALENT,
             "T_KSTE_3 identical vectors pass Tier-0");
    SP_CHECK(sp_kste_tier1(&a, &b) == SP_EQUIVALENT,
             "T_KSTE_3 identical vectors pass Tier-1 (same child multiset)");
    /* and the child byte regions are literally identical */
    SP_CHECK(memcmp(a.bytes + SP_KSTE_OFF_CHILDREN,
                    b.bytes + SP_KSTE_OFF_CHILDREN, 36) == 0,
             "T_KSTE_3 child region byte-identical on match");

    /* Near-miss: same GLOBAL order statistics (so Tier-0 ties / passes) but
     * different per-slice distribution, so Tier-1 must catch it.
     * [1..8] and its reverse share global min/max/percentile structure but the
     * three first-level slices carry different content. */
    int32_t fwd[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    int32_t rev[8] = { 8, 7, 6, 5, 4, 3, 2, 1 };
    sp_kste_tree_t tf, tr;
    sp_kste_encode(fwd, 8, &tf);
    sp_kste_encode(rev, 8, &tr);

    sp_dom_t t0 = sp_kste_tier0(&tf, &tr);
    sp_dom_t t1 = sp_kste_tier1(&tf, &tr);
    /* fwd and rev have identical sorted multiset -> identical root stats ->
     * Tier-0 EQUIVALENT. */
    SP_CHECK(t0 == SP_EQUIVALENT,
             "T_KSTE_3 near-miss passes Tier-0 (same root fingerprint)");
    /* Tier-1 must NOT call them equivalent: child multisets differ. */
    SP_CHECK(t1 != SP_EQUIVALENT,
             "T_KSTE_3 near-miss CAUGHT by Tier-1 (child multiset differs)");
    /* concretely, the child byte regions differ */
    SP_CHECK(memcmp(tf.bytes + SP_KSTE_OFF_CHILDREN,
                    tr.bytes + SP_KSTE_OFF_CHILDREN, 36) != 0,
             "T_KSTE_3 near-miss child region differs in bytes");

    /* Positive confirmation rule, stated directly: any two trees that are
     * Tier-0 EQUIVALENT AND Tier-1 EQUIVALENT share the same first-level child
     * multiset (identical canonical child bytes). Verify across random pairs. */
    rng_seed(0x4444444444444444ull);
    int confirmed = 1, saw_t1_equiv = 0;
    for (int i = 0; i < 20000; i++) {
        int32_t x[6], y[6];
        for (int j = 0; j < 6; j++) x[j] = rng_i32(3);   /* small range -> */
        /* half the time compare a vector to a fresh random one, half the time
         * to a permutation of itself. Permutations that re-bucket into the same
         * canonical child set are exactly the Tier-0&Tier-1 equivalent pairs we
         * want to observe; the small value range makes collisions frequent. */
        if (i & 1) {
            for (int j = 0; j < 6; j++) y[j] = x[5 - j];  /* reverse */
        } else {
            for (int j = 0; j < 6; j++) y[j] = rng_i32(3);
        }
        sp_kste_tree_t tx, ty;
        sp_kste_encode(x, 6, &tx);
        sp_kste_encode(y, 6, &ty);
        if (sp_kste_tier0(&tx, &ty) == SP_EQUIVALENT &&
            sp_kste_tier1(&tx, &ty) == SP_EQUIVALENT) {
            saw_t1_equiv = 1;
            if (memcmp(tx.bytes + SP_KSTE_OFF_CHILDREN,
                       ty.bytes + SP_KSTE_OFF_CHILDREN, 36) != 0)
                confirmed = 0;
        }
    }
    SP_CHECK(saw_t1_equiv, "T_KSTE_3 saw Tier-0&Tier-1 equivalent pairs");
    SP_CHECK(confirmed,
             "T_KSTE_3 Tier-0+Tier-1 equivalent => identical child multiset");
}

/* ---- T_KSTE_4: cross-platform byte anchor -------------------------------- */

static void T_KSTE_4(void) {
    /* Anchor input. The expected[] image is whatever this deterministic
     * encoder produced on the reference (MinGW gcc) build; CI on Linux/MSVC
     * diffs against it. Any future drift in the layout or quantization trips
     * this test. */
    int32_t anchor[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    sp_kste_tree_t t;
    sp_kste_encode(anchor, 8, &t);

    static const uint8_t expected[64] = {
        /* header: version, branch, depth, reserved, k=8 (u32 LE) */
        0x01, 0x03, 0x03, 0x00, 0x08, 0x00, 0x00, 0x00,
        /* Tier-0 root label: 1,2,3,5,6,8 (int16 LE) */
        0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x05, 0x00, 0x06, 0x00, 0x08, 0x00,
        /* Tier-1 child[0] = {1,1,1,2,2,3} */
        0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x02, 0x00, 0x02, 0x00, 0x03, 0x00,
        /* Tier-1 child[1] = {4,4,4,5,5,6} */
        0x04, 0x00, 0x04, 0x00, 0x04, 0x00, 0x05, 0x00, 0x05, 0x00, 0x06, 0x00,
        /* Tier-1 child[2] = {7,7,7,7,7,8} */
        0x07, 0x00, 0x07, 0x00, 0x07, 0x00, 0x07, 0x00, 0x07, 0x00, 0x08, 0x00,
        /* Tier-2 grandchild digest (8 x int8, coarse high-byte mins/root) */
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    };

    /* same input -> same bytes (intra-run determinism, always valid) */
    sp_kste_tree_t t2;
    sp_kste_encode(anchor, 8, &t2);
    SP_CHECK(tree_eq(&t, &t2), "T_KSTE_4 same input -> same bytes");

    int match = (memcmp(t.bytes, expected, 64) == 0);
    SP_CHECK(match, "T_KSTE_4 byte image matches frozen cross-platform anchor");
}

/* ---- T_KSTE_5: size enforcement ------------------------------------------ */

static void T_KSTE_5(void) {
    SP_CHECK_EQ_I64((int64_t)sizeof(sp_kste_tree_t), 64,
                    "T_KSTE_5 sizeof(sp_kste_tree_t) == 64");
}

int main(void) {
    SP_RUN(T_KSTE_1);
    SP_RUN(T_KSTE_2);
    SP_RUN(T_KSTE_3);
    SP_RUN(T_KSTE_4);
    SP_RUN(T_KSTE_5);
    return SP_DONE();
}
