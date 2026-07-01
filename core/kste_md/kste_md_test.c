/* kste_md_test.c — unit gate for the magnitude-as-depth encoder + Dickson order.
 * Uses the header-only sp/sp_test.h harness (T_KMD_* names -> ctest output). */
#include "sp/sp_test.h"
#include "sp/kste_md.h"
#include <string.h>

static void fill(int32_t *v, int k, int seed) {
    uint64_t s = (uint64_t)seed * 0x9E3779B97F4A7C15ULL + 1;
    for (int i = 0; i < k; i++) { s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        v[i] = (int32_t)((int64_t)(s * 0x2545F4914F6CDD1DULL >> 33) - 1000000000); }
}

/* determinism: same input -> byte-identical signature */
static void T_KMD_1(void) {
    int32_t v[128]; fill(v, 128, 7);
    sp_kste_md_sig_t a, b;
    sp_kste_md_encode(v, 128, &a);
    sp_kste_md_encode(v, 128, &b);
    SP_CHECK(memcmp(&a, &b, sizeof(a)) == 0, "encode deterministic");
}

/* tri-state dominance is fully reachable */
static void T_KMD_2(void) {
    sp_kste_md_sig_t a, b; memset(&a, 0, sizeof(a)); memset(&b, 0, sizeof(b));
    SP_CHECK(sp_kste_md_dom(&a, &b) == SP_EQUIVALENT, "equal -> EQUIVALENT");
    a.v[0] = 5;                                 /* a >= b, one strictly > */
    SP_CHECK(sp_kste_md_dom(&a, &b) == SP_DOMINATES, "a>=b -> DOMINATES");
    SP_CHECK(sp_kste_md_dom(&b, &a) == SP_DOMINATED, "b<=a -> DOMINATED");
    b.v[1] = 9;                                 /* now a>b on [0], a<b on [1] */
    SP_CHECK(sp_kste_md_dom(&a, &b) == SP_INCOMPARABLE, "mixed -> INCOMPARABLE");
}

/* sign accounting: flipping every sign swaps nB<->nC and BB<->CC */
static void T_KMD_3(void) {
    int32_t v[128], w[128]; fill(v, 128, 11);
    for (int i = 0; i < 128; i++) w[i] = (v[i] == INT32_MIN) ? INT32_MAX : -v[i];
    sp_kste_md_sig_t a, b; sp_kste_md_encode(v, 128, &a); sp_kste_md_encode(w, 128, &b);
    SP_CHECK_EQ_I64(a.v[1], b.v[2], "flip: nB(a)==nC(b)");
    SP_CHECK_EQ_I64(a.v[2], b.v[1], "flip: nC(a)==nB(b)");
    SP_CHECK_EQ_I64(a.v[9], b.v[13], "flip: BB(a)==CC(b)");
}

/* all-zero vector -> neutral zero signature; budget respected */
static void T_KMD_4(void) {
    int32_t z[128]; memset(z, 0, sizeof(z));
    sp_kste_md_sig_t a; sp_kste_md_encode(z, 128, &a);
    int allzero = 1; for (int i = 0; i < SP_KMD_SIGDIM; i++) if (a.v[i] != 0) allzero = 0;
    SP_CHECK(allzero, "zero vector -> zero signature");
    int32_t v[128]; fill(v, 128, 3); sp_kste_md_sig_t b; sp_kste_md_encode(v, 128, &b);
    SP_CHECK(b.v[4] <= b.v[0] + SP_KMD_NODEBUDGET, "ntotal within nA+node budget");
}

/* near-duplicate comparable, unrelated usually incomparable (spot check) */
static void T_KMD_5(void) {
    int32_t v[128], vn[128]; fill(v, 128, 21);
    for (int i = 0; i < 128; i++) vn[i] = v[i] + (i % 7 == 0 ? 3 : 0);  /* tiny perturb */
    sp_kste_md_sig_t a, b; sp_kste_md_encode(v, 128, &a); sp_kste_md_encode(vn, 128, &b);
    SP_CHECK(sp_kste_md_dom(&a, &b) != SP_INCOMPARABLE, "near-duplicate is comparable");
}

int main(void) {
    SP_RUN(T_KMD_1); SP_RUN(T_KMD_2); SP_RUN(T_KMD_3); SP_RUN(T_KMD_4); SP_RUN(T_KMD_5);
    return SP_DONE();
}
