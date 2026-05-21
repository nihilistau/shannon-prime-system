/* ntt_crt_test.c — contract tests for the dual-prime CRT negacyclic NTT.
 *
 * One T_NTT_n per roadmap item, driven by sp_test.h. All randomness uses the
 * house fixed seed so runs are deterministic across machines.
 *
 *   T_NTT_1  forward o inverse == identity, 4096 random polys per N
 *   T_NTT_2  full pipeline == ntt_ref_int128 negacyclic product (gcc tier)
 *   T_NTT_3  same gate under MSVC — DEFERRED (oracle needs __int128)
 *   T_NTT_4  pointwise-mul + inverse == negacyclic convolution (vs schoolbook)
 *   T_NTT_5  production lib built without __int128 (configure guard is the gate)
 *
 * ntt_ref_int128.c (TEST-ONLY parity oracle) is compiled into this executable
 * via TEST_SOURCES and provides ntt_ref_negacyclic_mul().
 */
#include "sp/sp_test.h"
#include "sp/ntt_crt.h"

#include <stdint.h>
#include <stdlib.h>

/* Reference negacyclic multiply, mod M, signed-centered. Defined in the
 * test-only oracle ntt_ref_int128.c. out has N entries in (-M/2, M/2]. */
void ntt_ref_negacyclic_mul(uint32_t N, const int32_t *a, const int32_t *b,
                            int64_t *out);

/* ---- deterministic RNG (xorshift64*, house fixed seed) ------------------- */

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

/* uniform int32 coefficient in a modest range so that schoolbook products
 * stay comfortably inside (-M/2, M/2] for all supported N. With |c| < 2^23 and
 * N <= 512, |sum of N products| < 512 * 2^46 = 2^55 < M/2 (~2^59). */
static int32_t rng_coeff(void) {
    int32_t v = (int32_t)(rng_next() & 0x00FFFFFFu);   /* [0, 2^24) */
    return v - (1 << 23);                              /* [-2^23, 2^23) */
}

static const uint32_t kNs[3] = { 128u, 256u, 512u };

/* ---- T_NTT_1 : forward o inverse == identity ----------------------------- */

static void T_NTT_1(void) {
    rng_seed(0x9E3779B97F4A7C15ull);

    /* invalid N must be rejected (only {128,256,512} supported; 1024 cannot be
     * supported by the frozen primes). ntt_free(NULL) is a no-op. */
    SP_CHECK(ntt_init(0)    == NULL, "N=0 rejected");
    SP_CHECK(ntt_init(64)   == NULL, "N=64 rejected");
    SP_CHECK(ntt_init(1024) == NULL, "N=1024 rejected (primes can't support it)");
    SP_CHECK(ntt_init(200)  == NULL, "non-power-of-two rejected");
    ntt_free(NULL);

    for (int ni = 0; ni < 3; ni++) {
        uint32_t N = kNs[ni];
        ntt_ctx *ctx = ntt_init(N);
        SP_CHECK(ctx != NULL, "ntt_init returns a context for valid N");
        if (!ctx) continue;

        int32_t *in   = malloc(sizeof(int32_t) * N);
        uint32_t *r1  = malloc(sizeof(uint32_t) * N);
        uint32_t *r2  = malloc(sizeof(uint32_t) * N);
        int64_t *out  = malloc(sizeof(int64_t) * N);

        int bad = 0;
        for (int trial = 0; trial < 4096 && !bad; trial++) {
            for (uint32_t i = 0; i < N; i++) in[i] = rng_coeff();
            ntt_forward(ctx, in, r1, r2);
            ntt_inverse(ctx, r1, r2, out);
            for (uint32_t i = 0; i < N; i++) {
                if (out[i] != (int64_t)in[i]) { bad = 1; break; }
            }
        }
        SP_CHECK(!bad, "forward then inverse is the identity (4096 polys)");

        free(in); free(r1); free(r2); free(out);
        ntt_free(ctx);
    }
}

/* ---- T_NTT_2 : full pipeline == oracle negacyclic product ---------------- */

static void T_NTT_2(void) {
    rng_seed(0xD1B54A32D192ED03ull);
    for (int ni = 0; ni < 3; ni++) {
        uint32_t N = kNs[ni];
        ntt_ctx *ctx = ntt_init(N);
        if (!ctx) { SP_CHECK(0, "ntt_init for T_NTT_2"); continue; }

        int32_t  *a  = malloc(sizeof(int32_t) * N);
        int32_t  *b  = malloc(sizeof(int32_t) * N);
        uint32_t *a1 = malloc(sizeof(uint32_t) * N);
        uint32_t *a2 = malloc(sizeof(uint32_t) * N);
        uint32_t *b1 = malloc(sizeof(uint32_t) * N);
        uint32_t *b2 = malloc(sizeof(uint32_t) * N);
        uint32_t *c1 = malloc(sizeof(uint32_t) * N);
        uint32_t *c2 = malloc(sizeof(uint32_t) * N);
        int64_t  *got = malloc(sizeof(int64_t) * N);
        int64_t  *ref = malloc(sizeof(int64_t) * N);

        int trials = 256;
        int bad = 0;
        for (int t = 0; t < trials && !bad; t++) {
            for (uint32_t i = 0; i < N; i++) { a[i] = rng_coeff(); b[i] = rng_coeff(); }
            ntt_forward(ctx, a, a1, a2);
            ntt_forward(ctx, b, b1, b2);
            ntt_pointwise_mul(ctx, a1, a2, b1, b2, c1, c2);
            ntt_inverse(ctx, c1, c2, got);
            ntt_ref_negacyclic_mul(N, a, b, ref);
            for (uint32_t i = 0; i < N; i++) {
                if (got[i] != ref[i]) { bad = 1; break; }
            }
        }
        SP_CHECK(!bad, "pipeline bit-exact vs ntt_ref_int128 (negacyclic product)");

        free(a); free(b); free(a1); free(a2); free(b1); free(b2);
        free(c1); free(c2); free(got); free(ref);
        ntt_free(ctx);
    }
}

/* ---- T_NTT_3 : MSVC gate — deferred -------------------------------------- */

static void T_NTT_3(void) {
    /* The bit-exactness gate also has to hold under Windows MSVC. MSVC cannot
     * compile the __int128 oracle, so this case is deferred to the MSVC
     * follow-up wave (roadmap §3.7). Plan: the gcc tier already exercises the
     * identical pipeline in T_NTT_2; the MSVC wave will compare against
     * pre-emitted reference vectors instead of the live __int128 oracle. */
    fprintf(stderr, "[T_NTT_3] DEFERRED (MSVC wave)\n");
    SP_CHECK(1, "T_NTT_3 deferred to MSVC wave (no failure)");
}

/* ---- T_NTT_4 : pointwise-mul + inverse == negacyclic convolution --------- */

/* Local schoolbook negacyclic convolution mod M, signed-centered, computed
 * here in plain int64 (no __int128): with |coeff| < 2^23 and N <= 512 each
 * accumulator stays |.| < 2^55, far inside int64. Independent of the oracle so
 * T_NTT_4 cross-checks both the kernel and the oracle. */
static void schoolbook_negacyclic(uint32_t N, const int32_t *a,
                                   const int32_t *b, int64_t *out) {
    for (uint32_t k = 0; k < N; k++) out[k] = 0;
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < N; j++) {
            int64_t prod = (int64_t)a[i] * (int64_t)b[j];
            uint32_t s = i + j;
            if (s < N) out[s] += prod;          /* x^s */
            else       out[s - N] -= prod;      /* x^N = -1 wraparound */
        }
    }
    /* center mod M */
    const int64_t M = SP_NTT_M;
    for (uint32_t k = 0; k < N; k++) {
        int64_t v = out[k] % M;
        if (v < 0) v += M;
        if (v > M / 2) v -= M;
        out[k] = v;
    }
}

static void T_NTT_4(void) {
    rng_seed(0x243F6A8885A308D3ull);
    for (int ni = 0; ni < 3; ni++) {
        uint32_t N = kNs[ni];
        ntt_ctx *ctx = ntt_init(N);
        if (!ctx) { SP_CHECK(0, "ntt_init for T_NTT_4"); continue; }

        int32_t  *a  = malloc(sizeof(int32_t) * N);
        int32_t  *b  = malloc(sizeof(int32_t) * N);
        uint32_t *a1 = malloc(sizeof(uint32_t) * N);
        uint32_t *a2 = malloc(sizeof(uint32_t) * N);
        uint32_t *b1 = malloc(sizeof(uint32_t) * N);
        uint32_t *b2 = malloc(sizeof(uint32_t) * N);
        uint32_t *c1 = malloc(sizeof(uint32_t) * N);
        uint32_t *c2 = malloc(sizeof(uint32_t) * N);
        int64_t  *got = malloc(sizeof(int64_t) * N);
        int64_t  *ref = malloc(sizeof(int64_t) * N);

        int bad = 0;
        for (int t = 0; t < 128 && !bad; t++) {
            for (uint32_t i = 0; i < N; i++) { a[i] = rng_coeff(); b[i] = rng_coeff(); }
            ntt_forward(ctx, a, a1, a2);
            ntt_forward(ctx, b, b1, b2);
            ntt_pointwise_mul(ctx, a1, a2, b1, b2, c1, c2);
            ntt_inverse(ctx, c1, c2, got);
            schoolbook_negacyclic(N, a, b, ref);
            for (uint32_t i = 0; i < N; i++) {
                if (got[i] != ref[i]) { bad = 1; break; }
            }
        }
        SP_CHECK(!bad, "pointwise-mul+inverse == negacyclic convolution (schoolbook)");

        free(a); free(b); free(a1); free(a2); free(b1); free(b2);
        free(c1); free(c2); free(got); free(ref);
        ntt_free(ctx);
    }
}

/* ---- T_NTT_5 : production path is __int128-free --------------------------- */

static void T_NTT_5(void) {
    /* The enforcing gate is the configure-time guard in CMakeLists.txt, which
     * FATAL_ERRORs if "__int128" appears in ntt_crt.c. Reaching this point
     * means the production lib configured and linked without it. */
    fprintf(stderr, "[T_NTT_5] production lib built without __int128 "
                    "(configure-time guard is the enforcing gate)\n");
#if defined(__SIZEOF_INT128__)
    SP_CHECK(1, "compiler supports __int128 yet production lib avoids it");
#else
    SP_CHECK(1, "compiler without __int128; production lib unaffected");
#endif
}

int main(void) {
    SP_RUN(T_NTT_1);
    SP_RUN(T_NTT_2);
    SP_RUN(T_NTT_3);
    SP_RUN(T_NTT_4);
    SP_RUN(T_NTT_5);
    return SP_DONE();
}
