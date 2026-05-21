/* poly_ring_test.c — contract tests for R_q polynomial-ring attention (1C).
 *
 * One T_PR_n per roadmap item, driven by sp/sp_test.h. All randomness uses the
 * house fixed seed (xorshift64*) so runs are deterministic across machines.
 *
 *   T_PR_1  sp_pr_mul == independent schoolbook negacyclic multiply
 *           (bit-exact, centered signed), N in {128,256,512}, 1024 pairs each.
 *   T_PR_2  attention fidelity: softmax over ring inner products has
 *           KL <= 1e-7 vs softmax over true integer Euclidean dot products.
 *   T_PR_3  negacyclic property of multiplication by the monomial x.
 *   T_PR_4  cross-N stability on a low-degree (non-wrapping) product prefix.
 *
 * Coefficient range is pinned to [-2^23, 2^23) (matching the ntt_crt tests):
 * with N <= 512, |schoolbook coeff| < 512 * 2^46 = 2^55 << M/2 (~2^59), so the
 * centered recovery equals the exact integer and the inner product never
 * overflows the (-M/2, M/2] window. Pinning this range is what makes T_PR_2's
 * KL bound meaningful: <q,k> is recovered exactly, so any softmax divergence is
 * pure float rounding.
 */
#include "sp/sp_test.h"
#include "sp/poly_ring.h"
#include "sp/ntt_crt.h"      /* SP_NTT_M for the schoolbook centering */

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

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
/* uniform int32 coefficient in [-2^23, 2^23). */
static int32_t rng_coeff(void) {
    int32_t v = (int32_t)(rng_next() & 0x00FFFFFFu);   /* [0, 2^24) */
    return v - (1 << 23);                              /* [-2^23, 2^23) */
}

static const uint32_t kNs[3] = { 128u, 256u, 512u };

/* ---- T_PR_1 : sp_pr_mul == schoolbook negacyclic product ----------------- */

/* Independent schoolbook negacyclic convolution mod M, signed-centered, in
 * plain int64 (no __int128): with |coeff| < 2^23 and N <= 512 each accumulator
 * stays |.| < 2^55, far inside int64. Written inline here so the test does not
 * depend on the kernel's own reference. */
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
    const int64_t M = SP_NTT_M;
    for (uint32_t k = 0; k < N; k++) {
        int64_t v = out[k] % M;
        if (v < 0) v += M;
        if (v > M / 2) v -= M;
        out[k] = v;
    }
}

static void T_PR_1(void) {
    rng_seed(0x123456789ABCDEF0ull);
    for (int ni = 0; ni < 3; ni++) {
        uint32_t N = kNs[ni];
        sp_pr_ctx *ctx = sp_pr_init(N);
        SP_CHECK(ctx != NULL, "sp_pr_init returns a context for valid N");
        if (!ctx) continue;
        SP_CHECK_EQ_I64(sp_pr_degree(ctx), N, "sp_pr_degree reports N");

        int32_t *a   = malloc(sizeof(int32_t) * N);
        int32_t *b   = malloc(sizeof(int32_t) * N);
        int64_t *got = malloc(sizeof(int64_t) * N);
        int64_t *ref = malloc(sizeof(int64_t) * N);

        int bad = 0;
        for (int t = 0; t < 1024 && !bad; t++) {
            for (uint32_t i = 0; i < N; i++) { a[i] = rng_coeff(); b[i] = rng_coeff(); }
            sp_pr_mul(ctx, a, b, got);
            schoolbook_negacyclic(N, a, b, ref);
            for (uint32_t i = 0; i < N; i++) {
                if (got[i] != ref[i]) { bad = 1; break; }
            }
        }
        SP_CHECK(!bad, "sp_pr_mul bit-exact vs schoolbook negacyclic (1024 pairs)");

        free(a); free(b); free(got); free(ref);
        sp_pr_free(ctx);
    }
    /* invalid N rejected through to the caller. */
    SP_CHECK(sp_pr_init(64)   == NULL, "N=64 rejected");
    SP_CHECK(sp_pr_init(1024) == NULL, "N=1024 rejected");
    sp_pr_free(NULL);
}

/* ---- T_PR_2 : attention fidelity (KL <= 1e-7) ---------------------------- */

/* Stable softmax over int64 scores into doubles. Same routine drives both the
 * ring-derived scores and the plain-integer reference, so any divergence is the
 * float rounding of identical inputs. */
static void softmax_i64(const int64_t *score, int n, double *probs) {
    int64_t mx = score[0];
    for (int i = 1; i < n; i++) if (score[i] > mx) mx = score[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double e = exp((double)(score[i] - mx));
        probs[i] = e;
        sum += e;
    }
    for (int i = 0; i < n; i++) probs[i] /= sum;
}

/* KL(p || q) = sum_i p_i log(p_i / q_i). */
static double kl_div(const double *p, const double *q, int n) {
    double kl = 0.0;
    for (int i = 0; i < n; i++) {
        if (p[i] > 0.0) kl += p[i] * log(p[i] / q[i]);
    }
    return kl;
}

static void T_PR_2(void) {
    rng_seed(0x0FEDCBA987654321ull);
    const uint32_t d = 256u;          /* attention dimension */
    const int n_keys = 32;

    sp_pr_ctx *ctx = sp_pr_init(d);
    SP_CHECK(ctx != NULL, "sp_pr_init(256) for T_PR_2");
    if (!ctx) return;

    int32_t *q = malloc(sizeof(int32_t) * d);
    int32_t **keys = malloc(sizeof(int32_t *) * (size_t)n_keys);
    for (int k = 0; k < n_keys; k++) keys[k] = malloc(sizeof(int32_t) * d);

    double *probs_ring = malloc(sizeof(double) * (size_t)n_keys);
    double *probs_true = malloc(sizeof(double) * (size_t)n_keys);
    int64_t *score_ring = malloc(sizeof(int64_t) * (size_t)n_keys);
    int64_t *score_true = malloc(sizeof(int64_t) * (size_t)n_keys);

    double worst_kl = 0.0;
    int rounds = 64;
    for (int r = 0; r < rounds; r++) {
        for (uint32_t i = 0; i < d; i++) q[i] = rng_coeff();
        for (int k = 0; k < n_keys; k++)
            for (uint32_t i = 0; i < d; i++) keys[k][i] = rng_coeff();

        /* Ring path: scores via the involution-recovered inner product, then
         * softmax. (Recompute the per-key inner products here too so we can
         * confirm sp_pr_inner == true dot exactly, alongside the attention API.) */
        for (int k = 0; k < n_keys; k++)
            score_ring[k] = sp_pr_inner(ctx, q, keys[k]);

        /* True path: plain integer Euclidean dot product. */
        for (int k = 0; k < n_keys; k++) {
            int64_t dot = 0;
            for (uint32_t i = 0; i < d; i++)
                dot += (int64_t)q[i] * (int64_t)keys[k][i];
            score_true[k] = dot;
        }

        /* The ring inner product must recover the dot product EXACTLY. */
        for (int k = 0; k < n_keys; k++)
            SP_CHECK_EQ_I64(score_ring[k], score_true[k],
                            "sp_pr_inner == integer Euclidean dot");

        softmax_i64(score_ring, n_keys, probs_ring);
        softmax_i64(score_true, n_keys, probs_true);

        /* Also exercise the public attention API; it must match the ring
         * softmax bit-for-bit (same scores, same routine). */
        double *probs_api = malloc(sizeof(double) * (size_t)n_keys);
        sp_pr_attention(ctx, q, (const int32_t *const *)keys, n_keys, probs_api);
        for (int k = 0; k < n_keys; k++)
            SP_CHECK(probs_api[k] == probs_ring[k],
                     "sp_pr_attention probs == reference ring softmax");
        free(probs_api);

        double kl = kl_div(probs_true, probs_ring, n_keys);
        if (kl > worst_kl) worst_kl = kl;
    }

    fprintf(stderr, "[T_PR_2] worst KL(true||ring) = %.3e over %d rounds\n",
            worst_kl, rounds);
    SP_CHECK(worst_kl <= 1e-7, "KL(true || ring) <= 1e-7");

    for (int k = 0; k < n_keys; k++) free(keys[k]);
    free(keys); free(q);
    free(probs_ring); free(probs_true); free(score_ring); free(score_true);
    sp_pr_free(ctx);
}

/* ---- T_PR_3 : negacyclic property of multiply-by-x ----------------------- */

static void T_PR_3(void) {
    rng_seed(0xA5A5A5A5DEADBEEFull);
    for (int ni = 0; ni < 3; ni++) {
        uint32_t N = kNs[ni];
        sp_pr_ctx *ctx = sp_pr_init(N);
        if (!ctx) { SP_CHECK(0, "sp_pr_init for T_PR_3"); continue; }

        int32_t *p   = malloc(sizeof(int32_t) * N);
        int32_t *x   = malloc(sizeof(int32_t) * N);   /* the monomial x */
        int64_t *out = malloc(sizeof(int64_t) * N);

        for (uint32_t i = 0; i < N; i++) { p[i] = rng_coeff(); x[i] = 0; }
        x[1] = 1;                                     /* x = 0 + 1*x */

        sp_pr_mul(ctx, x, p, out);

        /* (x*p)_0 = -p_{N-1}; (x*p)_i = p_{i-1} for i >= 1. */
        SP_CHECK_EQ_I64(out[0], -(int64_t)p[N - 1], "(x*p)_0 = -p_{N-1}");
        int bad = 0;
        for (uint32_t i = 1; i < N; i++)
            if (out[i] != (int64_t)p[i - 1]) { bad = 1; break; }
        SP_CHECK(!bad, "(x*p)_i = p_{i-1} for i >= 1");

        free(p); free(x); free(out);
        sp_pr_free(ctx);
    }
}

/* ---- T_PR_4 : cross-N stability on a non-wrapping prefix ------------------ */

static void T_PR_4(void) {
    rng_seed(0xC0FFEE1234567890ull);
    const uint32_t deg = 60u;         /* < 64: product degree < 120 < 128 */

    sp_pr_ctx *c256 = sp_pr_init(256);
    sp_pr_ctx *c512 = sp_pr_init(512);
    SP_CHECK(c256 && c512, "sp_pr_init(256) and (512) for T_PR_4");
    if (!c256 || !c512) { sp_pr_free(c256); sp_pr_free(c512); return; }

    int32_t *a256 = calloc(256, sizeof(int32_t));
    int32_t *b256 = calloc(256, sizeof(int32_t));
    int32_t *a512 = calloc(512, sizeof(int32_t));
    int32_t *b512 = calloc(512, sizeof(int32_t));
    int64_t *o256 = malloc(sizeof(int64_t) * 256);
    int64_t *o512 = malloc(sizeof(int64_t) * 512);

    int bad = 0;
    for (int t = 0; t < 64 && !bad; t++) {
        for (uint32_t i = 0; i < 256; i++) { a256[i] = 0; b256[i] = 0; }
        for (uint32_t i = 0; i < 512; i++) { a512[i] = 0; b512[i] = 0; }
        for (uint32_t i = 0; i < deg; i++) {
            int32_t av = rng_coeff(), bv = rng_coeff();
            a256[i] = a512[i] = av;
            b256[i] = b512[i] = bv;
        }
        sp_pr_mul(c256, a256, b256, o256);
        sp_pr_mul(c512, a512, b512, o512);
        /* Shared low-degree prefix [0, 2*deg) must agree; nothing wraps at
         * either N since 2*deg-2 = 118 < 128 <= N. */
        for (uint32_t i = 0; i < 2 * deg; i++)
            if (o256[i] != o512[i]) { bad = 1; break; }
        /* and the products vanish above the true degree at both N. */
        for (uint32_t i = 2 * deg; i < 256 && !bad; i++)
            if (o256[i] != 0 || o512[i] != 0) { bad = 1; break; }
    }
    SP_CHECK(!bad, "N=256 and N=512 products agree on the non-wrapping prefix");

    free(a256); free(b256); free(a512); free(b512); free(o256); free(o512);
    sp_pr_free(c256); sp_pr_free(c512);
}

int main(void) {
    SP_RUN(T_PR_1);
    SP_RUN(T_PR_2);
    SP_RUN(T_PR_3);
    SP_RUN(T_PR_4);
    return SP_DONE();
}
