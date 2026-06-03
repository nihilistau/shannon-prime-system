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
#include "sp/poly_ring_bluestein.h"   /* NTT.5a sibling API under test */
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

/* ========================================================================
 *  NTT.5a Bluestein wrapper gates (T_NTT5A_*)
 * ========================================================================
 *
 *  Tests the new sp_pr_bluestein_* API that extends polynomial-ring
 *  attention from N ∈ {128, 256, 512} (direct sp_pr_init) to all powers of 2
 *  N ∈ {2, 4, 8, 16, 32, 64, 128, 256} (Bluestein wrapper). N=512 stays on
 *  direct sp_pr_init; non-power-of-2 N is mathematically unsupported with
 *  our frozen primes regardless of algorithm.
 *
 *  Schoolbook oracle is identical in shape to the existing
 *  schoolbook_negacyclic above, just parameterized for the wider N range.
 *  The two oracles agree on the overlapping N ∈ {128, 256} (cross-check in
 *  Stage 1's T_NTT5A_SCHOOLBOOK_CROSSCHECK), which is what gives the
 *  Bluestein-vs-schoolbook gate its trust.
 */

/* Coefficient range pinned to [-2^14, 2^14) for the Bluestein gates. With
 * N ≤ 256, |inner product| < 256 * 2^28 = 2^36 ≪ M/2 ≈ 2^59, and each
 * linear-conv coefficient is also bounded by 2^36. This range is tighter
 * than the existing T_PR coefficient range (2^23) and gives extra headroom
 * vs the M_full/2 bit-exact recovery window. */
static int32_t rng_coeff_blue(void) {
    int32_t v = (int32_t)(rng_next() & 0x00007FFFu);    /* [0, 2^15) */
    return v - (1 << 14);                                /* [-2^14, 2^14) */
}

/* All Bluestein-admissible N values (powers of 2 in [2, 256]). N=512 is
 * intentionally NOT here — direct sp_pr_init handles it. */
static const uint32_t kBluesteinNs[8] = { 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u };

/* Schoolbook negacyclic inner product: c_0 of the negacyclic product, i.e.
 * sum_i q_i k_i. With our coefficient range and N ≤ 256, computed in plain
 * int64 with no modular reduction (the true integer fits and equals the
 * centered Z_M residue). */
static int64_t schoolbook_inner(uint32_t N, const int32_t *q, const int32_t *k) {
    int64_t acc = 0;
    for (uint32_t i = 0; i < N; i++) acc += (int64_t)q[i] * (int64_t)k[i];
    return acc;
}

/* Schoolbook negacyclic product, parameterized over Bluestein-admissible N.
 * Identical structure to the existing schoolbook_negacyclic at the top of
 * this file; reproduced here because the existing one is private to the
 * T_PR_1 section. Centered result mod M, identical to direct sp_pr_mul. */
static void schoolbook_neg_mul(uint32_t N, const int32_t *a, const int32_t *b,
                               int64_t *out) {
    for (uint32_t k = 0; k < N; k++) out[k] = 0;
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < N; j++) {
            int64_t prod = (int64_t)a[i] * (int64_t)b[j];
            uint32_t s = i + j;
            if (s < N) out[s] += prod;
            else       out[s - N] -= prod;
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

/* ---- T_NTT5A_SCHOOLBOOK_CROSSCHECK (Stage 1) -----------------------------
 * Cross-check the new schoolbook helpers against the EXISTING
 * schoolbook_negacyclic on the overlapping N ∈ {128, 256}. This is the
 * Stage 1 trust-the-oracle gate: if the two independent schoolbook
 * implementations disagree, the Bluestein-vs-schoolbook gate is unreliable.
 * The Stage 1 implementation skeleton of sp_pr_bluestein_init returns NULL
 * for every N, so no actual Bluestein behavior is exercised at Stage 1. */
static void T_NTT5A_SCHOOLBOOK_CROSSCHECK(void) {
    rng_seed(0xCAFEF00D5A1A5A1Aull);
    const uint32_t overlap_Ns[2] = { 128u, 256u };
    for (int ni = 0; ni < 2; ni++) {
        uint32_t N = overlap_Ns[ni];
        int32_t *a = malloc(sizeof(int32_t) * N);
        int32_t *b = malloc(sizeof(int32_t) * N);
        int64_t *out_a = malloc(sizeof(int64_t) * N);
        int64_t *out_b = malloc(sizeof(int64_t) * N);

        int bad = 0;
        for (int t = 0; t < 8 && !bad; t++) {
            for (uint32_t i = 0; i < N; i++) {
                a[i] = rng_coeff_blue();
                b[i] = rng_coeff_blue();
            }
            schoolbook_negacyclic(N, a, b, out_a);    /* existing helper */
            schoolbook_neg_mul   (N, a, b, out_b);    /* NTT.5a helper   */
            for (uint32_t i = 0; i < N; i++)
                if (out_a[i] != out_b[i]) { bad = 1; break; }

            /* And the inner-product helper agrees with c_0 of the negacyclic
             * product when k is the involuted form. */
            int32_t *kinv = malloc(sizeof(int32_t) * N);
            kinv[0] = b[0];
            for (uint32_t j = 1; j < N; j++) kinv[j] = -b[N - j];
            int64_t inner_ref = schoolbook_inner(N, a, b);
            schoolbook_neg_mul(N, a, kinv, out_b);
            if (out_b[0] != inner_ref) bad = 1;
            free(kinv);
        }
        SP_CHECK(!bad, "T_NTT5A schoolbook oracle agrees with T_PR oracle");

        free(a); free(b); free(out_a); free(out_b);
    }

    /* Stage 1 skeleton claim: sp_pr_bluestein_init returns NULL for every
     * N before Stage 2 wires in the real implementation. This gate flips
     * meaning at Stage 2 — at Stage 2+ we expect non-NULL for admissible N
     * (T_NTT5A_NULL_FOR_INADMISSIBLE_N picks that up). */
    SP_CHECK(sp_pr_bluestein_degree(NULL) == 0u,
             "sp_pr_bluestein_degree(NULL) == 0");
    sp_pr_bluestein_free(NULL);   /* must be safe */
}

/* ---- T_NTT5A_BLUESTEIN_BIT_EXACT_VS_SCHOOLBOOK (Stage 3) ----------------- */
static void T_NTT5A_BLUESTEIN_BIT_EXACT_VS_SCHOOLBOOK(void) {
    rng_seed(0xB1E5731411111111ull);
    const int SEEDS_PER_N = 100;
    int total_runs = 0;
    int total_divergences = 0;
    int per_N_divergences[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    int per_N_runs[8]        = { 0, 0, 0, 0, 0, 0, 0, 0 };

    for (int ni = 0; ni < 8; ni++) {
        uint32_t N = kBluesteinNs[ni];
        sp_pr_bluestein_ctx *ctx = sp_pr_bluestein_init(N);
        SP_CHECK(ctx != NULL, "sp_pr_bluestein_init returns ctx for admissible N");
        if (!ctx) continue;
        SP_CHECK_EQ_I64((int64_t)sp_pr_bluestein_degree(ctx), (int64_t)N,
                        "sp_pr_bluestein_degree reports N");

        int32_t *q = malloc(sizeof(int32_t) * N);
        int32_t *k = malloc(sizeof(int32_t) * N);

        for (int t = 0; t < SEEDS_PER_N; t++) {
            for (uint32_t i = 0; i < N; i++) {
                q[i] = rng_coeff_blue();
                k[i] = rng_coeff_blue();
            }
            int64_t got  = sp_pr_bluestein_inner(ctx, q, k);
            int64_t want = schoolbook_inner(N, q, k);
            if (got != want) {
                total_divergences++;
                per_N_divergences[ni]++;
            }
            total_runs++;
            per_N_runs[ni]++;
        }

        free(q); free(k);
        sp_pr_bluestein_free(ctx);
    }
    fprintf(stderr,
            "[T_NTT5A_BLUESTEIN_BIT_EXACT_VS_SCHOOLBOOK] runs=%d divergences=%d\n",
            total_runs, total_divergences);
    for (int ni = 0; ni < 8; ni++)
        fprintf(stderr, "  N=%u: %d/%d divergences\n",
                kBluesteinNs[ni], per_N_divergences[ni], per_N_runs[ni]);
    SP_CHECK(total_divergences == 0,
             "Bluestein inner is bit-exact vs schoolbook across all admissible N");
}

/* ---- T_NTT5A_VS_SP_PR_INNER_BIT_EXACT (Stage 3) -------------------------- */
static void T_NTT5A_VS_SP_PR_INNER_BIT_EXACT(void) {
    rng_seed(0x7E5701112201D0E5ull);
    const uint32_t overlap_Ns[2] = { 128u, 256u };
    const int SEEDS_PER_N = 100;
    int total_runs = 0;
    int total_divergences = 0;

    for (int ni = 0; ni < 2; ni++) {
        uint32_t N = overlap_Ns[ni];
        sp_pr_ctx *direct = sp_pr_init(N);
        sp_pr_bluestein_ctx *blue = sp_pr_bluestein_init(N);
        SP_CHECK(direct && blue, "both contexts non-NULL for overlap N");
        if (!direct || !blue) { sp_pr_free(direct); sp_pr_bluestein_free(blue); continue; }

        int32_t *q = malloc(sizeof(int32_t) * N);
        int32_t *k = malloc(sizeof(int32_t) * N);
        for (int t = 0; t < SEEDS_PER_N; t++) {
            for (uint32_t i = 0; i < N; i++) {
                q[i] = rng_coeff_blue();
                k[i] = rng_coeff_blue();
            }
            int64_t a = sp_pr_inner(direct, q, k);
            int64_t b = sp_pr_bluestein_inner(blue, q, k);
            if (a != b) total_divergences++;
            total_runs++;
        }
        free(q); free(k);
        sp_pr_free(direct);
        sp_pr_bluestein_free(blue);
    }
    fprintf(stderr,
            "[T_NTT5A_VS_SP_PR_INNER_BIT_EXACT] runs=%d divergences=%d\n",
            total_runs, total_divergences);
    SP_CHECK(total_divergences == 0,
             "Bluestein inner == sp_pr_inner on overlapping N ∈ {128, 256}");
}

/* ---- T_NTT5A_BLUESTEIN_MUL_BIT_EXACT (Stage 4) --------------------------- */
static void T_NTT5A_BLUESTEIN_MUL_BIT_EXACT(void) {
    rng_seed(0xB1E573004D11D04Aull);
    const int SEEDS_PER_N = 50;
    int total_runs = 0;
    int total_coeff_divergences = 0;
    int per_N_coeff_divergences[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

    for (int ni = 0; ni < 8; ni++) {
        uint32_t N = kBluesteinNs[ni];
        sp_pr_bluestein_ctx *ctx = sp_pr_bluestein_init(N);
        SP_CHECK(ctx != NULL, "sp_pr_bluestein_init for admissible N (mul gate)");
        if (!ctx) continue;

        int32_t *a   = malloc(sizeof(int32_t) * N);
        int32_t *b   = malloc(sizeof(int32_t) * N);
        int64_t *got = malloc(sizeof(int64_t) * N);
        int64_t *ref = malloc(sizeof(int64_t) * N);

        for (int t = 0; t < SEEDS_PER_N; t++) {
            for (uint32_t i = 0; i < N; i++) {
                a[i] = rng_coeff_blue();
                b[i] = rng_coeff_blue();
            }
            sp_pr_bluestein_mul(ctx, a, b, got);
            schoolbook_neg_mul (N, a, b, ref);
            for (uint32_t i = 0; i < N; i++) {
                if (got[i] != ref[i]) {
                    total_coeff_divergences++;
                    per_N_coeff_divergences[ni]++;
                }
            }
            total_runs++;
        }

        free(a); free(b); free(got); free(ref);
        sp_pr_bluestein_free(ctx);
    }
    fprintf(stderr,
            "[T_NTT5A_BLUESTEIN_MUL_BIT_EXACT] runs=%d total-coeff-divergences=%d\n",
            total_runs, total_coeff_divergences);
    for (int ni = 0; ni < 8; ni++)
        fprintf(stderr, "  N=%u: %d coeff divergences over %d runs\n",
                kBluesteinNs[ni], per_N_coeff_divergences[ni], SEEDS_PER_N);
    SP_CHECK(total_coeff_divergences == 0,
             "Bluestein mul is bit-exact per-coefficient vs schoolbook");
}

/* ---- T_NTT5B_BACKEND_FORWARD_PASSTHROUGH (Stage 1) -----------------------
 *
 * Validates that sp_pr_bluestein_set_backend correctly routes the per-prime
 * forward NTT through the supplied dispatch fn, with bit-exact output vs the
 * no-backend host path. The stub forward implementation calls the public
 * ntt_forward (which fuses both primes) and discards the wrong channel —
 * functionally identical to math-core's per-prime forward_one, so the
 * dispatched path must produce byte-identical output to the host path.
 *
 * Inverse is left NULL so pr_blue_convolve_M falls back to host ntt_inverse.
 * This covers forward dispatch wiring; full forward+inverse dispatch is
 * validated on-device by the Stage 3 smoke harness (which uses the real
 * Hexagon HVX kernels via FastRPC).
 *
 * Stub forward: reinterpret the u32 input as int32 (values in [0, qP) fit
 * trivially in int32 since q ≈ 2^30), call ntt_forward, keep the matching
 * residue output channel.
 */

#include "sp/ntt_crt.h"   /* ntt_init/ntt_free/ntt_forward — only the existing
                           * NTT.5a header chain pulls this in transitively */

/* Cached inner ntt_ctx per M to avoid per-call init; one per M ∈ {128,256,512}. */
typedef struct {
    ntt_ctx *m128;
    ntt_ctx *m256;
    ntt_ctx *m512;
} pr_blue_5b_stub_handle;

static int pr_blue_5b_stub_forward(void *handle, int q_idx, int N,
                                   const uint32_t *in, uint32_t *out) {
    pr_blue_5b_stub_handle *h = (pr_blue_5b_stub_handle *)handle;
    if (!h || !in || !out) return -1;
    ntt_ctx *ctx = NULL;
    if      (N == 128) ctx = h->m128;
    else if (N == 256) ctx = h->m256;
    else if (N == 512) ctx = h->m512;
    if (!ctx) return -1;

    /* Reinterpret u32 as int32: values in [0, qP) ⊂ [0, 2^30) → positive int32. */
    const int32_t *in_i32 = (const int32_t *)in;
    /* Per-prime: keep q_idx channel, discard the other. ntt_forward reduces
     * each input by qP internally, so feeding the q1-prepared u32 values
     * to the q2 channel would produce different (still valid) residues —
     * but we only KEEP the channel matching q_idx, so the discard is fine. */
    uint32_t *scratch = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)N);
    if (!scratch) return -1;
    if (q_idx == 0) {
        ntt_forward(ctx, in_i32, out, scratch);
    } else if (q_idx == 1) {
        ntt_forward(ctx, in_i32, scratch, out);
    } else {
        free(scratch);
        return -1;
    }
    free(scratch);
    return 0;
}

static void T_NTT5B_BACKEND_FORWARD_PASSTHROUGH(void) {
    rng_seed(0x5B7B5B7B5B7B5B7Bull);
    pr_blue_5b_stub_handle handle = {
        .m128 = ntt_init(128u),
        .m256 = ntt_init(256u),
        .m512 = ntt_init(512u),
    };
    SP_CHECK(handle.m128 && handle.m256 && handle.m512,
             "ntt_init succeeded for stub backend inner ctxs");

    const uint32_t Ns[8] = { 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u };
    const int SEEDS_PER_N = 20;
    int total_runs = 0;
    int total_coeff_div = 0;

    for (int ni = 0; ni < 8; ni++) {
        uint32_t N = Ns[ni];
        sp_pr_bluestein_ctx *host_ctx = sp_pr_bluestein_init(N);
        sp_pr_bluestein_ctx *disp_ctx = sp_pr_bluestein_init(N);
        SP_CHECK(host_ctx && disp_ctx,
                 "sp_pr_bluestein_init for both host + dispatch ctx");
        if (!host_ctx || !disp_ctx) {
            sp_pr_bluestein_free(host_ctx);
            sp_pr_bluestein_free(disp_ctx);
            continue;
        }
        /* Dispatch ctx routes forward through the stub; inverse stays NULL,
         * so it falls back to host ntt_inverse on the host_ctx baseline path. */
        sp_pr_bluestein_set_backend(disp_ctx, &handle,
                                    pr_blue_5b_stub_forward, NULL);

        int32_t *a   = malloc(sizeof(int32_t) * N);
        int32_t *b   = malloc(sizeof(int32_t) * N);
        int64_t *got = malloc(sizeof(int64_t) * N);
        int64_t *ref = malloc(sizeof(int64_t) * N);

        for (int t = 0; t < SEEDS_PER_N; t++) {
            for (uint32_t i = 0; i < N; i++) {
                a[i] = rng_coeff_blue();
                b[i] = rng_coeff_blue();
            }
            sp_pr_bluestein_mul(host_ctx, a, b, ref);
            sp_pr_bluestein_mul(disp_ctx, a, b, got);
            for (uint32_t i = 0; i < N; i++) {
                if (got[i] != ref[i]) total_coeff_div++;
            }
            total_runs++;
        }
        free(a); free(b); free(got); free(ref);
        sp_pr_bluestein_free(host_ctx);
        sp_pr_bluestein_free(disp_ctx);
    }
    ntt_free(handle.m128);
    ntt_free(handle.m256);
    ntt_free(handle.m512);

    fprintf(stderr,
            "[T_NTT5B_BACKEND_FORWARD_PASSTHROUGH] runs=%d coeff-divergences=%d\n",
            total_runs, total_coeff_div);
    SP_CHECK(total_coeff_div == 0,
             "Bluestein forward dispatch passthrough is bit-exact vs host path");
}

/* ---- T_NTT5A_NULL_FOR_INADMISSIBLE_N (Stage 5) --------------------------- */
static void T_NTT5A_NULL_FOR_INADMISSIBLE_N(void) {
    /* N values that must return NULL:
     *   - 1: degenerate
     *   - 3, 5, 6, 7, 9: small non-powers-of-2 (odd factor blocks 2N-th root)
     *   - 96, 100, 384: realistic but non-powers-of-2
     *   - 512: powers-of-2 but excluded (direct sp_pr_init handles this)
     *   - 1024: exceeds 2-adic valuation cap of the frozen primes
     */
    const uint32_t bad[] = { 1u, 3u, 5u, 6u, 7u, 9u, 96u, 100u, 384u, 512u, 1024u };
    const int n_bad = (int)(sizeof(bad) / sizeof(bad[0]));
    int all_null = 1;
    for (int i = 0; i < n_bad; i++) {
        sp_pr_bluestein_ctx *ctx = sp_pr_bluestein_init(bad[i]);
        if (ctx != NULL) {
            fprintf(stderr,
                    "  [T_NTT5A_NULL_FOR_INADMISSIBLE_N] N=%u returned non-NULL\n",
                    bad[i]);
            all_null = 0;
            sp_pr_bluestein_free(ctx);
        }
    }
    SP_CHECK(all_null,
             "sp_pr_bluestein_init returns NULL for every inadmissible N");
}


/* T_PR_KSTORE — the NTT-FUSION exactness contract: scoring against a stored
 * (write-once transformed) key block must equal sp_pr_inner to the BIT, for
 * every supported N, across random coefficient draws at the attention scale
 * (|coeff| ~ 2^16, the SP_NTT_ATTN_SCALE quantization range) plus adversarial
 * extremes. A single mismatch falsifies the no-inverse coefficient-0 recovery
 * (psi-convention or Garner error) — surface UPSTREAM, do not tune. */
static uint32_t ks_rng = 0x9E3779B9u;
static int32_t ks_coeff(int32_t mag) {
    ks_rng = ks_rng * 1664525u + 1013904223u;
    return (int32_t)(ks_rng % (uint32_t)(2 * mag + 1)) - mag;
}
static void T_PR_KSTORE(void) {
    static const uint32_t Ns[3] = { 128, 256, 512 };
    for (int ni = 0; ni < 3; ni++) {
        const uint32_t N = Ns[ni];
        sp_pr_ctx *ctx = sp_pr_init(N);
        SP_CHECK(ctx != NULL, "kstore ctx init");
        if (!ctx) continue;
        SP_CHECK_EQ_I64((int64_t)sp_pr_kstore_words(ctx), (int64_t)(2 * N),
                        "kstore block is 2N residues");
        int32_t *q  = malloc(sizeof(int32_t) * N);
        int32_t *k  = malloc(sizeof(int32_t) * N);
        uint32_t *kr = malloc(sizeof(uint32_t) * 2 * N);
        SP_CHECK(q && k && kr, "kstore scratch");
        int ok = 1;
        for (int trial = 0; trial < 32 && ok; trial++) {
            int32_t mag = (trial < 24) ? 65536 : 2097152;   /* attention scale + extremes */
            for (uint32_t i = 0; i < N; i++) { q[i] = ks_coeff(mag); k[i] = ks_coeff(mag); }
            int64_t want = sp_pr_inner(ctx, q, k);
            sp_pr_kstore_encode(ctx, k, kr);
            sp_pr_query_begin(ctx, q);
            int64_t got = sp_pr_score_kstore(ctx, kr);
            if (got != want) {
                fprintf(stderr, "    KSTORE MISMATCH N=%u trial=%d want=%lld got=%lld\n",
                        N, trial, (long long)want, (long long)got);
                ok = 0;
            }
        }
        SP_CHECK(ok, "kstore score == sp_pr_inner BIT-EXACT (32 trials)");
        /* one query, many keys: query_begin amortization is stateless-correct */
        sp_pr_query_begin(ctx, q);
        int multi_ok = 1;
        for (int t2 = 0; t2 < 4; t2++) {
            for (uint32_t i = 0; i < N; i++) k[i] = ks_coeff(65536);
            sp_pr_kstore_encode(ctx, k, kr);     /* encode clobbers ctx scratch... */
            sp_pr_query_begin(ctx, q);           /* ...so re-begin: documents the contract */
            if (sp_pr_score_kstore(ctx, kr) != sp_pr_inner(ctx, q, k)) multi_ok = 0;
        }
        SP_CHECK(multi_ok, "amortized query vs fresh keys stays bit-exact");
        free(q); free(k); free(kr);
        sp_pr_free(ctx);
    }
}


/* T_PR_KSTORE_BLUE — the Bluestein keystore exactness contract: scoring against
 * the stored (padded, post-chirp, post-weight) key block must equal
 * sp_pr_bluestein_inner to the BIT for every Bluestein-admissible N. The
 * weights were derived empirically at init through the public ntt_inverse; a
 * mismatch falsifies the linear-functional derivation — surface UPSTREAM. */
static void T_PR_KSTORE_BLUE(void) {
    static const uint32_t Ns[8] = { 2, 4, 8, 16, 32, 64, 128, 256 };
    for (int ni = 0; ni < 8; ni++) {
        const uint32_t N = Ns[ni];
        sp_pr_bluestein_ctx *ctx = sp_pr_bluestein_init(N);
        SP_CHECK(ctx != NULL, "bluestein kstore ctx init");
        if (!ctx) continue;
        int32_t *q  = malloc(sizeof(int32_t) * N);
        int32_t *k  = malloc(sizeof(int32_t) * N);
        uint32_t *kr = malloc(sizeof(uint32_t) * sp_pr_bluestein_kstore_words(ctx));
        SP_CHECK(q && k && kr, "bluestein kstore scratch");
        int ok = 1;
        for (int trial = 0; trial < 16 && ok; trial++) {
            int32_t mag = (trial < 12) ? 65536 : 1048576;
            for (uint32_t i = 0; i < N; i++) { q[i] = ks_coeff(mag); k[i] = ks_coeff(mag); }
            int64_t want = sp_pr_bluestein_inner(ctx, q, k);
            sp_pr_bluestein_kstore_encode(ctx, k, kr);
            sp_pr_bluestein_query_begin(ctx, q);
            int64_t got = sp_pr_bluestein_score_kstore(ctx, kr);
            if (got != want) {
                fprintf(stderr, "    BLUE KSTORE MISMATCH N=%u trial=%d want=%lld got=%lld\n",
                        N, trial, (long long)want, (long long)got);
                ok = 0;
            }
        }
        SP_CHECK(ok, "bluestein kstore score == sp_pr_bluestein_inner BIT-EXACT (16 trials)");
        free(q); free(k); free(kr);
        sp_pr_bluestein_free(ctx);
    }
}


/* T_PR_RESDOT — the hot-loop contract: deferred-reduction resdot == the naive
 * per-element %-loop, bit-exact, across lengths (incl chunk-boundary odd sizes)
 * and full-range residues. The engine's AVX2 override must satisfy the SAME
 * gate semantics (same integer result; chunking cannot change the value). */
static void T_PR_RESDOT(void) {
    static const uint32_t Ls[7] = { 1, 14, 15, 16, 127, 256, 601 };
    static uint32_t a[601], b[601];
    int ok = 1;
    for (int li = 0; li < 7 && ok; li++) {
        uint32_t n = Ls[li];
        for (int trial = 0; trial < 8 && ok; trial++) {
            for (uint32_t i = 0; i < n; i++) {
                ks_rng = ks_rng * 1664525u + 1013904223u; a[i] = ks_rng % SP_NTT_Q1;
                ks_rng = ks_rng * 1664525u + 1013904223u; b[i] = ks_rng % SP_NTT_Q1;
            }
            if (trial == 0) for (uint32_t i = 0; i < n; i++) { a[i] = SP_NTT_Q1 - 1u; b[i] = SP_NTT_Q1 - 1u; }
            uint64_t want = 0;
            for (uint32_t i = 0; i < n; i++)
                want = (want + (uint64_t)a[i] * b[i] % SP_NTT_Q1) % SP_NTT_Q1;
            uint32_t got = sp_pr_resdot(a, b, n, SP_NTT_Q1);
            if ((uint64_t)got != want) {
                fprintf(stderr, "    RESDOT MISMATCH n=%u trial=%d want=%llu got=%u\n",
                        n, trial, (unsigned long long)want, got);
                ok = 0;
            }
        }
    }
    SP_CHECK(ok, "resdot (deferred 15-chunk reduction) == naive modular dot, all lengths + extremes");
}

int main(void) {
    SP_RUN(T_PR_1);
    SP_RUN(T_PR_2);
    SP_RUN(T_PR_3);
    SP_RUN(T_PR_4);
    SP_RUN(T_NTT5A_SCHOOLBOOK_CROSSCHECK);
    SP_RUN(T_NTT5A_BLUESTEIN_BIT_EXACT_VS_SCHOOLBOOK);
    SP_RUN(T_NTT5A_VS_SP_PR_INNER_BIT_EXACT);
    SP_RUN(T_NTT5A_BLUESTEIN_MUL_BIT_EXACT);
    SP_RUN(T_NTT5B_BACKEND_FORWARD_PASSTHROUGH);
    SP_RUN(T_NTT5A_NULL_FOR_INADMISSIBLE_N);
    SP_RUN(T_PR_RESDOT);
    SP_RUN(T_PR_KSTORE);
    SP_RUN(T_PR_KSTORE_BLUE);
    return SP_DONE();
}
