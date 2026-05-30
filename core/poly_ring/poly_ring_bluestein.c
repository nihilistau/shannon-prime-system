/* poly_ring_bluestein.c — host-side Bluestein wrapper for negacyclic R_q
 * polynomial multiplication at arbitrary power-of-2 degree N ∈ [2, 256].
 *
 * Anti-contamination: this file is a NEW caller-side wrapper; it does NOT
 * modify poly_ring.h/c or ntt_crt.h/c. All ntt_crt access is through the
 * public surface in sp/ntt_crt.h (ntt_init, ntt_forward, ntt_pointwise_mul,
 * ntt_inverse, ntt_crt_recombine, SP_NTT_Q1, SP_NTT_Q2, SP_NTT_M). The
 * primitive-2N-th-root search and the small modular helpers below mirror
 * ntt_crt.c's algorithms (Fermat-style modinv via modpow, search from a=2
 * upward for psi with psi^N = -1), but operate over arbitrary Bluestein N
 * — not the inner-NTT N — so they cannot be borrowed wholesale from
 * ntt_crt.c without exposing its internals.
 *
 * Algorithm (Approach A, derived in tools/sp_compute_skel/docs/PLAN-NTT-5a.md):
 *
 *   negacyclic length-N product c = a (x) b over Z_q[x]/(x^N+1)
 *   per-prime (q1 then q2):
 *
 *   (1) psi-twist with ψ_N (primitive 2N-th root, ψ_N^N = -1 in Z_q):
 *           A_j = a_j · ψ_N^j   mod q
 *           B_j = b_j · ψ_N^j   mod q
 *       which converts the negacyclic length-N convolution into a
 *       length-N cyclic convolution C with c_k = C_k · ψ_N^{-k}.
 *
 *   (2) Zero-pad A, B to length M ∈ {128, 256, 512}, smallest with M ≥ 2N:
 *           M = 128 for N ∈ {2,4,8,16,32,64}
 *           M = 256 for N = 128
 *           M = 512 for N = 256
 *       and compute the length-M negacyclic product D via math-core's
 *       ntt_forward/ntt_pointwise_mul/ntt_inverse. Because A and B vanish
 *       beyond index N-1, the highest non-zero linear-conv index is
 *       2N-2 < M, so the length-M wraparound term (x^M = -1 sign flip)
 *       contributes zero, and D coincides with the linear convolution
 *       of the zero-padded inputs.
 *
 *   (3) Fold length-(2N-1) linear result to length-N cyclic:
 *           C_k = D_k + (k + N < M ? D_{k+N} : 0)   for k ∈ [0, N).
 *
 *   (4) Post-untwist:
 *           c_k = C_k · ψ_N^{-k}  mod q.
 *
 *   (5) Across the two primes, recombine residue pairs by signed Garner CRT
 *       (math-core's ntt_crt_recombine) into the signed centered int64
 *       output in (-M_full/2, M_full/2] where M_full = q1·q2 ~ 2^60.
 *
 * Note: math-core's ntt_inverse already does steps (3)-style scaling, the
 * Cooley-Tukey omega-inverse butterfly, AND the per-prime psi-twist (its
 * OWN psi for the INNER N=M, unrelated to ours), and a CRT recombine
 * internally. We use ntt_inverse to get the length-M product back into
 * signed integer form (per-prime residues recombined to int64). Then we
 * apply OUR length-N psi-twist (psi for the OUTER N), and our length-N
 * fold, on TOP of that signed int64 length-M result.
 *
 * In other words: math-core's inner NTT is treated as an opaque
 * "signed-integer length-M negacyclic multiplier"; ALL Bluestein-specific
 * machinery (zero-pad, outer-N psi-twist, fold) lives in this file. The
 * inner length-M psi/omega tables (kept in ntt_ctx) are never touched here.
 *
 * Wait — there's a subtle issue with the above plan. ntt_inverse returns
 * the SIGNED CENTERED int64 result of the length-M NEGACYCLIC product. But
 * we want the negacyclic-length-M product to actually be the LINEAR product
 * of our zero-padded inputs, which it is (zero-padding kills wraparound).
 * So this is fine; ntt_inverse's signed-centered output is exactly the
 * linear conv result in Z (provided |coefficient| < M_full/2).
 *
 * However, ntt_inverse runs the inner negacyclic NTT — meaning its
 * pre-weight already psi-twists by the INNER N=M's psi_M, and its
 * post-weight inverts that. We feed it inputs that have ALREADY been
 * outer-N psi-twisted (by our ψ_N for outer N). The inner ψ_M twist
 * happens to the OUTER-twisted values, which is fine because both twists
 * are mod-q linear operations and don't mix coefficients across positions
 * — each coefficient j carries its outer factor ψ_N^j and the inner
 * machinery applies/inverts ψ_M^j independently. The math-core's
 * length-M negacyclic NTT result is therefore the length-M negacyclic
 * product of (a_j · ψ_N^j) padded with zeros — which is what we want.
 */
#include "sp/poly_ring_bluestein.h"
#include "sp/ntt_crt.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

/* ---- Bluestein admissible-N table ---------------------------------------- */

/* Returns inner-NTT length M for an admissible Bluestein N, or 0 for
 * inadmissible. Admissible N values: {2, 4, 8, 16, 32, 64, 128, 256}.
 * M choice = smallest of {128, 256, 512} satisfying M ≥ 2N. */
static uint32_t bluestein_inner_M(uint32_t N) {
    switch (N) {
        case 2u: case 4u: case 8u: case 16u:
        case 32u: case 64u:
            return 128u;
        case 128u:
            return 256u;
        case 256u:
            return 512u;
        default:
            return 0u;   /* inadmissible */
    }
}

/* Is N a power of 2 in the admissible Bluestein range? */
static int bluestein_admissible(uint32_t N) {
    return bluestein_inner_M(N) != 0u;
}

/* ---- modular arithmetic (Barrett-like, no 128-bit type, mirrors ntt_crt.c)
 *
 * The math-core's modular primitives (barrett_reduce / modmul / modpow /
 * modinv / find_psi) are TU-private to ntt_crt.c. We replicate the same
 * algorithms here, restricted to our public-surface constants SP_NTT_Q1
 * / SP_NTT_Q2. ~80 LOC; cleanly localized; not load-bearing for the
 * production hot path (only used in init for the small psi search and the
 * O(N) twiddle table fill). */

#define POLY_RING_BLUE_Q_BITS 30u

static inline uint64_t pr_blue_barrett_reduce(uint64_t x, uint64_t q,
                                              uint64_t mu) {
    uint64_t qhat = ((x >> (POLY_RING_BLUE_Q_BITS - 1u)) * mu)
                    >> (POLY_RING_BLUE_Q_BITS + 1u);
    uint64_t r = x - qhat * q;
    if (r >= q) r -= q;
    if (r >= q) r -= q;
    return r;
}

static inline uint32_t pr_blue_modmul(uint32_t a, uint32_t b, uint32_t q,
                                      uint64_t mu) {
    uint64_t x = (uint64_t)a * (uint64_t)b;       /* < 2^60 since both < 2^30 */
    return (uint32_t)pr_blue_barrett_reduce(x, (uint64_t)q, mu);
}

static uint32_t pr_blue_modpow(uint32_t base, uint64_t e, uint32_t q,
                               uint64_t mu) {
    uint32_t result = 1u % q;
    uint32_t b = base % q;
    while (e) {
        if (e & 1u) result = pr_blue_modmul(result, b, q, mu);
        b = pr_blue_modmul(b, b, q, mu);
        e >>= 1;
    }
    return result;
}

static uint32_t pr_blue_modinv(uint32_t a, uint32_t q, uint64_t mu) {
    return pr_blue_modpow(a, (uint64_t)q - 2u, q, mu);   /* Fermat */
}

/* Find primitive 2N-th root ψ_N: smallest a≥2 with (a^((q-1)/(2N)))^N == -1
 * (mod q). Returns 0 if not found — unreachable for our frozen primes when
 * N ≤ 512 by v_2(q-1) = 10 (sanity-checked in pr_blue_prime_setup via
 * assert). */
static uint32_t pr_blue_find_psi(uint32_t N, uint32_t q, uint64_t mu) {
    uint64_t exp = ((uint64_t)q - 1u) / (2u * (uint64_t)N);
    for (uint32_t a = 2u; a < q; a++) {
        uint32_t psi = pr_blue_modpow(a, exp, q, mu);
        if (psi == 0u) continue;
        uint32_t pN = pr_blue_modpow(psi, (uint64_t)N, q, mu);
        if (pN == q - 1u) return psi;
    }
    return 0u;
}

/* Per-prime Bluestein side context: outer-N psi tables for the negacyclic
 * twist. */
typedef struct {
    uint32_t  q;            /* SP_NTT_Q1 or SP_NTT_Q2 */
    uint64_t  mu;           /* Barrett floor(2^60 / q) */
    uint32_t *psi_pow;      /* ψ_N^j      for j ∈ [0, N) — pre-weight  */
    uint32_t *ipsi_pow;     /* ψ_N^{-j}   for j ∈ [0, N) — post-weight */
} pr_blue_prime_ctx;

static int pr_blue_prime_setup(pr_blue_prime_ctx *pc, uint32_t N, uint32_t q) {
    pc->q  = q;
    pc->mu = ((uint64_t)1 << 60) / (uint64_t)q;

    uint32_t psi = pr_blue_find_psi(N, q, pc->mu);
    if (psi == 0u) return 0;

    /* Invariants: psi^N = -1, psi^{2N} = 1. */
    assert(pr_blue_modpow(psi, (uint64_t)N, q, pc->mu) == q - 1u);
    assert(pr_blue_modpow(psi, 2u * (uint64_t)N, q, pc->mu) == 1u % q);

    uint32_t ipsi = pr_blue_modinv(psi, q, pc->mu);

    pc->psi_pow  = (uint32_t *)malloc(sizeof(uint32_t) * N);
    pc->ipsi_pow = (uint32_t *)malloc(sizeof(uint32_t) * N);
    if (!pc->psi_pow || !pc->ipsi_pow) return 0;

    uint32_t acc = 1u % q;
    for (uint32_t j = 0; j < N; j++) {     /* psi^j  */
        pc->psi_pow[j] = acc;
        acc = pr_blue_modmul(acc, psi, q, pc->mu);
    }
    acc = 1u % q;
    for (uint32_t j = 0; j < N; j++) {     /* psi^{-j} */
        pc->ipsi_pow[j] = acc;
        acc = pr_blue_modmul(acc, ipsi, q, pc->mu);
    }
    return 1;
}

static void pr_blue_prime_free(pr_blue_prime_ctx *pc) {
    if (!pc) return;
    free(pc->psi_pow);  pc->psi_pow  = NULL;
    free(pc->ipsi_pow); pc->ipsi_pow = NULL;
}

/* ---- Bluestein context --------------------------------------------------- */

struct sp_pr_bluestein_ctx {
    uint32_t           N;        /* outer degree (Bluestein)           */
    uint32_t           M;        /* inner NTT length ∈ {128,256,512}   */
    ntt_ctx           *inner;    /* math-core CRT NTT context for M    */
    pr_blue_prime_ctx  p1;       /* outer-N psi tables for prime q1    */
    pr_blue_prime_ctx  p2;       /* outer-N psi tables for prime q2    */

    /* Per-call scratch, sized M. Kept on the context to avoid malloc on the
     * hot path (matching sp_pr_ctx's pattern). NOT thread-safe; one context
     * per thread. */
    int32_t  *a_pad;             /* outer-twisted, zero-padded a (M)   */
    int32_t  *b_pad;             /* outer-twisted, zero-padded b (M)   */
    uint32_t *a1_res, *a2_res;   /* per-prime residue domain (M each)  */
    uint32_t *b1_res, *b2_res;   /* per-prime residue domain (M each)  */
    uint32_t *c1_res, *c2_res;   /* per-prime pointwise product (M)    */
    int64_t  *D;                 /* length-M signed centered conv (M)  */
};

/* ---- public API --------------------------------------------------------- */

sp_pr_bluestein_ctx *sp_pr_bluestein_init(uint32_t N) {
    if (!bluestein_admissible(N)) return NULL;
    uint32_t M = bluestein_inner_M(N);

    ntt_ctx *inner = ntt_init(M);
    if (!inner) return NULL;     /* defensive — admissible M is always good */

    sp_pr_bluestein_ctx *ctx = (sp_pr_bluestein_ctx *)
        calloc(1, sizeof(*ctx));
    if (!ctx) { ntt_free(inner); return NULL; }
    ctx->N     = N;
    ctx->M     = M;
    ctx->inner = inner;

    if (!pr_blue_prime_setup(&ctx->p1, N, SP_NTT_Q1) ||
        !pr_blue_prime_setup(&ctx->p2, N, SP_NTT_Q2)) {
        sp_pr_bluestein_free(ctx);
        return NULL;
    }

    ctx->a_pad  = (int32_t  *)malloc(sizeof(int32_t)  * M);
    ctx->b_pad  = (int32_t  *)malloc(sizeof(int32_t)  * M);
    ctx->a1_res = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->a2_res = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->b1_res = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->b2_res = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->c1_res = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->c2_res = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->D      = (int64_t  *)malloc(sizeof(int64_t)  * M);

    if (!ctx->a_pad || !ctx->b_pad ||
        !ctx->a1_res || !ctx->a2_res || !ctx->b1_res || !ctx->b2_res ||
        !ctx->c1_res || !ctx->c2_res || !ctx->D) {
        sp_pr_bluestein_free(ctx);
        return NULL;
    }
    return ctx;
}

void sp_pr_bluestein_free(sp_pr_bluestein_ctx *ctx) {
    if (!ctx) return;
    ntt_free(ctx->inner);
    pr_blue_prime_free(&ctx->p1);
    pr_blue_prime_free(&ctx->p2);
    free(ctx->a_pad);  free(ctx->b_pad);
    free(ctx->a1_res); free(ctx->a2_res);
    free(ctx->b1_res); free(ctx->b2_res);
    free(ctx->c1_res); free(ctx->c2_res);
    free(ctx->D);
    free(ctx);
}

uint32_t sp_pr_bluestein_degree(const sp_pr_bluestein_ctx *ctx) {
    return ctx ? ctx->N : 0u;
}

/* Stage 2: inner / mul are NOT YET IMPLEMENTED. Stage 3 will wire them up.
 * Returning 0 / no-op so Stage 2 init/free can be smoke-tested without the
 * full pipeline. */

int64_t sp_pr_bluestein_inner(sp_pr_bluestein_ctx *ctx,
                              const int32_t *q, const int32_t *k) {
    (void)ctx; (void)q; (void)k;
    return 0;
}

void sp_pr_bluestein_mul(sp_pr_bluestein_ctx *ctx,
                         const int32_t *a, const int32_t *b,
                         int64_t *out) {
    (void)ctx; (void)a; (void)b; (void)out;
}
