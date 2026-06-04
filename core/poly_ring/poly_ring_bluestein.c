/* poly_ring_bluestein.c — host-side Bluestein wrapper for negacyclic R_q
 * polynomial multiplication at arbitrary power-of-2 degree N ∈ [2, 256].
 *
 * Anti-contamination: NEW caller-side wrapper. Does NOT modify
 * poly_ring.h/c or ntt_crt.h/c. All ntt_crt access is through the public
 * surface in sp/ntt_crt.h (ntt_init/ntt_forward/ntt_pointwise_mul/
 * ntt_inverse/ntt_crt_recombine + SP_NTT_Q1/Q2/M).
 *
 * ----------------------------------------------------------------------
 * Algorithm (Approach A; full derivation in
 * tools/sp_compute_skel/docs/PLAN-NTT-5a.md):
 *
 * Goal: negacyclic length-N product c = a (x) b over R_q = Z_q[x]/(x^N+1)
 * for N ∈ {2,4,8,16,32,64,128,256} (the inner ntt_init only admits
 * M ∈ {128,256,512}, so this Bluestein wrapper picks the smallest such M
 * with M ≥ 2N; the table is in bluestein_inner_M below).
 *
 * Step 1 (per-prime psi-twist, in coefficient domain):
 *   For each prime qP ∈ {SP_NTT_Q1, SP_NTT_Q2}, search a primitive 2N-th
 *   root ψ_NP and form
 *       a_tw_qP[j] = (a[j] · ψ_NP^j) mod qP   for j ∈ [0,N),  0 for j ∈ [N,M).
 *   Same for b. ψ_NP exists because 2N | (qP - 1) — v_2(qP - 1) = 10 in both
 *   frozen primes, so 2N up to 1024 is admissible (we use up to 2N = 512).
 *
 *   This twist converts the negacyclic length-N convolution we want
 *   (c = a (x)_neg b) into a CYCLIC length-N convolution C of (twisted a,
 *   twisted b), with the identity c_k = ψ_NP^{-k} · C_k mod qP, derived as
 *
 *     C_k = sum_{i+j ≡ k (mod N)} (a_i ψ_NP^i) (b_j ψ_NP^j)
 *         = sum_{i+j=k}   a_i b_j ψ_NP^k          (since i+j = k → no wrap)
 *         + sum_{i+j=k+N} a_i b_j ψ_NP^{k+N}       (negacyclic wraparound)
 *         = ψ_NP^k · (sum_{i+j=k} a_i b_j  -  sum_{i+j=k+N} a_i b_j)
 *           since ψ_NP^N = -1.
 *         = ψ_NP^k · c_k.
 *
 * Step 2 (length-M linear convolution, via math-core's CRT NTT):
 *   Zero-padding a_tw_qP / b_tw_qP to length M means the negacyclic
 *   length-M product equals the linear length-(2N-1) product (the
 *   wraparound term involves indices ≥ M, where both inputs vanish, so it
 *   contributes zero). To use math-core's CRT NTT (which fuses both primes
 *   on int32 inputs), we call ntt_forward twice per input — once with the
 *   q1-twisted vector (keeping only the q1 residue), once with the
 *   q2-twisted vector (keeping only the q2 residue). The pointwise multiply
 *   and inverse then run unchanged.
 *
 *     ntt_forward(inner, a_tw_q1, a_res_q1, _scratch_q2);   // keep a_res_q1
 *     ntt_forward(inner, a_tw_q2, _scratch_q1, a_res_q2);   // keep a_res_q2
 *     ntt_forward(inner, b_tw_q1, b_res_q1, _scratch_q2);
 *     ntt_forward(inner, b_tw_q2, _scratch_q1, b_res_q2);
 *     ntt_pointwise_mul(inner, a_res_q1, a_res_q2,
 *                              b_res_q1, b_res_q2,
 *                              c_res_q1, c_res_q2);
 *     ntt_inverse(inner, c_res_q1, c_res_q2, D);    // signed int64, length M
 *
 *   D is the length-M negacyclic product of the implied CRT-combined
 *   integers x = CRT(a_tw_q1, a_tw_q2), y = CRT(b_tw_q1, b_tw_q2). Because
 *   both x and y are zero for indices ≥ N, D is the linear product (no
 *   wraparound).
 *
 *   Cost is 4 × ntt_forward + 1 × ntt_pointwise_mul + 1 × ntt_inverse per
 *   Bluestein call — exactly 2× the "shape" of sp_pr_mul's pipeline (which
 *   is 2 forward + 1 mul + 1 inverse). Acceptable for a correctness-first
 *   sprint; further pruning is an NTT.5b optimization.
 *
 * Step 3 (per-prime fold to length N, in coefficient domain):
 *   D is a length-M signed centered int64. Reduce per-prime by mod qP, then
 *   fold:
 *     C_qP[k] = (D[k] mod qP + D[k+N] mod qP) mod qP   for k ∈ [0, N).
 *   For our M choices (smallest M ≥ 2N), k + N is in [N, 2N) ⊆ [0, M), so
 *   D[k+N] is always a valid index.
 *
 * Step 4 (per-prime post-untwist):
 *     c_qP[k] = (C_qP[k] · ψ_NP^{-k}) mod qP.
 *
 * Step 5 (CRT recombine to signed int64):
 *     out[k] = garner(c_q1[k], c_q2[k])
 *   via math-core's ntt_crt_recombine. Output is signed centered in
 *   (-M_full/2, M_full/2] where M_full = q1·q2 ~ 2^60.
 *
 * Bit-exactness invariant: |out[k]| < M_full / 2. For caller |coeff| < 2^14
 * and N ≤ 256, every linear-conv coefficient is bounded by N · (2^14)^2 ≤
 * 256 · 2^28 = 2^36 ≪ 2^59 ≈ M_full / 2. Recovery is exact in Z.
 *
 * Public-surface use only — no per-prime butterfly reimplementation, no
 * ntt_crt.c internal access.
 */
#include "sp/poly_ring_bluestein.h"
#include "sp/poly_ring.h"     /* sp_pr_resdot — the shared keystore hot loop */
#include "sp/ntt_crt.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>    /* memcpy — batched query staging */

/* ---- Bluestein admissible-N table ---------------------------------------- */

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
            return 0u;
    }
}

static int bluestein_admissible(uint32_t N) {
    return bluestein_inner_M(N) != 0u;
}

/* ---- modular arithmetic (mirrors ntt_crt.c's Barrett primitives, but
 *      this TU's primitives are not exposed; we own the duplication
 *      because the primitives are TU-private to ntt_crt.c).
 *
 * Both primes are 30-bit, so a*b < 2^60 fits uint64; Barrett reduces with
 * mu = floor(2^60 / q). Identical algorithm to barrett_reduce/modmul in
 * ntt_crt.c. */

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
    uint64_t x = (uint64_t)a * (uint64_t)b;
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

/* Primitive 2N-th root search — identical algorithm to ntt_crt.c's
 * find_psi but specialized for the OUTER Bluestein N (the INNER N=M still
 * uses math-core's own find_psi via ntt_init). Returns 0 if not found —
 * unreachable for our frozen primes when N ∈ {2..256}. */
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

/* Reduce a signed int64 into [0, q). Inputs land in (-M_full/2, M_full/2]
 * post ntt_inverse — well-inside int64 range — and modulo is computed via
 * the native % then adjusted for sign. Caller-friendly: handles any int64. */
static inline uint32_t pr_blue_reduce_i64_modq(int64_t x, uint32_t q) {
    int64_t r = x % (int64_t)q;
    if (r < 0) r += (int64_t)q;
    return (uint32_t)r;
}

/* Local Garner CRT recombine — mirrors ntt_crt.c's garner_one but
 * implemented here so we can drive an N-coefficient recombine without
 * over-reaching into ctx->inner->N entries (math-core's
 * ntt_crt_recombine iterates over the inner M, would overrun our
 * length-N output buffer). The Garner constants we need are q1, q2, and
 * q1^{-1} mod q2 — derived locally from public SP_NTT_Q1/Q2 plus our own
 * modular primitives. */
static inline int64_t pr_blue_garner(uint32_t x1, uint32_t x2,
                                     uint32_t q1, uint32_t q2,
                                     uint64_t mu2,
                                     uint64_t q1_inv_mod_q2) {
    /* t = (x2 - x1) mod q2, then r = x1 + q1 * t, center to (-M/2, M/2]. */
    uint32_t d;
    uint32_t x1_mod_q2 = x1 % q2;
    if (x2 >= x1_mod_q2) d = x2 - x1_mod_q2;
    else                 d = x2 + q2 - x1_mod_q2;
    uint32_t t = (uint32_t)pr_blue_barrett_reduce((uint64_t)d * q1_inv_mod_q2,
                                                  (uint64_t)q2, mu2);
    uint64_t r = (uint64_t)x1 + (uint64_t)q1 * (uint64_t)t;
    int64_t v = (int64_t)r;
    const int64_t M_full = SP_NTT_M;
    if (v > M_full / 2) v -= M_full;
    return v;
}

/* ---- per-prime side context (outer-N psi tables) ------------------------- */

typedef struct {
    uint32_t  q;
    uint64_t  mu;
    uint32_t *psi_pow;     /* ψ_NP^j      for j ∈ [0, N) */
    uint32_t *ipsi_pow;    /* ψ_NP^{-j}   for j ∈ [0, N) */
} pr_blue_prime_ctx;

static int pr_blue_prime_setup(pr_blue_prime_ctx *pc, uint32_t N, uint32_t q) {
    pc->q  = q;
    pc->mu = ((uint64_t)1 << 60) / (uint64_t)q;

    uint32_t psi = pr_blue_find_psi(N, q, pc->mu);
    if (psi == 0u) return 0;

    /* psi^N = -1, psi^{2N} = 1 — the negacyclic invariants. */
    assert(pr_blue_modpow(psi, (uint64_t)N, q, pc->mu) == q - 1u);
    assert(pr_blue_modpow(psi, 2u * (uint64_t)N, q, pc->mu) == 1u % q);

    uint32_t ipsi = pr_blue_modinv(psi, q, pc->mu);

    pc->psi_pow  = (uint32_t *)malloc(sizeof(uint32_t) * N);
    pc->ipsi_pow = (uint32_t *)malloc(sizeof(uint32_t) * N);
    if (!pc->psi_pow || !pc->ipsi_pow) return 0;

    uint32_t acc = 1u % q;
    for (uint32_t j = 0; j < N; j++) {
        pc->psi_pow[j] = acc;
        acc = pr_blue_modmul(acc, psi, q, pc->mu);
    }
    acc = 1u % q;
    for (uint32_t j = 0; j < N; j++) {
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
    pr_blue_prime_ctx  p1;
    pr_blue_prime_ctx  p2;
    uint64_t           q1_inv_mod_q2;   /* Garner constant — local, since
                                         * ntt_crt's q1_inv_mod_q2 lives
                                         * in the opaque ntt_ctx          */

    /* Per-call scratch (sized M); on the context to avoid malloc per call.
     * NOT thread-safe — one context per thread. */
    int32_t  *a_pad_q1, *a_pad_q2;   /* outer-twisted, zero-padded a   */
    int32_t  *b_pad_q1, *b_pad_q2;   /* outer-twisted, zero-padded b   */
    uint32_t *a_res_q1, *a_res_q2;   /* inner-NTT residue domain       */
    uint32_t *b_res_q1, *b_res_q2;
    uint32_t *c_res_q1, *c_res_q2;
    uint32_t *scratch_q1, *scratch_q2;   /* discarded channel from per-prime
                                          * forward calls — keeps the inner
                                          * ntt_forward API happy without
                                          * clobbering our keep channel */
    int64_t  *D;                          /* signed length-M lin-conv int64 */
    uint32_t *c_fold_q1, *c_fold_q2;     /* folded length-N residues       */

    /* ── Sprint NTT.5b: opt-in compute backend ──
     * NULL fields = host path (the NTT.5a baseline). When non-NULL, the
     * 4× ntt_forward + 1× ntt_inverse inner calls in pr_blue_convolve_M
     * route through (forward, inverse) instead of math-core's host
     * ntt_crt. Set via sp_pr_bluestein_set_backend. */
    void                       *backend_handle;
    sp_compute_ntt_dispatch_fn  backend_forward;
    sp_compute_ntt_dispatch_fn  backend_inverse;
    /* ── NTT-FUSION keystore state ──
     * w1/w2: per-index weights realising C^ -> (D[0]+D[N]) mod p, derived
     * empirically at init via unit vectors through the public ntt_inverse
     * (convention-proof). qf1/qf2: the current query's forward residues.
     * kstar: involution scratch (N). */
    uint32_t *w1, *w2;       /* per-prime weight tables (M each)   */
    uint32_t *qf1, *qf2;     /* transformed query (M each)         */
    int32_t  *kstar;         /* involuted key scratch (N)          */
    uint32_t *qb1, *qb2;     /* batched queries (nb × M each), grown on demand */
    int       qb_cap;

    /* Per-call u32 scratch for backend dispatch (sized M). Math-core's host
     * ntt_forward takes int32_t* in; the backend ABI takes uint32_t*. The
     * int32 pad buffers above already hold values in [0, qP) so a memcpy-
     * style reinterpret to u32 is safe. We keep dedicated u32 scratch to
     * keep the conversion explicit and avoid aliasing UB. */
    uint32_t *u32_pad_q1, *u32_pad_q2;   /* mirror of int32 pads as u32 */
};

/* ---- public API --------------------------------------------------------- */

sp_pr_bluestein_ctx *sp_pr_bluestein_init(uint32_t N) {
    if (!bluestein_admissible(N)) return NULL;
    uint32_t M = bluestein_inner_M(N);

    ntt_ctx *inner = ntt_init(M);
    if (!inner) return NULL;

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

    /* Garner constant: q1^{-1} mod q2, used by pr_blue_garner. */
    ctx->q1_inv_mod_q2 = pr_blue_modinv(SP_NTT_Q1 % SP_NTT_Q2, SP_NTT_Q2,
                                        ctx->p2.mu);

    ctx->a_pad_q1  = (int32_t  *)malloc(sizeof(int32_t)  * M);
    ctx->a_pad_q2  = (int32_t  *)malloc(sizeof(int32_t)  * M);
    ctx->b_pad_q1  = (int32_t  *)malloc(sizeof(int32_t)  * M);
    ctx->b_pad_q2  = (int32_t  *)malloc(sizeof(int32_t)  * M);
    ctx->a_res_q1  = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->a_res_q2  = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->b_res_q1  = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->b_res_q2  = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->c_res_q1  = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->c_res_q2  = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->scratch_q1 = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->scratch_q2 = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->D          = (int64_t  *)malloc(sizeof(int64_t)  * M);
    ctx->c_fold_q1  = (uint32_t *)malloc(sizeof(uint32_t) * N);
    ctx->c_fold_q2  = (uint32_t *)malloc(sizeof(uint32_t) * N);
    /* Sprint NTT.5b: dedicated u32 mirrors of the int32 pad buffers,
     * for the backend dispatch ABI (uint32_t* in/out). */
    ctx->u32_pad_q1 = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->u32_pad_q2 = (uint32_t *)malloc(sizeof(uint32_t) * M);

    ctx->w1    = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->w2    = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->qf1   = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->qf2   = (uint32_t *)malloc(sizeof(uint32_t) * M);
    ctx->kstar = (int32_t  *)malloc(sizeof(int32_t)  * N);

    if (!ctx->a_pad_q1 || !ctx->a_pad_q2 || !ctx->b_pad_q1 || !ctx->b_pad_q2 ||
        !ctx->a_res_q1 || !ctx->a_res_q2 || !ctx->b_res_q1 || !ctx->b_res_q2 ||
        !ctx->c_res_q1 || !ctx->c_res_q2 || !ctx->scratch_q1 || !ctx->scratch_q2 ||
        !ctx->D || !ctx->c_fold_q1 || !ctx->c_fold_q2 ||
        !ctx->u32_pad_q1 || !ctx->u32_pad_q2 ||
        !ctx->w1 || !ctx->w2 || !ctx->qf1 || !ctx->qf2 || !ctx->kstar) {
        sp_pr_bluestein_free(ctx);
        return NULL;
    }

    /* ── keystore weight derivation (one-time, ~2M public INTT calls) ──
     * The functional C^ -> (D[0] + D[N]) mod p is linear; its matrix row is
     * recovered by unit residue vectors. With C^_qOther = 0, the CRT output D
     * satisfies D == INTT_qP(e_j) (mod qP), so reducing D[0]/D[N] mod qP
     * yields exactly the per-prime weight — no assumption about the inner
     * kernel's ordering, twiddle folding, or scaling conventions. */
    {
        uint32_t j;
        for (j = 0; j < M; j++) { ctx->c_res_q1[j] = 0u; ctx->c_res_q2[j] = 0u; }
        for (j = 0; j < M; j++) {
            ctx->c_res_q1[j] = 1u;
            ntt_inverse(ctx->inner, ctx->c_res_q1, ctx->c_res_q2, ctx->D);
            uint32_t a = pr_blue_reduce_i64_modq(ctx->D[0], SP_NTT_Q1);
            uint32_t b = pr_blue_reduce_i64_modq(ctx->D[N], SP_NTT_Q1);
            uint32_t w = a + b; if (w >= SP_NTT_Q1) w -= SP_NTT_Q1;
            ctx->w1[j] = w;
            ctx->c_res_q1[j] = 0u;
        }
        for (j = 0; j < M; j++) {
            ctx->c_res_q2[j] = 1u;
            ntt_inverse(ctx->inner, ctx->c_res_q1, ctx->c_res_q2, ctx->D);
            uint32_t a = pr_blue_reduce_i64_modq(ctx->D[0], SP_NTT_Q2);
            uint32_t b = pr_blue_reduce_i64_modq(ctx->D[N], SP_NTT_Q2);
            uint32_t w = a + b; if (w >= SP_NTT_Q2) w -= SP_NTT_Q2;
            ctx->w2[j] = w;
            ctx->c_res_q2[j] = 0u;
        }
    }
    /* backend_* fields land NULL via calloc; explicit set_backend opts in. */
    return ctx;
}

void sp_pr_bluestein_free(sp_pr_bluestein_ctx *ctx) {
    if (!ctx) return;
    ntt_free(ctx->inner);
    pr_blue_prime_free(&ctx->p1);
    pr_blue_prime_free(&ctx->p2);
    free(ctx->a_pad_q1); free(ctx->a_pad_q2);
    free(ctx->b_pad_q1); free(ctx->b_pad_q2);
    free(ctx->a_res_q1); free(ctx->a_res_q2);
    free(ctx->b_res_q1); free(ctx->b_res_q2);
    free(ctx->c_res_q1); free(ctx->c_res_q2);
    free(ctx->scratch_q1); free(ctx->scratch_q2);
    free(ctx->D);
    free(ctx->c_fold_q1); free(ctx->c_fold_q2);
    free(ctx->u32_pad_q1); free(ctx->u32_pad_q2);
    free(ctx->w1); free(ctx->w2); free(ctx->qf1); free(ctx->qf2); free(ctx->kstar);
    free(ctx->qb1); free(ctx->qb2);
    free(ctx);
}

uint32_t sp_pr_bluestein_degree(const sp_pr_bluestein_ctx *ctx) {
    return ctx ? ctx->N : 0u;
}

/* ---- inner pipeline (shared by inner / mul) ------------------------------ */

/* Outer-twist input vector `in` (length N) into per-prime zero-padded
 * length-M scratch `pad_q1` / `pad_q2`. Values land in [0, qP) but encoded
 * as int32 (max ~2^30 < 2^31). */
static void pr_blue_twist_input(const sp_pr_bluestein_ctx *ctx,
                                const int32_t *in,
                                int32_t *pad_q1, int32_t *pad_q2) {
    const uint32_t N = ctx->N;
    const uint32_t M = ctx->M;
    const uint32_t q1 = ctx->p1.q;
    const uint32_t q2 = ctx->p2.q;
    const uint64_t mu1 = ctx->p1.mu;
    const uint64_t mu2 = ctx->p2.mu;
    const uint32_t *psi1 = ctx->p1.psi_pow;
    const uint32_t *psi2 = ctx->p2.psi_pow;

    for (uint32_t j = 0; j < N; j++) {
        int64_t v1 = (int64_t)in[j] % (int64_t)q1;
        if (v1 < 0) v1 += (int64_t)q1;
        int64_t v2 = (int64_t)in[j] % (int64_t)q2;
        if (v2 < 0) v2 += (int64_t)q2;
        pad_q1[j] = (int32_t)pr_blue_modmul((uint32_t)v1, psi1[j], q1, mu1);
        pad_q2[j] = (int32_t)pr_blue_modmul((uint32_t)v2, psi2[j], q2, mu2);
    }
    /* zero-pad to length M */
    for (uint32_t j = N; j < M; j++) {
        pad_q1[j] = 0;
        pad_q2[j] = 0;
    }
}

/* Drive the CRT NTT pipeline to compute the length-M signed-centered int64
 * linear convolution D of two outer-twisted, zero-padded inputs. Either path
 * (host vs backend) ends at the same D buffer; downstream fold/untwist/garner
 * is direction-agnostic.
 *
 * Host path: 4× ntt_forward (each fuses both primes from one int32 input;
 * we discard the wrong channel via the scratch_qP buffers), 1× ntt_pointwise_mul,
 * 1× ntt_inverse (which does its own CRT recombine internally).
 *
 * Backend path (NTT.5b): 4× backend_forward (one per (operand × prime) per the
 * uniform per-prime u32 ABI), 1× ntt_pointwise_mul (host-side; small + cheap),
 * 2× backend_inverse (one per prime; produces per-prime u32 residues in [0, q)),
 * 1× ntt_crt_recombine (host-side; produces the same D buffer ntt_inverse would
 * have produced).
 *
 * Per-direction NULL fallback: if backend_forward is NULL the forward path uses
 * host; same for backend_inverse independently. This lets us validate forward-
 * only or inverse-only dispatch during incremental bring-up. */
static void pr_blue_convolve_M(sp_pr_bluestein_ctx *ctx) {
    const uint32_t M = ctx->M;
    const int     iM = (int)M;
    void * const  h  = ctx->backend_handle;
    const sp_compute_ntt_dispatch_fn fwd = ctx->backend_forward;
    const sp_compute_ntt_dispatch_fn inv = ctx->backend_inverse;

    /* ── Forward NTTs (per operand, per prime) ── */
    if (fwd) {
        /* Backend dispatch path: uniform u32-LE in/out per prime. The pad
         * buffers already hold values in [0, qP) (per pr_blue_twist_input),
         * so the int32→u32 mirror is a value-preserving copy. */
        for (uint32_t j = 0; j < M; j++) {
            ctx->u32_pad_q1[j] = (uint32_t)ctx->a_pad_q1[j];
            ctx->u32_pad_q2[j] = (uint32_t)ctx->a_pad_q2[j];
        }
        if (fwd(h, 0, iM, ctx->u32_pad_q1, ctx->a_res_q1) != 0 ||
            fwd(h, 1, iM, ctx->u32_pad_q2, ctx->a_res_q2) != 0) {
            /* Dispatch failure: fall back to host for this single call. The
             * caller has no error channel from convolve_M; the smoke test
             * will see the divergence via bit-exact compare and surface it. */
            ntt_forward(ctx->inner, ctx->a_pad_q1, ctx->a_res_q1, ctx->scratch_q2);
            ntt_forward(ctx->inner, ctx->a_pad_q2, ctx->scratch_q1, ctx->a_res_q2);
        }
        for (uint32_t j = 0; j < M; j++) {
            ctx->u32_pad_q1[j] = (uint32_t)ctx->b_pad_q1[j];
            ctx->u32_pad_q2[j] = (uint32_t)ctx->b_pad_q2[j];
        }
        if (fwd(h, 0, iM, ctx->u32_pad_q1, ctx->b_res_q1) != 0 ||
            fwd(h, 1, iM, ctx->u32_pad_q2, ctx->b_res_q2) != 0) {
            ntt_forward(ctx->inner, ctx->b_pad_q1, ctx->b_res_q1, ctx->scratch_q2);
            ntt_forward(ctx->inner, ctx->b_pad_q2, ctx->scratch_q1, ctx->b_res_q2);
        }
    } else {
        /* Host path (NTT.5a baseline). */
        ntt_forward(ctx->inner, ctx->a_pad_q1, ctx->a_res_q1, ctx->scratch_q2);
        ntt_forward(ctx->inner, ctx->a_pad_q2, ctx->scratch_q1, ctx->a_res_q2);
        ntt_forward(ctx->inner, ctx->b_pad_q1, ctx->b_res_q1, ctx->scratch_q2);
        ntt_forward(ctx->inner, ctx->b_pad_q2, ctx->scratch_q1, ctx->b_res_q2);
    }

    /* ── Per-prime pointwise multiply (host-side in either path) ── */
    ntt_pointwise_mul(ctx->inner,
                      ctx->a_res_q1, ctx->a_res_q2,
                      ctx->b_res_q1, ctx->b_res_q2,
                      ctx->c_res_q1, ctx->c_res_q2);

    /* ── Inverse + CRT ── */
    if (inv) {
        /* Backend dispatch path: two per-prime INTTs into u32 [0, q) residues,
         * then host-side Garner via the public ntt_crt_recombine accessor.
         * scratch_q1/scratch_q2 hold the per-prime residue outputs. */
        if (inv(h, 0, iM, ctx->c_res_q1, ctx->scratch_q1) != 0 ||
            inv(h, 1, iM, ctx->c_res_q2, ctx->scratch_q2) != 0) {
            /* Dispatch failure → fall back to host */
            ntt_inverse(ctx->inner, ctx->c_res_q1, ctx->c_res_q2, ctx->D);
        } else {
            ntt_crt_recombine(ctx->inner, ctx->scratch_q1, ctx->scratch_q2, ctx->D);
        }
    } else {
        ntt_inverse(ctx->inner, ctx->c_res_q1, ctx->c_res_q2, ctx->D);
    }
}

/* Fold D[0..M) into per-prime length-N residue vectors C_qP[k]:
 *     C_qP[k] = (D[k] + (k+N < M ? D[k+N] : 0)) mod qP.
 * For our M choices (smallest M ≥ 2N), k+N is always < M for k ∈ [0,N). */
static void pr_blue_fold(const sp_pr_bluestein_ctx *ctx) {
    const uint32_t N = ctx->N;
    const uint32_t M = ctx->M;
    const uint32_t q1 = ctx->p1.q;
    const uint32_t q2 = ctx->p2.q;
    for (uint32_t k = 0; k < N; k++) {
        int64_t d_k = ctx->D[k];
        int64_t d_kn = (k + N < M) ? ctx->D[k + N] : 0;
        uint32_t r1_a = pr_blue_reduce_i64_modq(d_k,  q1);
        uint32_t r1_b = pr_blue_reduce_i64_modq(d_kn, q1);
        uint32_t s1   = r1_a + r1_b;
        if (s1 >= q1) s1 -= q1;
        ctx->c_fold_q1[k] = s1;

        uint32_t r2_a = pr_blue_reduce_i64_modq(d_k,  q2);
        uint32_t r2_b = pr_blue_reduce_i64_modq(d_kn, q2);
        uint32_t s2   = r2_a + r2_b;
        if (s2 >= q2) s2 -= q2;
        ctx->c_fold_q2[k] = s2;
    }
}

/* Per-prime post-untwist of c_fold_qP[k] by ψ_NP^{-k}. In place. */
static void pr_blue_untwist(const sp_pr_bluestein_ctx *ctx) {
    const uint32_t N = ctx->N;
    const uint32_t q1 = ctx->p1.q;
    const uint32_t q2 = ctx->p2.q;
    const uint64_t mu1 = ctx->p1.mu;
    const uint64_t mu2 = ctx->p2.mu;
    const uint32_t *ip1 = ctx->p1.ipsi_pow;
    const uint32_t *ip2 = ctx->p2.ipsi_pow;
    for (uint32_t k = 0; k < N; k++) {
        ctx->c_fold_q1[k] = pr_blue_modmul(ctx->c_fold_q1[k], ip1[k], q1, mu1);
        ctx->c_fold_q2[k] = pr_blue_modmul(ctx->c_fold_q2[k], ip2[k], q2, mu2);
    }
}

/* ---- public API: inner and mul ------------------------------------------ */

void sp_pr_bluestein_mul(sp_pr_bluestein_ctx *ctx,
                         const int32_t *a, const int32_t *b,
                         int64_t *out) {
    if (!ctx) return;
    pr_blue_twist_input(ctx, a, ctx->a_pad_q1, ctx->a_pad_q2);
    pr_blue_twist_input(ctx, b, ctx->b_pad_q1, ctx->b_pad_q2);
    pr_blue_convolve_M(ctx);
    pr_blue_fold(ctx);
    pr_blue_untwist(ctx);
    /* CRT recombine the N coefficients into signed centered int64. We
     * drive Garner manually (not via ntt_crt_recombine) because the
     * latter iterates ctx->inner->N entries == our inner M, and would
     * over-write the caller's length-N out buffer beyond bounds. */
    const uint32_t N = ctx->N;
    const uint32_t q1 = ctx->p1.q;
    const uint32_t q2 = ctx->p2.q;
    const uint64_t mu2 = ctx->p2.mu;
    const uint64_t q1inv = ctx->q1_inv_mod_q2;
    for (uint32_t k = 0; k < N; k++)
        out[k] = pr_blue_garner(ctx->c_fold_q1[k], ctx->c_fold_q2[k],
                                q1, q2, mu2, q1inv);
}

/* ── Sprint NTT.5b: opt-in compute backend setter ──
 * Caller-serialized. Storage is a plain assign (3 word-sized fields). NULL
 * combinations are accepted as-written; pr_blue_convolve_M checks each
 * direction's pointer at dispatch time and falls back per-direction. */
void sp_pr_bluestein_set_backend(sp_pr_bluestein_ctx *ctx,
                                 void *handle,
                                 sp_compute_ntt_dispatch_fn forward,
                                 sp_compute_ntt_dispatch_fn inverse) {
    if (!ctx) return;
    ctx->backend_handle  = handle;
    ctx->backend_forward = forward;
    ctx->backend_inverse = inverse;
}

int64_t sp_pr_bluestein_inner(sp_pr_bluestein_ctx *ctx,
                              const int32_t *q, const int32_t *k) {
    if (!ctx) return 0;
    /* Same involution as sp_pr_inner: <q,k> = (q (x) k*)_0 with
     * k*_0 = k_0, k*_j = -k_{N-j} for j > 0. We reuse the mul scratch by
     * computing the involuted k into the b_pad buffers — but b_pad is
     * an internal pre-padded twisted buffer, not the raw input. Cleanest
     * to compute kstar into a local stack-or-scratch int32 array of length
     * N, then call sp_pr_bluestein_mul, then read coefficient 0.
     *
     * For the inner fast path (read only c_0), we still pay the full mul
     * cost in this version; an optimization (skip the fold + untwist for
     * k > 0) is deferred to NTT.5b. */
    const uint32_t N = ctx->N;
    int32_t *kstar = (int32_t *)malloc(sizeof(int32_t) * N);
    if (!kstar) return 0;
    kstar[0] = k[0];
    for (uint32_t j = 1; j < N; j++) kstar[j] = -k[N - j];

    int64_t *prod = (int64_t *)malloc(sizeof(int64_t) * N);
    if (!prod) { free(kstar); return 0; }
    sp_pr_bluestein_mul(ctx, q, kstar, prod);

    int64_t c0 = prod[0];
    free(kstar);
    free(prod);
    return c0;
}

/* ── NTT-FUSION keystore (Bluestein side; contract in the header) ─────────── */

size_t sp_pr_bluestein_kstore_words(const sp_pr_bluestein_ctx *ctx) {
    return ctx ? (size_t)2u * ctx->M : 0u;
}

void sp_pr_bluestein_kstore_encode(sp_pr_bluestein_ctx *ctx, const int32_t *k,
                                   uint32_t *kres_out) {
    const uint32_t N = ctx->N, M = ctx->M;
    ctx->kstar[0] = k[0];
    for (uint32_t j = 1; j < N; j++) ctx->kstar[j] = -k[N - j];   /* involution */
    pr_blue_twist_input(ctx, ctx->kstar, ctx->b_pad_q1, ctx->b_pad_q2);
    ntt_forward(ctx->inner, ctx->b_pad_q1, ctx->b_res_q1, ctx->scratch_q2);
    ntt_forward(ctx->inner, ctx->b_pad_q2, ctx->scratch_q1, ctx->b_res_q2);
    /* fold the coefficient-0 weights into the stored key */
    for (uint32_t j = 0; j < M; j++) {
        kres_out[j]     = pr_blue_modmul(ctx->b_res_q1[j], ctx->w1[j], ctx->p1.q, ctx->p1.mu);
        kres_out[M + j] = pr_blue_modmul(ctx->b_res_q2[j], ctx->w2[j], ctx->p2.q, ctx->p2.mu);
    }
}

void sp_pr_bluestein_query_begin(sp_pr_bluestein_ctx *ctx, const int32_t *q) {
    pr_blue_twist_input(ctx, q, ctx->a_pad_q1, ctx->a_pad_q2);
    ntt_forward(ctx->inner, ctx->a_pad_q1, ctx->qf1, ctx->scratch_q2);
    ntt_forward(ctx->inner, ctx->a_pad_q2, ctx->scratch_q1, ctx->qf2);
}

int64_t sp_pr_bluestein_score_kstore(sp_pr_bluestein_ctx *ctx, const uint32_t *kres) {
    const uint32_t M = ctx->M;
    const uint32_t q1 = ctx->p1.q, q2 = ctx->p2.q;
    uint32_t s1 = sp_pr_resdot(ctx->qf1, kres,     M, q1);
    uint32_t s2 = sp_pr_resdot(ctx->qf2, kres + M, M, q2);
    return pr_blue_garner(s1, s2, q1, q2, ctx->p2.mu, ctx->q1_inv_mod_q2);
}

/* ── batched keystore (Bluestein arm; contract in poly_ring.h §batch) ───────
 * Correctness-path implementation: the chirp pre/post stages are per-item
 * scalar work, so the batch variants LOOP the proven single-call path into
 * per-item storage. Bit-exactness is inherited; the throughput lever lives in
 * the direct-N arm (sp_ntt_fwd_batch lanes). If a Bluestein-shaped model ever
 * becomes the speed target, batch the inner length-M transforms here too —
 * surface that upstream rather than widening this quietly. */

void sp_pr_bluestein_query_begin_batch(sp_pr_bluestein_ctx *ctx,
                                       const int32_t *q, size_t qstride, int nb) {
    const uint32_t M = ctx->M;
    if (nb <= 0) return;
    if (nb > ctx->qb_cap) {
        uint32_t *n1 = (uint32_t *)realloc(ctx->qb1, sizeof(uint32_t) * (size_t)nb * M);
        if (!n1) return;
        ctx->qb1 = n1;
        uint32_t *n2 = (uint32_t *)realloc(ctx->qb2, sizeof(uint32_t) * (size_t)nb * M);
        if (!n2) return;
        ctx->qb2 = n2;
        ctx->qb_cap = nb;
    }
    for (int i = 0; i < nb; i++) {
        sp_pr_bluestein_query_begin(ctx, q + (size_t)i * qstride);
        memcpy(ctx->qb1 + (size_t)i * M, ctx->qf1, sizeof(uint32_t) * M);
        memcpy(ctx->qb2 + (size_t)i * M, ctx->qf2, sizeof(uint32_t) * M);
    }
}

int64_t sp_pr_bluestein_score_kstore_b(sp_pr_bluestein_ctx *ctx, int i,
                                       const uint32_t *kres) {
    const uint32_t M = ctx->M;
    const uint32_t q1 = ctx->p1.q, q2 = ctx->p2.q;
    uint32_t s1 = sp_pr_resdot(ctx->qb1 + (size_t)i * M, kres,     M, q1);
    uint32_t s2 = sp_pr_resdot(ctx->qb2 + (size_t)i * M, kres + M, M, q2);
    return pr_blue_garner(s1, s2, q1, q2, ctx->p2.mu, ctx->q1_inv_mod_q2);
}

void sp_pr_bluestein_kstore_encode_batch(sp_pr_bluestein_ctx *ctx,
                                         const int32_t *k, size_t kstride,
                                         uint32_t *kres_out, size_t ostride,
                                         int nb) {
    for (int i = 0; i < nb; i++)
        sp_pr_bluestein_kstore_encode(ctx, k + (size_t)i * kstride,
                                      kres_out + (size_t)i * ostride);
}
