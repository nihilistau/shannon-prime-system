/* poly_ring_bluestein.c — host-side Bluestein wrapper for negacyclic R_q
 * polynomial multiplication at arbitrary power-of-2 degree N ∈ [2, 256].
 *
 * STAGE 1 SKELETON: header surface only. init returns NULL for every N until
 * Stage 2 wires in the inner ntt_ctx + psi-twist tables. Free is a safe
 * no-op. Degree returns 0 for NULL ctx. inner and mul intentionally not
 * reached at Stage 1 because init never produces a non-NULL ctx.
 *
 * Algorithm (Approach A, derived in tools/sp_compute_skel/docs/PLAN-NTT-5a.md):
 *   negacyclic length-N product over Z_q[x]/(x^N+1) via
 *   (1) psi-twist with ψ_N (2N-th root) → cyclic length-N convolution;
 *   (2) zero-pad to length M ∈ {128,256,512} with M ≥ 2N → length-M linear
 *       convolution via math-core's negacyclic NTT (no wraparound term
 *       contributes because zero-padded inputs make all length-M
 *       wraparound sums identically zero);
 *   (3) fold length-(2N-1) linear result to length-N cyclic via
 *       C_k = D_k + (k+N < M ? D_{k+N} : 0);
 *   (4) post-untwist by ψ_N^{-k};
 *   (5) symmetric Garner CRT recombine to signed centered int64.
 *
 * Anti-contamination: this file is a NEW caller-side wrapper; it does NOT
 * modify poly_ring.h/c or ntt_crt.h/c. All ntt_crt access is through the
 * public surface in sp/ntt_crt.h (ntt_init, ntt_forward, ntt_pointwise_mul,
 * ntt_inverse, ntt_crt_recombine, SP_NTT_Q1, SP_NTT_Q2, SP_NTT_M).
 */
#include "sp/poly_ring_bluestein.h"
#include "sp/ntt_crt.h"

#include <stdint.h>
#include <stdlib.h>

/* ---- Stage 1 skeleton — init always returns NULL --------------------------
 * Stage 2 will replace this with the real context allocator. Keeping the
 * skeleton compilable and the test harness wired now lets Stage 1's oracle
 * cross-check run independently of any Bluestein implementation. */

struct sp_pr_bluestein_ctx {
    uint32_t N;   /* placeholder so the struct has at least one field */
};

sp_pr_bluestein_ctx *sp_pr_bluestein_init(uint32_t N) {
    (void)N;
    /* Stage 1: always NULL. Stage 2 will accept admissible N. */
    return NULL;
}

void sp_pr_bluestein_free(sp_pr_bluestein_ctx *ctx) {
    /* Free is safe at all stages; ctx is NULL through Stage 1. */
    if (!ctx) return;
    free(ctx);
}

uint32_t sp_pr_bluestein_degree(const sp_pr_bluestein_ctx *ctx) {
    return ctx ? ctx->N : 0u;
}

int64_t sp_pr_bluestein_inner(sp_pr_bluestein_ctx *ctx,
                              const int32_t *q, const int32_t *k) {
    /* Stage 1: callers cannot reach this with a valid ctx. */
    (void)ctx; (void)q; (void)k;
    return 0;
}

void sp_pr_bluestein_mul(sp_pr_bluestein_ctx *ctx,
                         const int32_t *a, const int32_t *b,
                         int64_t *out) {
    /* Stage 1: callers cannot reach this with a valid ctx. */
    (void)ctx; (void)a; (void)b;
    /* Defensive zero-fill on the off-chance caller passes a non-NULL out. */
    (void)out;
}
