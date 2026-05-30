/* poly_ring_bluestein.h — host-side Bluestein wrapper for negacyclic R_q
 * polynomial multiplication at arbitrary power-of-2 degree N in [2, 256].
 *
 * Companion to sp/poly_ring.h. Same ring R_q = Z_q[x]/(x^N + 1), same dual-
 * prime CRT-NTT substrate (M = q1*q2 ~ 2^60, recovery in (-M/2, M/2]), same
 * coefficient bit-exactness invariant (|c_k| < M/2 ⇒ exact in Z).
 *
 * Direct sp_pr_init(N) is restricted to N ∈ {128, 256, 512} because math-
 * core's inner ntt_init(N) admits exactly those degrees with the frozen
 * primes (v_2(q_i - 1) = 10, max 2N = 1024). Bluestein wrapping extends the
 * admissible set to all powers of 2 up to 256, by:
 *
 *   1) twisting the negacyclic length-N convolution into a cyclic length-N
 *      convolution via the standard psi-twist (ψ_N a primitive 2N-th root,
 *      ψ_N^N = -1), which is admissible whenever 2N | (q-1);
 *   2) implementing the cyclic length-N convolution as a length-M linear
 *      convolution (zero-padded), where M ∈ {128, 256, 512} is the smallest
 *      admissible inner-NTT length satisfying M ≥ 2N;
 *   3) folding the length-M linear result back to length-N cyclic;
 *   4) post-untwisting by ψ_N^{-k} to recover the negacyclic coefficients;
 *   5) symmetric Garner CRT recombine into signed centered int64.
 *
 * N=512 is NOT exposed through this API — direct sp_pr_init(512) is more
 * efficient (M=1024 would be required by Bluestein but is mathematically
 * unsupported by the frozen primes). N values with odd factors > 1 (e.g.
 * 96, 384) are not admissible at all with the current primes, regardless of
 * algorithm — see reference-ntt-frozen-primes-N-cap. For those, the SP-
 * aligned answer is direct integer dot product with Barrett reduction; do
 * not propose mixed-radix or Good-Thomas (banned per
 * reference-ntt-bluestein-arbitrary-n-escape).
 *
 * Ownership / sizes: same conventions as sp/poly_ring.h. Coefficient inputs
 * are int32 arrays of exactly N entries; sp_pr_bluestein_mul writes N int64
 * entries; sp_pr_bluestein_inner returns int64. Contexts are NOT thread-safe
 * (they own per-call scratch); use one context per thread.
 */
#ifndef SP_POLY_RING_BLUESTEIN_H
#define SP_POLY_RING_BLUESTEIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sp_pr_bluestein_ctx sp_pr_bluestein_ctx;

/* Allocate a Bluestein-wrapped polynomial-ring context for degree N.
 * Admissible N values: {2, 4, 8, 16, 32, 64, 128, 256}.
 *   - Returns NULL for N=512 (direct sp_pr_init handles that case more
 *     efficiently, and the inner Bluestein NTT would need M=1024 which
 *     the frozen primes cannot support).
 *   - Returns NULL for any non-power-of-2 N (would require an N-th root
 *     of unity with odd factor, which our frozen primes lack).
 *   - Returns NULL for N=1 (no convolution).
 * Free with sp_pr_bluestein_free. */
sp_pr_bluestein_ctx *sp_pr_bluestein_init(uint32_t N);

/* Release a context. sp_pr_bluestein_free(NULL) is a no-op. */
void sp_pr_bluestein_free(sp_pr_bluestein_ctx *ctx);

/* Degree N the context was created with. Returns 0 for NULL ctx. */
uint32_t sp_pr_bluestein_degree(const sp_pr_bluestein_ctx *ctx);

/* Bluestein-wrapped inner product <q,k> = sum_i q_i k_i.
 *
 * Same semantics as sp_pr_inner: exact in Z provided |<q,k>| < M/2 where
 * M = q1*q2 ~ 2^60. With caller |coeff| < 2^14 and N ≤ 256, accumulator is
 * < 2^36 << M/2. q, k caller-owned with exactly N int32 entries. */
int64_t sp_pr_bluestein_inner(sp_pr_bluestein_ctx *ctx,
                              const int32_t *q, const int32_t *k);

/* Full negacyclic product `out = a (x) b` in R_q via Bluestein wrap.
 *
 * Same semantics as sp_pr_mul: `out` receives N signed centered int64
 * coefficients in (-M/2, M/2]. `a`, `b`, `out` are caller-owned with
 * exactly N entries; `out` must not alias `a` or `b`. */
void sp_pr_bluestein_mul(sp_pr_bluestein_ctx *ctx,
                         const int32_t *a, const int32_t *b,
                         int64_t *out);

#ifdef __cplusplus
}
#endif

#endif /* SP_POLY_RING_BLUESTEIN_H */
