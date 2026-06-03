/* poly_ring.h — polynomial-ring attention over R_q = Z_q[x]/(x^N+1) (Phase 1C).
 *
 * Built on the Phase-1B dual-prime CRT-NTT kernel (sp/ntt_crt.h). The ring is
 * negacyclic (x^N = -1) for N in {128,256,512}, carried by the two frozen
 * primes q1,q2 with M = q1*q2 (~2^60); coefficients are recovered signed and
 * centered in (-M/2, M/2].
 *
 * This module realises PPT Step 5/6 (FUSED_KQ / score formation): the attention
 * inner product <q,k> is computed *exactly* as one coefficient of a negacyclic
 * polynomial product, using the negacyclic involution of PPT-LAT-Theory §6.1
 *     k*_0 = k_0,   k*_j = -k_{N-j}  (j > 0).
 *
 * Derivation (also restated in poly_ring.c on sp_pr_inner). For the negacyclic
 * product c = a (x) b in R_q,
 *     c_k = sum_{i+j=k} a_i b_j  -  sum_{i+j=k+N} a_i b_j.
 * Take a = q, b = k* and read coefficient 0. Only the i+j=0 term (=q_0 k*_0)
 * and the i+j=N terms (i=1..N-1, j=N-i) contribute, and the latter enter with
 * a minus sign from x^N=-1:
 *     c_0 = q_0 k*_0 - sum_{i=1..N-1} q_i k*_{N-i}
 *         = q_0 k_0   - sum_{i=1..N-1} q_i (-k_i)
 *         = sum_{i=0..N-1} q_i k_i = <q,k>.
 * So <q,k> = (q (x) k*)_0, exact in Z (no wraparound contamination), as long as
 * |<q,k>| < M/2 — which holds with room to spare for the supported degrees and
 * any reasonable coefficient range (e.g. |coeff| < 2^23 gives |<q,k>| < 2^55).
 *
 * Ownership / sizes: every array argument is caller-owned. Coefficient vectors
 * (a, b, q, k, keys[i]) have exactly N entries of arbitrary int32; products
 * (`out`) have N int64 entries. `probs_out` for sp_pr_attention has n_keys
 * doubles. The context caches an ntt_ctx plus reusable scratch and is NOT
 * thread-safe (its scratch is mutated per call); use one context per thread.
 */
#ifndef SP_POLY_RING_H
#define SP_POLY_RING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sp_pr_ctx sp_pr_ctx;

/* Allocate a polynomial-ring context for degree N. N must be one of
 * {128,256,512}; any other value returns NULL (the underlying ntt_init rejects
 * it — the frozen primes admit no 2N-th root beyond N=512). Free with
 * sp_pr_free. */
sp_pr_ctx *sp_pr_init(uint32_t N);

/* Release a context. sp_pr_free(NULL) is a no-op. */
void sp_pr_free(sp_pr_ctx *ctx);

/* Degree N the context was created with. */
uint32_t sp_pr_degree(const sp_pr_ctx *ctx);

/* Negacyclic product out = a (x) b in R_q via the NTT pipeline (forward both,
 * pointwise-multiply per residue, inverse, CRT recombine). `out` receives N
 * signed centered coefficients in (-M/2, M/2]. `a`, `b`, `out` are caller-owned
 * with exactly N entries; `out` must not alias `a` or `b`. */
void sp_pr_mul(sp_pr_ctx *ctx, const int32_t *a, const int32_t *b,
               int64_t *out);

/* Attention inner product <q,k> = sum_i q_i k_i, recovered EXACTLY as
 * coefficient 0 of the negacyclic product of q with the involuted k
 * (k*_0=k_0, k*_j=-k_{N-j} for j>0). See the file-header derivation. Exact in
 * Z provided |<q,k>| < M/2. q, k are caller-owned with N entries each. */
int64_t sp_pr_inner(sp_pr_ctx *ctx, const int32_t *q, const int32_t *k);

/* Attention over a set of keys: score_i = <q, keys[i]> via sp_pr_inner, then a
 * standard numerically-stable softmax (subtract max, exp, normalise) over the
 * recovered integer scores. `keys` is an array of n_keys pointers, each to an
 * N-entry coefficient vector. `probs_out` receives n_keys doubles summing to 1.
 * If n_keys <= 0 this is a no-op. */
void sp_pr_attention(sp_pr_ctx *ctx, const int32_t *q,
                     const int32_t *const *keys, int n_keys,
                     double *probs_out);

/* ── NTT-FUSION keystore: K stored natively in the dual-prime residue domain ──
 *
 * The write-once/read-many factoring of sp_pr_inner. At K-write time the
 * involuted key k* is forward-transformed ONCE and stored as its dual-prime
 * residue block (the same object the QUIC mesh ships — u32 residues per prime).
 * At score time the query is transformed once per (head,step) and each stored
 * key costs ONE residue dot per prime + a scalar Garner — no per-pair forward
 * transform, no inverse butterflies:
 *     c_0 = psi^0 * INTT(q^ . k^*)_0 = N^{-1} * sum_j q^_j k^*_j  (mod p),
 * and sum_j is permutation-invariant, so the kernel's residue ordering is
 * immaterial. EXACTNESS CONTRACT: sp_pr_score_kstore(encode(k)) must equal
 * sp_pr_inner(q,k) to the BIT for all inputs (gate T_PR_KSTORE). */

/* Stored-key block layout: [N residues mod q1][N residues mod q2] = 2N u32. */
size_t sp_pr_kstore_words(const sp_pr_ctx *ctx);   /* == 2N */

/* Write-once: involute k (k*_0=k_0, k*_j=-k_{N-j}) and forward-transform into
 * kres_out[2N] (caller-owned; this is the residue block to cache/spill/ship). */
void sp_pr_kstore_encode(sp_pr_ctx *ctx, const int32_t *k, uint32_t *kres_out);

/* Per (head, step): forward-transform q once into the context scratch.
 * Subsequent sp_pr_score_kstore calls score against THIS query. */
void sp_pr_query_begin(sp_pr_ctx *ctx, const int32_t *q);

/* Exact <q,k> against a stored key block (after sp_pr_query_begin): residue
 * dot per prime, scaled by N^{-1} mod p, Garner-recombined to the signed
 * centered integer in (-M/2, M/2]. Bit-equal to sp_pr_inner(q,k). */
int64_t sp_pr_score_kstore(sp_pr_ctx *ctx, const uint32_t *kres);

/* ── the keystore hot loop: modular residue dot ──────────────────────────────
 * sum_i a[i]*b[i] mod q for residues a[i], b[i] in [0, q), q < 2^30. Result in
 * [0, q). DEFERRED-REDUCTION contract: products are < 2^60, so 15 accumulate
 * exactly in uint64 before one reduction — the portable reference (resdot.c,
 * its OWN archive member) reduces once per 15-element chunk. The ENGINE
 * overrides this symbol with an AVX2 kernel (same chunking, _mm256_mul_epu32
 * bulk) by the same always-pulled-object pattern as the sp_ dispatch shims;
 * both implementations are EXACT (integer adds below 2^64, mod-sum
 * associativity), gate T_PR_RESDOT. Both keystore score paths (direct +
 * Bluestein) route through this. */
uint32_t sp_pr_resdot(const uint32_t *a, const uint32_t *b, uint32_t n, uint32_t q);

#ifdef __cplusplus
}
#endif

#endif /* SP_POLY_RING_H */
