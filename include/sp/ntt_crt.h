/* ntt_crt.h — dual-prime CRT negacyclic NTT kernel (Phase 1B).
 *
 * Ring  R_q = Z_q[x]/(x^N + 1)  (negacyclic, x^N = -1) for N in {128,256,512}.
 *
 * Two frozen 30-bit primes carry the residue arithmetic:
 *     q1 = 1073738753, q2 = 1073732609,  M = q1*q2 (~2^60).
 * A polynomial is held as a pair of residue-domain vectors (mod q1, mod q2);
 * CRT (Garner) recombines them into the signed centered residue mod M.
 *
 * Production path (ntt_crt.c) uses NO 128-bit integer type: products of two
 * 30-bit residues fit uint64, and Barrett reduction keeps every intermediate
 * < 2^64. Only the TEST-ONLY parity oracle (ntt_ref_int128.c) uses __int128.
 * This header is MSVC-clean (no GNU extensions) so it compiles on every tier.
 *
 * Ownership / sizes: every array argument is caller-owned. Coefficient and
 * residue vectors have exactly N entries, where N is the value passed to
 * ntt_init(). Residue vectors hold values in [0, q1) resp. [0, q2).
 */
#ifndef SP_NTT_CRT_H
#define SP_NTT_CRT_H

#include <stdint.h>

/* Production sources must never pull in a 128-bit type. The configure-time
 * guard in core/ntt_crt/CMakeLists.txt is the real gate (T_NTT_5); this is a
 * documentary belt-and-braces note pinned to the public surface. */
/* The kernel relies on 64-bit products of two 30-bit residues (< 2^60) with
 * Barrett reduction, so a true 64-bit uint64_t is mandatory and a 128-bit type
 * is never needed in the production path. The enforcing T_NTT_5 gate lives in
 * core/ntt_crt/CMakeLists.txt (configure-time scan of ntt_crt.c). */
/* C11/C++11 static assert, guarded for C++ backend consumers. */
#ifdef __cplusplus
static_assert(sizeof(uint64_t) == 8,
              "ntt_crt needs a 64-bit uint64_t; production path is "
              "128-bit-type-free (T_NTT_5 guards ntt_crt.c at configure time)");
#else
_Static_assert(sizeof(uint64_t) == 8,
               "ntt_crt needs a 64-bit uint64_t; production path is "
               "128-bit-type-free (T_NTT_5 guards ntt_crt.c at configure time)");
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Frozen system parameters (mirrored as compile-time constants for callers
 * that need M without constructing a context). */
#define SP_NTT_Q1   ((uint32_t)1073738753u)
#define SP_NTT_Q2   ((uint32_t)1073732609u)
/* M = q1*q2 = 1152908312643096577 (~2^60). */
#define SP_NTT_M    ((int64_t)1152908312643096577)

typedef struct ntt_ctx ntt_ctx;

/* Allocate and initialise a context for transform length N.
 * N must be one of {128,256,512}; any other value (including 1024, which the
 * frozen primes cannot support) returns NULL. The returned context caches the
 * per-prime psi/omega twiddle tables and is read-only thereafter, so a single
 * context may be shared across threads for transforms. Free with ntt_free. */
ntt_ctx *ntt_init(uint32_t N);

/* Release a context. ntt_free(NULL) is a no-op. */
void ntt_free(ntt_ctx *ctx);

/* Forward negacyclic NTT. `in` holds N signed coefficients (arbitrary int32);
 * each is reduced into [0,q) per prime before transforming. Writes the two
 * residue-domain vectors (N entries each) to out1 (mod q1) and out2 (mod q2). */
void ntt_forward(const ntt_ctx *ctx, const int32_t *in,
                 uint32_t *out1, uint32_t *out2);

/* Inverse negacyclic NTT + CRT. Takes the two residue-domain vectors
 * (in1 mod q1, in2 mod q2, N entries each) produced/processed in the NTT
 * domain, runs the per-prime inverse transform, then CRT-recombines into N
 * signed centered coefficients in (-M/2, M/2], written to `out`. */
void ntt_inverse(const ntt_ctx *ctx, const uint32_t *in1, const uint32_t *in2,
                 int64_t *out);

/* Per-residue pointwise multiply in the NTT domain (NO cross-prime mixing):
 * out1[i] = a1[i]*b1[i] mod q1,  out2[i] = a2[i]*b2[i] mod q2. All vectors
 * have N entries. out may alias a or b. */
void ntt_pointwise_mul(const ntt_ctx *ctx,
                       const uint32_t *a1, const uint32_t *a2,
                       const uint32_t *b1, const uint32_t *b2,
                       uint32_t *out1, uint32_t *out2);

/* Garner CRT recombine of N residue pairs (x1 mod q1, x2 mod q2) into N signed
 * centered coefficients in (-M/2, M/2], written to `out`. Standalone form of
 * the recombine step ntt_inverse performs internally. */
void ntt_crt_recombine(const ntt_ctx *ctx, const uint32_t *x1,
                       const uint32_t *x2, int64_t *out);

#ifdef __cplusplus
}
#endif

#endif /* SP_NTT_CRT_H */
