/* ring3.h — the Ring-3 VSA (vector-symbolic) layer + NIGHTSHIFT consolidation,
 * NATIVE-C, built strictly on the exact-integer negacyclic CRT-NTT (sp/poly_ring.h).
 *
 * This is the deploy-enabler port of the host-Python Ring-3 stack:
 *   shannon-prime-system-engine/tools/ring3/ok_bind.py        (bind/unbind/carrier/idvec/cos)
 *   shannon-prime-system-engine/tools/ring3/g_r3_nightshift.py (the consolidation state machine)
 * — no Python in the resident loop. The negacyclic arithmetic stays native
 * sp_pr_mul (the NTT is NOT reimplemented here); this module is the algebraic
 * VSA layer (carrier/id generators, bind/unbind tiling, superpose, recall/cleanup)
 * plus the SELECT->BIND->SHADOW-GATE->PROMOTE+EVICT->SEAL driver above it.
 *
 * Bit-exactness contract: every bind/unbind/superpose routes through sp_pr_mul,
 * so the M-coefficient vectors are BIT-IDENTICAL (int64, exact) to the Python
 * ok_bind reference once both sides share THE canonical ±1 generator. That
 * generator is splitmix64 (sp_r3_carrier/sp_r3_idvec below); ok_bind.py is
 * updated to the same splitmix64 so Python and C produce bit-identical ±1
 * vectors. The cleanup metric is cosine in double, computed exactly as
 * ok_bind.cos (a.b / (||a|| ||b|| + 1e-12)) so argmax+margin match.
 *
 * Sizes / ownership: a logical dim D is tiled as a DIRECT SUM of negacyclic
 * blocks in {128,256,512} (D=1024 = 512 (+) 512), each bound with the exact
 * engine product. Carriers/ids are ±1 (int8). M (the superposition) holds D
 * int64 coefficients. All array arguments are caller-owned. Contexts are NOT
 * thread-safe (the underlying sp_pr_ctx scratch is mutated per call).
 */
#ifndef SP_RING3_H
#define SP_RING3_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The id-seed offset matching ok_bind.py: idvec uses seed ^ SP_R3_ID_XOR. */
#define SP_R3_ID_XOR 0xABCDEFull

/* CAP = the pre-registered safety cap (R3.2 budget @ D=1024), matching
 * g_r3_nightshift.py CAP=32. */
#define SP_R3_CAP 32

/* ── canonical ±1 generators (splitmix64) ───────────────────────────────────
 * sp_r3_carrier(seed, D, out): out[i] in {-1,+1}, i in [0,D). splitmix64 stream
 *   seeded by `seed`; out[i] = +1 if (z & 1) else -1 (matches g_r3_nightshift.smix).
 * sp_r3_idvec(seed, D, out): identical, but seeded by (seed ^ SP_R3_ID_XOR)
 *   (matches ok_bind.py idvec). Both write D int8 entries. */
void sp_r3_carrier(uint64_t seed, int D, int8_t *out);
void sp_r3_idvec(uint64_t seed, int D, int8_t *out);

/* Is D tileable into {128,256,512} negacyclic blocks? (D in {128,256,512,1024}
 * and the general r>=512 decomposition of ok_bind._blocks). Returns 1/0. */
int sp_r3_dim_ok(int D);

/* ── bind / unbind (exact, native sp_pr_mul) ────────────────────────────────
 * sp_r3_bind(addr,idv,D,M_out): M_out[0:D] = addr (x) id, per negacyclic block.
 * sp_r3_unbind(M,addr,D,out):   out[0:D]  = M (x) addr*, per block (involution
 *   applied to addr before the product). int8 carriers in, int64 coeffs out.
 * Return 0 on success, nonzero if D is not tileable / context init fails. */
int sp_r3_bind(const int8_t *addr, const int8_t *idv, int D, int64_t *M_out);
int sp_r3_unbind(const int64_t *M, const int8_t *addr, int D, int64_t *out);

/* M += (addr (x) id), in place (int64). Returns 0 / nonzero (as sp_r3_bind). */
int sp_r3_superpose(int64_t *M, const int8_t *addr, const int8_t *idv, int D);

/* cosine of two vectors in double: a.b / (||a|| ||b|| + 1e-12). Two overloads
 * for the int64/int8 operand shapes the cleanup needs. */
double sp_r3_cos_i64(const int64_t *a, const int64_t *b, int D);
double sp_r3_cos_i64_i8(const int64_t *a, const int8_t *b, int D);

/* ── the NIGHTSHIFT consolidation state machine ─────────────────────────────
 * An ActiveVector accumulates bound (addr,id) episodes into M under the
 * SHADOW-GATE: a candidate bind is committed only if EVERY episode already in
 * the vector (plus the candidate) still recalls@1 (its id is argmax over the
 * vector's ids) with margin > 0. On gate-fail or count==CAP the vector seals
 * read-only and the episode retries in a fresh vector. */

#define SP_R3_MAX_EPS SP_R3_CAP   /* per-vector episode capacity */

typedef struct {
    int       D;
    int       n;                  /* episodes bound */
    int       sealed;             /* read-only once sealed */
    int64_t  *M;                  /* D coefficients (owned)         */
    uint64_t  seeds[SP_R3_MAX_EPS];
    int8_t   *addr[SP_R3_MAX_EPS];/* D int8 each (owned)            */
    int8_t   *idv [SP_R3_MAX_EPS];/* D int8 each (owned)            */
} sp_r3_vector;

/* Initialize an empty ActiveVector for dimension D (allocates M). Returns 0 on
 * success. Free with sp_r3_vector_free. */
int  sp_r3_vector_init(sp_r3_vector *v, int D);
void sp_r3_vector_free(sp_r3_vector *v);
void sp_r3_vector_seal(sp_r3_vector *v);

/* Whole-set recall@1 gate over the vector's CURRENT (M, episodes): every bound
 * episode's id must be argmax (margin>0) under unbind(M, addr_e). Returns 1/0. */
int sp_r3_recall_ok(const sp_r3_vector *v);

/* Try to bind episode (seed) into the vector under the shadow-gate. Returns:
 *   1  = PROMOTED (committed to M, episode count incremented)
 *   0  = gate-fail or vector full/sealed (caller seals + retries elsewhere)
 * The candidate's carrier/id are derived from `seed` via sp_r3_carrier/idvec. */
int sp_r3_try_bind(sp_r3_vector *v, uint64_t seed);

/* Verdict of one nightshift run. */
typedef struct {
    int n_vectors;
    int sizes[64];                /* episodes per vector (capped at 64 vectors) */
    int consolidated;             /* total episodes consolidated                */
    int sealed;                   /* how many vectors sealed                    */
    int all_recall_ok;            /* every consolidated episode recalls@1       */
    int cap_seal;                 /* D>=1024: non-last vectors sealed at CAP     */
    int gate_seal;                /* D<1024 : max vector size < CAP             */
} sp_r3_nightshift_result;

/* Drive the full consolidation loop over `n` episode seeds at dimension D
 * (CAP = SP_R3_CAP). Mirrors g_r3_nightshift.nightshift(). Returns 0 on
 * success (result populated), nonzero on allocation/dim error. */
int sp_r3_nightshift(int D, const uint64_t *seeds, int n,
                     sp_r3_nightshift_result *out);

#ifdef __cplusplus
}
#endif

#endif /* SP_RING3_H */
