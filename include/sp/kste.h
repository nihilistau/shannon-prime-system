/* sp/kste.h — Knight-Spinor Tree Encoder (KSTE) + Tier-0/Tier-1 dominance.
 * (Phase 1F of the math core.)
 *
 * KSTE deterministically maps a K-vector (K signed 32-bit integer components)
 * into a fixed 64-byte packed tree in the family T_{60,3}: a rooted tree of at
 * most 60 nodes, branching factor up to 3, up to 3 levels (depth 0..2). The
 * concrete shape we freeze for v1 is the *full* fixed-arity tree
 *
 *        root (level 0)                       1 node
 *        |-- 3 children (level 1)             3 nodes
 *            |-- 3 grandchildren each (lvl 2) 9 nodes
 *
 * = 13 nodes total, well inside the 60-node / arity-3 / depth-3 envelope. Using
 * a *fixed* shape (rather than a data-dependent one) is what makes the wire
 * image a flat, padding-free, byte-identical signature on every platform.
 *
 * Each node carries a label: a tuple of 6 quantized order statistics computed
 * from the slice of the K-vector that node covers (the slice min, the 20th,
 * 40th, 60th, 80th percentile sample, and the max — six independent sample
 * points, NOT algebraically derived from one another, so the componentwise
 * order below has genuine incomparable pairs). Labels are stored as little-
 * endian int16 via explicit shifts, never memcpy of a wider type, so the bytes
 * are identical regardless of host endianness or compiler.
 *
 * Dominance is the truncated Friedman-Kruskal homeomorphic-embedding order,
 * realized for Phase 1 as a *componentwise* (product) partial order on the
 * quantized label tuples:
 *
 *     label tuple A <= label tuple B   iff   A[i] <= B[i] for every component i.
 *
 * The componentwise order is reflexive, antisymmetric up to equality, and
 * transitive (each follows directly from the same property of <= on int16).
 * Because components move independently, two tuples can each have some
 * component strictly below the other -> SP_INCOMPARABLE, so all four tri-state
 * outcomes are reachable.
 *
 *   - Tier-0 dominance compares the ROOT labels (the coarse root fingerprint).
 *   - Tier-1 dominance compares the FIRST-LEVEL CHILD SET. Children are sorted
 *     into a canonical order at encode time (lexicographically on their label
 *     tuple), so "same first-level child multiset" is byte-detectable and
 *     Tier-1 is a well-defined componentwise order on the concatenated,
 *     canonically-ordered child tuples.
 *
 * The two tiers are deliberately separate: the Lattice dedup buckets by Tier-0
 * and confirms with Tier-1. Do not collapse to a single tier.
 *
 * Input element type: int32_t. We deliberately do NOT accept float — float
 * encoding cannot be made byte-identical across platforms (subnormals, NaN
 * payloads, -0.0, FMA reordering). Callers holding floats quantize at the door.
 *
 * This header is part of the shared public API root (include/). Include it as
 *   #include "sp/kste.h"
 */
#ifndef SP_KSTE_H
#define SP_KSTE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Frozen wire-format version. v1. Moving, resizing, or reordering ANY field in
 * the 64-byte layout below — including the per-node label tuple width, the
 * child sort key, or the tier byte ranges — REQUIRES bumping this constant.
 * Defined as a macro (not a const int) so _Static_assert and the preprocessor
 * can use it. */
#define SP_KSTE_LAYOUT_VERSION 1

/* Frozen tree-shape parameters (T_{60,3} instance). */
#define SP_KSTE_BRANCHING   3   /* children per node                          */
#define SP_KSTE_DEPTH       3   /* levels 0..2                                */
#define SP_KSTE_LABEL_DIM   6   /* order-statistic components per node label  */

/* The 64-byte packed tree. The struct IS the wire form: it is exactly the 64
 * serialized bytes, no hidden padding, no multi-byte fields that could pick up
 * implementation-defined byte order. Treat .bytes as the canonical signature
 * (hash it, transmit it, diff it across platforms).
 *
 * Byte layout (frozen v1):
 *   [ 0]      version  = SP_KSTE_LAYOUT_VERSION
 *   [ 1]      branching = SP_KSTE_BRANCHING (3)
 *   [ 2]      depth     = SP_KSTE_DEPTH (3)
 *   [ 3]      reserved  = 0
 *   [ 4.. 7]  k (input length) as little-endian uint32
 *   [ 8..19]  Tier-0 ROOT label   : 6 x int16 LE  (12 bytes)
 *   [20..55]  Tier-1 CHILD labels  : 3 children x (6 x int16 LE) = 36 bytes,
 *             children in canonical (lexicographic-on-tuple) order
 *   [56..63]  Tier-2 grandchild digest: 8 x int8, reserved coarse summary
 */
typedef struct sp_kste_tree_s {
    uint8_t bytes[64];
} sp_kste_tree_t;

/* C11 _Static_assert / C++11 static_assert — guarded so the math core stays
 * includable from the C++ backends (CUDA/Vulkan/Hexagon). */
#ifdef __cplusplus
static_assert(sizeof(sp_kste_tree_t) == 64,
              "sp_kste_tree_t must be exactly 64 bytes (frozen wire form)");
#else
_Static_assert(sizeof(sp_kste_tree_t) == 64,
               "sp_kste_tree_t must be exactly 64 bytes (frozen wire form)");
#endif

/* Byte offsets of the tier regions (frozen v1). */
#define SP_KSTE_OFF_VERSION   0
#define SP_KSTE_OFF_BRANCH    1
#define SP_KSTE_OFF_DEPTH     2
#define SP_KSTE_OFF_RESERVED  3
#define SP_KSTE_OFF_K         4    /* uint32 LE                              */
#define SP_KSTE_OFF_ROOT      8    /* Tier-0: 6 x int16 LE = 12 bytes        */
#define SP_KSTE_OFF_CHILDREN  20   /* Tier-1: 3 x 6 x int16 LE = 36 bytes    */
#define SP_KSTE_OFF_GRAND     56   /* Tier-2: 8 x int8 = 8 bytes             */

/* Tri-state result of a partial-order comparison.
 *
 *   sp_kste_tierN(a, b) reports the relation of a TO b:
 *     SP_DOMINATES   : a >= b   (a's label tuple componentwise >= b's, a != b)
 *     SP_DOMINATED   : a <= b   (a's label tuple componentwise <= b's, a != b)
 *     SP_EQUIVALENT  : a == b   (every component equal)
 *     SP_INCOMPARABLE: neither  (some component of a < b, some > b)
 */
typedef enum {
    SP_INCOMPARABLE = 0,
    SP_DOMINATES    = 1,
    SP_DOMINATED    = 2,
    SP_EQUIVALENT   = 3
} sp_dom_t;

/* Encode a K-vector into its 64-byte packed tree.
 *
 * vec : pointer to k signed 32-bit components (may be NULL iff k == 0).
 * k   : number of components, 0 <= k. (k is stored as uint32 LE in the image.)
 * out : destination tree (must be non-NULL); fully overwritten.
 *
 * Deterministic and pure: the same (vec, k) always yields byte-identical out on
 * every platform. */
void sp_kste_encode(const int32_t *vec, int k, sp_kste_tree_t *out);

/* Tier-0 dominance: componentwise partial order on the two ROOT labels. */
sp_dom_t sp_kste_tier0(const sp_kste_tree_t *a, const sp_kste_tree_t *b);

/* Tier-1 dominance: componentwise partial order on the concatenated, canonical-
 * order FIRST-LEVEL CHILD labels. SP_EQUIVALENT here means a and b share the
 * same first-level child multiset (byte-identical child region). */
sp_dom_t sp_kste_tier1(const sp_kste_tree_t *a, const sp_kste_tree_t *b);

#ifdef __cplusplus
}
#endif

#endif /* SP_KSTE_H */
