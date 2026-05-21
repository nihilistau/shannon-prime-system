/* kste_dominance.c — Tier-0 / Tier-1 dominance for KSTE (Phase 1F).
 *
 * Both tiers are the SAME componentwise (product) partial order, applied to a
 * different byte region of the packed tree:
 *   - Tier-0 over the 6-component root label (12 bytes at SP_KSTE_OFF_ROOT),
 *   - Tier-1 over the 18 components of the 3 canonical child labels
 *     (36 bytes at SP_KSTE_OFF_CHILDREN).
 *
 * Componentwise order on integer tuples is a genuine partial order:
 *   reflexive    — a[i] <= a[i] for all i  -> EQUIVALENT,
 *   antisymmetric— a<=b and b<=a forces a[i]==b[i] for all i -> EQUIVALENT,
 *   transitive   — a<=b<=c gives a[i]<=c[i] by transitivity of <= on int.
 * Because components vary independently, mixed pairs (some a[i]<b[i], some
 * a[i]>b[i]) are INCOMPARABLE, so all four tri-states are reachable.
 *
 * The labels are read back from the wire image with explicit little-endian
 * shifts — the mirror of the encoder — so the comparison is byte-order
 * independent and matches across platforms.
 */
#include "sp/kste.h"

#include <stdint.h>

/* read a little-endian int16 from two bytes */
static int16_t get_i16_le(const uint8_t *p) {
    uint16_t u = (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
    return (int16_t)u;   /* well-defined: int16_t holds the 2's-complement bits */
}

/* Componentwise comparison over `dim` int16 components starting at byte offset
 * `off` in each tree image. */
static sp_dom_t compare_region(const sp_kste_tree_t *a, const sp_kste_tree_t *b,
                               int off, int dim) {
    int any_lt = 0;   /* some a-component strictly < b-component */
    int any_gt = 0;   /* some a-component strictly > b-component */

    for (int i = 0; i < dim; i++) {
        int16_t av = get_i16_le(a->bytes + off + 2 * i);
        int16_t bv = get_i16_le(b->bytes + off + 2 * i);
        if (av < bv) any_lt = 1;
        else if (av > bv) any_gt = 1;
    }

    if (any_lt && any_gt) return SP_INCOMPARABLE;
    if (any_gt)           return SP_DOMINATES;   /* a >= b, a != b */
    if (any_lt)           return SP_DOMINATED;   /* a <= b, a != b */
    return SP_EQUIVALENT;                        /* all components equal */
}

sp_dom_t sp_kste_tier0(const sp_kste_tree_t *a, const sp_kste_tree_t *b) {
    return compare_region(a, b, SP_KSTE_OFF_ROOT, SP_KSTE_LABEL_DIM);
}

sp_dom_t sp_kste_tier1(const sp_kste_tree_t *a, const sp_kste_tree_t *b) {
    return compare_region(a, b, SP_KSTE_OFF_CHILDREN,
                          SP_KSTE_BRANCHING * SP_KSTE_LABEL_DIM);
}
