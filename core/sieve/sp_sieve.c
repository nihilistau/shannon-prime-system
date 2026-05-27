/* sp_sieve.c — Friedman Sieve: Pareto-frontier maintenance (Phase 5 PoUW).
 *
 * Pure C99 scalar.  No platform-specific headers.  The daemon layer wraps
 * sp_sieve_event_t into ed25519-signed receipts (Rust ed25519-dalek).
 *
 * Dominance rule (both tiers must agree to classify):
 *   - c "absorbed" by f  : tier0(f,c) and tier1(f,c) are each ≥ (DOMINATES
 *                          or EQUIVALENT) — c adds nothing new; discard.
 *   - c "folds" f        : tier0(c,f) and tier1(c,f) are each ≥, with at
 *                          least one being DOMINATES — f is replaced; emit event.
 *   - otherwise          : INCOMPARABLE — c extends the frontier, no event.
 *
 * The frontier is maintained as a valid Pareto-optimal set (no element
 * dominates another), so the absorbed and fold paths are mutually exclusive
 * across a single call to sp_sieve_evaluate.
 */
#include "sp/sp_sieve.h"
#include "sp/kste.h"
#include "sp/sp_hash.h"
#include "sp/sp_status.h"

#include <stdint.h>
#include <string.h>

/* c strictly dominates f if c >= f at both tiers with at least one strict. */
static int strictly_dominates(const sp_kste_tree_t *c, const sp_kste_tree_t *f) {
    sp_dom_t d0 = sp_kste_tier0(c, f);
    sp_dom_t d1 = sp_kste_tier1(c, f);
    int gte0 = (d0 == SP_DOMINATES || d0 == SP_EQUIVALENT);
    int gte1 = (d1 == SP_DOMINATES || d1 == SP_EQUIVALENT);
    return gte0 && gte1 && (d0 == SP_DOMINATES || d1 == SP_DOMINATES);
}

/* c is absorbed by f if f >= c at both tiers (f weakly dominates c). */
static int absorbed_by(const sp_kste_tree_t *c, const sp_kste_tree_t *f) {
    sp_dom_t d0 = sp_kste_tier0(f, c);
    sp_dom_t d1 = sp_kste_tier1(f, c);
    return (d0 == SP_DOMINATES || d0 == SP_EQUIVALENT) &&
           (d1 == SP_DOMINATES || d1 == SP_EQUIVALENT);
}

sp_status sp_sieve_evaluate(
    const sp_kste_tree_t *candidates,
    size_t                n,
    sp_kste_tree_t       *frontier,
    size_t               *frontier_n,
    size_t                frontier_cap,
    sp_sieve_event_t     *events_out,
    size_t               *n_events_out)
{
    if (!candidates || !frontier || !frontier_n || !events_out || !n_events_out)
        return SP_EBADARG;

    *n_events_out = 0;

    for (size_t ci = 0; ci < n; ci++) {
        const sp_kste_tree_t *c = &candidates[ci];

        /* Pass 1: check whether any frontier member weakly dominates c. */
        int discard = 0;
        for (size_t fi = 0; fi < *frontier_n; fi++) {
            if (absorbed_by(c, &frontier[fi])) {
                discard = 1;
                break;
            }
        }
        if (discard)
            continue;

        /* Pass 2: in-place compact — keep only members c does not dominate. */
        size_t old_n = *frontier_n;
        size_t new_n = 0;
        for (size_t fi = 0; fi < *frontier_n; fi++) {
            if (!strictly_dominates(c, &frontier[fi]))
                frontier[new_n++] = frontier[fi];
        }
        *frontier_n = new_n;

        /* Sieve fold: at least one frontier member was dominated. */
        if (old_n > new_n) {
            sp_sieve_event_t *ev = &events_out[*n_events_out];
            ev->sig   = *c;
            ev->round = (uint32_t)(*n_events_out);
            sp_sha256(c->bytes, sizeof c->bytes, ev->seq_hash);
            (*n_events_out)++;
        }

        /* Add c to the frontier. */
        if (*frontier_n >= frontier_cap)
            return SP_ESIEVE_FULL;
        frontier[(*frontier_n)++] = *c;
    }

    return SP_OK;
}
