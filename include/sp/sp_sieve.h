/* sp/sp_sieve.h — Friedman Sieve: PoUW Phase-5 Pareto-frontier maintenance.
 *
 * Maintains a Pareto-optimal frontier of KSTE signatures under the combined
 * Tier-0 + Tier-1 dominance partial order.  Each time a candidate strictly
 * dominates a frontier member a sieve-fold event is emitted.  The daemon
 * assembles sieve-fold events into cryptographically signed receipts using
 * its ed25519 node keypair.
 *
 * Receipt wire format (frozen v1, 152 bytes):
 *   [  0..  7]  magic        "SPRCPT01" (8 bytes, no NUL)
 *   [  8.. 71]  kste_sig     64-byte KSTE tree  (sp_kste_tree_t.bytes)
 *   [ 72..103]  seq_hash     SHA-256(kste_sig.bytes) — receipt binding
 *   [104..135]  pubkey       ed25519 public key (32 bytes, filled by daemon)
 *   [136..143]  round        uint64 LE, sieve fold counter  (filled by daemon)
 *   [144..151]  minted_at_ns uint64 LE, CLOCK_REALTIME nanoseconds (daemon)
 *
 * The math-core provides kste_sig + seq_hash via sp_sieve_event_t.  The
 * daemon fills pubkey, round, and minted_at_ns when serialising the receipt.
 */
#ifndef SP_SIEVE_H
#define SP_SIEVE_H

#include <stddef.h>
#include <stdint.h>
#include "sp/kste.h"
#include "sp/sp_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Receipt byte offsets (frozen v1). */
#define SP_RECEIPT_MAGIC         "SPRCPT01"
#define SP_RECEIPT_BYTES          152U
#define SP_RECEIPT_OFF_MAGIC        0U
#define SP_RECEIPT_OFF_SIG          8U
#define SP_RECEIPT_OFF_SEQ_HASH    72U
#define SP_RECEIPT_OFF_PUBKEY     104U
#define SP_RECEIPT_OFF_ROUND      136U
#define SP_RECEIPT_OFF_MINTED     144U

/* Sieve-fold event: emitted when a candidate strictly dominates a frontier
 * member under the combined Tier-0 + Tier-1 partial order.
 *
 * "Strictly dominates" means: Tier-0(c,f) and Tier-1(c,f) are each
 * SP_DOMINATES or SP_EQUIVALENT, with at least one being SP_DOMINATES.
 *
 * sig:      the dominating KSTE signature.
 * seq_hash: SHA-256(sig.bytes); carried verbatim into the receipt wire image.
 * round:    0-based fold index within the current sp_sieve_evaluate call.
 *           The daemon adds its own monotone global fold counter on top.
 */
typedef struct {
    sp_kste_tree_t  sig;
    uint8_t         seq_hash[32];
    uint32_t        round;
} sp_sieve_event_t;

/* Evaluate a batch of KSTE candidates against a Pareto frontier.
 *
 * For each candidate:
 *   - Dominated by (or equivalent to) any frontier member  → discard silently.
 *   - Strictly dominates one or more frontier members       → replace them,
 *     emit a sieve-fold event (sig + seq_hash + round).
 *   - Incomparable to all frontier members                 → add to frontier,
 *     no event.
 *
 * Parameters:
 *   candidates    pointer to n KSTE trees (processed in order)
 *   n             count of candidates
 *   frontier      caller-managed Pareto-optimal set; updated in place
 *   frontier_n    IN/OUT: current frontier size
 *   frontier_cap  maximum allowed frontier size; returns SP_ESIEVE_FULL if an
 *                 incomparable candidate would exceed it
 *   events_out    pre-allocated event array; must hold at least n entries
 *   n_events_out  OUT: count of sieve-fold events emitted this call
 *
 * Returns SP_OK, SP_ESIEVE_FULL, or SP_EBADARG (NULL pointer in required arg).
 */
sp_status sp_sieve_evaluate(
    const sp_kste_tree_t *candidates,
    size_t                n,
    sp_kste_tree_t       *frontier,
    size_t               *frontier_n,
    size_t                frontier_cap,
    sp_sieve_event_t     *events_out,
    size_t               *n_events_out);

#ifdef __cplusplus
}
#endif
#endif /* SP_SIEVE_H */
