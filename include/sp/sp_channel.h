/* sp_channel.h — GF(2) memory-channel oracle (Phase TS.MAP §16.1).
 *
 * Recovers the host memory controller's undocumented channel-select hash as a
 * k×N binary matrix M via empirical hedge-read timing (Laurie's TailSlayer
 * methodology, re-derived in math-core idioms).
 *
 * SAFETY CONTRACT:
 *   TailSlayer is a PERF OVERLAY, never a correctness dependency.
 *   sp_channel_map_build returns SP_OK with mode=SP_CHANNEL_DISABLED in any
 *   VM / container / huge-page-denied environment.  sp_channel_of returns
 *   SP_CHANNEL_UNSPECIFIED on a DISABLED map.  All lattice math continues with
 *   SP_CHANNEL_UNSPECIFIED; only the hedge allocator's placement decisions are
 *   gated on this module.
 */
#ifndef SP_CHANNEL_H
#define SP_CHANNEL_H

#include <stdint.h>
#include <stddef.h>
#include "sp/sp_status.h"
#include "sp/spinor_block.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Returned by sp_channel_of when the map is DISABLED or on any error. */
#define SP_CHANNEL_UNSPECIFIED 0xFFFFFFFFu

typedef enum {
    SP_CHANNEL_DISABLED = 0,  /* VM/container/huge-page-denied: oracle inactive */
    SP_CHANNEL_LIVE     = 1,  /* bare-metal: M recovered, sp_channel_of is valid */
} sp_channel_mode;

/* Opaque handle. sp_channel_map_build / sp_channel_map_load_cached allocate;
 * the caller destroys with sp_channel_map_free. */
typedef struct sp_channel_map sp_channel_map;

/* ── Build ──────────────────────────────────────────────────────────────────
 * Probe the memory controller and construct M.  Always returns SP_OK on
 * success; the mode field of *out distinguishes LIVE from DISABLED.  Returns
 * SP_ENOMEM only when the map struct itself cannot be allocated (out-of-memory
 * on the calling process).  All VM / huge-page-denied paths return SP_OK with
 * mode=DISABLED — never a non-OK status.
 *
 * Set env SP_FORCE_VIRT_DETECTION=1 to force DISABLED (for CI / test use). */
sp_status sp_channel_map_build(sp_channel_map **out);

/* ── Query ──────────────────────────────────────────────────────────────────
 * Compute M · addr_bits mod 2 → channel index in [0, 2^k).
 * Returns SP_CHANNEL_UNSPECIFIED if m is NULL or mode == DISABLED. */
uint32_t sp_channel_of(const sp_channel_map *m, uintptr_t addr);

/* ── Cache ──────────────────────────────────────────────────────────────────
 * Cache paths:
 *   POSIX:   $HOME/.cache/shannon-prime/channel_map_<fingerprint>.bin
 *   Windows: %LOCALAPPDATA%\shannon-prime\channel_map_<fingerprint>.bin
 *
 * sp_channel_map_load_cached:
 *   SP_OK + *out != NULL  → cache hit, valid map loaded
 *   SP_OK + *out == NULL  → cache miss (file absent — not an error)
 *   SP_EIO                → file present but corrupt or unreadable
 *
 * sp_channel_map_save_cached:
 *   SP_OK   → written successfully
 *   SP_EIO  → write failed (non-fatal; caller may continue with live map)
 */
sp_status sp_channel_map_load_cached(const char *host_fingerprint,
                                     sp_channel_map **out);
sp_status sp_channel_map_save_cached(const sp_channel_map *m,
                                     const char *host_fingerprint);

/* ── Introspection ──────────────────────────────────────────────────────────
 * sp_channel_map_mode: SP_CHANNEL_DISABLED or SP_CHANNEL_LIVE.
 * sp_channel_map_dims: k = channel-select bits (rows of M); n = address bits
 *   probed (columns of M).  Returns SP_EBADARG on NULL args.
 */
sp_channel_mode sp_channel_map_mode(const sp_channel_map *m);
sp_status       sp_channel_map_dims(const sp_channel_map *m,
                                    uint32_t *k_out, uint32_t *n_out);

/* ── Host fingerprint ───────────────────────────────────────────────────────
 * Derives a short (8-hex-char, null-terminated) ASCII fingerprint from CPU
 * model + installed memory size.  buf must be >= 16 bytes.  Used as the cache
 * key.  Returns SP_EBADARG on bad args, SP_OK otherwise. */
sp_status sp_channel_host_fingerprint(char *buf, size_t buf_len);

/* Destroy. Safe to call with NULL. */
void sp_channel_map_free(sp_channel_map *m);

/* ── Channel-pair allocator ─────────────────────────────────────────────────
 * Allocates two cache-line-aligned pointers guaranteed to reside on distinct
 * memory channels (LIVE mode) or falls back to malloc with a warning (DISABLED
 * mode, VM/CI environments).
 *
 * LIVE path: allocates a 4 × huge-page arena (to keep virtual-bit arithmetic
 *   valid) and scans it with sp_channel_of() to find two addresses on distinct
 *   channels.  *arena_out holds the arena handle for sp_free_channel_pair().
 *
 * DISABLED/fallback path: logs SP_WARN and returns two malloc'd pointers with
 *   channels = SP_CHANNEL_UNSPECIFIED.  sp_free_channel_pair() still works.
 *
 * Returns SP_ENOMEM on allocation failure.  Never returns a non-OK status for
 * the DISABLED path — same safety contract as sp_channel_map_build. */
typedef struct sp_channel_pair_arena sp_channel_pair_arena;

sp_status sp_alloc_channel_pair(const sp_channel_map *m,
                                void **ptr_a_out,
                                void **ptr_b_out,
                                sp_channel_pair_arena **arena_out);

/* Free the arena returned by sp_alloc_channel_pair.  Safe to call with NULL. */
void sp_free_channel_pair(sp_channel_pair_arena *arena);

/* ── §16.3 TS.HEDGE: persistent-pool hedge-read primitives ─────────────────
 * Pattern: hedged_reader.hpp:124,138-152 (Laurie TailSlayer).
 * N worker threads, each pinned to a dedicated core at startup.  Hot path =
 * atomic publication slot + worker memcpy + atomic completion count.
 * No CLFLUSH, no TSC rendezvous, no LFENCE, no per-read thread spawn.
 */

/* Opaque pool handle.  Created once at module startup; destroyed at teardown. */
typedef struct sp_hedge_pool sp_hedge_pool;

/* Create persistent pool.  Spawns n_channels threads; pins each to core_ids[i].
 * max_bytes: maximum n_bytes accepted by sp_hedge_read_pair (SP_EBADARG if exceeded).
 * n_channels=1: no thread spawned; sp_hedge_read_pair is a direct memcpy of side a.
 * Returns SP_ENOMEM on allocation failure, SP_EBADARG on NULL/bad args. */
sp_status sp_hedge_pool_create(sp_hedge_pool **out,
                               const int *core_ids,
                               size_t n_channels,
                               size_t max_bytes);

void sp_hedge_pool_destroy(sp_hedge_pool *pool);

/* Hedged pair read.  Publishes (a, b) to worker slots via atomic store-release;
 * workers memcpy into local buffers on their pinned cores; caller spins on
 * completion count (acquire) then copies results to out_a, out_b.
 * NOT reentrant.  n_bytes must be <= max_bytes passed to pool_create. */
sp_status sp_hedge_read_pair(sp_hedge_pool *pool,
                             const void *a, const void *b,
                             size_t n_bytes,
                             void *out_a, void *out_b);

/* Spinor wrapper: 63-byte blocks, both sides.  out_a and out_b are independent. */
sp_status sp_hedge_read_spinor(sp_hedge_pool *pool,
                               const sp_spinor_block_t *a,
                               const sp_spinor_block_t *b,
                               sp_spinor_block_t *out_a,
                               sp_spinor_block_t *out_b);

/* §16.3.1 batch hedge: workers memcpy n_bytes directly into caller-provided
 * dst_a / dst_b in parallel.  Amortizes pool overhead (~300 cyc/call) across
 * the whole transfer, unlike sp_hedge_read_pair which pays one round-trip
 * per call.  n_bytes is unbounded (caller manages buffer sizes).
 * N=1: direct memcpy(dst_a, src_a, n_bytes); src_b/dst_b ignored.
 * NOT reentrant (single pool serves one bulk read at a time). */
sp_status sp_hedge_read_bulk(sp_hedge_pool *pool,
                             const void *src_a, void *dst_a,
                             const void *src_b, void *dst_b,
                             size_t n_bytes);

#ifdef __cplusplus
}
#endif
#endif /* SP_CHANNEL_H */
