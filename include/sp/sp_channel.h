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

/* ── Hedge-read primitives (§16.3 TS.HEDGE) ────────────────────────────────
 * Single-thread PREFETCH + volatile LOAD through channel-paired addresses.
 * No pauses, no thread races, no TSC rendezvous on the hot path.
 *
 * Callers are responsible for channel placement (sp_alloc_channel_pair).
 * These functions return bitwise-correct data even when a and b are on the
 * SAME channel (CI/DISABLED path): correctness is channel-independent.
 *
 * Prefetch hint: NTA (non-temporal, streaming).  For reused hot sets
 * (§16.5 KSTE upper tier), a T0-hinted variant is Phase F7+ scope.
 */

/* Replica hedge: a and b carry IDENTICAL data (caller's algebraic invariant).
 * Both channels are prefetched; data is loaded from a; channel B is warmed for
 * the NEXT iteration in a stream of reads (load-balancing, not winner-takes-all). */
void sp_hedge_read64_replica(const void *a, const void *b, uint64_t *out);

/* Pair hedge: a and b carry INDEPENDENT data (e.g. q1 / q2 CRT residues).
 * Both channels prefetched simultaneously; latency ≈ max(lat_A, lat_B)
 * rather than lat_A + lat_B. */
void sp_hedge_read_pair64(const void *a, const void *b,
                          uint64_t *out_a, uint64_t *out_b);

/* Block hedge: general n_bytes variant; interleaves PREFETCH with LOAD in
 * 64-byte (cache-line) strides.  Writes n_bytes from a→out_a, b→out_b. */
void sp_hedge_read_block(const void *a, const void *b, size_t n_bytes,
                         uint8_t *out_a, uint8_t *out_b);

/* Spinor hedge: 63-byte replica hedge.  Caller's invariant: a and b carry the
 * same sp_spinor_block_t on independent channels.  Writes a's content to *out;
 * channel B is warmed for the next call in a KV-read stream. */
void sp_hedge_read_spinor(const sp_spinor_block_t *a,
                          const sp_spinor_block_t *b,
                          sp_spinor_block_t *out);

#ifdef __cplusplus
}
#endif
#endif /* SP_CHANNEL_H */
