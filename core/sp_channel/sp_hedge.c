/* sp_hedge.c — §16.3 TS.HEDGE production hedge-read primitives.
 *
 * Single-thread PREFETCH + volatile LOAD through channel-paired addresses.
 * No pauses, no thread races, no TSC rendezvous.  The oracle apparatus
 * (two-thread + spin barrier + LFENCE) lives in sp_channel_probe.c only.
 *
 * "TailSlayer's runtime path (sp_alloc_channel_pair) loads the cached bin
 *  file and does O(1) channel selection with zero probe or pause overhead."
 *  — bench_ts_hedge.c header (canonical oracle-vs-production disclaimer).
 */

#include "sp/sp_channel.h"
#include "sp/spinor_block.h"
#include <stdint.h>
#include <stddef.h>

/* ── Prefetch macro ──────────────────────────────────────────────────────────
 * NTA hint: non-temporal, streaming — correct for Q8/Q4 arena and Spinor KV
 * reads (each byte accessed once per kernel call, no intra-call reuse).
 * §16.5 KSTE upper-tier (reused hot set) may warrant T0; defer to that phase.
 */
#if defined(_MSC_VER)
#  include <intrin.h>
#  define SP_PREFETCH(p) _mm_prefetch((const char *)(p), _MM_HINT_NTA)
#elif defined(__GNUC__) || defined(__clang__)
#  define SP_PREFETCH(p) __builtin_prefetch((const void *)(p), 0, 0)
#else
#  define SP_PREFETCH(p) ((void)(p))
#endif

/* ── sp_hedge_read64_replica ──────────────────────────────────────────────── */

void sp_hedge_read64_replica(const void *a, const void *b, uint64_t *out)
{
    SP_PREFETCH(a);
    SP_PREFETCH(b);
    /* Load from a.  Volatile load of b warms channel B for the NEXT call in a
     * stream of hedge reads (stream load-balancing, not winner-takes-all). */
    *out = *(const volatile uint64_t *)a;
    (void)*(const volatile uint64_t *)b;
}

/* ── sp_hedge_read_pair64 ────────────────────────────────────────────────── */

void sp_hedge_read_pair64(const void *a, const void *b,
                          uint64_t *out_a, uint64_t *out_b)
{
    SP_PREFETCH(a);
    SP_PREFETCH(b);
    /* Both fetches are in-flight simultaneously; latency ≈ max(lat_A, lat_B)
     * instead of the serial lat_A + lat_B.  Required for CRT residue pairs. */
    *out_a = *(const volatile uint64_t *)a;
    *out_b = *(const volatile uint64_t *)b;
}

/* ── sp_hedge_read_block ────────────────────────────────────────────────── */

void sp_hedge_read_block(const void *a, const void *b, size_t n_bytes,
                         uint8_t *out_a, uint8_t *out_b)
{
    const volatile uint8_t *va = (const volatile uint8_t *)a;
    const volatile uint8_t *vb = (const volatile uint8_t *)b;

    /* Prefetch the first cache line of both channels before entering the loop. */
    SP_PREFETCH(va);
    SP_PREFETCH(vb);

    for (size_t i = 0; i < n_bytes; i++) {
        /* At every cache-line boundary, pre-fetch the next line of both
         * channels so they travel through their respective DDR channels in
         * parallel with the current-line stores. */
        if (i > 0 && (i & 63u) == 0u && i + 64u < n_bytes) {
            SP_PREFETCH(va + i + 64u);
            SP_PREFETCH(vb + i + 64u);
        }
        out_a[i] = va[i];
        out_b[i] = vb[i];
    }
}

/* ── sp_hedge_read_spinor ────────────────────────────────────────────────── */

void sp_hedge_read_spinor(const sp_spinor_block_t *a,
                          const sp_spinor_block_t *b,
                          sp_spinor_block_t *out)
{
    const volatile uint8_t *va = (const volatile uint8_t *)a;
    const volatile uint8_t *vb = (const volatile uint8_t *)b;

    SP_PREFETCH(va);
    SP_PREFETCH(vb);

    /* Load all 63 bytes from channel A into output. */
    uint8_t *po = (uint8_t *)out;
    for (int i = 0; i < 63; i++) {
        po[i] = va[i];
    }
    /* Warm channel B: one volatile load commits the B prefetch for the next
     * Spinor block in the KV-read stream (stream load-balancing). */
    (void)vb[0];
}
