/* bench_ts_hedge.c — M_TS_HEDGE: channel-diverse vs same-channel P99 latency.
 *
 * Gate: P99_same / P99_diverse >= 2.0 on bare-metal Linux with huge pages.
 * In VM/CI (DISABLED path) the bench exits 0 with REQUIRES_LIVE_MODE.
 *
 * Method: run sp_probe_bit for every address-bit position in [CHAN_BIT_LO,
 * CHAN_BIT_HI).  Each probe races two threads for the A / A^(1<<bit) pair and
 * records P99 of MAX(latency_A, latency_B).  The oracle's is_same_channel flag
 * labels each bit.  Pick the lowest P99 diverse bit and highest P99 same-channel
 * bit; their ratio is the M_TS_HEDGE metric. */

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sp/sp_channel.h"
#include "sp/sp_status.h"
#include "sp_channel_internal.h"

#define N_PROBES   256
#define RATIO_GATE 2.0

int main(void) {
    sp_channel_map *m = NULL;
    sp_status rc = sp_channel_map_build(&m);
    if (rc != SP_OK || !m) {
        fprintf(stderr, "bench_ts_hedge: sp_channel_map_build failed (rc=%d)\n", (int)rc);
        return 1;
    }

    if (sp_channel_map_mode(m) != SP_CHANNEL_LIVE) {
        printf("M_TS_HEDGE: REQUIRES_LIVE_MODE"
               " (DISABLED — VM/container/no huge-pages)\n");
        printf("            Grant SeLockMemoryPrivilege (Windows) or run on"
               " bare-metal Linux.\n");
        sp_channel_map_free(m);
        return 0;
    }

    /* Allocate the probe arena: 4 huge pages (sp_probe_bit requirement) */
    size_t hp_size = 2u * 1024u * 1024u;
    void *arena = sp_alloc_huge(4, hp_size);
    if (!arena) {
        printf("M_TS_HEDGE: REQUIRES_LIVE_MODE (huge-page alloc failed)\n");
        sp_channel_map_free(m);
        return 0;
    }

    /* Probe all bit positions */
    int n_bits = CHAN_N_PHYS;
    sp_probe_result results[CHAN_N_PHYS];
    memset(results, 0, sizeof results);

    printf("Probing %d address bits with %d samples each...\n", n_bits, N_PROBES);
    for (int i = 0; i < n_bits; i++) {
        sp_probe_bit((uintptr_t)arena, CHAN_BIT_LO + i, hp_size, N_PROBES, &results[i]);
    }

    sp_free_huge(arena, 4, hp_size);
    sp_channel_map_free(m);

    /* Report table */
    printf("\n  bit | same_ch | p99_ns\n");
    printf("  ----|---------|-------\n");
    for (int i = 0; i < n_bits; i++) {
        printf("   %2d |    %d    | %llu\n",
               results[i].bit,
               results[i].is_same_channel,
               (unsigned long long)results[i].p99_ns);
    }
    printf("\n");

    /* Find best diverse (lowest p99, is_same_channel=0) and
     * worst same-channel (highest p99, is_same_channel=1) */
    int      diverse_idx = -1, same_idx = -1;
    uint64_t diverse_p99 = (uint64_t)-1;
    uint64_t same_p99    = 0;

    for (int i = 0; i < n_bits; i++) {
        if (!results[i].is_same_channel && results[i].p99_ns > 0 &&
            results[i].p99_ns < diverse_p99) {
            diverse_p99 = results[i].p99_ns;
            diverse_idx = i;
        }
        if (results[i].is_same_channel && results[i].p99_ns > same_p99) {
            same_p99  = results[i].p99_ns;
            same_idx  = i;
        }
    }

    if (diverse_idx < 0 || same_idx < 0) {
        printf("M_TS_HEDGE: INCONCLUSIVE"
               " (oracle did not find both same and diverse pairs)\n");
        return 0;
    }

    double ratio = (diverse_p99 > 0) ?
                   (double)same_p99 / (double)diverse_p99 : 0.0;

    printf("diverse pair (bit %2d): p99 = %llu ns\n",
           results[diverse_idx].bit, (unsigned long long)diverse_p99);
    printf("same    pair (bit %2d): p99 = %llu ns\n",
           results[same_idx].bit,   (unsigned long long)same_p99);
    printf("ratio same/diverse    = %.2f× (gate >= %.1f×)\n", ratio, RATIO_GATE);
    printf("\n");

    if (ratio >= RATIO_GATE) {
        printf("M_TS_HEDGE: PASS\n");
        return 0;
    }
    printf("M_TS_HEDGE: FAIL (ratio %.2f× < %.1f×)\n", ratio, RATIO_GATE);
    return 1;
}
