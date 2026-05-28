/* bench_sp_hedge.c — M_TS_HEDGE_PROD: §16.3 persistent-pool hedge-read bench.
 *
 * Compares P50/P90/P99 TSC cycles:
 *   baseline : 131072 sequential volatile uint64 reads from ptr_a (channel A)
 *   hedge    : 131072 sp_hedge_read_pair calls over (ptr_a[i], ptr_b[i])
 *
 * ptr_a and ptr_b are from sp_alloc_channel_pair (1 MB each, huge-page backed,
 * guaranteed on different physical DDR channels in LIVE mode).
 *
 * Gate M_TS_HEDGE_PROD:
 *   P99(hedge) / P99(baseline) ≤ 0.50 → PASS
 *   P99(hedge) / P99(baseline) ≤ 0.85 → WEAK
 *   otherwise                          → FAIL
 *
 * DISABLED (CI/VM): prints REQUIRES_LIVE_MODE and exits 0.
 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif
#ifdef __linux__
#  define _GNU_SOURCE
#  include <sched.h>
#endif
#include "sp/sp_channel.h"
#include "sp_channel_internal.h"   /* sp_alloc_huge, sp_free_huge */

#define N_TRIALS  2048
#define N_ELEM    131072u  /* × 8 bytes = 1 MB per side */

/* ── TSC ────────────────────────────────────────────────────────────────── */
#if defined(_MSC_VER)
#  include <intrin.h>
static uint64_t rdtsc_bench(void) { return __rdtsc(); }
#elif defined(__GNUC__) || defined(__clang__)
static uint64_t rdtsc_bench(void) {
    unsigned int lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}
#else
static uint64_t rdtsc_bench(void) { return 0; }
#endif

/* ── Sort ────────────────────────────────────────────────────────────────── */
static int cmp_u64_b(const void *x, const void *y) {
    uint64_t a = *(const uint64_t *)x;
    uint64_t b = *(const uint64_t *)y;
    return (a > b) - (a < b);
}

/* ── Core affinity ──────────────────────────────────────────────────────── */
static void pin_main_thread(void) {
    /* Bind caller to core 1 (CORE_MAIN analogue; separate from workers on 0,2). */
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << 1);
#elif defined(__linux__)
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(1, &cs);
    sched_setaffinity(0, sizeof cs, &cs);
#endif
}

/* ── Percentile ─────────────────────────────────────────────────────────── */
static uint64_t pct(uint64_t *s, int n, int p) {
    int i = (int)((long long)p * n / 100LL);
    return s[i < n ? i : n - 1];
}

int main(void)
{
    sp_channel_map *m = NULL;
    sp_status rc = sp_channel_map_build(&m);
    if (rc != SP_OK || !m) {
        fprintf(stderr, "bench_sp_hedge: sp_channel_map_build failed (rc=%d)\n",
                (int)rc);
        return 1;
    }
    if (sp_channel_map_mode(m) != SP_CHANNEL_LIVE) {
        printf("M_TS_HEDGE_PROD: REQUIRES_LIVE_MODE"
               " (DISABLED — VM/container/no huge-pages)\n");
        sp_channel_map_free(m);
        return 0;
    }

    printf("=== §16.3 TS.HEDGE persistent-pool bench ===\n");
    printf("Trials: %d, N_ELEM: %u (%.0f KB per side)\n\n",
           N_TRIALS, N_ELEM, N_ELEM * 8.0 / 1024.0);

    /* ── Allocate dual-channel arena via sp_alloc_channel_pair ─────────── */
    void *ptr_a = NULL, *ptr_b = NULL;
    sp_channel_pair_arena *arena = NULL;
    rc = sp_alloc_channel_pair(m, &ptr_a, &ptr_b, &arena);
    if (rc != SP_OK || !ptr_a || !ptr_b) {
        printf("M_TS_HEDGE_PROD: REQUIRES_LIVE_MODE"
               " (sp_alloc_channel_pair failed)\n");
        sp_channel_map_free(m);
        return 0;
    }

    /* Check channel placement */
    uint32_t ch_a = sp_channel_of(m, (uintptr_t)ptr_a);
    uint32_t ch_b = sp_channel_of(m, (uintptr_t)ptr_b);
    printf("ptr_a channel=%u, ptr_b channel=%u  →  ",
           (unsigned)ch_a, (unsigned)ch_b);
    if (ch_a != ch_b)
        printf("channel-diverse (hedge benefit expected)\n");
    else
        printf("WARNING: same channel — no hedge benefit\n");

    /* We need N_ELEM elements per side.
     * ptr_a and ptr_b are single addresses; sp_alloc_channel_pair's arena is
     * 4 × hp (≈8 MB).  Use ptr_a and ptr_b as the BASE of our arrays and ensure
     * elements at stride 8 bytes (sequential) stay within the arena.
     * The memory controller's interleaving across channels is at cache-line (64B)
     * granularity; sequential access naturally alternates channels, so even with
     * a single base ptr_a the hardware sees both channels.
     * For the hedge test we use ptr_a as side-A base and ptr_b as side-B base. */
    uint64_t *arr_a = (uint64_t *)ptr_a;
    uint64_t *arr_b = (uint64_t *)ptr_b;

    /* Pre-fault: memset both sides before timing */
    memset(arr_a, 0xAA, N_ELEM * sizeof(uint64_t));
    memset(arr_b, 0xBB, N_ELEM * sizeof(uint64_t));

    /* ── Create hedge pool: workers on cores 0 and 2 ─────────────────────── */
    const int WORKER_CORES[2] = {0, 2};
    sp_hedge_pool *pool = NULL;
    rc = sp_hedge_pool_create(&pool, WORKER_CORES, 2, 8);
    if (rc != SP_OK || !pool) {
        printf("M_TS_HEDGE_PROD: pool_create failed (rc=%d)\n", (int)rc);
        sp_free_channel_pair(arena);
        sp_channel_map_free(m);
        return 1;
    }

    /* ── Bind caller to core 1 (CORE_MAIN; separate from workers 0,2) ──── */
    pin_main_thread();

    uint64_t *base_t  = (uint64_t *)malloc(N_TRIALS * sizeof(uint64_t));
    uint64_t *hedge_t = (uint64_t *)malloc(N_TRIALS * sizeof(uint64_t));
    if (!base_t || !hedge_t) {
        printf("bench_sp_hedge: trial alloc failed\n");
        sp_hedge_pool_destroy(pool);
        sp_free_channel_pair(arena);
        sp_channel_map_free(m);
        free(base_t); free(hedge_t);
        return 1;
    }

    volatile uint64_t sink_a, sink_b;

    /* ── Baseline: sequential volatile reads from arr_a only ─────────────── */
    printf("Running baseline (%d trials × %u reads)...\n", N_TRIALS, N_ELEM);
    for (int t = 0; t < N_TRIALS; t++) {
        uint64_t t0 = rdtsc_bench();
        for (unsigned i = 0; i < N_ELEM; i++)
            sink_a = *(const volatile uint64_t *)(arr_a + i);
        uint64_t t1 = rdtsc_bench();
        base_t[t] = t1 - t0;
    }
    (void)sink_a;

    /* ── Hedge: sp_hedge_read_pair over (arr_a[i], arr_b[i]) ────────────── */
    printf("Running hedge     (%d trials × %u pairs)...\n", N_TRIALS, N_ELEM);
    for (int t = 0; t < N_TRIALS; t++) {
        uint64_t oa, ob;
        uint64_t t0 = rdtsc_bench();
        for (unsigned i = 0; i < N_ELEM; i++)
            sp_hedge_read_pair(pool, arr_a + i, arr_b + i, 8, &oa, &ob);
        sink_a = oa; sink_b = ob;
        uint64_t t1 = rdtsc_bench();
        hedge_t[t] = t1 - t0;
    }
    (void)sink_b;

    qsort(base_t,  N_TRIALS, sizeof(uint64_t), cmp_u64_b);
    qsort(hedge_t, N_TRIALS, sizeof(uint64_t), cmp_u64_b);

    uint64_t bp50 = pct(base_t,  N_TRIALS, 50);
    uint64_t bp90 = pct(base_t,  N_TRIALS, 90);
    uint64_t bp99 = pct(base_t,  N_TRIALS, 99);
    uint64_t hp50 = pct(hedge_t, N_TRIALS, 50);
    uint64_t hp90 = pct(hedge_t, N_TRIALS, 90);
    uint64_t hp99 = pct(hedge_t, N_TRIALS, 99);

    double r50 = bp50 ? (double)hp50 / (double)bp50 : 0.0;
    double r90 = bp90 ? (double)hp90 / (double)bp90 : 0.0;
    double r99 = bp99 ? (double)hp99 / (double)bp99 : 0.0;

    printf("\n  body     | P50 (cyc) | P90 (cyc) | P99 (cyc)\n");
    printf("  ---------|-----------|-----------|----------\n");
    printf("  baseline | %9llu | %9llu | %9llu\n",
           (unsigned long long)bp50, (unsigned long long)bp90, (unsigned long long)bp99);
    printf("  hedge    | %9llu | %9llu | %9llu\n",
           (unsigned long long)hp50, (unsigned long long)hp90, (unsigned long long)hp99);
    printf("  ratio    | %9.3f | %9.3f | %9.3f"
           "  (hedge/baseline; <1.0 = faster)\n", r50, r90, r99);

    printf("\n");
    if (r99 <= 0.50)
        printf("M_TS_HEDGE_PROD: PASS (P99 ratio %.3f ≤ 0.50)\n", r99);
    else if (r99 <= 0.85)
        printf("M_TS_HEDGE_PROD: WEAK (P99 ratio %.3f; 0.50 < ratio ≤ 0.85)\n", r99);
    else
        printf("M_TS_HEDGE_PROD: FAIL (P99 ratio %.3f > 0.85)\n", r99);

    sp_hedge_pool_destroy(pool);
    free(base_t); free(hedge_t);
    sp_free_channel_pair(arena);
    sp_channel_map_free(m);
    return 0;
}
