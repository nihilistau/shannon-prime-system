/* bench_sp_hedge.c — M_TS_HEDGE_PROD: §16.3 production hedge-read bench.
 *
 * Compares P50/P90/P99 of:
 *   baseline : N_ELEM sequential volatile u64 reads from arr_a[] only
 *   hedge    : N_ELEM sp_hedge_read_pair64(&arr_a[i], &arr_b[i]) calls
 *
 * arr_a and arr_b are flat contiguous arrays (no pointer-array indirection).
 * sp_channel_of verifies whether they are on different logical channels.
 *
 * Timing: whole-loop TSC cycles per trial.  2048 trials; P99 = trial[20].
 * No per-element RDTSC, no LFENCE inside the loop, no _mm_pause.
 *
 * Gate M_TS_HEDGE_PROD (P99 ratio = hedge / baseline):
 *   ≤ 0.50 → PASS
 *   ≤ 0.85 → WEAK
 *   > 0.85 → FAIL
 *
 * Expected on Beast Canyon (L3=12 MB): WEAK — 1 MB bench data fits in L3
 * after trial 1; DRAM-channel benefit masked by L3 hits.  See plan §5.
 *
 * In VM/CI (DISABLED): prints REQUIRES_LIVE_MODE and exits 0.
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
#include "sp_channel_internal.h"

#define N_TRIALS  2048
/* 512 KB per side: large enough to show some DRAM effect at P99 tails.
 * 8 MB L3 residency concern documented in plan §5 and closure note. */
#define N_ELEM    65536   /* × 8 bytes = 512 KB per side */

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
static void sort_u64(uint64_t *a, int n) {
    for (int i = 1; i < n; i++) {
        uint64_t key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) { a[j+1] = a[j]; j--; }
        a[j+1] = key;
    }
}

/* ── Core affinity ──────────────────────────────────────────────────────── */
static void pin_to_pcore(void) {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << 0);
#elif defined(__linux__)
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(0, &cs);
    sched_setaffinity(0, sizeof cs, &cs);
#endif
}

/* ── Percentile ─────────────────────────────────────────────────────────── */
static uint64_t pct(uint64_t *sorted, int n, int p) {
    int idx = (int)((long long)p * n / 100LL);
    if (idx >= n) idx = n - 1;
    return sorted[idx];
}

int main(void)
{
    sp_channel_map *m = NULL;
    sp_status rc = sp_channel_map_build(&m);
    if (rc != SP_OK || !m) {
        fprintf(stderr, "bench_sp_hedge: sp_channel_map_build failed (rc=%d)\n", (int)rc);
        return 1;
    }

    if (sp_channel_map_mode(m) != SP_CHANNEL_LIVE) {
        printf("M_TS_HEDGE_PROD: REQUIRES_LIVE_MODE"
               " (DISABLED — VM/container/no huge-pages)\n");
        sp_channel_map_free(m);
        return 0;
    }

    printf("=== §16.3 TS.HEDGE production bench ===\n");
    printf("Trials: %d, N_ELEM: %d (%zu KB per side)\n\n",
           N_TRIALS, N_ELEM, (size_t)N_ELEM * 8 / 1024);

    /* ── Flat array allocation ─────────────────────────────────────────── */
    /* Use huge-page-backed memory so pages are physically mapped */
    size_t hp_size = 2u * 1024u * 1024u;  /* 2 MB huge page */
    void *arena_a = sp_alloc_huge(1, hp_size);
    void *arena_b = sp_alloc_huge(1, hp_size);
    if (!arena_a || !arena_b) {
        /* Fall back to aligned malloc — pages may not be physically contiguous */
        if (arena_a) sp_free_huge(arena_a, 1, hp_size); else arena_a = NULL;
        if (arena_b) sp_free_huge(arena_b, 1, hp_size); else arena_b = NULL;
        arena_a = malloc((size_t)N_ELEM * 8 + 64);
        arena_b = malloc((size_t)N_ELEM * 8 + 64);
        if (!arena_a || !arena_b) {
            fprintf(stderr, "bench_sp_hedge: malloc failed\n");
            free(arena_a); free(arena_b);
            sp_channel_map_free(m);
            return 1;
        }
        printf("NOTE: huge-page alloc failed; using regular malloc (less DRAM signal)\n");
    }

    uint64_t *arr_a = (uint64_t *)arena_a;
    uint64_t *arr_b = (uint64_t *)arena_b;

    /* Pre-fault: bring all pages into physical memory before timing */
    memset(arr_a, 0x42, (size_t)N_ELEM * 8);
    memset(arr_b, 0xBE, (size_t)N_ELEM * 8);

    /* Check channels: are arr_a and arr_b on different logical channels? */
    uint32_t ch_a = sp_channel_of(m, (uintptr_t)arr_a);
    uint32_t ch_b = sp_channel_of(m, (uintptr_t)arr_b);
    printf("arr_a base channel: %u,  arr_b base channel: %u\n", ch_a, ch_b);
    if (ch_a == ch_b) {
        printf("NOTE: arr_a and arr_b start on same channel (%u).\n"
               "      Physical DDR interleaving may still provide some benefit;\n"
               "      bench result reflects real hardware, not forced topology.\n", ch_a);
    } else {
        printf("Channel-diverse placement confirmed (a=%u, b=%u).\n", ch_a, ch_b);
    }
    printf("\n");

    /* ── Bind to P-core ─────────────────────────────────────────────────── */
    pin_to_pcore();

    uint64_t *base_trials  = (uint64_t *)malloc(N_TRIALS * sizeof(uint64_t));
    uint64_t *hedge_trials = (uint64_t *)malloc(N_TRIALS * sizeof(uint64_t));
    if (!base_trials || !hedge_trials) {
        fprintf(stderr, "bench_sp_hedge: trial alloc failed\n");
        free(base_trials); free(hedge_trials);
        sp_channel_map_free(m);
        return 1;
    }

    volatile uint64_t sink_a = 0, sink_b = 0;

    /* ── Baseline: sequential volatile reads from arr_a[] only ───────────── */
    printf("Running baseline (%d trials × %d reads)...\n", N_TRIALS, N_ELEM);
    for (int t = 0; t < N_TRIALS; t++) {
        uint64_t t0 = rdtsc_bench();
        for (int i = 0; i < N_ELEM; i++) {
            sink_a = *(const volatile uint64_t *)(arr_a + i);
        }
        uint64_t t1 = rdtsc_bench();
        base_trials[t] = t1 - t0;
    }
    (void)sink_a;

    /* ── Hedge: sp_hedge_read_pair64 over (arr_a[i], arr_b[i]) ──────────── */
    printf("Running hedge     (%d trials × %d pairs)...\n", N_TRIALS, N_ELEM);
    for (int t = 0; t < N_TRIALS; t++) {
        uint64_t oa = 0, ob = 0;
        uint64_t t0 = rdtsc_bench();
        for (int i = 0; i < N_ELEM; i++) {
            sp_hedge_read_pair64(arr_a + i, arr_b + i,
                                 (uint64_t *)&oa, (uint64_t *)&ob);
        }
        sink_a = oa; sink_b = ob;
        uint64_t t1 = rdtsc_bench();
        hedge_trials[t] = t1 - t0;
    }
    (void)sink_b;

    /* ── Sort + percentiles ─────────────────────────────────────────────── */
    sort_u64(base_trials,  N_TRIALS);
    sort_u64(hedge_trials, N_TRIALS);

    uint64_t bp50 = pct(base_trials,  N_TRIALS, 50);
    uint64_t bp90 = pct(base_trials,  N_TRIALS, 90);
    uint64_t bp99 = pct(base_trials,  N_TRIALS, 99);
    uint64_t hp50 = pct(hedge_trials, N_TRIALS, 50);
    uint64_t hp90 = pct(hedge_trials, N_TRIALS, 90);
    uint64_t hp99 = pct(hedge_trials, N_TRIALS, 99);

    double r50 = bp50 > 0 ? (double)hp50 / (double)bp50 : 0.0;
    double r90 = bp90 > 0 ? (double)hp90 / (double)bp90 : 0.0;
    double r99 = bp99 > 0 ? (double)hp99 / (double)bp99 : 0.0;

    printf("\n  body     | P50 (cyc) | P90 (cyc) | P99 (cyc)\n");
    printf("  ---------|-----------|-----------|----------\n");
    printf("  baseline | %9llu | %9llu | %9llu\n",
           (unsigned long long)bp50,
           (unsigned long long)bp90,
           (unsigned long long)bp99);
    printf("  hedge    | %9llu | %9llu | %9llu\n",
           (unsigned long long)hp50,
           (unsigned long long)hp90,
           (unsigned long long)hp99);
    printf("  ratio    | %9.3f | %9.3f | %9.3f"
           "  (hedge/baseline; <1 = faster)\n",
           r50, r90, r99);

    /* ── Gate ────────────────────────────────────────────────────────────── */
    printf("\n");
    if (r99 <= 0.50) {
        printf("M_TS_HEDGE_PROD: PASS (P99 ratio %.3f ≤ 0.50)\n", r99);
    } else if (r99 <= 0.85) {
        printf("M_TS_HEDGE_PROD: WEAK (P99 ratio %.3f; 0.50 < ratio ≤ 0.85)\n", r99);
        printf("  NOTE: benchmark data likely in L3 (12 MB) after trial 1 —\n"
               "  hedge benefit is DRAM-channel-level; L3 hits mask it.\n"
               "  See session-plan §5 (Path A) and closure note §4.\n");
    } else {
        printf("M_TS_HEDGE_PROD: FAIL (P99 ratio %.3f > 0.85)\n", r99);
        if (ch_a == ch_b) {
            printf("  FINDING: arr_a and arr_b on same channel — no channel\n"
                   "  diversity possible with current malloc placement.\n");
        }
    }

    sp_free_huge(arena_a, 1, hp_size);
    sp_free_huge(arena_b, 1, hp_size);
    free(base_trials); free(hedge_trials);
    sp_channel_map_free(m);
    return 0;
}
