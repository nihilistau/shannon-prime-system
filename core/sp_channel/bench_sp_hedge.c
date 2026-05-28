/* bench_sp_hedge.c — M_TS_HEDGE_PROD: §16.3.1 batch hedge throughput bench.
 *
 * Compares P50/P90/P99 TSC cycles per trial; both bodies do the same total
 * memory work (256 MB read + 256 MB write):
 *   baseline : single-threaded SERIAL memcpy of src_a→dst_a AND src_b→dst_b
 *              (256 MB total through one thread)
 *   hedge    : one sp_hedge_read_bulk(src_a→dst_a, src_b→dst_b) — workers
 *              copy 128 MB each in PARALLEL on dedicated cores (256 MB total
 *              split across 2 channels)
 *
 * If DDR channels are independent, parallel work completes in ~half the wall
 * time → ratio P99(hedge) / P99(baseline) ≈ 0.5×.  Same total work; the
 * gate measures channel-parallel speedup, not throughput.
 *
 * Expected on Beast Canyon dual-channel DDR4: baseline ~20 ms/trial (256 MB
 * serial); hedge ~10 ms/trial (256 MB split 128 MB per channel) → P99 ≈ 0.5×.
 *
 * Bench corrections preserved from rework: 128 MB arena past 24 MB L3;
 * direct sp_alloc_huge with hard-abort + 1-page probe diagnostic.
 * F1 fix from rework session: setvbuf(stdout, NULL, _IONBF, 0) so per-trial
 * progress prints flush immediately when stdout is redirected.
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
#include "sp_channel_internal.h"   /* sp_alloc_huge / sp_free_huge */

#define N_TRIALS  2048
#define N_BYTES   (128u * 1024u * 1024u)  /* 128 MB per side */
#define HP_SIZE   (2u * 1024u * 1024u)
#define N_PAGES   (N_BYTES / HP_SIZE)     /* = 64 */

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

/* ── Sort + percentile ──────────────────────────────────────────────────── */
static int cmp_u64_b(const void *x, const void *y) {
    uint64_t a = *(const uint64_t *)x;
    uint64_t b = *(const uint64_t *)y;
    return (a > b) - (a < b);
}
static uint64_t pct(uint64_t *s, int n, int p) {
    int i = (int)((long long)p * n / 100LL);
    return s[i < n ? i : n - 1];
}

/* ── Pin main thread to core 1 (workers on 0,2) ─────────────────────────── */
static void pin_main_thread(void) {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << 1);
#elif defined(__linux__)
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(1, &cs);
    sched_setaffinity(0, sizeof cs, &cs);
#endif
}

/* ── Diagnostic-aware huge-page allocation ─────────────────────────────── */
static void *alloc_arena_or_diagnose(const char *side_name) {
    void *probe = sp_alloc_huge(1, HP_SIZE);
    if (!probe) {
        fprintf(stderr,
            "M_TS_HEDGE_PROD: REQUIRES_LIVE_MODE — 1-page huge alloc FAILED (%s)\n"
            "  Root cause: SeLockMemoryPrivilege not in process token.\n"
            "  Fix: secpol.msc → Lock pages in memory → add user → logoff/logon.\n",
            side_name);
        return NULL;
    }
    sp_free_huge(probe, 1, HP_SIZE);
    void *arena = sp_alloc_huge((size_t)N_PAGES, HP_SIZE);
    if (!arena) {
        fprintf(stderr,
            "M_TS_HEDGE_PROD: REQUIRES_LIVE_MODE — %zu-page huge alloc FAILED (%s)\n"
            "  Root cause: Hyper-V SLAT fragmentation (1-page OK, %zu-page failed).\n"
            "  Fix: reboot fresh; run bench early in uptime.\n",
            (size_t)N_PAGES, side_name, (size_t)N_PAGES);
        return NULL;
    }
    return arena;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);   /* F1 fix: unbuffered stdout */

    sp_channel_map *m = NULL;
    sp_status rc = sp_channel_map_build(&m);
    if (rc != SP_OK || !m) {
        fprintf(stderr, "bench_sp_hedge: sp_channel_map_build failed (rc=%d)\n",
                (int)rc);
        return 1;
    }
    if (sp_channel_map_mode(m) != SP_CHANNEL_LIVE) {
        printf("M_TS_HEDGE_PROD: REQUIRES_LIVE_MODE (channel map DISABLED)\n");
        sp_channel_map_free(m);
        return 0;
    }

    printf("=== §16.3.1 TS.HEDGE batch bench ===\n");
    printf("N_BYTES = %u (%u MB per side); N_PAGES = %u × 2MB; N_TRIALS = %d\n\n",
           N_BYTES, N_BYTES / (1024u * 1024u), N_PAGES, N_TRIALS);

    /* ── Allocate two 128 MB source arenas ──────────────────────────────── */
    void *src_a = alloc_arena_or_diagnose("src A");
    if (!src_a) { sp_channel_map_free(m); return 0; }
    void *src_b = alloc_arena_or_diagnose("src B");
    if (!src_b) {
        sp_free_huge(src_a, (size_t)N_PAGES, HP_SIZE);
        sp_channel_map_free(m);
        return 0;
    }
    /* ── Destination buffers (caller-side; bulk writes into these) ─────── */
    void *dst_a = alloc_arena_or_diagnose("dst A");
    if (!dst_a) {
        sp_free_huge(src_a, (size_t)N_PAGES, HP_SIZE);
        sp_free_huge(src_b, (size_t)N_PAGES, HP_SIZE);
        sp_channel_map_free(m);
        return 0;
    }
    void *dst_b = alloc_arena_or_diagnose("dst B");
    if (!dst_b) {
        sp_free_huge(src_a, (size_t)N_PAGES, HP_SIZE);
        sp_free_huge(src_b, (size_t)N_PAGES, HP_SIZE);
        sp_free_huge(dst_a, (size_t)N_PAGES, HP_SIZE);
        sp_channel_map_free(m);
        return 0;
    }

    /* Pre-fault all four arenas */
    printf("Pre-faulting 4 × 128 MB = 512 MB arenas...\n");
    memset(src_a, 0x42, N_BYTES);
    memset(src_b, 0xBE, N_BYTES);
    memset(dst_a, 0,    N_BYTES);
    memset(dst_b, 0,    N_BYTES);

    /* ── Create hedge pool: workers on cores 0 and 2 ────────────────────── */
    const int WORKER_CORES[2] = {0, 2};
    sp_hedge_pool *pool = NULL;
    rc = sp_hedge_pool_create(&pool, WORKER_CORES, 2, 8);
    if (rc != SP_OK || !pool) {
        printf("M_TS_HEDGE_PROD: pool_create failed (rc=%d)\n", (int)rc);
        sp_free_huge(src_a, (size_t)N_PAGES, HP_SIZE);
        sp_free_huge(src_b, (size_t)N_PAGES, HP_SIZE);
        sp_free_huge(dst_a, (size_t)N_PAGES, HP_SIZE);
        sp_free_huge(dst_b, (size_t)N_PAGES, HP_SIZE);
        sp_channel_map_free(m);
        return 1;
    }
    pin_main_thread();

    uint64_t *base_t  = (uint64_t *)malloc(N_TRIALS * sizeof(uint64_t));
    uint64_t *hedge_t = (uint64_t *)malloc(N_TRIALS * sizeof(uint64_t));
    if (!base_t || !hedge_t) {
        printf("bench_sp_hedge: trial alloc failed\n");
        sp_hedge_pool_destroy(pool);
        sp_free_huge(src_a, (size_t)N_PAGES, HP_SIZE);
        sp_free_huge(src_b, (size_t)N_PAGES, HP_SIZE);
        sp_free_huge(dst_a, (size_t)N_PAGES, HP_SIZE);
        sp_free_huge(dst_b, (size_t)N_PAGES, HP_SIZE);
        sp_channel_map_free(m);
        free(base_t); free(hedge_t);
        return 1;
    }

    /* ── Baseline: SERIAL memcpy of BOTH sides (256 MB total per trial) ──── */
    printf("Running baseline (%d trials × 256 MB serial memcpy)...\n", N_TRIALS);
    for (int t = 0; t < N_TRIALS; t++) {
        uint64_t t0 = rdtsc_bench();
        memcpy(dst_a, src_a, N_BYTES);
        memcpy(dst_b, src_b, N_BYTES);
        uint64_t t1 = rdtsc_bench();
        base_t[t] = t1 - t0;
        if ((t & 0x1F) == 0) printf("  base %d/%d\r", t, N_TRIALS);
    }
    printf("  base done                 \n");

    /* ── Hedge: one sp_hedge_read_bulk per trial (parallel 2-channel) ────── */
    printf("Running hedge    (%d trials × 256 MB parallel bulk via 2 workers)...\n", N_TRIALS);
    for (int t = 0; t < N_TRIALS; t++) {
        uint64_t t0 = rdtsc_bench();
        sp_hedge_read_bulk(pool, src_a, dst_a, src_b, dst_b, N_BYTES);
        uint64_t t1 = rdtsc_bench();
        hedge_t[t] = t1 - t0;
        if ((t & 0x1F) == 0) printf("  hedge %d/%d\r", t, N_TRIALS);
    }
    printf("  hedge done                \n");

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

    printf("\n  body     |    P50 (cyc)    |    P90 (cyc)    |    P99 (cyc)\n");
    printf("  ---------|-----------------|-----------------|----------------\n");
    printf("  baseline | %15llu | %15llu | %15llu\n",
           (unsigned long long)bp50, (unsigned long long)bp90, (unsigned long long)bp99);
    printf("  hedge    | %15llu | %15llu | %15llu\n",
           (unsigned long long)hp50, (unsigned long long)hp90, (unsigned long long)hp99);
    printf("  ratio    | %15.3f | %15.3f | %15.3f"
           "   (hedge/baseline; <1.0 = faster)\n", r50, r90, r99);

    printf("\n");
    if (r99 <= 0.50)
        printf("M_TS_HEDGE_PROD: PASS (P99 ratio %.3f ≤ 0.50)\n", r99);
    else if (r99 <= 0.85)
        printf("M_TS_HEDGE_PROD: WEAK (P99 ratio %.3f; 0.50 < ratio ≤ 0.85)\n", r99);
    else
        printf("M_TS_HEDGE_PROD: FAIL (P99 ratio %.3f > 0.85)\n", r99);

    sp_hedge_pool_destroy(pool);
    free(base_t); free(hedge_t);
    sp_free_huge(src_a, (size_t)N_PAGES, HP_SIZE);
    sp_free_huge(src_b, (size_t)N_PAGES, HP_SIZE);
    sp_free_huge(dst_a, (size_t)N_PAGES, HP_SIZE);
    sp_free_huge(dst_b, (size_t)N_PAGES, HP_SIZE);
    sp_channel_map_free(m);
    return 0;
}
