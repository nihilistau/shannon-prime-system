/* bench_sp_hedge.c — M_TS_HEDGE_PROD: §16.3 persistent-pool hedge-read bench.
 *
 * Compares P50/P90/P99 TSC cycles over 2048 trials:
 *   baseline : N_ELEM sequential volatile uint64 reads from arena_a (channel A)
 *   hedge    : N_ELEM sp_hedge_read_pair calls over (arena_a[i], arena_b[i])
 *
 * Bench correction 1: arena size scaled past L3.
 *   Beast Canyon i9-11900KB L3 = 24 MB (Intel ARK); 4× margin = 96 MB; using
 *   128 MB per side. 16M × 8 = 128 MB; 64 × 2 MB huge pages per side.
 *
 * Bench correction 2: direct sp_alloc_huge (NOT sp_alloc_channel_pair —
 *   which has a silent malloc fallback at line 654 of sp_channel_map.c).
 *   Hard-abort with diagnostic on failure. A 1-page probe distinguishes
 *   missing-privilege (1-page fails) from Hyper-V SLAT fragmentation
 *   (1-page OK but 64-page fails).
 *
 * Bench correction 3: pre-fault via memset before timing (evicts first-access
 *   page-fault latency outliers).
 *
 * Pattern source: in-tree pool at sp_hedge.c (1727f88); reference
 * hedged_reader.hpp:124,138-152.
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
#define N_ELEM    (16u * 1024u * 1024u)   /* 16M × 8 bytes = 128 MB per side */
#define HP_SIZE   (2u * 1024u * 1024u)    /* 2 MB huge page */
#define N_PAGES   (((N_ELEM * 8u) + HP_SIZE - 1u) / HP_SIZE)  /* = 64 */

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

/* ── Sort helper ────────────────────────────────────────────────────────── */
static int cmp_u64_b(const void *x, const void *y) {
    uint64_t a = *(const uint64_t *)x;
    uint64_t b = *(const uint64_t *)y;
    return (a > b) - (a < b);
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

/* ── Percentile ─────────────────────────────────────────────────────────── */
static uint64_t pct(uint64_t *s, int n, int p) {
    int i = (int)((long long)p * n / 100LL);
    return s[i < n ? i : n - 1];
}

/* ── Diagnostic-aware huge-page allocation ─────────────────────────────── *
 * Returns NULL on failure; bench should hard-abort.  Prints diagnostic
 * distinguishing privilege failure from fragmentation. */
static void *alloc_arena_or_diagnose(const char *side_name) {
    void *probe = sp_alloc_huge(1, HP_SIZE);
    if (!probe) {
        fprintf(stderr,
            "M_TS_HEDGE_PROD: REQUIRES_LIVE_MODE — 1-page huge alloc FAILED (%s)\n",
            side_name);
        fprintf(stderr,
            "  Root cause: SeLockMemoryPrivilege not in process token.\n"
            "  Fix:\n"
            "    1. Run secpol.msc → Security Settings → Local Policies →\n"
            "       User Rights Assignment → Lock pages in memory →\n"
            "       Add user account.\n"
            "    2. Log out and log back in (token cache is per-session).\n"
            "    3. Re-run bench as Administrator (right-click → Run as Admin).\n");
        return NULL;
    }
    sp_free_huge(probe, 1, HP_SIZE);

    void *arena = sp_alloc_huge((size_t)N_PAGES, HP_SIZE);
    if (!arena) {
        fprintf(stderr,
            "M_TS_HEDGE_PROD: REQUIRES_LIVE_MODE — %zu-page huge alloc FAILED (%s)\n",
            (size_t)N_PAGES, side_name);
        fprintf(stderr,
            "  Root cause: 1-page allocation succeeded, so the privilege IS\n"
            "  present.  The %zu-page contiguous allocation failed → Hyper-V\n"
            "  SLAT fragmentation of the kernel's large-page free list.\n"
            "  Fix:\n"
            "    1. Reboot fresh; run this bench EARLY in uptime.\n"
            "    2. OR temporarily disable Hyper-V: bcdedit /set\n"
            "       hypervisorlaunchtype off; reboot; run bench; then\n"
            "       bcdedit /set hypervisorlaunchtype auto; reboot.\n",
            (size_t)N_PAGES);
        return NULL;
    }
    return arena;
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
               " (channel map DISABLED — VM/container)\n");
        sp_channel_map_free(m);
        return 0;
    }

    printf("=== §16.3 TS.HEDGE persistent-pool bench (REWORKED) ===\n");
    printf("N_ELEM = %u (%zu MB per side); N_PAGES = %zu × 2MB; N_TRIALS = %d\n\n",
           N_ELEM, (size_t)(N_ELEM * 8u) / (1024u * 1024u),
           (size_t)N_PAGES, N_TRIALS);

    /* ── Allocate two 128 MB arenas; hard-abort on failure ──────────────── */
    void *arena_a = alloc_arena_or_diagnose("side A");
    if (!arena_a) { sp_channel_map_free(m); return 0; }

    void *arena_b = alloc_arena_or_diagnose("side B");
    if (!arena_b) {
        sp_free_huge(arena_a, (size_t)N_PAGES, HP_SIZE);
        sp_channel_map_free(m);
        return 0;
    }

    /* Pre-fault both sides */
    memset(arena_a, 0x42, (size_t)N_ELEM * 8);
    memset(arena_b, 0xBE, (size_t)N_ELEM * 8);

    /* Channel-of distribution sample: cache-line granularity over 16 samples */
    int ch_a_hist[16] = {0}, ch_b_hist[16] = {0};
    for (int i = 0; i < 256; i++) {
        uintptr_t aa = (uintptr_t)arena_a + (uintptr_t)i * 64u * 4096u;
        uintptr_t bb = (uintptr_t)arena_b + (uintptr_t)i * 64u * 4096u;
        uint32_t ca = sp_channel_of(m, aa);
        uint32_t cb = sp_channel_of(m, bb);
        if (ca != SP_CHANNEL_UNSPECIFIED && ca < 16) ch_a_hist[ca]++;
        if (cb != SP_CHANNEL_UNSPECIFIED && cb < 16) ch_b_hist[cb]++;
    }
    int n_chan_a = 0, n_chan_b = 0;
    for (int i = 0; i < 16; i++) {
        if (ch_a_hist[i]) n_chan_a++;
        if (ch_b_hist[i]) n_chan_b++;
    }
    printf("Channel-of distribution (256-sample stride, cache-line × 4 KB):\n");
    printf("  arena_a touches %d distinct channels; arena_b touches %d.\n",
           n_chan_a, n_chan_b);
    printf("  Hardware interleaving guarantees diversity within each arena;\n"
           "  arena_a and arena_b base addresses are NOT the operative property.\n\n");

    /* ── Create hedge pool: workers on cores 0 and 2 ────────────────────── */
    const int WORKER_CORES[2] = {0, 2};
    sp_hedge_pool *pool = NULL;
    rc = sp_hedge_pool_create(&pool, WORKER_CORES, 2, 8);
    if (rc != SP_OK || !pool) {
        printf("M_TS_HEDGE_PROD: pool_create failed (rc=%d)\n", (int)rc);
        sp_free_huge(arena_a, (size_t)N_PAGES, HP_SIZE);
        sp_free_huge(arena_b, (size_t)N_PAGES, HP_SIZE);
        sp_channel_map_free(m);
        return 1;
    }

    /* Bind caller to core 1 (CORE_MAIN; separate from workers 0, 2). */
    pin_main_thread();

    uint64_t *base_t  = (uint64_t *)malloc(N_TRIALS * sizeof(uint64_t));
    uint64_t *hedge_t = (uint64_t *)malloc(N_TRIALS * sizeof(uint64_t));
    if (!base_t || !hedge_t) {
        printf("bench_sp_hedge: trial alloc failed\n");
        sp_hedge_pool_destroy(pool);
        sp_free_huge(arena_a, (size_t)N_PAGES, HP_SIZE);
        sp_free_huge(arena_b, (size_t)N_PAGES, HP_SIZE);
        sp_channel_map_free(m);
        free(base_t); free(hedge_t);
        return 1;
    }

    volatile uint64_t sink_a = 0, sink_b = 0;
    const uint64_t *arr_a = (const uint64_t *)arena_a;
    const uint64_t *arr_b = (const uint64_t *)arena_b;

    /* ── Baseline ──────────────────────────────────────────────────────── */
    printf("Running baseline (%d trials × %u reads)...\n", N_TRIALS, N_ELEM);
    for (int t = 0; t < N_TRIALS; t++) {
        uint64_t t0 = rdtsc_bench();
        for (unsigned i = 0; i < N_ELEM; i++)
            sink_a = *(const volatile uint64_t *)(arr_a + i);
        uint64_t t1 = rdtsc_bench();
        base_t[t] = t1 - t0;
    }
    (void)sink_a;

    /* ── Hedge ─────────────────────────────────────────────────────────── */
    printf("Running hedge     (%d trials × %u pairs)...\n", N_TRIALS, N_ELEM);
    for (int t = 0; t < N_TRIALS; t++) {
        uint64_t oa = 0, ob = 0;
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

    printf("\n  body     |   P50 (cyc)   |   P90 (cyc)   |   P99 (cyc)\n");
    printf("  ---------|---------------|---------------|--------------\n");
    printf("  baseline | %13llu | %13llu | %13llu\n",
           (unsigned long long)bp50, (unsigned long long)bp90, (unsigned long long)bp99);
    printf("  hedge    | %13llu | %13llu | %13llu\n",
           (unsigned long long)hp50, (unsigned long long)hp90, (unsigned long long)hp99);
    printf("  ratio    | %13.3f | %13.3f | %13.3f"
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
    sp_free_huge(arena_a, (size_t)N_PAGES, HP_SIZE);
    sp_free_huge(arena_b, (size_t)N_PAGES, HP_SIZE);
    sp_channel_map_free(m);
    return 0;
}
