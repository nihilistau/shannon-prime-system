/* sp_channel_probe.c — hedge-read timing probes with persistent thread pool.
 *
 * Replacing the old per-sample CreateThread/pthread_create approach that added
 * 1–100 µs thread-creation jitter — 1000× the ~100 ns DRAM contention signal.
 *
 * Two reader threads are spawned once per probe_pool.  A lock-free spin-barrier
 * (volatile int cmd/done + RDTSC) synchronises each sample with ~20-cycle
 * overhead instead of microseconds.  Workers are pinned to distinct physical
 * cores (best-effort) to avoid L1/L2 bandwidth bottleneck before the memory
 * controller sees contention. */

#if !defined(_WIN32) && !defined(_GNU_SOURCE)
#  define _GNU_SOURCE   /* pthread_setaffinity_np, cpu_set_t */
#endif

#define _CRT_SECURE_NO_WARNINGS
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  ifdef _MSC_VER
#    include <intrin.h>   /* __rdtsc() */
#  endif
#else
#  include <pthread.h>
#  include <sched.h>
#  include <time.h>
#endif

#include "sp_channel_internal.h"
#include <stdlib.h>
#include <string.h>

/* ── Platform alignment ───────────────────────────────────────────────────── */

#if defined(_MSC_VER)
#  define SP_CHAN_ALIGN64 __declspec(align(64))
#elif defined(__GNUC__) || defined(__clang__)
#  define SP_CHAN_ALIGN64 __attribute__((aligned(64)))
#else
#  define SP_CHAN_ALIGN64
#endif

/* ── Monotonic nanosecond clock (non-x86 fallback only) ─────────────────── */
/* Compiled only on non-x86 platforms where rdtsc_now() delegates to it.     */
#if !defined(_M_X64) && !defined(_M_IX86) && !defined(__x86_64__) && !defined(__i386__)
static uint64_t mono_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, t;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    LONGLONG q = t.QuadPart / freq.QuadPart;
    LONGLONG r = t.QuadPart % freq.QuadPart;
    return (uint64_t)(q * 1000000000LL + r * 1000000000LL / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}
#endif /* non-x86 guard */

/* ── Cache-line flush and memory fence ───────────────────────────────────── */

static void cache_flush(volatile void *p) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("clflush (%0)" :: "r"(p) : "memory");
#elif defined(_WIN32) && (defined(_M_X64) || defined(_M_IX86))
    _mm_clflush(p);
#else
    (void)p;
    __asm__ volatile("" ::: "memory");
#endif
}

static void mfence(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("mfence" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

/* ── Worker context: 128 bytes, 2 separate cache lines, no false sharing ─── *
 *
 * Line 0 (offset 0–63):  quit + cmd + done + 52 bytes padding
 * Line 1 (offset 64–127): addr(8) + lat(8) + fire_tsc(8) + 40 bytes padding
 *
 * fire_tsc: TSC rendezvous timestamp.  Both workers spin until rdtsc_now() >=
 * fire_tsc before issuing their DRAM loads, ensuring simultaneous access to A
 * and B.  Without synchronisation, worker A starts 100-300 cycles before B;
 * A's read completes before B arrives at the memory controller, eliminating
 * any channel-contention signal.  x86 TSO guarantees that if a worker sees
 * cmd=1 it also sees the prior fire_tsc store, so the field is always valid
 * when the worker enters the GO phase. */

typedef struct {
    volatile int       quit;        /* line 0: control words             */
    volatile int       cmd;         /* 0 = IDLE, 1 = GO                  */
    volatile int       done;
    char              _ctl_pad[52]; /* pad line 0 to 64 bytes: 64-3×4=52 */
    volatile char     *addr;        /* line 1: per-sample payload         */
    volatile uint64_t  lat;         /* TSC cycle count for this read      */
    volatile uint64_t  fire_tsc;    /* rendezvous: spin until RDTSC >= this */
    char              _dat_pad[40]; /* pad line 1 to 64 bytes: 64-8-8-8=40 */
} SP_CHAN_ALIGN64 probe_worker_ctx;

_Static_assert(sizeof(probe_worker_ctx) == 128,
               "probe_worker_ctx must be 128 bytes (2 cache lines)");

/* ── Pool (opaque to callers; forward-declared in sp_channel_internal.h) ─── */

struct sp_probe_pool {
    probe_worker_ctx wA;      /* 128 bytes at offset 0   */
    probe_worker_ctx wB;      /* 128 bytes at offset 128 */
    uint64_t         tsc_hz;  /* TSC cycles per second   */
#ifdef _WIN32
    HANDLE    hA, hB;
#else
    pthread_t tA, tB;
#endif
};

/* ── RDTSC: ~20-cycle overhead vs QPC ~100-300 cycles ────────────────────── */

static uint64_t rdtsc_now(void) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    return __rdtsc();
#elif defined(__x86_64__) || defined(__i386__)
    unsigned hi, lo;
    __asm__ volatile("rdtsc" : "=d"(hi), "=a"(lo));
    return ((uint64_t)hi << 32) | (uint64_t)lo;
#else
    return mono_ns();  /* non-x86: treat ns as cycles; calibrate_tsc_hz → 1e9 */
#endif
}

/* ── Spin-wait hint ───────────────────────────────────────────────────────── */

static void sp_pause(void) {
#if defined(_MSC_VER)
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#endif
}

/* LFENCE: serialises RDTSC against OOO loads on Tiger Lake / Alder Lake.
 * Without it, the second rdtsc can retire before the volatile load on
 * aggressive superscalar pipelines, collapsing the measured interval to ~0. */
static void sp_lfence(void) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_lfence();
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("lfence" ::: "memory");
#endif
}

/* ── Cooperative yield: lets a co-scheduled thread run on the same CPU ─────── *
 * Used in both main's done-wait and the worker's IDLE-reset spin so that      *
 * threads sharing the same logical CPU hand off to each other without waiting  *
 * for the OS timer interrupt (~15 ms).  In the cross-core case sp_yield on    *
 * an otherwise-idle core returns immediately with no meaningful overhead.      */
static void sp_yield(void) {
#ifdef _WIN32
    SwitchToThread();
#else
    sched_yield();
#endif
}

/* ── TSC frequency calibration ───────────────────────────────────────────── */

static uint64_t calibrate_tsc_hz(void) {
#if defined(_WIN32)
    /* QPC on Windows may be backed by HPET (10 MHz) or ACPI (3.58 MHz), NOT
     * the invariant TSC.  Measure RDTSC cycles that elapse in a 10 ms QPC
     * interval to get the true TSC rate regardless of the QPC source. */
    LARGE_INTEGER qpf, t0_qpc, t1_qpc;
    QueryPerformanceFrequency(&qpf);
    QueryPerformanceCounter(&t0_qpc);
    uint64_t c0 = rdtsc_now();
    uint64_t wait_ticks = (uint64_t)qpf.QuadPart / 100;  /* 10 ms */
    do { QueryPerformanceCounter(&t1_qpc); }
    while ((uint64_t)(t1_qpc.QuadPart - t0_qpc.QuadPart) < wait_ticks);
    uint64_t c1 = rdtsc_now();
    uint64_t elapsed = (uint64_t)(t1_qpc.QuadPart - t0_qpc.QuadPart);
    if (elapsed == 0) return 3000000000ULL;
    /* tsc_hz = (c1-c0) × qpf / elapsed_qpc_ticks */
    uint64_t hz = (c1 - c0) * (uint64_t)qpf.QuadPart / elapsed;
    return hz > 0 ? hz : 3000000000ULL;
#elif defined(__x86_64__) || defined(__i386__)
    /* 5 ms busy-spin to measure TSC tick rate against CLOCK_MONOTONIC */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    uint64_t c0 = rdtsc_now();
    uint64_t ns0 = (uint64_t)t0.tv_sec * 1000000000ULL + (uint64_t)t0.tv_nsec;
    uint64_t ns_elapsed;
    do {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        uint64_t ns1 = (uint64_t)t1.tv_sec * 1000000000ULL + (uint64_t)t1.tv_nsec;
        ns_elapsed = ns1 - ns0;
    } while (ns_elapsed < 5000000ULL);
    uint64_t c1 = rdtsc_now();
    if (ns_elapsed == 0) return 3000000000ULL;
    return (c1 - c0) * 1000000000ULL / ns_elapsed;
#else
    return 1000000000ULL;  /* rdtsc_now() returns ns directly on non-x86 */
#endif
}

/* ── Worker thread: spin-barrier IDLE → GO → DONE → IDLE ─────────────────── */

#ifdef _WIN32
static DWORD WINAPI probe_worker_fn(LPVOID arg_v) {
#else
static void *probe_worker_fn(void *arg_v) {
#endif
    probe_worker_ctx *ctx = (probe_worker_ctx *)arg_v;
    for (;;) {
        /* IDLE spin: wait for a new GO (cmd=1 AND done=0).
         * x86 TSO guarantees that if we observe cmd=1 we also observe all
         * prior main-thread stores, including the done=0 reset — so the
         * two-field test is race-free without any extra fence.  Using done=0
         * as a co-condition also eliminates the old second spin: after setting
         * done=1 we re-enter IDLE immediately; done=1 keeps us parked here
         * until main resets done=0 for the next probe. */
        while (!ctx->quit && !(ctx->cmd && !ctx->done)) sp_pause();
        if (ctx->quit) break;

        /* GO: wait for the TSC rendezvous so both workers issue their DRAM loads
         * simultaneously.  Without this, worker A starts 100-300 cycles before B
         * (coherence skew), and A's read completes before B even arrives at the
         * memory controller — eliminating the channel-contention signal. */
        while (rdtsc_now() < (uint64_t)ctx->fire_tsc) sp_pause();

        /* LFENCE before t0 serialises RDTSC against speculative execution;
         * LFENCE after the load prevents the second RDTSC from retiring before
         * the volatile read (Tiger Lake OOO). */
        sp_lfence();
        uint64_t t0 = rdtsc_now();
        volatile char x = *ctx->addr; (void)x;
        sp_lfence();
        ctx->lat  = rdtsc_now() - t0;
        ctx->done = 1;
    }
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

/* ── Public pool API ──────────────────────────────────────────────────────── */

probe_pool *sp_probe_pool_create(void) {
    probe_pool *pool;
#ifdef _WIN32
    pool = (probe_pool *)_aligned_malloc(sizeof *pool, 64);
#else
    if (posix_memalign((void **)&pool, 64, sizeof *pool) != 0) pool = NULL;
#endif
    if (!pool) return NULL;
    memset(pool, 0, sizeof *pool);

    pool->tsc_hz = calibrate_tsc_hz();

#ifdef _WIN32
    pool->hA = CreateThread(NULL, 0, probe_worker_fn, &pool->wA, 0, NULL);
    pool->hB = CreateThread(NULL, 0, probe_worker_fn, &pool->wB, 0, NULL);
    if (!pool->hA || !pool->hB) {
        if (pool->hA) { pool->wA.quit = 1; WaitForSingleObject(pool->hA, 1000); CloseHandle(pool->hA); }
        if (pool->hB) { pool->wB.quit = 1; WaitForSingleObject(pool->hB, 1000); CloseHandle(pool->hB); }
        _aligned_free(pool);
        return NULL;
    }
    /* Pin workers to P-cores 0 and 2.  On Tiger Lake / Beast Canyon these are
     * both P-cores on the same Ring Bus, giving the best cache-coherence signal.
     * Core 0 receives DPCs/IRQs but we now use P90 (not P99) so the ~1%
     * interrupt rate does not pollute the percentile we measure. */
    (void)SetThreadAffinityMask(pool->hA, (DWORD_PTR)1 << 0);
    (void)SetThreadAffinityMask(pool->hB, (DWORD_PTR)1 << 2);
#else
    if (pthread_create(&pool->tA, NULL, probe_worker_fn, &pool->wA) != 0) {
        free(pool); return NULL;
    }
    if (pthread_create(&pool->tB, NULL, probe_worker_fn, &pool->wB) != 0) {
        pool->wA.quit = 1;
        pthread_join(pool->tA, NULL);
        free(pool); return NULL;
    }
    /* Best-effort: pin workers to distinct physical cores */
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset); CPU_SET(0, &cpuset);
        (void)pthread_setaffinity_np(pool->tA, sizeof cpuset, &cpuset);
        CPU_ZERO(&cpuset); CPU_SET(2, &cpuset);
        (void)pthread_setaffinity_np(pool->tB, sizeof cpuset, &cpuset);
    }
#endif
    return pool;
}

void sp_probe_pool_destroy(probe_pool *pool) {
    if (!pool) return;
    /* Signal workers to exit from either spin (IDLE or IDLE-reset) */
    pool->wA.quit = 1; pool->wB.quit = 1;
    pool->wA.cmd  = 0; pool->wB.cmd  = 0;  /* release any cmd==1 spin */
#ifdef _WIN32
    if (pool->hA) { WaitForSingleObject(pool->hA, 2000); CloseHandle(pool->hA); }
    if (pool->hB) { WaitForSingleObject(pool->hB, 2000); CloseHandle(pool->hB); }
    _aligned_free(pool);
#else
    pthread_join(pool->tA, NULL);
    pthread_join(pool->tB, NULL);
    free(pool);
#endif
}

/* ── Pooled hedge pair: flush → GO spin-barrier → collect max cycles ─────── */

static void hedge_pair_pooled(probe_pool *pool,
                               volatile char *A, volatile char *B,
                               uint64_t *cycles_out) {
    /* Reset done flags BEFORE signalling GO so main's done-spin is not stale */
    pool->wA.done = 0;
    pool->wB.done = 0;
    cache_flush(A); cache_flush(B); mfence();
    pool->wA.addr = A;
    pool->wB.addr = B;
    /* Set rendezvous 1000 cycles ahead: enough for cache coherence to propagate
     * cmd=1 to both cores and for both workers to enter the fire_tsc spin before
     * the rendezvous fires.  1000 cycles ≈ 312 ns at 3.2 GHz.  x86 TSO ensures
     * workers that see cmd=1 also see the prior fire_tsc stores. */
    uint64_t fire = rdtsc_now() + 1000;
    pool->wA.fire_tsc = fire;
    pool->wB.fire_tsc = fire;
    pool->wA.cmd = 1;   /* GO: wake worker A */
    pool->wB.cmd = 1;   /* GO: wake worker B */
    /* Adaptive wait: spin first, yield only when workers need more time.
     * Workers finish in ~300 ns when cross-core; the 5000-pause threshold
     * is never reached in that case so sp_yield never fires.  If workers
     * share a physical core with the main thread the threshold acts as a
     * fallback to let the OS schedule the worker. */
    /* 200 PAUSEs ≈ 620 ns at 3.2 GHz — enough for cross-core workers (~300 ns)
     * to finish without yielding.  If workers share a core with main, we yield
     * quickly (not after a 22 µs stall). */
    for (int _sp = 0; !pool->wA.done || !pool->wB.done; _sp++) {
        if (_sp < 200) { sp_pause(); } else { sp_yield(); _sp = 0; }
    }
    *cycles_out = (pool->wA.lat > pool->wB.lat) ? pool->wA.lat : pool->wB.lat;
    pool->wA.cmd = 0;   /* IDLE reset */
    pool->wB.cmd = 0;
}

/* ── qsort helper ────────────────────────────────────────────────────────── */

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* ── Public probe function ────────────────────────────────────────────────── */

int sp_probe_bit(uintptr_t base_addr, int bit, size_t huge_page_size,
                 int n_probes, probe_pool *pool, sp_probe_result *result_out) {
    if (!result_out || !pool || n_probes <= 0 || bit < CHAN_BIT_LO || bit >= CHAN_BIT_HI)
        return 1;

    uintptr_t A   = base_addr + huge_page_size;
    uintptr_t B   = A ^ ((uintptr_t)1 << bit);
    uintptr_t end = base_addr + (uintptr_t)4u * huge_page_size;

    if (B < base_addr || B >= end) {
        result_out->bit             = bit;
        result_out->is_same_channel = 1;
        result_out->p99_ns          = 0;
        return 0;
    }

    uint64_t *samples = (uint64_t *)malloc((size_t)n_probes * sizeof *samples);
    if (!samples) return 1;

    for (int i = 0; i < n_probes; i++) {
        uint64_t cycles = 0;
        hedge_pair_pooled(pool, (volatile char *)A, (volatile char *)B, &cycles);
        /* Convert TSC cycles to nanoseconds */
        samples[i] = (pool->tsc_hz > 0) ? (cycles * 1000000000ULL / pool->tsc_hz) : cycles;
    }

    qsort(samples, (size_t)n_probes, sizeof *samples, cmp_u64);

    uint64_t p50 = samples[(size_t)(n_probes / 2)];
    /* P90 instead of P99: with 512 samples, P90 = 52nd-from-top.
     * Windows DPC/interrupt rate is ~1% → ~5 spikes per 512 samples → P99 is
     * contaminated, P90 is clean.  P90 faithfully reflects DRAM latency. */
    uint64_t p90 = samples[(size_t)((n_probes * 90) / 100)];
    free(samples);

    /* Same-channel heuristic: P90 > 1.5 × P50 */
    int same = (p50 > 0u && p90 > p50 + p50 / 2u) ? 1 : 0;

    result_out->bit             = bit;
    result_out->is_same_channel = same;
    result_out->p50_ns          = p50;
    result_out->p90_ns          = p90;
    result_out->p99_ns          = p90;   /* compat alias — actually P90 */
    return 0;
}
