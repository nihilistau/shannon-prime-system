/* sp_channel_probe.c — hedge-read timing probes for GF(2) channel recovery.
 *
 * Each probe: flush two addresses A and A^(1<<bit) from cache, then race two
 * OS threads to read them.  The P99 of MAX(latency_A, latency_B) over N samples
 * discriminates same-channel (high-tail contention) from different-channel
 * (low-tail parallel service).
 *
 * Thread failure (privilege denied / resource limit) returns non-zero so the
 * caller can fall back to DISABLED rather than producing garbage data. */

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#include <time.h>
#endif

#include "sp_channel_internal.h"
#include <stdlib.h>
#include <string.h>

/* ── Monotonic nanosecond clock ───────────────────────────────────────────── */

static uint64_t mono_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, t;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    /* Avoid __int128: split into quotient and remainder parts */
    LONGLONG q = t.QuadPart / freq.QuadPart;
    LONGLONG r = t.QuadPart % freq.QuadPart;
    return (uint64_t)(q * 1000000000LL + r * 1000000000LL / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

/* ── Cache-line flush ─────────────────────────────────────────────────────── */

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

/* ── Hedge-read threads ───────────────────────────────────────────────────── */

typedef struct {
    volatile char *addr;
    uint64_t       latency_ns;
} hedge_arg;

#ifdef _WIN32
static DWORD WINAPI hedge_reader(LPVOID arg_v) {
    hedge_arg *a = (hedge_arg *)arg_v;
    uint64_t t0  = mono_ns();
    volatile char x = *a->addr; (void)x;
    a->latency_ns   = mono_ns() - t0;
    return 0;
}
#else
static void *hedge_reader(void *arg_v) {
    hedge_arg *a = (hedge_arg *)arg_v;
    uint64_t t0  = mono_ns();
    volatile char x = *a->addr; (void)x;
    a->latency_ns   = mono_ns() - t0;
    return NULL;
}
#endif

/* Flush both addresses, race two threads, return the MAX latency.
 * Returns 0 on success, non-zero if thread creation failed. */
static int hedge_pair(volatile char *A, volatile char *B, uint64_t *max_lat) {
    cache_flush(A); cache_flush(B); mfence();

    hedge_arg argA = { A, 0 };
    hedge_arg argB = { B, 0 };

#ifdef _WIN32
    HANDLE hA = CreateThread(NULL, 0, hedge_reader, &argA, 0, NULL);
    HANDLE hB = CreateThread(NULL, 0, hedge_reader, &argB, 0, NULL);
    if (!hA || !hB) {
        if (hA) CloseHandle(hA);
        if (hB) CloseHandle(hB);
        return 1;
    }
    HANDLE hs[2] = { hA, hB };
    WaitForMultipleObjects(2, hs, TRUE, 2000);
    CloseHandle(hA); CloseHandle(hB);
#else
    pthread_t tA, tB;
    if (pthread_create(&tA, NULL, hedge_reader, &argA) != 0) return 1;
    if (pthread_create(&tB, NULL, hedge_reader, &argB) != 0) {
        pthread_join(tA, NULL); return 1;
    }
    pthread_join(tA, NULL);
    pthread_join(tB, NULL);
#endif

    *max_lat = (argA.latency_ns > argB.latency_ns)
               ? argA.latency_ns : argB.latency_ns;
    return 0;
}

/* ── qsort helper ────────────────────────────────────────────────────────── */

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* ── Public probe function ────────────────────────────────────────────────── */

int sp_probe_bit(uintptr_t base_addr, int bit, size_t huge_page_size,
                 int n_probes, sp_probe_result *result_out) {
    if (!result_out || n_probes <= 0 || bit < CHAN_BIT_LO || bit >= CHAN_BIT_HI)
        return 1;

    /* Place A one huge page into the allocation so flipping any bit in
     * [12, 24) keeps B within the 4-page region regardless of direction. */
    uintptr_t A   = base_addr + huge_page_size;
    uintptr_t B   = A ^ ((uintptr_t)1 << bit);
    uintptr_t end = base_addr + (uintptr_t)4u * huge_page_size;

    if (B < base_addr || B >= end) {
        /* B out of range — cannot probe; conservatively mark same-channel */
        result_out->bit             = bit;
        result_out->is_same_channel = 1;
        result_out->p99_ns          = 0;
        return 0;
    }

    uint64_t *samples = (uint64_t *)malloc((size_t)n_probes * sizeof *samples);
    if (!samples) return 1;

    int fail = 0;
    for (int i = 0; i < n_probes; i++) {
        uint64_t lat = 0;
        if (hedge_pair((volatile char *)A, (volatile char *)B, &lat) != 0) {
            fail = 1; break;
        }
        samples[i] = lat;
    }

    if (fail) { free(samples); return 1; }

    qsort(samples, (size_t)n_probes, sizeof *samples, cmp_u64);

    uint64_t p50 = samples[(size_t)(n_probes / 2)];
    uint64_t p99 = samples[(size_t)((n_probes * 99) / 100)];
    free(samples);

    /* Same-channel heuristic: P99 > 1.5 × P50.
     * Written as p99 > p50 + p50/2 to avoid large multiplications. */
    int same = (p50 > 0u && p99 > p50 + p50 / 2u) ? 1 : 0;

    result_out->bit             = bit;
    result_out->is_same_channel = same;
    result_out->p99_ns          = p99;
    return 0;
}
