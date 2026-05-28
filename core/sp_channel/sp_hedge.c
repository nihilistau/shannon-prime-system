/* sp_hedge.c — §16.3 TS.HEDGE: persistent worker-pool hedge-read primitives.
 *
 * Production pattern: hedged_reader.hpp:124,138-152 (Laurie TailSlayer).
 * Pool framework: sp_channel_probe.c lifecycle (spawn+pin+destroy shape).
 *
 * N worker threads (one per channel) spawned ONCE at pool_create, each pinned
 * to a dedicated core.  Hot path: atomic publication → worker memcpy → atomic
 * completion count.  No CLFLUSH, TSC rendezvous, or LFENCE on the hot path.
 *
 * v1 limits: n_channels ≤ 2, n_bytes ≤ 64 (covers uint64 + Spinor 63 bytes).
 * Supersedes commit 416417b (PREFETCH+LOAD single-thread — wrong pattern).
 */
#if !defined(_WIN32) && !defined(_GNU_SOURCE)
#  define _GNU_SOURCE
#endif
#define _CRT_SECURE_NO_WARNINGS
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <pthread.h>
#  include <sched.h>
#  include <unistd.h>
#endif
#include "sp/sp_channel.h"
#include "sp/spinor_block.h"
#include "sp/sp_status.h"
#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* ── Platform alignment ─────────────────────────────────────────────────── */
#if defined(_MSC_VER)
#  define SP_HEDGE_ALIGN64 __declspec(align(64))
#elif defined(__GNUC__) || defined(__clang__)
#  define SP_HEDGE_ALIGN64 __attribute__((aligned(64)))
#else
#  define SP_HEDGE_ALIGN64
#endif

/* v1 limit: result buffer size. Covers uint64 (8) and Spinor (63). */
#define SP_HEDGE_MAX_BYTES 64u

/* ── Per-worker context: 2 cache lines (128 bytes), no false sharing ────── *
 * Line 0 (bytes 0-63):  control (should_exit, core_id, n_bytes, src_addr,
 *                        completion_ptr) + 32-byte pad.
 * Line 1 (bytes 64-127): 64-byte result buffer (result_inline).            */
typedef struct {
    atomic_int            should_exit;    /*  4 bytes */
    int                   core_id;        /*  4 bytes */
    size_t                n_bytes;        /*  8 bytes */
    atomic_uintptr_t      src_addr;       /*  8 bytes — NULL=idle, else work addr */
    atomic_int           *completion_ptr; /*  8 bytes — points to pool->completion */
    char                  _pad[32];       /* 32 bytes — line 0 total = 64 bytes */
    uint8_t               result_inline[64]; /* line 1 */
} SP_HEDGE_ALIGN64 sp_hedge_worker_ctx;

_Static_assert(sizeof(sp_hedge_worker_ctx) == 128,
               "sp_hedge_worker_ctx must be 128 bytes (2 cache lines)");

/* ── Pool struct ────────────────────────────────────────────────────────── */
struct sp_hedge_pool {
    size_t               n_channels;   /* 1 or 2 */
    atomic_int           completion;   /* shared counter written by workers */
    sp_hedge_worker_ctx *workers;      /* aligned array; NULL if n_channels==1 */
#ifdef _WIN32
    HANDLE               hA, hB;
#else
    pthread_t            tA, tB;
    int                  tA_valid, tB_valid;
#endif
};

/* ── Core affinity ──────────────────────────────────────────────────────── */
static void sp_hedge_pin(int core_id) {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << core_id);
#elif defined(__linux__)
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(core_id, &cs);
    (void)pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
#else
    (void)core_id;
#endif
}

/* ── Worker function ──────────────────────────────────────────────────────
 * hedged_reader.hpp:157-176: pin → idle spin → read (final_work) → loop.
 * INVARIANT: always drain (memcpy + completion++) after seeing non-NULL src.
 * Never re-check should_exit between acquiring src and completing — prevents
 * caller livelock on the completion count. */
#ifdef _WIN32
static DWORD WINAPI sp_hedge_worker_fn(LPVOID arg_v) {
#else
static void *sp_hedge_worker_fn(void *arg_v) {
#endif
    sp_hedge_worker_ctx *ctx = (sp_hedge_worker_ctx *)arg_v;
    sp_hedge_pin(ctx->core_id);

    for (;;) {
        /* Idle spin — no _mm_pause: cores are dedicated, sub-µs response needed. */
        uintptr_t raw;
        for (;;) {
            raw = atomic_load_explicit(&ctx->src_addr, memory_order_acquire);
            if (raw != 0u) break;
            if (atomic_load_explicit(&ctx->should_exit, memory_order_relaxed))
                goto done;
        }
        /* Drain: always complete after acquiring non-NULL src. */
        memcpy(ctx->result_inline, (const void *)raw, ctx->n_bytes);
        atomic_store_explicit(&ctx->src_addr, 0u, memory_order_relaxed);
        atomic_fetch_add_explicit(ctx->completion_ptr, 1, memory_order_release);
    }
done:
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

/* ── sp_hedge_pool_create ──────────────────────────────────────────────── */
sp_status sp_hedge_pool_create(sp_hedge_pool **out,
                               const int *core_ids,
                               size_t n_channels,
                               size_t max_bytes)
{
    if (!out || !core_ids || n_channels == 0 || max_bytes == 0)
        return SP_EBADARG;
    if (n_channels > 2 || max_bytes > SP_HEDGE_MAX_BYTES)
        return SP_EBADARG;   /* v1 limit; extend in §16.4 */

    sp_hedge_pool *pool = (sp_hedge_pool *)calloc(1, sizeof *pool);
    if (!pool) return SP_ENOMEM;
    pool->n_channels = n_channels;
    atomic_init(&pool->completion, 0);

    /* N=1: no threads; read_pair is direct memcpy. */
    if (n_channels == 1) { *out = pool; return SP_OK; }

    /* Allocate worker contexts (cache-line aligned). */
#ifdef _WIN32
    pool->workers = (sp_hedge_worker_ctx *)_aligned_malloc(
        2 * sizeof(sp_hedge_worker_ctx), 64);
#else
    if (posix_memalign((void **)&pool->workers, 64,
                       2 * sizeof(sp_hedge_worker_ctx)) != 0)
        pool->workers = NULL;
#endif
    if (!pool->workers) { free(pool); return SP_ENOMEM; }
    memset(pool->workers, 0, 2 * sizeof(sp_hedge_worker_ctx));

    for (int i = 0; i < 2; i++) {
        atomic_init(&pool->workers[i].should_exit, 0);
        atomic_init(&pool->workers[i].src_addr, 0u);
        pool->workers[i].core_id        = core_ids[i];
        pool->workers[i].completion_ptr = &pool->completion;
    }

    /* Spawn two workers; pin inside the thread function (hedged_reader.hpp:158). */
#ifdef _WIN32
    pool->hA = CreateThread(NULL, 0, sp_hedge_worker_fn, &pool->workers[0], 0, NULL);
    pool->hB = CreateThread(NULL, 0, sp_hedge_worker_fn, &pool->workers[1], 0, NULL);
    if (!pool->hA || !pool->hB) {
        sp_hedge_pool_destroy(pool);
        return SP_ENOMEM;
    }
    Sleep(10);  /* let workers reach idle spin; hedged_reader.hpp:126 (usleep 10ms) */
#else
    pool->tA_valid = (pthread_create(&pool->tA, NULL,
                                     sp_hedge_worker_fn, &pool->workers[0]) == 0);
    pool->tB_valid = (pthread_create(&pool->tB, NULL,
                                     sp_hedge_worker_fn, &pool->workers[1]) == 0);
    if (!pool->tA_valid || !pool->tB_valid) {
        sp_hedge_pool_destroy(pool);
        return SP_ENOMEM;
    }
    usleep(10000);
#endif
    *out = pool;
    return SP_OK;
}

/* ── sp_hedge_pool_destroy ─────────────────────────────────────────────── */
void sp_hedge_pool_destroy(sp_hedge_pool *pool) {
    if (!pool) return;
    if (pool->workers) {
        atomic_store_explicit(&pool->workers[0].should_exit, 1, memory_order_relaxed);
        atomic_store_explicit(&pool->workers[1].should_exit, 1, memory_order_relaxed);
    }
#ifdef _WIN32
    if (pool->hA) { WaitForSingleObject(pool->hA, 2000); CloseHandle(pool->hA); }
    if (pool->hB) { WaitForSingleObject(pool->hB, 2000); CloseHandle(pool->hB); }
    if (pool->workers) _aligned_free(pool->workers);
#else
    if (pool->tA_valid) pthread_join(pool->tA, NULL);
    if (pool->tB_valid) pthread_join(pool->tB, NULL);
    if (pool->workers) free(pool->workers);
#endif
    free(pool);
}

/* ── sp_hedge_read_pair ────────────────────────────────────────────────── */
sp_status sp_hedge_read_pair(sp_hedge_pool *pool,
                             const void *a, const void *b,
                             size_t n_bytes,
                             void *out_a, void *out_b)
{
    if (!pool || n_bytes > SP_HEDGE_MAX_BYTES) return SP_EBADARG;

    /* N=1 fallback: direct memcpy of side a (b ignored). */
    if (pool->n_channels == 1) {
        if (out_a) memcpy(out_a, a, n_bytes);
        return SP_OK;
    }

    const void *srcs[2] = {a, b};

    /* Reset completion (relaxed; ordered by release-acquire on src_addr). */
    atomic_store_explicit(&pool->completion, 0, memory_order_relaxed);

    /* Publish n_bytes then src_addr (release ensures n_bytes visible to workers). */
    pool->workers[0].n_bytes = n_bytes;
    pool->workers[1].n_bytes = n_bytes;
    atomic_store_explicit(&pool->workers[0].src_addr,
                          (uintptr_t)srcs[0], memory_order_release);
    atomic_store_explicit(&pool->workers[1].src_addr,
                          (uintptr_t)srcs[1], memory_order_release);

    /* Spin on completion (acquire — ensures result_inline writes are visible). */
    while (atomic_load_explicit(&pool->completion, memory_order_acquire) < 2)
        ;

    if (out_a) memcpy(out_a, pool->workers[0].result_inline, n_bytes);
    if (out_b) memcpy(out_b, pool->workers[1].result_inline, n_bytes);
    return SP_OK;
}

/* ── sp_hedge_read_spinor ─────────────────────────────────────────────── */
sp_status sp_hedge_read_spinor(sp_hedge_pool *pool,
                               const sp_spinor_block_t *a,
                               const sp_spinor_block_t *b,
                               sp_spinor_block_t *out_a,
                               sp_spinor_block_t *out_b)
{
    return sp_hedge_read_pair(pool, a, b, sizeof *a, out_a, out_b);
}
