/* test_sp_hedge.c — §16.3 TS.HEDGE correctness + pool-lifecycle tests.
 *
 * T_HEDGE_POOL_CREATE_DESTROY — create + immediate destroy; clean teardown
 * T_HEDGE_PAIR_1              — read_pair returns byte-identical to src
 * T_HEDGE_SPINOR_1            — read_spinor returns correct 63-byte blocks
 * T_HEDGE_N1_FALLBACK         — N=1 pool: memcpy of side a; b ignored
 * T_HEDGE_REPEATED            — 10000 sequential reads through same pool
 *
 * All tests use plain malloc arenas — no channel map, no huge pages.
 * Correctness is channel-independent (gate is bitwise correctness, not speed).
 */
#include "sp/sp_channel.h"
#include "sp/spinor_block.h"
#include "sp/sp_test.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void fill_rand(void *buf, size_t n, unsigned int seed) {
    unsigned char *p = (unsigned char *)buf;
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        p[i] = (unsigned char)(seed >> 16);
    }
}

/* Default core_ids for the 2-worker pool in tests */
static const int TEST_CORES[2] = {0, 2};

/* ── T_HEDGE_POOL_CREATE_DESTROY ─────────────────────────────────────────*/
static void T_HEDGE_POOL_CREATE_DESTROY(void) {
    sp_hedge_pool *pool = NULL;
    sp_status rc = sp_hedge_pool_create(&pool, TEST_CORES, 2, 8);
    SP_CHECK(rc == SP_OK, "pool_create returns SP_OK");
    SP_CHECK(pool != NULL, "pool non-NULL after create");
    sp_hedge_pool_destroy(pool);
    /* If we reach here without hang/crash: pool teardown is clean */
    SP_CHECK(1, "pool_destroy completed without hang");
}

/* ── T_HEDGE_PAIR_1 ───────────────────────────────────────────────────── */
static void T_HEDGE_PAIR_1(void) {
    sp_hedge_pool *pool = NULL;
    sp_status rc = sp_hedge_pool_create(&pool, TEST_CORES, 2, 8);
    SP_CHECK(rc == SP_OK, "pair: pool_create");
    if (!pool) return;

    /* Read uint64 pair */
    uint64_t src_a = 0xDEADBEEFCAFEBABEull;
    uint64_t src_b = 0x0123456789ABCDEFull;
    uint64_t out_a = 0, out_b = 0;
    rc = sp_hedge_read_pair(pool, &src_a, &src_b, 8, &out_a, &out_b);
    SP_CHECK(rc == SP_OK, "pair: read_pair SP_OK");
    SP_CHECK_EQ_I64(out_a, src_a, "pair: out_a == src_a");
    SP_CHECK_EQ_I64(out_b, src_b, "pair: out_b == src_b");

    /* Sweep 16 random pairs */
    uint64_t arr_a[16], arr_b[16], res_a[16], res_b[16];
    fill_rand(arr_a, sizeof arr_a, 0x1234);
    fill_rand(arr_b, sizeof arr_b, 0x5678);
    for (int i = 0; i < 16; i++) {
        res_a[i] = res_b[i] = 0;
        sp_hedge_read_pair(pool, &arr_a[i], &arr_b[i], 8, &res_a[i], &res_b[i]);
        SP_CHECK_EQ_I64(res_a[i], arr_a[i], "pair sweep: a");
        SP_CHECK_EQ_I64(res_b[i], arr_b[i], "pair sweep: b");
    }
    sp_hedge_pool_destroy(pool);
}

/* ── T_HEDGE_SPINOR_1 ────────────────────────────────────────────────── */
static void T_HEDGE_SPINOR_1(void) {
    sp_hedge_pool *pool = NULL;
    sp_status rc = sp_hedge_pool_create(&pool, TEST_CORES, 2, 64);
    SP_CHECK(rc == SP_OK, "spinor: pool_create");
    if (!pool) return;

    sp_spinor_block_t sa, sb, oa, ob;
    fill_rand((uint8_t *)&sa, 63, 0xBEEF);
    fill_rand((uint8_t *)&sb, 63, 0xCAFE);
    memset(&oa, 0, 63); memset(&ob, 0, 63);

    rc = sp_hedge_read_spinor(pool, &sa, &sb, &oa, &ob);
    SP_CHECK(rc == SP_OK, "spinor: read_spinor SP_OK");
    SP_CHECK(memcmp(&oa, &sa, 63) == 0, "spinor: out_a == src_a");
    SP_CHECK(memcmp(&ob, &sb, 63) == 0, "spinor: out_b == src_b");

    sp_hedge_pool_destroy(pool);
}

/* ── T_HEDGE_N1_FALLBACK ─────────────────────────────────────────────── */
static void T_HEDGE_N1_FALLBACK(void) {
    int cores[1] = {0};
    sp_hedge_pool *pool = NULL;
    sp_status rc = sp_hedge_pool_create(&pool, cores, 1, 8);
    SP_CHECK(rc == SP_OK, "n1: pool_create");
    if (!pool) return;

    uint64_t src_a = 0xAAAABBBBCCCCDDDDull;
    uint64_t src_b = 0xEEEEFFFF00001111ull;
    uint64_t out_a = 0, out_b = 0xFFFFFFFFFFFFFFFFull; /* sentinel */
    rc = sp_hedge_read_pair(pool, &src_a, &src_b, 8, &out_a, &out_b);
    SP_CHECK(rc == SP_OK, "n1: read_pair SP_OK");
    SP_CHECK_EQ_I64(out_a, src_a, "n1: out_a == src_a (memcpy of a)");
    /* out_b must be unchanged (b ignored in N=1 path) */
    SP_CHECK_EQ_I64(out_b, (int64_t)0xFFFFFFFFFFFFFFFFull,
                    "n1: out_b unchanged (b ignored)");

    sp_hedge_pool_destroy(pool);
}

/* ── T_HEDGE_REPEATED ────────────────────────────────────────────────── */
static void T_HEDGE_REPEATED(void) {
    sp_hedge_pool *pool = NULL;
    sp_status rc = sp_hedge_pool_create(&pool, TEST_CORES, 2, 8);
    SP_CHECK(rc == SP_OK, "repeated: pool_create");
    if (!pool) return;

    int fail_count = 0;
    for (int iter = 0; iter < 10000; iter++) {
        uint64_t a = (uint64_t)iter * 0x9E3779B97F4A7C15ull;
        uint64_t b = (uint64_t)iter * 0x6C62272E07BB0142ull;
        uint64_t ra = 0, rb = 0;
        sp_hedge_read_pair(pool, &a, &b, 8, &ra, &rb);
        if (ra != a || rb != b) fail_count++;
    }
    SP_CHECK(fail_count == 0, "repeated: 10000 reads all bitwise-correct");

    sp_hedge_pool_destroy(pool);
}

/* ── T_HEDGE_BULK_SMALL — §16.3.1 ─────────────────────────────────────── */
static void T_HEDGE_BULK_SMALL(void) {
    sp_hedge_pool *pool = NULL;
    sp_status rc = sp_hedge_pool_create(&pool, TEST_CORES, 2, 8);
    SP_CHECK(rc == SP_OK, "bulk_small: pool_create");
    if (!pool) return;

    const size_t N = 8192;
    uint8_t *src_a = (uint8_t *)malloc(N);
    uint8_t *src_b = (uint8_t *)malloc(N);
    uint8_t *dst_a = (uint8_t *)malloc(N);
    uint8_t *dst_b = (uint8_t *)malloc(N);
    SP_CHECK(src_a && src_b && dst_a && dst_b, "bulk_small: alloc");
    if (src_a && src_b && dst_a && dst_b) {
        fill_rand(src_a, N, 0xCAFE);
        fill_rand(src_b, N, 0xBEEF);
        memset(dst_a, 0, N);
        memset(dst_b, 0, N);
        rc = sp_hedge_read_bulk(pool, src_a, dst_a, src_b, dst_b, N);
        SP_CHECK(rc == SP_OK, "bulk_small: read_bulk SP_OK");
        SP_CHECK(memcmp(dst_a, src_a, N) == 0, "bulk_small: dst_a == src_a");
        SP_CHECK(memcmp(dst_b, src_b, N) == 0, "bulk_small: dst_b == src_b");
    }
    free(src_a); free(src_b); free(dst_a); free(dst_b);
    sp_hedge_pool_destroy(pool);
}

/* ── T_HEDGE_BULK_LARGE — 1 MB transfer ─────────────────────────────── */
static void T_HEDGE_BULK_LARGE(void) {
    sp_hedge_pool *pool = NULL;
    sp_status rc = sp_hedge_pool_create(&pool, TEST_CORES, 2, 8);
    SP_CHECK(rc == SP_OK, "bulk_large: pool_create");
    if (!pool) return;

    const size_t N = 1024 * 1024;
    uint8_t *src_a = (uint8_t *)malloc(N);
    uint8_t *src_b = (uint8_t *)malloc(N);
    uint8_t *dst_a = (uint8_t *)malloc(N);
    uint8_t *dst_b = (uint8_t *)malloc(N);
    SP_CHECK(src_a && src_b && dst_a && dst_b, "bulk_large: alloc");
    if (src_a && src_b && dst_a && dst_b) {
        fill_rand(src_a, N, 0x1234);
        fill_rand(src_b, N, 0x5678);
        memset(dst_a, 0, N);
        memset(dst_b, 0, N);
        rc = sp_hedge_read_bulk(pool, src_a, dst_a, src_b, dst_b, N);
        SP_CHECK(rc == SP_OK, "bulk_large: read_bulk SP_OK");
        SP_CHECK(memcmp(dst_a, src_a, N) == 0, "bulk_large: dst_a == src_a (1 MB)");
        SP_CHECK(memcmp(dst_b, src_b, N) == 0, "bulk_large: dst_b == src_b (1 MB)");
    }
    free(src_a); free(src_b); free(dst_a); free(dst_b);
    sp_hedge_pool_destroy(pool);
}

/* ── T_HEDGE_BULK_N1_FALLBACK ─────────────────────────────────────────── */
static void T_HEDGE_BULK_N1_FALLBACK(void) {
    int cores[1] = {0};
    sp_hedge_pool *pool = NULL;
    sp_status rc = sp_hedge_pool_create(&pool, cores, 1, 8);
    SP_CHECK(rc == SP_OK, "bulk_n1: pool_create");
    if (!pool) return;

    const size_t N = 256;
    uint8_t src_a[256], src_b[256], dst_a[256], dst_b[256];
    fill_rand(src_a, N, 0xAA);
    fill_rand(src_b, N, 0xBB);
    memset(dst_a, 0, N);
    memset(dst_b, 0xFF, N);  /* sentinel: b should not be touched */

    rc = sp_hedge_read_bulk(pool, src_a, dst_a, src_b, dst_b, N);
    SP_CHECK(rc == SP_OK, "bulk_n1: read_bulk SP_OK");
    SP_CHECK(memcmp(dst_a, src_a, N) == 0, "bulk_n1: dst_a == src_a");
    /* dst_b must be unchanged (all 0xFF) */
    uint8_t sentinel[256];
    memset(sentinel, 0xFF, N);
    SP_CHECK(memcmp(dst_b, sentinel, N) == 0, "bulk_n1: dst_b unchanged (b ignored)");

    sp_hedge_pool_destroy(pool);
}

int main(void) {
    SP_RUN(T_HEDGE_POOL_CREATE_DESTROY);
    SP_RUN(T_HEDGE_PAIR_1);
    SP_RUN(T_HEDGE_SPINOR_1);
    SP_RUN(T_HEDGE_N1_FALLBACK);
    SP_RUN(T_HEDGE_REPEATED);
    SP_RUN(T_HEDGE_BULK_SMALL);
    SP_RUN(T_HEDGE_BULK_LARGE);
    SP_RUN(T_HEDGE_BULK_N1_FALLBACK);
    return SP_DONE();
}
