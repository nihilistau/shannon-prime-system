/* test_sp_hedge.c — §16.3 TS.HEDGE bitwise correctness tests.
 *
 * T_HEDGE_PAIR_1    sp_hedge_read_pair64 returns byte-identical to serial reads
 * T_HEDGE_REPLICA_1 sp_hedge_read64_replica returns a's value when a == b
 * T_HEDGE_BLOCK_1   sp_hedge_read_block matches memcpy for sizes 1..4096
 * T_HEDGE_SPINOR_1  sp_hedge_read_spinor returns byte-identical 63-byte block
 * T_HEDGE_DISABLED  correctness on plain malloc arenas (channel-independent)
 *
 * No channel map, no sp_alloc_channel_pair.  Correctness is independent of
 * channel topology: same primitives work in CI/DISABLED mode.
 */
#include "sp/sp_channel.h"
#include "sp/spinor_block.h"
#include "sp/sp_test.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void fill_rand(void *buf, size_t n, unsigned int seed)
{
    unsigned char *p = (unsigned char *)buf;
    for (size_t i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        p[i] = (unsigned char)(seed >> 16);
    }
}

static void T_HEDGE_PAIR_1(void)
{
    uint64_t va = 0xDEADBEEFCAFEBABEull;
    uint64_t vb = 0x0123456789ABCDEFull;
    uint64_t oa = 0, ob = 0;
    sp_hedge_read_pair64(&va, &vb, &oa, &ob);
    SP_CHECK_EQ_I64(oa, va, "pair64: out_a == src_a");
    SP_CHECK_EQ_I64(ob, vb, "pair64: out_b == src_b");

    /* Sweep 16 random pairs */
    uint64_t arr_a[16], arr_b[16];
    fill_rand(arr_a, sizeof arr_a, 0x1234);
    fill_rand(arr_b, sizeof arr_b, 0x5678);
    for (int i = 0; i < 16; i++) {
        uint64_t ra = 0, rb = 0;
        sp_hedge_read_pair64(&arr_a[i], &arr_b[i], &ra, &rb);
        SP_CHECK_EQ_I64(ra, arr_a[i], "pair64 sweep a");
        SP_CHECK_EQ_I64(rb, arr_b[i], "pair64 sweep b");
    }
}

static void T_HEDGE_REPLICA_1(void)
{
    uint64_t val = 0xFEDCBA9876543210ull;
    uint64_t out = 0;
    sp_hedge_read64_replica(&val, &val, &out);
    SP_CHECK_EQ_I64(out, val, "replica: out == src when a==b");

    uint64_t arr[32];
    fill_rand(arr, sizeof arr, 0xABCD);
    for (int i = 0; i < 32; i++) {
        uint64_t r = 0;
        sp_hedge_read64_replica(&arr[i], &arr[i], &r);
        SP_CHECK_EQ_I64(r, arr[i], "replica sweep");
    }
}

static void T_HEDGE_BLOCK_1(void)
{
    static const size_t sizes[] = {1, 7, 8, 63, 64, 65, 256, 4096};
    int n = (int)(sizeof sizes / sizeof sizes[0]);

    for (int s = 0; s < n; s++) {
        size_t sz = sizes[s];
        uint8_t *src_a = (uint8_t *)malloc(sz + 64);
        uint8_t *src_b = (uint8_t *)malloc(sz + 64);
        uint8_t *out_a = (uint8_t *)malloc(sz);
        uint8_t *out_b = (uint8_t *)malloc(sz);
        SP_CHECK(src_a && src_b && out_a && out_b, "block alloc");
        if (!src_a || !src_b || !out_a || !out_b) {
            free(src_a); free(src_b); free(out_a); free(out_b);
            continue;
        }
        fill_rand(src_a, sz, (unsigned int)(s * 31 + 17));
        fill_rand(src_b, sz, (unsigned int)(s * 31 + 53));
        memset(out_a, 0, sz);
        memset(out_b, 0, sz);

        sp_hedge_read_block(src_a, src_b, sz, out_a, out_b);

        SP_CHECK(memcmp(out_a, src_a, sz) == 0, "block: out_a == src_a");
        SP_CHECK(memcmp(out_b, src_b, sz) == 0, "block: out_b == src_b");

        free(src_a); free(src_b); free(out_a); free(out_b);
    }
}

static void T_HEDGE_SPINOR_1(void)
{
    sp_spinor_block_t sa, sb, so;
    fill_rand((uint8_t*)&sa, 63, 0xBEEF);
    memcpy(&sb, &sa, 63);   /* replica: sb == sa */
    memset(&so, 0, 63);

    sp_hedge_read_spinor(&sa, &sb, &so);
    SP_CHECK(memcmp(&so, &sa, 63) == 0, "spinor replica: out == a");

    /* Non-replica b: output must still equal a */
    fill_rand((uint8_t*)&sb, 63, 0xCAFE);
    memset(&so, 0, 63);
    sp_hedge_read_spinor(&sa, &sb, &so);
    SP_CHECK(memcmp(&so, &sa, 63) == 0, "spinor non-replica b: out still == a");
}

static void T_HEDGE_DISABLED(void)
{
    /* Plain malloc arenas — no channel map or privilege required.
     * Verifies correctness on the CI/DISABLED path. */
    uint64_t a64 = 0x1122334455667788ull;
    uint64_t b64 = 0xAABBCCDDEEFF0011ull;
    uint64_t oa = 0, ob = 0;

    sp_hedge_read_pair64(&a64, &b64, &oa, &ob);
    SP_CHECK_EQ_I64(oa, a64, "disabled: pair out_a");
    SP_CHECK_EQ_I64(ob, b64, "disabled: pair out_b");

    uint64_t r = 0;
    sp_hedge_read64_replica(&a64, &a64, &r);
    SP_CHECK_EQ_I64(r, a64, "disabled: replica");

    uint8_t ba[32], bb[32], oa2[32], ob2[32];
    fill_rand(ba, 32, 0x99);
    fill_rand(bb, 32, 0x77);
    sp_hedge_read_block(ba, bb, 32, oa2, ob2);
    SP_CHECK(memcmp(oa2, ba, 32) == 0, "disabled: block a");
    SP_CHECK(memcmp(ob2, bb, 32) == 0, "disabled: block b");

    sp_spinor_block_t sa, sb, so;
    fill_rand((uint8_t*)&sa, 63, 0x55);
    memcpy(&sb, &sa, 63);
    sp_hedge_read_spinor(&sa, &sb, &so);
    SP_CHECK(memcmp(&so, &sa, 63) == 0, "disabled: spinor");
}

int main(void)
{
    SP_RUN(T_HEDGE_PAIR_1);
    SP_RUN(T_HEDGE_REPLICA_1);
    SP_RUN(T_HEDGE_BLOCK_1);
    SP_RUN(T_HEDGE_SPINOR_1);
    SP_RUN(T_HEDGE_DISABLED);
    return SP_DONE();
}
