/* spinor_test.c — contract tests for the frozen 63-byte Spinor block.
 *
 * Phase 1D, roadmap-named cases T_VHT_1 .. T_VHT_6. One executable; each
 * SP_RUN names the exact contract item in ctest output.
 *
 * TDD: written before the implementation. Build watches these fail, then
 * spinor_block.c / vht2.c / mobius_reorder.c make them green.
 */
#include "sp/sp_test.h"
#include "sp/spinor_block.h"

#include <stdint.h>
#include <string.h>
#include <math.h>

/* ---- fixed-seed RNG (xorshift32), so every run is byte-reproducible ------ */
static uint32_t g_rng;
static void rng_seed(uint32_t s) { g_rng = s ? s : 0xDEADBEEFu; }
static uint32_t rng_u32(void) {
    uint32_t x = g_rng;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    g_rng = x;
    return x;
}

/* Fill a struct's bytes with random data via its 63-byte image (exercises the
 * full container, including header sub-fields, with arbitrary bit patterns). */
static void rand_block(sp_spinor_block_t *b) {
    uint8_t img[63];
    for (int i = 0; i < 63; i++) img[i] = (uint8_t)(rng_u32() & 0xFFu);
    (void)sp_spinor_unpack(img, b);
}

static int bytes_eq(const uint8_t *a, const uint8_t *b, int n) {
    return memcmp(a, b, (size_t)n) == 0;
}

/* T_VHT_1 — container (de)serialization is exactly bijective:
 * for 65,536 random blocks, pack -> unpack -> pack is byte-identical. */
static void T_VHT_1(void) {
    rng_seed(0x5D1A0B01u);
    int all_ok = 1;
    for (int t = 0; t < 65536; t++) {
        sp_spinor_block_t b;
        rand_block(&b);
        uint8_t img1[63], img2[63];
        sp_spinor_pack(&b, img1);
        sp_spinor_block_t b2;
        (void)sp_spinor_unpack(img1, &b2);
        sp_spinor_pack(&b2, img2);
        if (!bytes_eq(img1, img2, 63)) { all_ok = 0; break; }
    }
    SP_CHECK(all_ok, "pack->unpack->pack byte-identical over 65536 blocks");
}

/* T_VHT_2 — Mobius reorder is a true bijection: every output position is hit
 * exactly once (no collisions), and reorder then inverse is the identity. */
static void T_VHT_2(void) {
    int collisions_ok = 1, identity_ok = 1;
    const int sizes[] = { 1, 2, 7, 11, 55, 64, 128 };
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        int n = sizes[s];
        int32_t in[128], out[128], back[128];
        for (int i = 0; i < n; i++) in[i] = i * 1000 + 7; /* distinct payloads */

        sp_mobius_reorder(in, out, n);

        /* every output slot written exactly once: out is a permutation of in */
        int seen[200];
        for (int i = 0; i < n; i++) seen[i] = 0;
        for (int i = 0; i < n; i++) {
            int v = (out[i] - 7) / 1000;   /* recover original index */
            if (v < 0 || v >= n || seen[v]) { collisions_ok = 0; break; }
            seen[v] = 1;
        }

        sp_mobius_reorder_inv(out, back, n);
        for (int i = 0; i < n; i++) if (back[i] != in[i]) { identity_ok = 0; break; }
    }
    SP_CHECK(collisions_ok, "Mobius reorder hits every output position once");
    SP_CHECK(identity_ok, "Mobius reorder then inverse is the identity");
}

/* T_VHT_3 — frozen container size. */
static void T_VHT_3(void) {
    SP_CHECK_EQ_I64(sizeof(sp_spinor_block_t), 63, "sizeof spinor block == 63");
}

/* T_VHT_4 — CRC-8 detects any single-bit corruption in header||body. */
static void T_VHT_4(void) {
    rng_seed(0xC0FFEE11u);
    int all_detected = 1;
    for (int t = 0; t < 4096 && all_detected; t++) {
        sp_spinor_block_t b;
        rand_block(&b);
        uint8_t payload[62];
        memcpy(payload, b.vht2_header, 7);
        memcpy(payload + 7, b.mobius_body, 55);
        uint8_t good = sp_crc8(payload, 62);
        /* sweep every one of the 62*8 single-bit flips */
        for (int bit = 0; bit < 62 * 8 && all_detected; bit++) {
            uint8_t corrupt[62];
            memcpy(corrupt, payload, 62);
            corrupt[bit >> 3] ^= (uint8_t)(1u << (bit & 7));
            if (sp_crc8(corrupt, 62) == good) all_detected = 0;
        }
    }
    SP_CHECK(all_detected, "CRC-8 differs for every single-bit flip");
}

/* Build a header byte image with explicit little-endian field placement, the
 * canonical v1 layout, so we can assert the implementation matches bit-exact. */
static void expect_header(uint8_t hdr[7], float scale, int8_t expo,
                          uint8_t basis_sel, uint8_t reserved) {
    union { float f; uint32_t u; } cv;
    cv.f = scale;
    hdr[0] = (uint8_t)(cv.u & 0xFFu);
    hdr[1] = (uint8_t)((cv.u >> 8) & 0xFFu);
    hdr[2] = (uint8_t)((cv.u >> 16) & 0xFFu);
    hdr[3] = (uint8_t)((cv.u >> 24) & 0xFFu);
    hdr[4] = (uint8_t)expo;
    hdr[5] = basis_sel;
    hdr[6] = reserved;
}

/* T_VHT_5 — byte determinism + explicit little-endian header serialization. */
static void T_VHT_5(void) {
    /* (a) same input vector encodes to identical bytes, twice. */
    float vec[40];
    rng_seed(0xABCD1234u);
    for (int i = 0; i < 40; i++) {
        /* deterministic floats in roughly [-2,2] */
        vec[i] = (float)((int32_t)(rng_u32() & 0xFFFFu) - 32768) / 16384.0f;
    }
    sp_spinor_block_t e1, e2;
    sp_spinor_encode(vec, 40, &e1);
    sp_spinor_encode(vec, 40, &e2);
    uint8_t i1[63], i2[63];
    sp_spinor_pack(&e1, i1);
    sp_spinor_pack(&e2, i2);
    SP_CHECK(bytes_eq(i1, i2, 63), "same input vector -> identical 63 bytes");

    /* (b) known-field-value fixture: the packed header is exactly the explicit
     * little-endian byte image (cross-platform-identity anchor). We construct a
     * block by hand, pack it, and compare the first 7 bytes to expect_header. */
    sp_spinor_block_t k;
    memset(&k, 0, sizeof k);
    float scale = 0.15625f;          /* exactly representable: 5 * 2^-5 */
    expect_header(k.vht2_header, scale, (int8_t)-3,
                  SP_VHT2_BASIS_CANONICAL, 0u);
    for (int i = 0; i < 55; i++) k.mobius_body[i] = (uint8_t)(i * 3 + 1);
    {
        uint8_t payload[62];
        memcpy(payload, k.vht2_header, 7);
        memcpy(payload + 7, k.mobius_body, 55);
        k.checksum = sp_crc8(payload, 62);
    }
    uint8_t img[63];
    sp_spinor_pack(&k, img);

    uint8_t want_hdr[7];
    expect_header(want_hdr, scale, (int8_t)-3, SP_VHT2_BASIS_CANONICAL, 0u);
    SP_CHECK(bytes_eq(img, want_hdr, 7),
             "header packs as explicit little-endian image");
    /* spot-check the LE float byte order directly: 0.15625f = 0x3E200000 */
    SP_CHECK(img[0] == 0x00 && img[1] == 0x00 && img[2] == 0x20 && img[3] == 0x3E,
             "float32 scale serialized little-endian (0x3E200000)");
    SP_CHECK(img[4] == (uint8_t)(int8_t)-3, "exponent byte = int8 two's complement");
    SP_CHECK(img[5] == SP_VHT2_BASIS_CANONICAL, "basis selector byte");
    SP_CHECK(img[6] == 0u, "reserved byte is 0 in v1");

    /* (c) encode/decode round-trips approximately (lossy quantization). */
    float dec[40];
    int rc = sp_spinor_decode(&e1, dec, 40);
    SP_CHECK(rc == 0, "decode of valid block returns 0");
    int approx_ok = 1;
    for (int i = 0; i < 40; i++) {
        if (fabsf(dec[i] - vec[i]) > 0.05f) { approx_ok = 0; break; }
    }
    SP_CHECK(approx_ok, "encode->decode within int8 quantization tolerance");

    /* (d) decode rejects a corrupted block (checksum mismatch -> nonzero). */
    sp_spinor_block_t bad = e1;
    bad.mobius_body[0] ^= 0xFFu;
    SP_CHECK(sp_spinor_decode(&bad, dec, 40) != 0,
             "decode of corrupted block returns nonzero");
}

/* T_VHT_6 — frozen version guard.
 *
 * FREEZE NOTE: SP_SPINOR_LAYOUT_VERSION is 1. Any move/resize/reorder of a
 * field in vht2_header or mobius_body, any change to the CRC-8 polynomial, the
 * Mobius permutation, or the quantization convention, REQUIRES bumping this
 * constant AND adding a migration note. Bumping the constant here forces this
 * test (and any reader) to re-confirm the on-wire format intentionally changed.
 */
static void T_VHT_6(void) {
    SP_CHECK_EQ_I64(SP_SPINOR_LAYOUT_VERSION, 1, "frozen layout version == 1");
}

int main(void) {
    SP_RUN(T_VHT_1);
    SP_RUN(T_VHT_2);
    SP_RUN(T_VHT_3);
    SP_RUN(T_VHT_4);
    SP_RUN(T_VHT_5);
    SP_RUN(T_VHT_6);
    return SP_DONE();
}
