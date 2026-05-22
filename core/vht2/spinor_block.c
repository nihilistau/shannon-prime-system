/* spinor_block.c — assembly of the frozen 63-byte Spinor block.
 *
 * encode: VHT2 project+quantize -> Mobius reorder of the int8 coeffs -> pack
 *         the header (explicit little-endian) and body + CRC-8 trailer.
 * decode: verify CRC-8 -> Mobius inverse -> de-quantize.
 * pack/unpack: exact bijective (de)serialization of the container to 63 bytes
 *         (header sub-fields are already a fixed byte image, so this is plain
 *         concatenation; the little-endian field assembly happens in encode).
 *
 * All header multi-byte fields are serialized by explicit byte shifts in
 * little-endian order — never memcpy of a wider type — so the byte image is
 * identical on every platform/ABI. Frozen v1; see include/sp/spinor_block.h.
 */
#include "sp/spinor_block.h"

#include <string.h>
#include <stdint.h>

/* Internal VHT2 helpers, defined in vht2.c (forward-declared identically there
 * — kept out of the public header on purpose; not API). */
void sp_vht2_project_quantize(const float *vec, int k,
                              int8_t coeff[SP_SPINOR_BODY_LEN],
                              float *out_scale, int8_t *out_exp);
void sp_vht2_dequantize(const int8_t coeff[SP_SPINOR_BODY_LEN],
                        float scale, int8_t expo,
                        float *vec, int k);

/* ---- header sub-field offsets (v1 frozen layout) ------------------------- */
#define SP_HDR_SCALE_OFF  0   /* float32 LE, bytes [0..3] */
#define SP_HDR_EXP_OFF    4   /* int8                     */
#define SP_HDR_BASIS_OFF  5   /* uint8                    */
#define SP_HDR_RSVD_OFF   6   /* uint8, must be 0 in v1   */

/* Write a float32 into dst[0..3] as IEEE-754 bits, little-endian, via bit
 * extraction (union) + explicit shifts. No memcpy of the float. */
static void put_f32_le(uint8_t *dst, float f) {
    union { float f; uint32_t u; } cv;
    cv.f = f;
    dst[0] = (uint8_t)(cv.u & 0xFFu);
    dst[1] = (uint8_t)((cv.u >> 8) & 0xFFu);
    dst[2] = (uint8_t)((cv.u >> 16) & 0xFFu);
    dst[3] = (uint8_t)((cv.u >> 24) & 0xFFu);
}

/* Read a little-endian float32 back from src[0..3] (inverse of put_f32_le). */
static float get_f32_le(const uint8_t *src) {
    union { float f; uint32_t u; } cv;
    cv.u =  (uint32_t)src[0]
         | ((uint32_t)src[1] << 8)
         | ((uint32_t)src[2] << 16)
         | ((uint32_t)src[3] << 24);
    return cv.f;
}

/* CRC-8 over vht2_header[0..6] || mobius_body[0..54] of a struct. */
static uint8_t block_crc(const sp_spinor_block_t *b) {
    uint8_t payload[62];
    memcpy(payload, b->vht2_header, 7);
    memcpy(payload + 7, b->mobius_body, 55);
    return sp_crc8(payload, 62);
}

void sp_spinor_encode(const float *vec, int k, sp_spinor_block_t *out) {
    int8_t coeff[SP_SPINOR_BODY_LEN];
    float scale;
    int8_t expo;
    sp_vht2_project_quantize(vec, k, coeff, &scale, &expo);

    /* Mobius-reorder the int8 coefficients via the int32 permutation, then
     * narrow back to int8 in the body. body[mobius(i)] = coeff[i]. */
    int32_t in32[SP_SPINOR_BODY_LEN], out32[SP_SPINOR_BODY_LEN];
    for (int i = 0; i < SP_SPINOR_BODY_LEN; i++) in32[i] = coeff[i];
    sp_mobius_reorder(in32, out32, SP_SPINOR_BODY_LEN);

    memset(out, 0, sizeof *out);
    put_f32_le(&out->vht2_header[SP_HDR_SCALE_OFF], scale);
    out->vht2_header[SP_HDR_EXP_OFF]   = (uint8_t)expo;
    out->vht2_header[SP_HDR_BASIS_OFF] = SP_VHT2_BASIS_CANONICAL;
    out->vht2_header[SP_HDR_RSVD_OFF]  = 0u;
    for (int i = 0; i < SP_SPINOR_BODY_LEN; i++) {
        out->mobius_body[i] = (uint8_t)(int8_t)out32[i];
    }
    out->checksum = block_crc(out);
}

int sp_spinor_decode(const sp_spinor_block_t *in, float *vec, int k) {
    if (block_crc(in) != in->checksum) return 1;   /* corruption */

    /* invert the Mobius reorder: coeff[i] = body[mobius(i)]. */
    int32_t in32[SP_SPINOR_BODY_LEN], out32[SP_SPINOR_BODY_LEN];
    for (int i = 0; i < SP_SPINOR_BODY_LEN; i++) {
        in32[i] = (int32_t)(int8_t)in->mobius_body[i];
    }
    sp_mobius_reorder_inv(in32, out32, SP_SPINOR_BODY_LEN);

    int8_t coeff[SP_SPINOR_BODY_LEN];
    for (int i = 0; i < SP_SPINOR_BODY_LEN; i++) coeff[i] = (int8_t)out32[i];

    float scale = get_f32_le(&in->vht2_header[SP_HDR_SCALE_OFF]);
    int8_t expo = (int8_t)in->vht2_header[SP_HDR_EXP_OFF];
    sp_vht2_dequantize(coeff, scale, expo, vec, k);
    return 0;
}

void sp_spinor_pack(const sp_spinor_block_t *blk, uint8_t out[63]) {
    /* Exact byte image: header || body || checksum. The container is all
     * uint8_t with no padding, so this is a deterministic concatenation. */
    memcpy(out, blk->vht2_header, 7);
    memcpy(out + 7, blk->mobius_body, 55);
    out[62] = blk->checksum;
}

int sp_spinor_unpack(const uint8_t in[63], sp_spinor_block_t *blk) {
    memcpy(blk->vht2_header, in, 7);
    memcpy(blk->mobius_body, in + 7, 55);
    blk->checksum = in[62];
    return 0;   /* pure inverse of pack; validation is sp_spinor_decode's job */
}

/* ── Multi-block KV head codec (frozen balanced split) ───────────────────────
 * nblk = ceil(k/55); the k coordinates are split into nblk contiguous chunks
 * sized base or base+1, with the first `extra` chunks taking the +1 (k=128,
 * nblk=3 -> 43/43/42). Pure integer split, no data dependence, so it is identical
 * on every backend. */
int sp_spinor_blocks_for(int k) {
    if (k <= SP_SPINOR_BODY_LEN) return 1;
    return (k + SP_SPINOR_BODY_LEN - 1) / SP_SPINOR_BODY_LEN;
}

void sp_spinor_encode_vec(const float *vec, int k, sp_spinor_block_t *blocks) {
    int nblk = sp_spinor_blocks_for(k);
    int base = k / nblk, extra = k % nblk, off = 0;
    for (int b = 0; b < nblk; b++) {
        int len = base + (b < extra ? 1 : 0);
        sp_spinor_encode(vec + off, len, &blocks[b]);
        off += len;
    }
}

int sp_spinor_decode_vec(const sp_spinor_block_t *blocks, int k, float *vec) {
    int nblk = sp_spinor_blocks_for(k);
    int base = k / nblk, extra = k % nblk, off = 0, rc = 0;
    for (int b = 0; b < nblk; b++) {
        int len = base + (b < extra ? 1 : 0);
        if (sp_spinor_decode(&blocks[b], vec + off, len)) rc = 1;
        off += len;
    }
    return rc;
}
