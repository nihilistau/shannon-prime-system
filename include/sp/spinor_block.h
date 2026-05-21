/* sp/spinor_block.h — VHT2 + Mobius-reorder + 63-byte Spinor block.
 * (Phase 1D of the math core.)
 *
 * This is the FROZEN on-disk / on-wire KV-cache record format. A K vector is
 *   (1) projected onto fixed lattice anchors (VHT2) and quantized to small
 *       integers per anchor (int8);
 *   (2) the per-anchor coefficients are Mobius-reordered — a fixed bijective
 *       permutation of coefficient positions;
 *   (3) packed into mobius_body[55], with vht2_header[7] holding the scalar
 *       metadata (a norm/scale + an exponent + a basis selector) and a CRC-8
 *       trailer over header||body.
 *
 * Part of the shared public API root (include/). Include as
 *   #include "sp/spinor_block.h"
 *
 * ============================================================================
 *  FROZEN v1 LAYOUT — DO NOT MODIFY WITHOUT BUMPING SP_SPINOR_LAYOUT_VERSION.
 * ============================================================================
 * The 63-byte container, the 7/55/1 split, SP_SPINOR_LAYOUT_VERSION, and the
 * "CRC-8 of (header||body)" trailer are FROZEN per roadmap S4.5/S7.9.
 *
 * Any field move, resize, reorder, or semantic change — including changing the
 * CRC polynomial, the Mobius permutation, the header sub-field bit assignment,
 * or the quantization convention — REQUIRES bumping SP_SPINOR_LAYOUT_VERSION
 * and writing a migration note. T_VHT_6 guards the version constant; T_VHT_3
 * guards the size; T_VHT_5 guards the exact byte image of the header.
 *
 * ---------------------------------------------------------------------------
 *  vht2_header[7] sub-field layout (v1) — all multi-byte fields LITTLE-ENDIAN,
 *  serialized by explicit byte shifts (never memcpy of a wider type), so the
 *  byte image is identical on every platform / ABI:
 *
 *    byte [0..3]  scale       float32, IEEE-754 bits in little-endian order.
 *                             The de-quantization magnitude: a decoded anchor
 *                             coefficient q (int8) maps back to a float as
 *                             q * scale * 2^exponent.
 *    byte [4]     exponent    int8 (two's complement), the binary power-of-two
 *                             shift applied on top of `scale`. Together
 *                             (scale, exponent) form the "norm/scale and an
 *                             exponent" pair of the freeze spec.
 *    byte [5]     basis_sel   uint8, VHT2 basis selector. v1 defines exactly
 *                             one basis: SP_VHT2_BASIS_CANONICAL == 0
 *                             (anchor i = canonical coordinate i). Other values
 *                             are reserved for future bases (version-gated).
 *    byte [6]     reserved    uint8, MUST be 0 in v1. Reserved for flags.
 *
 *  mobius_body[55]  one int8 (two's complement) quantized anchor coefficient
 *                   per byte, stored at Mobius-permuted positions:
 *                   body[ sp_mobius(i) ] = quant(anchor_coeff[i]).
 *
 *  checksum         CRC-8 over the 62 bytes vht2_header[0..6] || mobius_body
 *                   (in that on-wire order). Polynomial: CRC-8/SMBus —
 *                   poly 0x07, init 0x00, no input/output reflection, xorout 0.
 * ---------------------------------------------------------------------------
 *
 *  Mobius permutation (v1): affine bijection on Z/nZ,
 *      sp_mobius(i)     = (SP_MOBIUS_A * i)            mod n
 *      sp_mobius_inv(j) = (a_inv * j)                  mod n,  a_inv*A == 1 (mod n)
 *  with SP_MOBIUS_A = 17. 17 is coprime to 55 (=5*11) and to every n the body
 *  uses, so the map is a bijection for any n>=1 (a_inv computed per n via the
 *  extended Euclidean algorithm). Pure index permutation, no data dependence.
 *
 *  VHT2 projection (v1, basis = CANONICAL): anchor i is canonical coordinate i.
 *  Encoding of a length-k float vector into 55 anchors:
 *    - scale = max_i |vec[i]| over i in [0,k) (0 -> treated as 1 to avoid /0);
 *      exponent = 0 in v1 (reserved hook for block float-exponent extraction).
 *    - anchor_coeff[i] = round_half_away( vec[i] / (scale*2^exp) * 127 ),
 *      saturated to [-127, 127]; for i in [k,55) (k<55) the anchor is 0
 *      (zero-pad). For k>55 the tail is truncated (lossy). The /127 maps the
 *      normalized [-1,1] range onto the int8 magnitude range.
 *    Decoding inverts: vec[i] = q[i]/127 * scale * 2^exp.
 */
#ifndef SP_SPINOR_BLOCK_H
#define SP_SPINOR_BLOCK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SP_SPINOR_LAYOUT_VERSION 1u

/* VHT2 basis selectors (vht2_header byte [5]). v1 defines only CANONICAL. */
#define SP_VHT2_BASIS_CANONICAL 0u

/* Mobius affine multiplier (see header doc). Coprime to 55 = 5*11. */
#define SP_MOBIUS_A 17

/* Number of int8 anchor coefficients carried by mobius_body. */
#define SP_SPINOR_BODY_LEN 55

typedef struct {              /* 63 bytes total, packed; all uint8_t => no padding on any ABI */
    uint8_t vht2_header[7];   /* norm + exponent + VHT2 basis selector (sub-fields above) */
    uint8_t mobius_body[55];  /* Mobius-reordered, packed anchor coefficients */
    uint8_t checksum;         /* CRC-8 of vht2_header || mobius_body */
} sp_spinor_block_t;

_Static_assert(sizeof(sp_spinor_block_t) == 63, "frozen Spinor layout");

/* CRC-8/SMBus over `len` bytes: poly 0x07, init 0x00, no reflection, xorout 0. */
uint8_t sp_crc8(const uint8_t *data, int len);

/* Mobius reorder (a fixed bijective permutation) and its inverse, over the
 * index range [0,n). Both are pure index bijections: out[sp_mobius(i)] = in[i].
 * sp_mobius_reorder writes out[sp_mobius(i)] = in[i]; the inverse undoes it. */
void sp_mobius_reorder(const int32_t *in, int32_t *out, int n);
void sp_mobius_reorder_inv(const int32_t *in, int32_t *out, int n);

/* VHT2 project -> quantize -> Mobius reorder -> pack + checksum.
 * `vec` has length `k` (>=0). See header doc for the k<55 / k>55 handling. */
void sp_spinor_encode(const float *vec, int k, sp_spinor_block_t *out);

/* Inverse of sp_spinor_encode (lossy). Writes up to `k` floats into `vec`.
 * Returns 0 on success, nonzero if the block's CRC-8 does not verify. */
int sp_spinor_decode(const sp_spinor_block_t *in, float *vec, int k);

/* Exact, bijective container (de)serialization (this is what T_VHT_1
 * round-trips). pack writes the 63 on-wire bytes; unpack reconstructs the
 * struct. unpack returns 0 always (no validation; it is the pure inverse of
 * pack — checksum validation is sp_spinor_decode's job). */
void sp_spinor_pack(const sp_spinor_block_t *blk, uint8_t out[63]);
int  sp_spinor_unpack(const uint8_t in[63], sp_spinor_block_t *blk);

#ifdef __cplusplus
}
#endif

#endif /* SP_SPINOR_BLOCK_H */
