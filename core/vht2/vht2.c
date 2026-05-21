/* vht2.c — VHT2 lattice projection / quantization + the CRC-8 trailer.
 *
 * VHT2 (v1, basis = CANONICAL): anchor i is canonical coordinate i. The
 * projection of a length-k float vector onto 55 anchors is the identity on
 * coordinates (zero-padding for k<55, truncation for k>55), followed by a
 * uniform per-block scale + symmetric int8 quantization.
 *
 * CRC-8/SMBus: poly 0x07, init 0x00, no input/output reflection, xorout 0.
 * Detects all single-bit errors over the 62-byte payload (T_VHT_4).
 *
 * These primitives are frozen v1: changing the polynomial or the quantization
 * convention requires bumping SP_SPINOR_LAYOUT_VERSION.
 */
#include "sp/spinor_block.h"

#include <math.h>
#include <stdint.h>

/* Internal VHT2 helpers shared with spinor_block.c. Declared here (and matched
 * by an identical forward declaration in spinor_block.c) rather than in a
 * separate header, to keep this module's file set to exactly the owned files.
 * Not part of the public API. */
void sp_vht2_project_quantize(const float *vec, int k,
                              int8_t coeff[SP_SPINOR_BODY_LEN],
                              float *out_scale, int8_t *out_exp);
void sp_vht2_dequantize(const int8_t coeff[SP_SPINOR_BODY_LEN],
                        float scale, int8_t expo,
                        float *vec, int k);

uint8_t sp_crc8(const uint8_t *data, int len) {
    uint8_t crc = 0x00u;                      /* init */
    for (int i = 0; i < len; i++) {
        crc ^= data[i];
        for (int b = 0; b < 8; b++) {
            if (crc & 0x80u) {
                crc = (uint8_t)((crc << 1) ^ 0x07u);  /* poly 0x07 */
            } else {
                crc = (uint8_t)(crc << 1);
            }
        }
    }
    return crc;                               /* xorout 0, no reflection */
}

/* Round half away from zero, deterministically (no dependence on the FE
 * rounding mode, unlike lrint/nearbyint). */
static int32_t round_half_away(float x) {
    float r = (x >= 0.0f) ? floorf(x + 0.5f) : ceilf(x - 0.5f);
    return (int32_t)r;
}

/* Symmetric saturation into the int8 range used for anchor coefficients.
 * We use [-127, 127] (not -128) so the magnitude range is symmetric, matching
 * the q/127 de-quantization. */
static int8_t sat_q7(int32_t v) {
    if (v >  127) return  127;
    if (v < -127) return -127;
    return (int8_t)v;
}

void sp_vht2_project_quantize(const float *vec, int k,
                              int8_t coeff[SP_SPINOR_BODY_LEN],
                              float *out_scale, int8_t *out_exp) {
    /* scale = max |vec[i]| over the represented coordinates; 0 -> 1 to avoid
     * division by zero (an all-zero vector quantizes to all-zero coeffs). */
    float maxabs = 0.0f;
    int lim = (k < SP_SPINOR_BODY_LEN) ? k : SP_SPINOR_BODY_LEN;
    if (lim < 0) lim = 0;
    for (int i = 0; i < lim; i++) {
        float a = fabsf(vec[i]);
        if (a > maxabs) maxabs = a;
    }
    float scale = (maxabs > 0.0f) ? maxabs : 1.0f;
    int8_t expo = 0;                          /* v1: exponent reserved, fixed 0 */

    float denom = scale; /* * 2^expo, expo==0 */
    for (int i = 0; i < SP_SPINOR_BODY_LEN; i++) {
        if (i < lim) {
            float norm = vec[i] / denom;      /* in [-1,1] */
            coeff[i] = sat_q7(round_half_away(norm * 127.0f));
        } else {
            coeff[i] = 0;                      /* zero-pad k<55 */
        }
    }
    *out_scale = scale;
    *out_exp = expo;
}

void sp_vht2_dequantize(const int8_t coeff[SP_SPINOR_BODY_LEN],
                        float scale, int8_t expo,
                        float *vec, int k) {
    float mul = scale * ldexpf(1.0f, (int)expo) / 127.0f;
    int lim = (k < SP_SPINOR_BODY_LEN) ? k : SP_SPINOR_BODY_LEN;
    if (lim < 0) lim = 0;
    for (int i = 0; i < lim; i++) {
        vec[i] = (float)coeff[i] * mul;
    }
    /* coordinates [55,k) (k>55) are unrecoverable (truncated at encode): 0. */
    for (int i = SP_SPINOR_BODY_LEN; i < k; i++) vec[i] = 0.0f;
}
