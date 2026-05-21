/* frobenius_lift.c — Q8 weight storage: per-row-scaled int8 packing with an
 * inline dequant ("Frobenius lift").  Phase 1E.  See sp/frobenius_lift.h for
 * the full contract and the rounding-rule / error-bound documentation.
 */
#include "sp/frobenius_lift.h"

#include <math.h>
#include <stddef.h>

/* Per-row Frobenius scale: max_c |row[c]|.  0 for an all-zero or empty row. */
float sp_frob_row_scale(const float *row, int cols) {
    float m = 0.0f;
    if (cols <= 0) return 0.0f;
    for (size_t c = 0; c < (size_t)cols; c++) {
        float a = fabsf(row[c]);
        if (a > m) m = a;
    }
    return m;
}

/* Ceiling-shift (round-half-away-from-zero) quantiser, clamped to
 * [-SP_FROB_QMAX, +SP_FROB_QMAX].  Deterministic; uses floorf/ceilf so the
 * result never depends on the hardware FP rounding mode.  s == 0 -> code 0. */
int8_t sp_frob_quant1(float v, float scale) {
    if (scale == 0.0f) return 0;
    float x = v / scale * 127.0f;
    /* round half away from zero: +0.5 -> +1, -0.5 -> -1 */
    float r = (x >= 0.0f) ? floorf(x + 0.5f) : ceilf(x - 0.5f);
    if (r >  (float)SP_FROB_QMAX) r =  (float)SP_FROB_QMAX;
    if (r < -(float)SP_FROB_QMAX) r = -(float)SP_FROB_QMAX;
    return (int8_t)r;
}

/* Inline lift for one code: v_hat = q * (s / 127). */
float sp_frob_dequant1(int8_t q, float scale) {
    return (float)q * (scale / 127.0f);
}

void sp_frob_quantize(const float *w, int rows, int cols,
                      int8_t *packed, float *row_scale) {
    if (rows <= 0 || cols <= 0) return;
    for (size_t r = 0; r < (size_t)rows; r++) {
        const float *row = w + r * (size_t)cols;
        float s = sp_frob_row_scale(row, cols);
        row_scale[r] = s;
        int8_t *prow = packed + r * (size_t)cols;
        for (size_t c = 0; c < (size_t)cols; c++)
            prow[c] = sp_frob_quant1(row[c], s);
    }
}

void sp_frob_dequantize(const int8_t *packed, const float *row_scale,
                        int rows, int cols, float *out) {
    if (rows <= 0 || cols <= 0) return;
    for (size_t r = 0; r < (size_t)rows; r++) {
        float s = row_scale[r];
        const int8_t *prow = packed + r * (size_t)cols;
        float *orow = out + r * (size_t)cols;
        for (size_t c = 0; c < (size_t)cols; c++)
            orow[c] = sp_frob_dequant1(prow[c], s);
    }
}

size_t sp_frob_packed_bytes(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return 0;
    return (size_t)rows * (size_t)cols + (size_t)rows * sizeof(float);
}

double sp_frob_ratio(int rows, int cols, int src_dtype_bytes) {
    size_t packed = sp_frob_packed_bytes(rows, cols);
    if (packed == 0) return 0.0;
    double src = (double)src_dtype_bytes * (double)rows * (double)cols;
    return src / (double)packed;
}
