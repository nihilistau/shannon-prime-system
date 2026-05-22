/* frobenius_lift.c — Q8 weight storage: per-row-scaled int8 packing with an
 * inline dequant ("Frobenius lift").  Phase 1E.  See sp/frobenius_lift.h for
 * the full contract and the rounding-rule / error-bound documentation.
 */
#include "sp/frobenius_lift.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

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

/* ── Q4 (4-bit) variant ──────────────────────────────────────────────────── */

int8_t sp_frob_quant1_q4(float v, float scale) {
    if (scale == 0.0f) return 0;
    float x = v / scale * 7.0f;
    float r = (x >= 0.0f) ? floorf(x + 0.5f) : ceilf(x - 0.5f);   /* round half away */
    if (r >  (float)SP_FROB_QMAX4) r =  (float)SP_FROB_QMAX4;
    if (r < -(float)SP_FROB_QMAX4) r = -(float)SP_FROB_QMAX4;
    return (int8_t)r;
}

float sp_frob_dequant1_q4(int8_t q, float scale) {
    return (float)q * (scale / 7.0f);
}

void sp_frob_q4_pack(const int8_t *codes, int n, uint8_t *nib) {
    for (int i = 0; i < n; i += 2) {
        uint8_t lo = (uint8_t)(codes[i] & 0xF);
        uint8_t hi = (i + 1 < n) ? (uint8_t)(codes[i + 1] & 0xF) : 0u;
        nib[i >> 1] = (uint8_t)(lo | (hi << 4));
    }
}

void sp_frob_q4_unpack(const uint8_t *nib, int n, int8_t *codes) {
    for (int i = 0; i < n; i++) {
        uint8_t v = (i & 1) ? (uint8_t)(nib[i >> 1] >> 4) : (uint8_t)(nib[i >> 1] & 0xF);
        codes[i] = (int8_t)((v & 0x8) ? (int)v - 16 : (int)v);   /* sign-extend 4-bit */
    }
}

float sp_frob_q4_row_relerr(const float *row, int cols) {
    if (cols <= 0) return 0.0f;
    float s = sp_frob_row_scale(row, cols);
    if (s == 0.0f) return 0.0f;
    double e2 = 0.0, n2 = 0.0;
    for (size_t c = 0; c < (size_t)cols; c++) {
        int8_t q = sp_frob_quant1_q4(row[c], s);
        double d = (double)row[c] - (double)sp_frob_dequant1_q4(q, s);
        e2 += d * d;
        n2 += (double)row[c] * (double)row[c];
    }
    return (n2 > 0.0) ? (float)sqrt(e2 / n2) : 0.0f;
}

size_t sp_frob_q4_packed_bytes(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return 0;
    return (size_t)rows * (size_t)((cols + 1) / 2) + (size_t)rows * sizeof(float);
}

/* ── Mixed-precision packed tensor (arena layout) ────────────────────────────
 * Per-row: pick the Frobenius scale, optionally promote a Q4 row to Q8 by its
 * round-trip rel-error, quantize, and append the codes at a per-row byte offset.
 * Mirrors the inline dequant exactly (code * scale / qmax) so the lift is a true
 * inverse up to the quantization step. */
void sp_frob_packed_free(sp_frob_packed_tensor *t) {
    if (!t) return;
    free(t->row_prec); free(t->row_scale); free(t->row_off); free(t->codes);
    memset(t, 0, sizeof *t);
}

int sp_frob_pack_tensor(int rows, int cols, int precision, float promote,
                        sp_frob_row_fn get_row, void *ctx,
                        sp_frob_packed_tensor *out, long *promoted) {
    if (!out) return 1;
    memset(out, 0, sizeof *out);
    if (!get_row || rows <= 0 || cols <= 0 || (precision != 8 && precision != 4))
        return 1;

    float  *wrow = (float *)malloc((size_t)cols * sizeof(float));
    int8_t *tmp  = (int8_t *)malloc((size_t)cols);                  /* Q4 pack scratch */
    out->row_prec  = (uint8_t *)malloc((size_t)rows);
    out->row_scale = (float *)malloc((size_t)rows * sizeof(float));
    out->row_off   = (size_t *)malloc((size_t)rows * sizeof(size_t));
    out->codes     = (uint8_t *)malloc((size_t)rows * (size_t)cols);   /* upper bound (all Q8) */
    if (!wrow || !tmp || !out->row_prec || !out->row_scale || !out->row_off || !out->codes) {
        free(wrow); free(tmp); sp_frob_packed_free(out); return 1;
    }
    out->rows = rows; out->cols = cols;

    size_t off = 0;
    int rc = 0;
    for (int j = 0; j < rows; j++) {
        if (get_row(ctx, j, wrow)) { rc = 1; break; }
        float s = sp_frob_row_scale(wrow, cols);
        int p = precision;
        if (precision == 4 && sp_frob_q4_row_relerr(wrow, cols) > promote) {
            p = 8; if (promoted) (*promoted)++;
        }
        out->row_prec[j]  = (uint8_t)p;
        out->row_scale[j] = s;
        out->row_off[j]   = off;
        uint8_t *dst = out->codes + off;
        if (p == 8) {
            int8_t *q = (int8_t *)dst;
            for (int i = 0; i < cols; i++) q[i] = sp_frob_quant1(wrow[i], s);
            off += (size_t)cols;
        } else {
            for (int i = 0; i < cols; i++) tmp[i] = sp_frob_quant1_q4(wrow[i], s);
            sp_frob_q4_pack(tmp, cols, dst);
            off += (size_t)((cols + 1) / 2);
        }
    }
    free(wrow); free(tmp);
    if (rc) { sp_frob_packed_free(out); return 1; }

    out->codes_bytes = off;
    uint8_t *shr = (uint8_t *)realloc(out->codes, off ? off : 1);   /* shrink to actual */
    if (shr) out->codes = shr;
    return 0;
}

int sp_frob_packed_dequant_row(const sp_frob_packed_tensor *t, int r, float *dst) {
    if (!t || !dst || r < 0 || r >= t->rows) return 1;
    const uint8_t *rc = t->codes + t->row_off[r];
    if (t->row_prec[r] == 8) {
        const int8_t *cp = (const int8_t *)rc;
        float inv = t->row_scale[r] / (float)SP_FROB_QMAX;
        for (int i = 0; i < t->cols; i++) dst[i] = (float)cp[i] * inv;
    } else {
        float inv = t->row_scale[r] / (float)SP_FROB_QMAX4;
        for (int i = 0; i < t->cols; i++) {
            uint8_t b = (i & 1) ? (uint8_t)(rc[i >> 1] >> 4) : (uint8_t)(rc[i >> 1] & 0xF);
            int8_t v = (int8_t)((b & 0x8) ? (int)b - 16 : (int)b);   /* sign-extend 4-bit */
            dst[i] = (float)v * inv;
        }
    }
    return 0;
}

size_t sp_frob_packed_tensor_bytes(const sp_frob_packed_tensor *t) {
    if (!t) return 0;
    return t->codes_bytes + (size_t)t->rows * (sizeof(float) + sizeof(size_t) + 1);
}
