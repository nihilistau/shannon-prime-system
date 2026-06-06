/* weight_dtype.c -- portable reference; see sp/weight_dtype.h. The F16<->F32 bit
 * conversions and the GGUF per-row dequant (F32 / F16 / Q8_0), lifted verbatim (in
 * arithmetic) out of the engine's forward/model.c head. No engine deps -- the dtype
 * tags are mirrored locally as sp_weight_dtype. The engine's matmul/embed/arena (and
 * the k-quant cases) remain behind until the model-representation layer migrates. */
#include "sp/weight_dtype.h"
#include <string.h>

float sp_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) { f = sign; }                 /* +/-0 */
        else {                                      /* subnormal -> normalize */
            exp = 127u - 15u + 1u;
            while ((man & 0x400u) == 0) { man <<= 1; exp--; }
            man &= 0x3FFu;
            f = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1Fu) {                       /* inf / nan */
        f = sign | 0x7F800000u | (man << 13);
    } else {                                         /* normal */
        f = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    float out; memcpy(&out, &f, 4); return out;
}

uint16_t sp_f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t e8   = (x >> 23) & 0xFFu;
    uint32_t man  = x & 0x7FFFFFu;
    if (e8 == 0xFFu)                                /* inf / nan */
        return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u));
    int32_t exp = (int32_t)e8 - 127 + 15;           /* rebias to half */
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);   /* overflow -> inf */
    if (exp <= 0) {                                 /* subnormal / underflow */
        if (exp < -10) return (uint16_t)sign;       /* magnitude too small -> 0 */
        man |= 0x800000u;                           /* restore implicit 1 */
        int shift = 14 - exp;                       /* shift in [14,24] */
        uint32_t half = man >> shift;
        uint32_t rem  = man & ((1u << shift) - 1u);
        uint32_t mid  = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1u))) half++;  /* nearest-even */
        return (uint16_t)(sign | half);
    }
    uint16_t h   = (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
    uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;  /* carry ripples into exp */
    return h;
}

/* ggml get_scale_min_k4: unpack the 6-bit scale/min from the packed scales[12]. */
static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4));
    }
}

int sp_dequant_row(const void *src, uint32_t type, int n, float *dst) {
    if (n < 0) return 1;
    switch (type) {
        case SP_WDT_F32:
            memcpy(dst, src, (size_t)n * sizeof(float));
            return 0;
        case SP_WDT_F16: {
            const uint16_t *h = (const uint16_t *)src;
            for (int i = 0; i < n; i++) dst[i] = sp_f16_to_f32(h[i]);
            return 0;
        }
        case SP_WDT_Q4_0: {
            /* block_q4_0 = { f16 d; u8 qs[16]; } = 18 bytes / 32 elems.
             * GGML layout: byte j holds elem j (LOW nibble) and elem j+16
             * (HIGH nibble); value = (nib - 8) * d. The QAT-Q4_0 release
             * (gemma-4-12B) carries ALL matmuls in this type. */
            if (n % 32 != 0) return 1;
            const uint8_t *p = (const uint8_t *)src;
            int nb = n / 32;
            for (int b = 0; b < nb; b++) {
                uint16_t d16; memcpy(&d16, p, 2);
                float d = sp_f16_to_f32(d16);
                const uint8_t *qs = p + 2;
                float *y = dst + (size_t)b * 32;
                for (int j = 0; j < 16; j++) {
                    y[j]      = (float)((int)(qs[j] & 0xF) - 8) * d;
                    y[j + 16] = (float)((int)(qs[j] >>  4) - 8) * d;
                }
                p += 18;
            }
            return 0;
        }
        case SP_WDT_Q8_0: {
            /* block_q8_0 = { f16 d; int8 qs[32]; } = 34 bytes / 32 elems */
            if (n % 32 != 0) return 1;
            const uint8_t *p = (const uint8_t *)src;
            int nb = n / 32;
            for (int b = 0; b < nb; b++) {
                uint16_t d16; memcpy(&d16, p, 2);
                float d = sp_f16_to_f32(d16);
                const int8_t *qs = (const int8_t *)(p + 2);
                for (int i = 0; i < 32; i++) dst[b * 32 + i] = (float)qs[i] * d;
                p += 34;
            }
            return 0;
        }
        case SP_WDT_Q4_K: {
            /* block_q4_K = { f16 d; f16 dmin; u8 scales[12]; u8 qs[128]; } = 144 B / 256 elems */
            if (n % 256 != 0) return 1;
            const uint8_t *p = (const uint8_t *)src;
            int nb = n / 256;
            for (int b = 0; b < nb; b++) {
                uint16_t dh, mh; memcpy(&dh, p, 2); memcpy(&mh, p + 2, 2);
                float d = sp_f16_to_f32(dh), dmin = sp_f16_to_f32(mh);
                const uint8_t *scales = p + 4;
                const uint8_t *q = p + 16;
                float *y = dst + (size_t)b * 256;
                int is = 0;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc, mm;
                    get_scale_min_k4(is + 0, scales, &sc, &mm); float d1 = d * sc, m1 = dmin * mm;
                    get_scale_min_k4(is + 1, scales, &sc, &mm); float d2 = d * sc, m2 = dmin * mm;
                    for (int l = 0; l < 32; l++) y[l]      = d1 * (float)(q[l] & 0xF) - m1;
                    for (int l = 0; l < 32; l++) y[l + 32] = d2 * (float)(q[l] >>  4) - m2;
                    y += 64; q += 32; is += 2;
                }
                p += 144;
            }
            return 0;
        }
        case SP_WDT_Q6_K: {
            /* block_q6_K = { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d; } = 210 B / 256 elems */
            if (n % 256 != 0) return 1;
            const uint8_t *p = (const uint8_t *)src;
            int nb = n / 256;
            for (int b = 0; b < nb; b++) {
                const uint8_t *ql = p;
                const uint8_t *qh = p + 128;
                const int8_t  *sc = (const int8_t *)(p + 192);
                uint16_t dh; memcpy(&dh, p + 208, 2);
                float d = sp_f16_to_f32(dh);
                float *y = dst + (size_t)b * 256;
                for (int n2 = 0; n2 < 256; n2 += 128) {
                    for (int l = 0; l < 32; l++) {
                        int is = l / 16;
                        int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        int8_t q3 = (int8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        int8_t q4 = (int8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        y[l +  0] = d * sc[is + 0] * q1;
                        y[l + 32] = d * sc[is + 2] * q2;
                        y[l + 64] = d * sc[is + 4] * q3;
                        y[l + 96] = d * sc[is + 6] * q4;
                    }
                    ql += 64; qh += 32; sc += 8; y += 128;
                }
                p += 210;
            }
            return 0;
        }
        default:
            return 1;  /* unsupported quant type */
    }
}
