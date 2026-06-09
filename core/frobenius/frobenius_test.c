/* frobenius_test.c — contract tests for the Q8 Frobenius lift (Phase 1E).
 *
 * One T_FRO_n per roadmap item, driven by sp_test.h.  All randomness is seeded
 * with fixed constants so runs are deterministic and reproducible.
 */
#include "sp/sp_test.h"
#include "sp/frobenius_lift.h"

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/* ---- deterministic RNG (xorshift64*, fixed seed) ------------------------- */

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static void rng_seed(uint64_t s) { rng_state = s ? s : 0x9E3779B97F4A7C15ull; }

static uint64_t rng_next(void) {
    uint64_t x = rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1Dull;
}

/* uniform float in [-mag, mag] */
static float rng_float(float mag) {
    /* 24 random bits -> [0,1) -> [-1,1) -> scaled */
    uint32_t bits = (uint32_t)(rng_next() >> 40) & 0xFFFFFFu;
    float u = (float)bits / (float)(1u << 24);   /* [0,1) */
    return (u * 2.0f - 1.0f) * mag;
}

/* ---- independent reference implementations (test-side oracle) ------------ */

/* Reference per-row scale: simplest possible max-abs scan. */
static float ref_row_scale(const float *row, int cols) {
    float m = 0.0f;
    for (int c = 0; c < cols; c++) {
        float a = fabsf(row[c]);
        if (a > m) m = a;
    }
    return m;
}

/* Reference ceiling-shift (round-half-away-from-zero) quantiser, written the
 * simplest possible way and independently of the production code.  This is the
 * bit-for-bit oracle T_FRO_1 compares against. */
static int8_t ref_quant1(float v, float scale) {
    if (scale == 0.0f) return 0;
    float x = v / scale * 127.0f;
    float r;
    if (x >= 0.0f) r = floorf(x + 0.5f);   /* +0.5 -> +1 */
    else           r = ceilf(x - 0.5f);    /* -0.5 -> -1 */
    if (r >  127.0f) r =  127.0f;
    if (r < -127.0f) r = -127.0f;
    return (int8_t)r;
}

/* ---- T_FRO_1: scale picker + ceiling-shift quantiser correctness --------- */

static void T_FRO_1(void) {
    rng_seed(0xF101F101ull);

    /* (a) per-row scale picker matches the reference across many random rows */
    for (int trial = 0; trial < 4096; trial++) {
        int cols = 1 + (int)(rng_next() % 64);   /* 1..64 */
        float row[64];
        for (int c = 0; c < cols; c++) row[c] = rng_float(10.0f);
        float got  = sp_frob_row_scale(row, cols);
        float want = ref_row_scale(row, cols);
        SP_CHECK(got == want, "T_FRO_1 row scale matches reference bit-for-bit");
    }

    /* (b) ceiling-shift quant1 matches the reference bit-for-bit, many rows */
    for (int trial = 0; trial < 4096; trial++) {
        float scale = 0.01f + rng_float(10.0f);
        if (scale <= 0.0f) scale = 0.5f;          /* keep scale positive */
        for (int k = 0; k < 32; k++) {
            /* values within and beyond the row range (to exercise clamp) */
            float v = rng_float(scale * 1.5f);
            SP_CHECK_EQ_I64(sp_frob_quant1(v, scale), ref_quant1(v, scale),
                            "T_FRO_1 quant1 matches reference bit-for-bit");
        }
    }

    /* (c) exact half-points round AWAY from zero (no round-half-up bias).
     * With scale s, x = v/s*127; pick v so x is exactly +/-0.5, +/-1.5, ... */
    {
        float s = 4.0f;
        /* x = m + 0.5 for integer m; choose v = (m+0.5)/127*s */
        for (int m = 0; m <= 100; m++) {
            float xp =  (float)m + 0.5f;
            float xn = -((float)m + 0.5f);
            float vp = xp / 127.0f * s;
            float vn = xn / 127.0f * s;
            int8_t qp = sp_frob_quant1(vp, s);
            int8_t qn = sp_frob_quant1(vn, s);
            /* away-from-zero: +0.5 -> +1, +1.5 -> +2, ...; symmetric negatives */
            SP_CHECK_EQ_I64(qp, ref_quant1(vp, s), "T_FRO_1 +half away-from-zero");
            SP_CHECK_EQ_I64(qn, ref_quant1(vn, s), "T_FRO_1 -half away-from-zero");
            SP_CHECK_EQ_I64(qp, -qn, "T_FRO_1 quantiser symmetric about 0");
        }
    }

    /* (d) zero row: scale 0 -> all codes 0, no divide-by-zero / nan */
    {
        float zrow[16] = {0};
        float zs = sp_frob_row_scale(zrow, 16);
        SP_CHECK(zs == 0.0f, "T_FRO_1 zero row scale is 0");
        for (int c = 0; c < 16; c++)
            SP_CHECK_EQ_I64(sp_frob_quant1(zrow[c], zs), 0,
                            "T_FRO_1 zero-row code is 0");
    }

    /* (e) mean quantisation error near zero on symmetric input (unbiased).
     * Quantise a large symmetric sample; the signed error v_hat - v should
     * average to ~0 (round-half-up would push it systematically positive). */
    {
        float s = 3.0f;
        double sum_err = 0.0;
        long n = 0;
        for (int i = 0; i < 200000; i++) {
            float v = rng_float(s);                 /* in [-s, s] */
            int8_t q = sp_frob_quant1(v, s);
            float vhat = sp_frob_dequant1(q, s);
            sum_err += (double)(vhat - v);
            n++;
        }
        double mean_err = sum_err / (double)n;
        /* bias well under a hundredth of a quant step (s/127 ~ 0.0236) */
        SP_CHECK(fabs(mean_err) < (double)(s / 127.0f) * 0.02,
                 "T_FRO_1 mean quant error ~ 0 (no round-half-up bias)");
    }
}

/* ---- T_FRO_2: round-trip error bound + idempotence ----------------------- */
/*
 * Bound proved here: for any element v in a row with scale s = max|row|, we
 * have |v| <= s so x = v/s*127 lies in [-127,127] and the clamp never fires.
 * The ceiling-shift rule then gives |q - x| <= 0.5 exactly, hence
 *     |v_hat - v| = (s/127) * |q - x| <= s/(2*127).
 * Allowing a tiny slack for the two fp32 roundings (the divide-multiply in
 * encode and the multiply in decode), the per-element bound asserted is
 *     |v_hat - v| <= (s/127) * (0.5 + 1e-6).
 * Idempotence: re-quantising v_hat reproduces the same code q, because
 * v_hat = q*(s/127) maps back through v_hat/s*127 to (within <<0.5) the
 * integer q, which the rounding rule snaps to exactly.
 */
static void T_FRO_2(void) {
    rng_seed(0xF202F202ull);

    enum { R = 37, C = 53 };
    static float w[R * C];
    static int8_t packed[R * C];
    static float scales[R];
    static float vhat[R * C];

    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            w[r * C + c] = rng_float(2.0f) + rng_float(0.1f);  /* mixed magnitudes */

    sp_frob_quantize(w, R, C, packed, scales);
    sp_frob_dequantize(packed, scales, R, C, vhat);

    int bound_ok = 1, idem_ok = 1;
    for (int r = 0; r < R; r++) {
        float s = scales[r];
        float bound = s / 127.0f * (0.5f + 1e-6f);
        for (int c = 0; c < C; c++) {
            float err = fabsf(vhat[r * C + c] - w[r * C + c]);
            if (err > bound) bound_ok = 0;
            /* idempotence: re-quantise the dequantised value -> same code */
            int8_t requant = sp_frob_quant1(vhat[r * C + c], s);
            if (requant != packed[r * C + c]) idem_ok = 0;
        }
    }
    SP_CHECK(bound_ok, "T_FRO_2 max|vhat-v| <= (s/127)*(0.5+eps) per element");
    SP_CHECK(idem_ok,  "T_FRO_2 re-quantising vhat reproduces same int8 codes");

    /* extreme: the row max maps to code +/-127 and reconstructs exactly */
    {
        float row[5] = { -1.0f, 0.25f, 0.0f, 0.9f, 1.0f };  /* max abs = 1.0 */
        float s = sp_frob_row_scale(row, 5);
        SP_CHECK(s == 1.0f, "T_FRO_2 row scale = max abs");
        SP_CHECK_EQ_I64(sp_frob_quant1(1.0f, s), 127,  "T_FRO_2 +max -> +127");
        SP_CHECK_EQ_I64(sp_frob_quant1(-1.0f, s), -127, "T_FRO_2 -max -> -127");
        SP_CHECK(sp_frob_dequant1(127, s) == 1.0f,  "T_FRO_2 +127 -> +s exactly");
        SP_CHECK(sp_frob_dequant1(-127, s) == -1.0f, "T_FRO_2 -127 -> -s exactly");
    }
}

/* ---- T_FRO_3: compression ratio on a 4096x4096 matrix -------------------- */
/*
 * Q8 is 1 byte per coefficient, so the honest ratio vs fp32 is ~4x, NOT 8x.
 * For 4096x4096:
 *   packed = 4096*4096*1 + 4096*4 = 16777216 + 16384 = 16793600 bytes
 *   fp32   = 4096*4096*4         = 67108864 bytes  -> ratio 67108864/16793600
 *                                                   = 3.9960975610...
 *   fp64   = 4096*4096*8         = 134217728 bytes -> ratio 134217728/16793600
 *                                                   = 7.9921951220...
 * The roadmap's "8x" figure is the fp64 comparison (an 8-byte arena).
 */
static void T_FRO_3(void) {
    const int N = 4096;

    size_t packed = sp_frob_packed_bytes(N, N);
    size_t expect = (size_t)N * (size_t)N + (size_t)N * sizeof(float);
    SP_CHECK_EQ_I64((int64_t)packed, (int64_t)expect,
                    "T_FRO_3 packed bytes = rows*cols + rows*4");

    double r32 = sp_frob_ratio(N, N, 4);
    double r64 = sp_frob_ratio(N, N, 8);

    fprintf(stderr, "    [T_FRO_3] packed=%zu B  fp32=%lld B (ratio %.8f)  "
            "fp64=%lld B (ratio %.8f)\n",
            packed, (long long)((int64_t)N * N * 4), r32,
            (long long)((int64_t)N * N * 8), r64);

    SP_CHECK(r32 >= 3.99, "T_FRO_3 ratio vs fp32 >= 3.99x (~4x, honest Q8)");
    SP_CHECK(r64 >= 7.99, "T_FRO_3 ratio vs fp64 >= 7.99x (~8x, the roadmap fig)");

    /* exact expected constants (1-byte Q8, not a fudged literal 8x vs fp32):
     *   r32 = 67108864/16793600  = 3.99609756097...
     *   r64 = 134217728/16793600 = 7.99219512195... (exactly 2*r32) */
    SP_CHECK(fabs(r32 - (67108864.0 / 16793600.0)) < 1e-12,
             "T_FRO_3 fp32 ratio exact (= 67108864/16793600)");
    SP_CHECK(fabs(r64 - (134217728.0 / 16793600.0)) < 1e-12,
             "T_FRO_3 fp64 ratio exact (= 134217728/16793600 = 2*fp32)");
}

/* ---- reference Q4 quantiser (independent oracle) ------------------------- */
static int8_t ref_quant1_q4(float v, float scale) {
    if (scale == 0.0f) return 0;
    float x = v / scale * 7.0f;
    float r = (x >= 0.0f) ? floorf(x + 0.5f) : ceilf(x - 0.5f);
    if (r >  7.0f) r =  7.0f;
    if (r < -7.0f) r = -7.0f;
    return (int8_t)r;
}

/* ---- T_FRO_Q4: 4-bit codec — quant/dequant, pack/unpack, calibration ----- */
/*
 * Q4 mirrors the Q8 contract at 4 bits: symmetric codes [-7,7], per-row scale,
 * round-half-away, dequant v_hat = q*(s/7). Shared by every engine backend
 * (the engine matmul and the future CUDA/Vulkan/Hexagon paths call these).
 */
static void T_FRO_Q4(void) {
    rng_seed(0xF404F404ull);

    /* (a) quant1_q4 matches the reference bit-for-bit across many rows */
    for (int trial = 0; trial < 4096; trial++) {
        float scale = 0.01f + rng_float(10.0f);
        if (scale <= 0.0f) scale = 0.5f;
        for (int k = 0; k < 32; k++) {
            float v = rng_float(scale * 1.5f);                 /* exercises the clamp */
            SP_CHECK_EQ_I64(sp_frob_quant1_q4(v, scale), ref_quant1_q4(v, scale),
                            "T_FRO_Q4 quant1_q4 matches reference bit-for-bit");
        }
    }

    /* (b) every code is in [-7,7]; pack->unpack is the identity on valid codes */
    {
        int8_t codes[129], back[129];
        uint8_t nib[65];
        for (int n = 1; n <= 129; n++) {
            for (int i = 0; i < n; i++) {
                float s = 2.0f, v = rng_float(s * 1.5f);
                codes[i] = sp_frob_quant1_q4(v, s);
                SP_CHECK(codes[i] >= -7 && codes[i] <= 7, "T_FRO_Q4 code in [-7,7]");
            }
            sp_frob_q4_pack(codes, n, nib);
            sp_frob_q4_unpack(nib, n, back);
            int idok = 1;
            for (int i = 0; i < n; i++) if (back[i] != codes[i]) idok = 0;
            SP_CHECK(idok, "T_FRO_Q4 pack->unpack is the identity on Q4 codes");
        }
    }

    /* (c) extremes: row max -> +/-7, dequant +/-7 -> +/-s exactly */
    {
        float row[5] = { -1.0f, 0.25f, 0.0f, 0.9f, 1.0f };
        float s = sp_frob_row_scale(row, 5);
        SP_CHECK_EQ_I64(sp_frob_quant1_q4( 1.0f, s),  7, "T_FRO_Q4 +max -> +7");
        SP_CHECK_EQ_I64(sp_frob_quant1_q4(-1.0f, s), -7, "T_FRO_Q4 -max -> -7");
        SP_CHECK(sp_frob_dequant1_q4( 7, s) ==  1.0f, "T_FRO_Q4 +7 -> +s exactly");
        SP_CHECK(sp_frob_dequant1_q4(-7, s) == -1.0f, "T_FRO_Q4 -7 -> -s exactly");
    }

    /* (d) zero row: scale 0 -> code 0, rel-error 0 (no nan/divide-by-zero) */
    {
        float zrow[16] = {0};
        SP_CHECK_EQ_I64(sp_frob_quant1_q4(0.0f, 0.0f), 0, "T_FRO_Q4 zero-row code 0");
        SP_CHECK(sp_frob_q4_row_relerr(zrow, 16) == 0.0f, "T_FRO_Q4 zero-row relerr 0");
    }

    /* (e) per-element round-trip bound |v_hat - v| <= (s/7)*(0.5+eps) */
    {
        rng_seed(0xF4E5F4E5ull);
        int bad = 0;
        float row[64];
        for (int trial = 0; trial < 512; trial++) {
            int cols = 1 + (int)(rng_next() % 64);
            for (int c = 0; c < cols; c++) row[c] = rng_float(3.0f);
            float s = sp_frob_row_scale(row, cols);
            float bound = (s == 0.0f) ? 0.0f : s / 7.0f * (0.5f + 1e-6f);
            for (int c = 0; c < cols; c++) {
                float vhat = sp_frob_dequant1_q4(sp_frob_quant1_q4(row[c], s), s);
                if (fabsf(vhat - row[c]) > bound) bad = 1;
            }
        }
        SP_CHECK(!bad, "T_FRO_Q4 |vhat-v| <= (s/7)*(0.5+eps) per element");
    }

    /* (f) calibration metric: a coarse row has larger Q4 rel-error than a row
     * that lands near the grid; both are finite and in [0,1]-ish range. */
    {
        float coarse[8] = { 1.0f, 0.13f, 0.51f, 0.27f, 0.88f, 0.04f, 0.62f, 0.39f };
        float e = sp_frob_q4_row_relerr(coarse, 8);
        SP_CHECK(e > 0.0f && e < 1.0f, "T_FRO_Q4 row relerr finite and positive");
    }

    /* (g) packed byte size: rows*ceil(cols/2) + rows*4 (4-bit codes) */
    {
        size_t got = sp_frob_q4_packed_bytes(37, 53);
        size_t want = (size_t)37 * ((53 + 1) / 2) + (size_t)37 * sizeof(float);
        SP_CHECK_EQ_I64((int64_t)got, (int64_t)want, "T_FRO_Q4 packed bytes formula");
        /* ~2x denser than Q8, ~8x vs fp32 on a wide matrix */
        SP_CHECK(sp_frob_q4_packed_bytes(4096, 4096) < sp_frob_packed_bytes(4096, 4096),
                 "T_FRO_Q4 denser than Q8");
    }
}

/* ---- T_FRO_5: mixed-precision packed tensor (arena layout) --------------- */

/* Row reader over a flat rows*cols f32 matrix held in ctx (the test stand-in for
 * the engine's GGUF-row dequant). */
typedef struct { const float *w; int cols; } mat_ctx;
static int mat_row(void *ctx, int j, float *dst) {
    const mat_ctx *m = (const mat_ctx *)ctx;
    const float *row = m->w + (size_t)j * (size_t)m->cols;
    for (int i = 0; i < m->cols; i++) dst[i] = row[i];
    return 0;
}

static void T_FRO_5(void) {
    /* frozen arena layout guard — bumping forces a conscious migration.
     * v1 -> v2: the OK_Q4B per-32 f16 block-scale migration (core 85aadd3,
     * gold-campaign-ratified 2026-06-08, ledger 06-R9/R10). This guard now
     * pins v2; the next bump again requires a formal migration. */
    SP_CHECK_EQ_I64(SP_FROB_ARENA_LAYOUT_VERSION, 2, "T_FRO_5 frozen arena layout version == 2 (bscale migration 85aadd3)");

    const int rows = 40, cols = 53;
    float *w = (float *)malloc((size_t)rows * (size_t)cols * sizeof(float));
    SP_CHECK(w != NULL, "T_FRO_5 alloc synthetic matrix");
    if (!w) return;
    rng_seed(0xABCDEF0123456789ull);
    for (int j = 0; j < rows; j++) {
        float mag = (j % 5 == 0) ? 4.0f : 1.0f;     /* a few rows with wider dynamic range */
        float *row = w + (size_t)j * (size_t)cols;
        for (int i = 0; i < cols; i++) row[i] = rng_float(mag);
    }
    mat_ctx ctx = { w, cols };
    float dq[64];   /* cols=53 fits */

    /* (a) Q8: every row Q8; dequant_row bit-matches per-element quant1->dequant1. */
    {
        sp_frob_packed_tensor t; long promoted = 0;
        int rc = sp_frob_pack_tensor(rows, cols, 8, 0.0f, mat_row, &ctx, &t, &promoted);
        SP_CHECK(rc == 0 && promoted == 0, "T_FRO_5 pack Q8 ok, nothing promoted");
        int bad = 0;
        for (int j = 0; j < rows && !bad; j++) {
            if (t.row_prec[j] != 8) { bad = 1; break; }
            sp_frob_packed_dequant_row(&t, j, dq);
            const float *row = w + (size_t)j * (size_t)cols;
            float s = sp_frob_row_scale(row, cols);
            for (int i = 0; i < cols; i++)
                if (dq[i] != sp_frob_dequant1(sp_frob_quant1(row[i], s), s)) { bad = 1; break; }
        }
        SP_CHECK(!bad, "T_FRO_5 Q8 dequant_row == quant1->dequant1 per element");
        SP_CHECK_EQ_I64((int64_t)sp_frob_packed_tensor_bytes(&t),
                        (int64_t)((size_t)rows * (size_t)cols + (size_t)rows * (sizeof(float) + sizeof(size_t) + 1)),
                        "T_FRO_5 Q8 packed tensor bytes formula");
        sp_frob_packed_free(&t);
        SP_CHECK(t.codes == NULL && t.rows == 0, "T_FRO_5 packed_free zeroes the descriptor");
    }

    /* (b) Q4 mixed: promotion count equals the rows whose relerr exceeds the
     *     threshold; dequant_row bit-matches the per-row codec (q4, or q8 if promoted). */
    {
        const float thr = 0.25f;
        long want = 0;
        for (int j = 0; j < rows; j++)
            if (sp_frob_q4_row_relerr(w + (size_t)j * (size_t)cols, cols) > thr) want++;
        sp_frob_packed_tensor t; long promoted = 0;
        int rc = sp_frob_pack_tensor(rows, cols, 4, thr, mat_row, &ctx, &t, &promoted);
        SP_CHECK(rc == 0, "T_FRO_5 pack Q4 ok");
        SP_CHECK_EQ_I64(promoted, want, "T_FRO_5 Q4 promotion count matches relerr threshold");
        int bad = 0;
        for (int j = 0; j < rows && !bad; j++) {
            sp_frob_packed_dequant_row(&t, j, dq);
            const float *row = w + (size_t)j * (size_t)cols;
            float s = sp_frob_row_scale(row, cols);
            for (int i = 0; i < cols; i++) {
                float want_v = (t.row_prec[j] == 8)
                    ? sp_frob_dequant1(sp_frob_quant1(row[i], s), s)
                    : sp_frob_dequant1_q4(sp_frob_quant1_q4(row[i], s), s);
                if (dq[i] != want_v) { bad = 1; break; }
            }
        }
        SP_CHECK(!bad, "T_FRO_5 Q4-mixed dequant_row == per-row codec per element");
        sp_frob_packed_free(&t);
    }

    /* (c) bad args rejected, descriptor left zeroed. */
    {
        sp_frob_packed_tensor t;
        SP_CHECK(sp_frob_pack_tensor(rows, cols, 5, 0.0f, mat_row, &ctx, &t, NULL) != 0 && t.codes == NULL,
                 "T_FRO_5 pack rejects precision != 8/4");
        SP_CHECK(sp_frob_pack_tensor(0, cols, 8, 0.0f, mat_row, &ctx, &t, NULL) != 0,
                 "T_FRO_5 pack rejects rows <= 0");
    }

    free(w);
}

/* ---- T_FRO_4: DEFERRED to Phase 2 ---------------------------------------- */

static void T_FRO_4(void) {
    /* Gemma3-1B PPL-within-0.1% needs a forward pass + a loaded model that do
     * not exist until Phase 2 (roadmap §7.5, amended 2026-05-21).  Deferred;
     * this case intentionally performs no checks and never fails. */
    fprintf(stderr, "[T_FRO_4] DEFERRED (Phase 2 — needs forward pass)\n");
}

int main(void) {
    SP_RUN(T_FRO_1);
    SP_RUN(T_FRO_2);
    SP_RUN(T_FRO_3);
    SP_RUN(T_FRO_Q4);
    SP_RUN(T_FRO_5);
    SP_RUN(T_FRO_4);
    return SP_DONE();
}
