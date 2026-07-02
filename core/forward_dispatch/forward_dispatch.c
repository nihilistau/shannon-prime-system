/* forward_dispatch.c — model-coupled weight-access kernels (see sp/forward_dispatch.h).
 * Relocated out of the engine's src/forward/kernels.c. The pure reference primitives
 * that headed that file (dot/rmsnorm/rope/attn) are NOT here — they are the math core's
 * forward_kernels and the orchestration calls them. Only the model-storage-coupled
 * weight lift moves here: the dequant-on-demand / arena-inline matmul, the embedding
 * lookup, and the norm-tensor accessor, plus the runtime weight-path gate-knob state.
 *
 * The one behavioral edit vs the engine original: the pure-f32 path's dot is the
 * scalar reference sp_dot_f32 (forward_kernels) — the engine's #ifdef AVX2 reduction
 * was a CPU-backend variant, not the L1 reference, so it is dropped. The Frob/Q4 and
 * F16-activation branches are relocated verbatim (they only call already-migrated
 * frobenius / weight-dtype primitives); the Qwen3 regression and the forward round-trip
 * guard behavior-preservation. */
#define _CRT_SECURE_NO_WARNINGS   /* getenv (gate knobs) is fine here (MSVC C4996) */
#include "sp/forward_dispatch.h"
#include "sp/forward_kernels.h"   /* sp_dot_f32 — the reference accumulation order */
#include "sp/arena.h"             /* sp_arena_find / sp_arena_dequant_row / sp_arena_tensor */
#include "sp/frobenius_lift.h"    /* sp_frob_* + SP_FROB_QMAX (inline-lift / on-the-fly Frob) */
#include "sp/weight_dtype.h"      /* sp_dequant_row, sp_f16_to_f32 / sp_f32_to_f16 (F16-act mimic) */

#include <stdlib.h>

/* ── runtime gate knobs honored by these kernels (default OFF = pure-f32 reference) ── */
static int   g_f16_act = 0;   /* SP_ENGINE_F16_ACT=1 rounds activations to F16 (ggml-faithful) */
static int   g_frob    = 0;   /* SP_ENGINE_FROB: 0 f32, 1/2 Q8 inline/dequant, 3/4 Q4 inline/dequant */
static float g_q4_promote = 0.25f;   /* promote a Q4 row to Q8 if its round-trip rel-L2 exceeds this */
static long  g_q4_promoted = 0;
static long  g_q4_rows = 0;

void sp_kernels_read_env(void) {
    { const char *e = getenv("SP_ENGINE_F16_ACT");  g_f16_act  = (e && e[0] == '1'); }
    { const char *e = getenv("SP_ENGINE_FROB");     g_frob     = e ? atoi(e) : 0; }
    { const char *e = getenv("SP_Q4_PROMOTE");      if (e) g_q4_promote = (float)atof(e);
      g_q4_promoted = 0; g_q4_rows = 0; }
    /* SP_CPU_SCALAR is moot in the math core: the L1 reference dot (sp_dot_f32) is
     * already the scalar sequential reduction; vectorization is a backend concern. */
}

void qwen3_q4_stats(long *promoted, long *rows) {
    if (promoted) *promoted = g_q4_promoted;
    if (rows)     *rows     = g_q4_rows;
}

/* bytes occupied by `n` contiguous elements of a ggml weight row
 * (F32/F16/Q8_0 + the Q4_K/Q6_K k-quants used by qwen35moe Q4_K_M). */
static size_t row_bytes(uint32_t type, int n) {
    switch (type) {
        case GGML_T_F32:  return (size_t)n * 4;
        case GGML_T_F16:  return (size_t)n * 2;
        case GGML_T_Q8_0: return (size_t)(n / 32) * 34;
        case GGML_T_Q4_K: return (size_t)(n / 256) * 144;
        case GGML_T_Q6_K: return (size_t)(n / 256) * 210;
        default:          return 0;
    }
}

/* NORTHSTAR brick 5 (2026-07-02): SIMD i8*f32 dot for the arena row loops. AVX2+FMA
 * when compiled with /arch:AVX2 (the perf build); scalar fallback otherwise = the
 * null floor. NOTE: SIMD changes the ACCUMULATION ORDER, so the gate for builds
 * carrying this flag is top-1/sequence equivalence (G-MOE-SIMD-TOKS), not
 * bit-parity — the N1b rule applied within-path. */
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
static float q36_dot_i8xf32(const int8_t *cp, const float *x, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i b  = _mm_loadl_epi64((const __m128i *)(cp + i));
        __m256  vf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
        acc = _mm256_fmadd_ps(vf, _mm256_loadu_ps(x + i), acc);
    }
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    float r = _mm_cvtss_f32(s);
    for (; i < n; i++) r += (float)cp[i] * x[i];
    return r;
}
#else
static float q36_dot_i8xf32(const int8_t *cp, const float *x, int n) {
    float r = 0.0f;
    for (int i = 0; i < n; i++) r += (float)cp[i] * x[i];
    return r;
}
#endif

/* Exported single-row packed dot (dequant-free): unpack-if-Q4 into caller scratch,
 * SIMD dot, apply the row/block scales. Used by expert_mm (qwen36 MoE) instead of
 * sp_frob_packed_dequant_row -> f32 dot (skips writing the f32 row entirely). */
float sp_arena_row_dot(const sp_frob_packed_tensor *pt, int j, const float *x,
                       int in, int8_t *unp) {
    const uint8_t *rc = pt->codes + pt->row_off[j];
    if (pt->row_prec[j] != 8 && pt->bscale) {
        sp_frob_q4_unpack(rc, in, unp);
        const uint16_t *bs = pt->bscale + (size_t)j * (size_t)pt->bs_nblk;
        float facc = 0.0f;
        for (int b = 0; b * 32 < in; b++) {
            int i0 = b * 32, i1 = i0 + 32 < in ? i0 + 32 : in;
            facc += q36_dot_i8xf32(unp + i0, x + i0, i1 - i0) * sp_f16_to_f32(bs[b]);
        }
        return facc;
    }
    if (pt->row_prec[j] == 8)
        return q36_dot_i8xf32((const int8_t *)rc, x, in) * (pt->row_scale[j] / 127.0f);
    sp_frob_q4_unpack(rc, in, unp);
    return q36_dot_i8xf32(unp, x, in) * (pt->row_scale[j] / 7.0f);
}

/* Arena matmul: inline-lift the packed Q8/Q4 codes (the §4.8 production path). The
 * per-row scale is applied once after the integer-code accumulation.
 * NORTHSTAR brick 4 (2026-07-02): rows are independent — OpenMP over j with a
 * PER-THREAD unpack scratch. Each row's accumulation order is untouched, so the
 * result is BIT-IDENTICAL to the serial loop (parity re-gated, G-MOE-STATE-PARITY).
 * Compiled without /openmp the pragmas are inert = the serial null floor. */
static int matmul_arena(const sp_arena_tensor *at, const float *X,
                        int n_tok, int in, int out, float *Y) {
    const sp_frob_packed_tensor *pt = &at->pt;
    if (pt->rows != out || pt->cols != in) return 1;
    int fail = 0;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int8_t *unp = (int8_t *)malloc((size_t)in);   /* Q4 unpack scratch (per thread) */
        if (!unp) {
            fail = 1;
        } else {
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 16)
#endif
            for (int j = 0; j < out; j++) {
                const uint8_t *rc = pt->codes + pt->row_off[j];
                const int8_t *cp;
                float inv;
                if (pt->row_prec[j] != 8 && pt->bscale) {
                    /* OK_Q4B row (arena v2): per-32-block f16 scales — accumulate per block,
                     * apply each block's scale, no trailing row scale. */
                    sp_frob_q4_unpack(rc, in, unp); cp = unp;
                    const uint16_t *bs = pt->bscale + (size_t)j * (size_t)pt->bs_nblk;
                    for (int t = 0; t < n_tok; t++) {
                        const float *x = X + (size_t)t * in;
                        float facc = 0.0f;
                        for (int b = 0; b * 32 < in; b++) {
                            int i0 = b * 32, i1 = i0 + 32 < in ? i0 + 32 : in;
                            facc += q36_dot_i8xf32(cp + i0, x + i0, i1 - i0) * sp_f16_to_f32(bs[b]);
                        }
                        Y[(size_t)t * out + j] = facc;
                    }
                    continue;
                }
                if (pt->row_prec[j] == 8) { cp = (const int8_t *)rc; inv = pt->row_scale[j] / 127.0f; }
                else { sp_frob_q4_unpack(rc, in, unp); cp = unp; inv = pt->row_scale[j] / 7.0f; }
                for (int t = 0; t < n_tok; t++) {
                    const float *x = X + (size_t)t * in;
                    Y[(size_t)t * out + j] = q36_dot_i8xf32(cp, x, in) * inv;
                }
            }
            free(unp);
        }
    }
    return fail;
}

int sp_matmul(const qwen3_model *m, const gguf_tensor *W,
              const float *X, int n_tok, int in, int out, float *Y) {
    if (m->arena) {                            /* packed-weight arena (§4.8) takes precedence */
        const sp_arena_tensor *at = sp_arena_find(m->arena, W->name);
        if (at) return matmul_arena(at, X, n_tok, in, out, Y);
        /* not arena-ized (e.g. token_embd in 1a): fall through to the GGUF path */
    }
    const uint8_t *base = (const uint8_t *)gguf_tensor_data(m->gguf, W);
    size_t rb = row_bytes(W->type, in);
    if (!base || rb == 0) return 1;
    float *wrow = (float *)malloc((size_t)in * sizeof(float));
    if (!wrow) return 1;
    /* When mimicking ggml's F16 src1 downcast, round the activation rows once
     * up front (the same rounded x is reused across all `out` weight rows). */
    float *xr = NULL;
    if (g_f16_act) {
        xr = (float *)malloc((size_t)n_tok * in * sizeof(float));
        if (!xr) { free(wrow); return 1; }
        for (size_t i = 0; i < (size_t)n_tok * in; i++)
            xr[i] = sp_f16_to_f32(sp_f32_to_f16(X[i]));
        X = xr;
    }
    int8_t  *codes = NULL;          /* per-row int8/int4 codes */
    uint8_t *nib   = NULL;          /* Q4 nibble-packed storage */
    if (g_frob) {
        codes = (int8_t *)malloc((size_t)in);
        if (!codes) { free(wrow); free(xr); return 1; }
        if (g_frob >= 3) {
            nib = (uint8_t *)malloc((size_t)(in + 1) / 2);
            if (!nib) { free(wrow); free(xr); free(codes); return 1; }
        }
    }
    const int q4_path = (g_frob == 3 || g_frob == 4);
    for (int j = 0; j < out; j++) {
        if (sp_dequant_row(base + (size_t)j * rb, W->type, in, wrow)) {
            free(wrow); free(xr); free(codes); free(nib); return 1;
        }
        if (g_frob) {
            float s = sp_frob_row_scale(wrow, in);
            float inv;
            if (!q4_path) {                                   /* Q8 (modes 1/2) */
                for (int i = 0; i < in; i++) codes[i] = sp_frob_quant1(wrow[i], s);
                inv = s / (float)SP_FROB_QMAX;
            } else {                                          /* Q4 (modes 3/4) + calibration */
                g_q4_rows++;
                if (sp_frob_q4_row_relerr(wrow, in) > g_q4_promote) {
                    for (int i = 0; i < in; i++) codes[i] = sp_frob_quant1(wrow[i], s);  /* promote -> Q8 */
                    inv = s / (float)SP_FROB_QMAX;
                    g_q4_promoted++;
                } else {
                    for (int i = 0; i < in; i++) codes[i] = sp_frob_quant1_q4(wrow[i], s);
                    sp_frob_q4_pack(codes, in, nib);          /* round-trip real 4-bit storage */
                    sp_frob_q4_unpack(nib, in, codes);
                    inv = s / 7.0f;
                }
            }
            if (g_frob == 2 || g_frob == 4)                   /* dequant reference: lift to f32 */
                for (int i = 0; i < in; i++) wrow[i] = (float)codes[i] * inv;
            for (int t = 0; t < n_tok; t++) {
                const float *x = X + (size_t)t * in;
                float acc = 0.0f;
                if (g_frob == 2 || g_frob == 4) {             /* plain f32 dot of lifted weights */
                    for (int i = 0; i < in; i++) acc += wrow[i] * x[i];
                    Y[(size_t)t * out + j] = acc;
                } else {                                      /* inline lift: scale once */
                    for (int i = 0; i < in; i++) acc += (float)codes[i] * x[i];
                    Y[(size_t)t * out + j] = acc * inv;
                }
            }
        } else {
            /* pure-f32 reference path: scalar sequential dot (the L1 reference order). */
            for (int t = 0; t < n_tok; t++)
                Y[(size_t)t * out + j] = sp_dot_f32(wrow, X + (size_t)t * in, in);
        }
    }
    free(wrow);
    free(xr);
    free(codes);
    free(nib);
    return 0;
}

const float *sp_as_f32(const qwen3_model *m, const gguf_tensor *t) {
    if (m->released) {
        for (int i = 0; i < m->n_norm; i++) if (m->norm_src[i] == t) return m->norm_buf[i];
        return NULL;   /* every norm the forward reads was copied in release */
    }
    return (const float *)gguf_tensor_data(m->gguf, t);
}

int sp_embed_row(const qwen3_model *m, int32_t tok, int E, float *dst) {
    const sp_arena_tensor *at = m->arena ? sp_arena_find(m->arena, m->token_embd->name) : NULL;
    if (at) return sp_arena_dequant_row(at, (int)tok, dst);
    const uint8_t *emb = (const uint8_t *)gguf_tensor_data(m->gguf, m->token_embd);
    size_t rb = row_bytes(m->token_embd->type, E);
    if (!emb || rb == 0) return 1;
    return sp_dequant_row(emb + (size_t)tok * rb, m->token_embd->type, E, dst);
}

/* Generalised row dequant: row `row` (length `len`) of an arbitrary weight tensor
 * W -> dst[len], from the arena (if W is packed there) else the GGUF mapping.
 * Mirrors sp_embed_row for any [len, n_rows] tensor (Gemma4 per_layer_token_embd). */
int sp_weight_row(const qwen3_model *m, const gguf_tensor *W, int row, int len, float *dst) {
    if (!W) return 1;
    const sp_arena_tensor *at = m->arena ? sp_arena_find(m->arena, W->name) : NULL;
    if (at) return sp_arena_dequant_row(at, row, dst);
    const uint8_t *p = (const uint8_t *)gguf_tensor_data(m->gguf, W);
    size_t rb = row_bytes(W->type, len);
    if (!p || rb == 0) return 1;
    return sp_dequant_row(p + (size_t)row * rb, W->type, len, dst);
}
