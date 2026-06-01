/* forward_kernels.c -- portable scalar reference; see sp/forward_kernels.h. The
 * pure-math kernels lifted verbatim (in arithmetic) out of the engine's forward path
 * so accelerated backends gate to them. No x86 intrinsics: the math core stays
 * portable across every backend's host (this is why the engine's AVX2 dot stayed
 * behind as a CPU-backend optimisation rather than moving here). */
#include "sp/forward_kernels.h"

#include <math.h>
#include <stddef.h>

float sp_dot_f32(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

void sp_rmsnorm(const float *x, const float *w, int n, float eps, float *out) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

void sp_rmsnorm_head(float *v, const float *w, int d, float eps) {
    double ss = 0.0;
    for (int i = 0; i < d; i++) ss += (double)v[i] * v[i];
    float scale = 1.0f / sqrtf((float)(ss / d) + eps);
    for (int i = 0; i < d; i++) v[i] = v[i] * scale * w[i];
}

void sp_rope_neox(float *v, int d, int p, float base) {
    int half = d / 2;
    for (int i = 0; i < half; i++) {
        float freq  = powf(base, -2.0f * (float)i / (float)d);
        float theta = (float)p * freq;
        float c = cosf(theta), s = sinf(theta);
        float a = v[i], b = v[i + half];
        v[i]        = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}

/* Gemma4 global-layer proportional RoPE: theta_i = p * base^(-2i/d) / ff[i].
 * ff==NULL is identical to sp_rope_neox. freq_factors semantics to be validated
 * against the ggml rope_ext oracle when gemma4_forward is wired (Stage 2). */
void sp_rope_neox_freqs(float *v, int d, int p, float base, const float *ff) {
    if (!ff) { sp_rope_neox(v, d, p, base); return; }
    int half = d / 2;
    for (int i = 0; i < half; i++) {
        float freq  = powf(base, -2.0f * (float)i / (float)d) / ff[i];
        float theta = (float)p * freq;
        float c = cosf(theta), s = sinf(theta);
        float a = v[i], b = v[i + half];
        v[i]        = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}

void sp_attn_head(const float *qh, const float *KC, const float *VC,
                  int pos, int KVD, int kvh, int HD, float ascale, int win,
                  float *sc, float *out) {
    int s0 = (win >= 0 && pos - win + 1 > 0) ? pos - win + 1 : 0;
    float maxs = -INFINITY;
    for (int s = s0; s <= pos; s++) {
        const float *kh = KC + (size_t)s * KVD + (size_t)kvh * HD;
        float acc = 0.0f;
        for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
        float dscore = acc * ascale;
        sc[s] = dscore;
        if (dscore > maxs) maxs = dscore;
    }
    float sum = 0.0f;
    for (int s = s0; s <= pos; s++) { sc[s] = expf(sc[s] - maxs); sum += sc[s]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < HD; i++) out[i] = 0.0f;
    for (int s = s0; s <= pos; s++) {
        float w = sc[s] * inv;
        const float *vh = VC + (size_t)s * KVD + (size_t)kvh * HD;
        for (int i = 0; i < HD; i++) out[i] += w * vh[i];
    }
}
