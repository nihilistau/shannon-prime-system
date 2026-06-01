/* gemma4.c — Gemma4 f32 reference forward pass (the math core's L1 Gemma4 forward).
 *
 * Mirrors gemma3.c on the migrated kernel stack (forward_dispatch sp_matmul /
 * sp_embed_row / sp_weight_row / sp_as_f32 + forward_kernels sp_rmsnorm /
 * sp_rmsnorm_head / sp_rope_neox_freqs / sp_attn_head). Only the Gemma4 deltas vs
 * Gemma3 live here. The authoritative spec is PPT-LAT-Roadmap §3-G4 Stage 1
 * (extracted from llama.cpp build_gemma4 @ 5dcb711 — reference, not copied):
 *
 *   - Attention scale = 1.0 (NOT 1/sqrt(head_dim); f_attention_scale=1.0).
 *   - Per-layer head geometry, dispatched on (L % g4_swa_period == period-1):
 *       GLOBAL  : head_dim/n_head/n_head_kv = cfg.head_dim/n_head/n_head_kv
 *                 (512/4/1), RoPE base cfg.rope_freq_base (1e6) WITH the shared
 *                 rope_freqs proportional freq-factor table, full causal.
 *       SWA     : g4_hd_swa/g4_nh_swa/g4_nkv_swa (256/8/2), RoPE base
 *                 g4_rope_base_swa (1e4), no freq factors, sliding window.
 *     Q/K/V projection widths are constant (QD=2048, KVD=512); the per-layer
 *     difference is the head split + RoPE base/factors + mask.
 *   - QK-norm per (per-layer) head_dim, before RoPE. V gets a WEIGHTLESS RMSNorm
 *     (no learned weight, no RoPE) — a delta from gemma3.
 *   - Shared-KV: layers [0, g4_n_kv_from_start) own K/V; trailing layers reuse an
 *     earlier owner's stored K/V (shared SWA -> owner kvfs-2, shared global ->
 *     kvfs-1) and skip their own K/V projection.
 *   - Sandwich norms (attn_norm/post_attn_norm + ffn_norm/post_ffw_norm) identical
 *     to gemma3; GeGLU FFN.
 *   - AltUp per-layer-input injection after the FFN residual (precomputed
 *     inp_per_layer from per_layer_token_embd + per_layer_model_proj). Then a
 *     per-layer scalar out_scale.
 *   - Tied LM head, final-logit softcap tanh(z/cap)*cap.
 *
 * NOTE (Stage 1b): the rope_freqs proportional-RoPE semantics, the AltUp scale
 * constants, and the weightless V-norm are validated bit-faithfully against the
 * gemma4 oracle by the M_GEMMA4 gate; this file is the f32 reference they grade.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/model.h"
#include "sp/forward_dispatch.h"   /* sp_matmul / sp_embed_row / sp_weight_row / sp_as_f32 */
#include "sp/forward_kernels.h"    /* sp_rmsnorm / sp_rmsnorm_head / sp_rope_neox_freqs / sp_attn_head */

#include <stdlib.h>
#include <math.h>
#include <string.h>

/* GELU tanh approximation (Gemma FFN + AltUp gate = ggml_gelu, F16-LUT in the
 * oracle; the f32 closed form matches it to the §8.6.1 floor — same as gemma3.c). */
static float g4_gelu(float x) {
    const float k = 0.7978845608028654f;   /* sqrt(2/pi) */
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

/* weightless RMSNorm over a length-d vector, in place (V-norm; sum of squares in
 * double, the reference precision). */
static void g4_rmsnorm_noweight(float *v, int d, float eps) {
    double ss = 0.0;
    for (int i = 0; i < d; i++) ss += (double)v[i] * (double)v[i];
    float inv = 1.0f / sqrtf((float)(ss / (double)d) + eps);
    for (int i = 0; i < d; i++) v[i] *= inv;
}

int gemma4_forward(const qwen3_model *m, const int32_t *tokens, int n_tok, float *logits) {
    const qwen3_config *c = &m->cfg;
    const int   E  = (int)c->n_embd, FF = (int)c->n_ff, V = (int)c->n_vocab;
    const int   NL = (int)c->n_layers;
    const float eps = c->rms_eps;
    const float embscale = sqrtf((float)E);
    const int   PL     = (int)c->g4_n_embd_per_layer;            /* AltUp width; 0 = none */
    const int   kvfs   = (int)c->g4_n_kv_from_start ? (int)c->g4_n_kv_from_start : NL;
    const int   period = (int)c->g4_swa_period ? (int)c->g4_swa_period : 6;
    const float softcap = c->g4_logit_softcap;
    /* global vs SWA geometry */
    const int   g_nh = (int)c->n_head,    g_nkv = (int)c->n_head_kv,    g_hd = (int)c->head_dim;
    const int   s_nh = (int)c->g4_nh_swa, s_nkv = (int)c->g4_nkv_swa,   s_hd = (int)c->g4_hd_swa;
    const float g_base = c->rope_freq_base, s_base = c->g4_rope_base_swa;
    const int   SW = (int)c->sliding_window;
    const int   QD  = g_nh * g_hd;     /* == s_nh*s_hd (constant proj width) */
    const int   KVD = g_nkv * g_hd;    /* == s_nkv*s_hd (constant proj width) */

    sp_kernels_read_env();
    int rc = 1;

    float  *x   = (float *)malloc((size_t)n_tok * E   * sizeof(float)); /* residual stream */
    float  *nx  = (float *)malloc((size_t)n_tok * E   * sizeof(float)); /* norm scratch */
    float  *q   = (float *)malloc((size_t)n_tok * QD  * sizeof(float));
    float  *ao  = (float *)malloc((size_t)n_tok * QD  * sizeof(float));
    float  *ap  = (float *)malloc((size_t)n_tok * E   * sizeof(float));
    float  *g   = (float *)malloc((size_t)n_tok * FF  * sizeof(float));
    float  *up  = (float *)malloc((size_t)n_tok * FF  * sizeof(float));
    float  *dn  = (float *)malloc((size_t)n_tok * E   * sizeof(float));
    float  *sc  = (float *)malloc((size_t)n_tok       * sizeof(float));
    float **Kst = (float **)calloc((size_t)NL, sizeof(float *)); /* per-owner K (shared idx stay NULL) */
    float **Vst = (float **)calloc((size_t)NL, sizeof(float *));
    /* AltUp scratch */
    float  *ipl   = PL ? (float *)malloc((size_t)n_tok * NL * PL * sizeof(float)) : NULL;
    float  *pgate = PL ? (float *)malloc((size_t)n_tok * PL * sizeof(float)) : NULL;
    float  *pproj = PL ? (float *)malloc((size_t)n_tok * E  * sizeof(float)) : NULL;
    float  *ple   = PL ? (float *)malloc((size_t)NL * PL * sizeof(float)) : NULL;
    if (!x || !nx || !q || !ao || !ap || !g || !up || !dn || !sc || !Kst || !Vst ||
        (PL && (!ipl || !pgate || !pproj || !ple))) goto done;

    /* embedding lookup; scale by sqrt(n_embd) */
    for (int t = 0; t < n_tok; t++) {
        if (sp_embed_row(m, tokens[t], E, x + (size_t)t * E)) goto done;
        float *xt = x + (size_t)t * E;
        for (int i = 0; i < E; i++) xt[i] *= embscale;
    }

    /* ── AltUp precompute: project_per_layer_inputs ──
     * ipl[t,L,:] = ( rmsnorm_{proj_norm}( (per_layer_model_proj·x)·(1/√E) )
     *               + (per_layer_token_embd row(tok)·√PL) ) · (1/√2) */
    if (PL) {
        if (sp_matmul(m, m->per_layer_model_proj, x, n_tok, E, NL * PL, ipl)) goto done; /* reuse ipl as proj scratch */
        const float proj_scale = 1.0f / sqrtf((float)E);
        const float in_scale   = 1.0f / sqrtf(2.0f);
        const float ple_scale  = sqrtf((float)PL);
        const float *pn = sp_as_f32(m, m->per_layer_proj_norm); /* [PL] */
        for (int t = 0; t < n_tok; t++) {
            if (sp_weight_row(m, m->per_layer_token_embd, tokens[t], NL * PL, ple)) goto done;
            for (int L = 0; L < NL; L++) {
                float *row = ipl + ((size_t)t * NL + L) * PL;
                double ss = 0.0;
                for (int i = 0; i < PL; i++) { row[i] *= proj_scale; ss += (double)row[i] * row[i]; }
                float inv = 1.0f / sqrtf((float)(ss / (double)PL) + eps);
                const float *pleL = ple + (size_t)L * PL;
                for (int i = 0; i < PL; i++)
                    row[i] = (row[i] * inv * pn[i] + pleL[i] * ple_scale) * in_scale;
            }
        }
    }

    for (int L = 0; L < NL; L++) {
        const qwen3_layer *ly = &m->layers[L];
        const int   global = ((L % period) == period - 1);
        const int   nh  = global ? g_nh  : s_nh;
        const int   nkv = global ? g_nkv : s_nkv;
        const int   hd  = global ? g_hd  : s_hd;
        const int   grp = nh / nkv;
        const float rbase = global ? g_base : s_base;
        const float *ff = global ? sp_as_f32(m, m->rope_freqs) : NULL; /* [hd/2] proportional factors */
        const int   win = global ? -1 : SW;
        const float ascale = 1.0f;   /* Gemma4: self.scaling = 1.0 */

        /* ── attention ── */
        for (int t = 0; t < n_tok; t++)
            sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, ly->attn_norm), E, eps, nx + (size_t)t * E);

        if (sp_matmul(m, ly->attn_q, nx, n_tok, E, QD, q)) goto done;
        const float *qn = sp_as_f32(m, ly->attn_q_norm);
        for (int t = 0; t < n_tok; t++)
            for (int h = 0; h < nh; h++) {
                float *qh = q + (size_t)t * QD + (size_t)h * hd;
                sp_rmsnorm_head(qh, qn, hd, eps);
                sp_rope_neox_freqs(qh, hd, t, rbase, ff);
            }

        float *Kuse, *Vuse;
        if (L < kvfs) {
            float *K  = (float *)malloc((size_t)n_tok * KVD * sizeof(float));
            float *Vb = (float *)malloc((size_t)n_tok * KVD * sizeof(float));
            if (!K || !Vb) { free(K); free(Vb); goto done; }
            if (sp_matmul(m, ly->attn_k, nx, n_tok, E, KVD, K))  { free(K); free(Vb); goto done; }
            if (sp_matmul(m, ly->attn_v, nx, n_tok, E, KVD, Vb)) { free(K); free(Vb); goto done; }
            const float *kn = sp_as_f32(m, ly->attn_k_norm);
            for (int t = 0; t < n_tok; t++)
                for (int h = 0; h < nkv; h++) {
                    float *kh = K  + (size_t)t * KVD + (size_t)h * hd;
                    sp_rmsnorm_head(kh, kn, hd, eps);
                    sp_rope_neox_freqs(kh, hd, t, rbase, ff);
                    g4_rmsnorm_noweight(Vb + (size_t)t * KVD + (size_t)h * hd, hd, eps);
                }
            Kst[L] = K; Vst[L] = Vb; Kuse = K; Vuse = Vb;
        } else {
            const int src = kvfs - (global ? 1 : 2);
            Kuse = Kst[src]; Vuse = Vst[src];
            if (!Kuse || !Vuse) goto done;
        }

        for (int t = 0; t < n_tok; t++)
            for (int h = 0; h < nh; h++)
                sp_attn_head(q + (size_t)t * QD + (size_t)h * hd, Kuse, Vuse, t, KVD,
                             h / grp, hd, ascale, win, sc, ao + (size_t)t * QD + (size_t)h * hd);

        if (sp_matmul(m, ly->attn_output, ao, n_tok, QD, E, ap)) goto done;
        for (int t = 0; t < n_tok; t++) {
            sp_rmsnorm(ap + (size_t)t * E, sp_as_f32(m, ly->post_attn_norm), E, eps, nx + (size_t)t * E);
            float *xt = x + (size_t)t * E; const float *pt = nx + (size_t)t * E;
            for (int i = 0; i < E; i++) xt[i] += pt[i];
        }

        /* ── FFN (GeGLU) + post_ffw_norm residual ── */
        for (int t = 0; t < n_tok; t++)
            sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, ly->ffn_norm), E, eps, nx + (size_t)t * E);
        if (sp_matmul(m, ly->ffn_gate, nx, n_tok, E, FF, g)) goto done;
        if (sp_matmul(m, ly->ffn_up,   nx, n_tok, E, FF, up)) goto done;
        for (size_t i = 0; i < (size_t)n_tok * FF; i++) g[i] = g4_gelu(g[i]) * up[i];
        if (sp_matmul(m, ly->ffn_down, g, n_tok, FF, E, dn)) goto done;
        for (int t = 0; t < n_tok; t++) {
            sp_rmsnorm(dn + (size_t)t * E, sp_as_f32(m, ly->post_ffw_norm), E, eps, nx + (size_t)t * E);
            float *xt = x + (size_t)t * E; const float *pt = nx + (size_t)t * E;
            for (int i = 0; i < E; i++) xt[i] += pt[i];
        }

        /* ── AltUp per-layer-input injection ── */
        if (PL) {
            if (sp_matmul(m, ly->per_layer_inp_gate, x, n_tok, E, PL, pgate)) goto done;
            for (int t = 0; t < n_tok; t++) {
                float *pg = pgate + (size_t)t * PL;
                const float *iplL = ipl + ((size_t)t * NL + L) * PL;
                for (int i = 0; i < PL; i++) pg[i] = g4_gelu(pg[i]) * iplL[i];
            }
            if (sp_matmul(m, ly->per_layer_proj, pgate, n_tok, PL, E, pproj)) goto done;
            for (int t = 0; t < n_tok; t++) {
                sp_rmsnorm(pproj + (size_t)t * E, sp_as_f32(m, ly->per_layer_post_norm), E, eps, nx + (size_t)t * E);
                float *xt = x + (size_t)t * E; const float *pt = nx + (size_t)t * E;
                for (int i = 0; i < E; i++) xt[i] += pt[i];
            }
        }

        /* ── per-layer output scale (scalar) ── */
        if (ly->out_scale) {
            const float *os = sp_as_f32(m, ly->out_scale);
            if (os) { float s = os[0]; for (size_t i = 0; i < (size_t)n_tok * E; i++) x[i] *= s; }
        }
    }

    /* ── final norm + tied LM head + softcap ── */
    for (int t = 0; t < n_tok; t++)
        sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, m->output_norm), E, eps, nx + (size_t)t * E);
    if (sp_matmul(m, m->output, nx, n_tok, E, V, logits)) goto done;
    if (softcap > 0.0f)
        for (size_t i = 0; i < (size_t)n_tok * V; i++)
            logits[i] = tanhf(logits[i] / softcap) * softcap;

    rc = 0;
done:
    free(x); free(nx); free(q); free(ao); free(ap); free(g); free(up); free(dn); free(sc);
    if (Kst) { for (int L = 0; L < NL; L++) free(Kst[L]); free(Kst); }
    if (Vst) { for (int L = 0; L < NL; L++) free(Vst[L]); free(Vst); }
    free(ipl); free(pgate); free(pproj); free(ple);
    return rc;
}

/* NTT.5c-style backend-aware wrapper. Gemma4 has no NTT-attention overlay yet
 * (same rationale as gemma3_forward_ex2); the triple is accepted for ABI
 * uniformity and ignored — gemma4 always takes the host f32 path. */
int gemma4_forward_ex2(const qwen3_model *m, const int32_t *tokens, int n_tok,
                       float *logits,
                       void *backend_handle,
                       sp_compute_ntt_dispatch_fn backend_forward,
                       sp_compute_ntt_dispatch_fn backend_inverse) {
    (void)backend_handle; (void)backend_forward; (void)backend_inverse;
    return gemma4_forward(m, tokens, n_tok, logits);
}
