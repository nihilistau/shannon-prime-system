/* qwen25.c — Qwen2.5 f32 reference forward pass.
 * Deltas vs Qwen3 (forward.c): no embedding scale, no QK norms, QKV biases
 * added after projection, SwiGLU FFN (same activation as Qwen3), no sandwich
 * norms, no sliding-window attention. All layers use rope_freq_base = 1e6.
 *
 * NTT.5c: qwen25_forward grew an NTT-attention overlay that mirrors the
 * forward.c (qwen3) overlay. The overlay is gated by SP_ENGINE_NTT_ATTN=1
 * and uses sp_pr_bluestein for HD ∈ {2..64,128,256} (Qwen2.5-Coder-0.5B
 * Memory model is HD=64 — the actual unblocking target) and direct sp_pr
 * for HD ∈ {128,256,512}. When a session has a registered compute backend
 * (NTT.5b path), the backend is plumbed into the Bluestein ctx so inner
 * length-M NTT calls route through the cDSP/HVX dispatcher. */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/model.h"
#include "sp/forward_dispatch.h"
#include "sp/forward_kernels.h"
#include "sp/poly_ring.h"           /* NTT.5c: direct sp_pr (HD ∈ {128,256,512}) */
#include "sp/poly_ring_bluestein.h" /* NTT.5c: Bluestein (HD ∈ {2..256} ∖ {512}) */

#include <stdlib.h>
#include <stdio.h>                  /* NTT.5c: getenv via _CRT_SECURE_NO_WARNINGS */
#include <math.h>

/* NTT.5c: identical scale to forward.c so the qwen25 path matches the qwen3
 * path's int32 quantization domain (|coeff| < 2^21, |<q,k>| < 2^49 << M/2). */
#define SP_NTT_ATTN_SCALE 65536.0

int qwen25_forward_ex2(const qwen3_model *m, const int32_t *tokens, int n_tok,
                       float *logits,
                       void *backend_handle,
                       sp_compute_ntt_dispatch_fn backend_forward,
                       sp_compute_ntt_dispatch_fn backend_inverse) {
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv;
    const int QD = NH * HD;
    const int KVD = NKV * HD;
    const int group = NH / NKV;
    const int V = (int)c->n_vocab;
    const float eps = c->rms_eps;
    const float base = c->rope_freq_base;
    const float ascale = 1.0f / sqrtf((float)HD);

    sp_kernels_read_env();
    /* NTT.5c: read SP_ENGINE_NTT_ATTN runtime gate here too (qwen25 didn't
     * have a g_ntt_attn before; mirror forward.c's read_env_knobs pattern
     * but local to this forward call since qwen25 has no module-level state). */
    int g_ntt_attn = 0;
    { const char *e = getenv("SP_ENGINE_NTT_ATTN"); g_ntt_attn = (e && e[0] == '1'); }

    int rc = 1;
    /* NTT.5c: HD-dispatched poly-ring ctx (identical logic to forward.c). */
    sp_pr_ctx           *pr   = NULL;
    sp_pr_bluestein_ctx *pr_b = NULL;
    int overlay_active = 0;
    int32_t *qi = NULL, *ki = NULL;
    float *x   = (float *)malloc((size_t)n_tok * E * sizeof(float));
    float *nx  = (float *)malloc((size_t)n_tok * E * sizeof(float));
    float *q   = (float *)malloc((size_t)n_tok * QD * sizeof(float));
    float *k   = (float *)malloc((size_t)n_tok * KVD * sizeof(float));
    float *vv  = (float *)malloc((size_t)n_tok * KVD * sizeof(float));
    float *ao  = (float *)malloc((size_t)n_tok * QD * sizeof(float));
    float *ap  = (float *)malloc((size_t)n_tok * E * sizeof(float));
    float *g   = (float *)malloc((size_t)n_tok * FF * sizeof(float));
    float *up  = (float *)malloc((size_t)n_tok * FF * sizeof(float));
    float *dn  = (float *)malloc((size_t)n_tok * E * sizeof(float));
    float *sc  = (float *)malloc((size_t)n_tok * sizeof(float));
    if (!x || !nx || !q || !k || !vv || !ao || !ap || !g || !up || !dn || !sc) goto done;

    if (g_ntt_attn) {
        if (HD == 128 || HD == 256 || HD == 512) {
            pr = sp_pr_init((uint32_t)HD);
            if (pr) overlay_active = 1;
        } else if (HD >= 2 && HD <= 256 && (HD & (HD - 1)) == 0) {
            pr_b = sp_pr_bluestein_init((uint32_t)HD);
            if (pr_b) {
                overlay_active = 1;
                if (backend_handle || backend_forward || backend_inverse)
                    sp_pr_bluestein_set_backend(pr_b, backend_handle,
                                                backend_forward,
                                                backend_inverse);
            }
        }
        if (overlay_active) {
            qi = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
            ki = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
            if (!qi || !ki) goto done;
        }
    }

    for (int t = 0; t < n_tok; t++)
        if (sp_embed_row(m, tokens[t], E, x + (size_t)t * E)) goto done;

    for (uint32_t L = 0; L < c->n_layers; L++) {
        const qwen3_layer *ly = &m->layers[L];

        /* ── attention block ── */
        for (int t = 0; t < n_tok; t++)
            sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, ly->attn_norm), E, eps, nx + (size_t)t * E);

        if (sp_matmul(m, ly->attn_q, nx, n_tok, E, QD, q)) goto done;
        if (sp_matmul(m, ly->attn_k, nx, n_tok, E, KVD, k)) goto done;
        if (sp_matmul(m, ly->attn_v, nx, n_tok, E, KVD, vv)) goto done;

        /* add QKV biases */
        const float *qb = sp_as_f32(m, ly->attn_q_bias);
        const float *kb = sp_as_f32(m, ly->attn_k_bias);
        const float *vb = sp_as_f32(m, ly->attn_v_bias);
        for (int t = 0; t < n_tok; t++) {
            float *qt = q  + (size_t)t * QD;
            float *kt = k  + (size_t)t * KVD;
            float *vt = vv + (size_t)t * KVD;
            for (int i = 0; i < QD;  i++) qt[i] += qb[i];
            for (int i = 0; i < KVD; i++) kt[i] += kb[i];
            for (int i = 0; i < KVD; i++) vt[i] += vb[i];
        }

        /* NEOX RoPE (no QK norm) */
        for (int t = 0; t < n_tok; t++) {
            for (int h = 0; h < NH; h++)
                sp_rope_neox(q + (size_t)t * QD + (size_t)h * HD, HD, t, base);
            for (int h = 0; h < NKV; h++)
                sp_rope_neox(k + (size_t)t * KVD + (size_t)h * HD, HD, t, base);
        }

        /* GQA causal attention (full causal). NTT.5c: overlay matches forward.c. */
        for (int t = 0; t < n_tok; t++) {
            for (int h = 0; h < NH; h++) {
                int kvh = h / group;
                const float *qh = q + (size_t)t * QD + (size_t)h * HD;
                float *out = ao + (size_t)t * QD + (size_t)h * HD;
                if (!g_ntt_attn || !overlay_active) {
                    sp_attn_head(qh, k, vv, t, KVD, kvh, HD, ascale, -1, sc, out);
                    continue;
                }
                for (int i = 0; i < HD; i++) qi[i] = (int32_t)lrintf(qh[i] * (float)SP_NTT_ATTN_SCALE);
                float maxs = -INFINITY;
                for (int s = 0; s <= t; s++) {
                    const float *kh = k + (size_t)s * KVD + (size_t)kvh * HD;
                    for (int i = 0; i < HD; i++) ki[i] = (int32_t)lrintf(kh[i] * (float)SP_NTT_ATTN_SCALE);
                    int64_t ip = pr_b ? sp_pr_bluestein_inner(pr_b, qi, ki)
                                      : sp_pr_inner(pr, qi, ki);
                    float d = (float)((double)ip / (SP_NTT_ATTN_SCALE * SP_NTT_ATTN_SCALE)) * ascale;
                    sc[s] = d;
                    if (d > maxs) maxs = d;
                }
                float sum = 0.0f;
                for (int s = 0; s <= t; s++) { sc[s] = expf(sc[s] - maxs); sum += sc[s]; }
                float inv = 1.0f / sum;
                for (int i = 0; i < HD; i++) out[i] = 0.0f;
                for (int s = 0; s <= t; s++) {
                    float w = sc[s] * inv;
                    const float *vh = vv + (size_t)s * KVD + (size_t)kvh * HD;
                    for (int i = 0; i < HD; i++) out[i] += w * vh[i];
                }
            }
        }

        if (sp_matmul(m, ly->attn_output, ao, n_tok, QD, E, ap)) goto done;
        for (size_t i = 0; i < (size_t)n_tok * E; i++) x[i] += ap[i];

        /* ── FFN block (SwiGLU) ── */
        for (int t = 0; t < n_tok; t++)
            sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, ly->ffn_norm), E, eps, nx + (size_t)t * E);
        if (sp_matmul(m, ly->ffn_gate, nx, n_tok, E, FF, g)) goto done;
        if (sp_matmul(m, ly->ffn_up,   nx, n_tok, E, FF, up)) goto done;
        for (size_t i = 0; i < (size_t)n_tok * FF; i++) {
            float gv = g[i];
            g[i] = gv / (1.0f + expf(-gv)) * up[i];
        }
        if (sp_matmul(m, ly->ffn_down, g, n_tok, FF, E, dn)) goto done;
        for (size_t i = 0; i < (size_t)n_tok * E; i++) x[i] += dn[i];
    }

    /* ── final norm + LM head ── */
    for (int t = 0; t < n_tok; t++)
        sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, m->output_norm), E, eps, nx + (size_t)t * E);
    if (sp_matmul(m, m->output, nx, n_tok, E, V, logits)) goto done;

    rc = 0;
done:
    free(x); free(nx); free(q); free(k); free(vv); free(ao); free(ap);
    free(g); free(up); free(dn); free(sc);
    free(qi); free(ki);
    sp_pr_free(pr);
    sp_pr_bluestein_free(pr_b);
    return rc;
}

/* NTT.5c: legacy entry point becomes thin wrapper passing all-NULL backend triple. */
int qwen25_forward(const qwen3_model *m, const int32_t *tokens, int n_tok, float *logits) {
    return qwen25_forward_ex2(m, tokens, n_tok, logits, NULL, NULL, NULL);
}
