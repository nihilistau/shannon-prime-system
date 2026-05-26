/* qwen25.c — Qwen2.5 f32 reference forward pass.
 * Deltas vs Qwen3 (forward.c): no embedding scale, no QK norms, QKV biases
 * added after projection, SwiGLU FFN (same activation as Qwen3), no sandwich
 * norms, no sliding-window attention. All layers use rope_freq_base = 1e6. */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/model.h"
#include "sp/forward_dispatch.h"
#include "sp/forward_kernels.h"

#include <stdlib.h>
#include <math.h>

int qwen25_forward(const qwen3_model *m, const int32_t *tokens, int n_tok, float *logits) {
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

    int rc = 1;
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

        /* GQA causal attention (full causal) */
        for (int t = 0; t < n_tok; t++)
            for (int h = 0; h < NH; h++)
                sp_attn_head(q + (size_t)t * QD + (size_t)h * HD, k, vv, t, KVD,
                             h / group, HD, ascale, -1, sc,
                             ao + (size_t)t * QD + (size_t)h * HD);

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
    return rc;
}
