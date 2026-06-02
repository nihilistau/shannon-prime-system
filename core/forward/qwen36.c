/* qwen36.c — Qwen3.6 / qwen35moe f32 reference forward (prefill, causal).
 *
 * A Gated DeltaNet (Qwen3-Next family) + MoE hybrid. Per-layer bifurcation:
 * full-attn iff (L+1)%full_attn_interval==0, else a GDN linear-attention block.
 * MoE FFN (routed + shared expert) on every layer. See papers/SPEC-qwen35moe-GDN.md
 * and the per-layer oracle fingerprints in papers/qwen35moe-oracle-fingerprints.txt.
 *
 * Validation: SP_Q36_DBG=1 prints per-layer block fingerprints (sum/absum/v0..2 of
 * the last-token column) matching the llama.cpp g35moe_oracle_dbg format, for
 * block-by-block localization (relative, not bit-exact: oracle Q4 vs SP f32).
 *
 * STATUS: GDN block implemented + validated. Full-attn (Stage 2b) and MoE FFN
 * (Stage 2c) are pass-through stubs in this revision. */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/model.h"
#include "sp/forward_dispatch.h"   /* sp_matmul / sp_embed_row / sp_as_f32 / sp_kernels_read_env */
#include "sp/forward_kernels.h"    /* sp_rmsnorm / sp_rmsnorm_head */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

static float silu_f(float x)     { return x / (1.0f + expf(-x)); }
static float sigmoid_f(float x)  { return 1.0f / (1.0f + expf(-x)); }
static float softplus_f(float x) { return (x > 20.0f) ? x : log1pf(expf(x)); }

/* ggml_l2_norm: v <- v / sqrt(sum(v^2) + eps), over d contiguous elements. */
static void l2norm(float *v, int d, float eps) {
    double s = 0.0;
    for (int i = 0; i < d; i++) s += (double)v[i] * v[i];
    float inv = 1.0f / sqrtf((float)s + eps);
    for (int i = 0; i < d; i++) v[i] *= inv;
}

static void dbg_fp(const char *name, int il, const float *col, int n) {
    double s = 0, a = 0;
    for (int i = 0; i < n; i++) { s += col[i]; a += (col[i] < 0 ? -col[i] : col[i]); }
    fprintf(stderr, "SP  %-14s-%d ne=[%d,?] sum=%.5f absum=%.5f v0=%.6f v1=%.6f v2=%.6f\n",
            name, il, n, s, a, col[0], n > 1 ? col[1] : 0, n > 2 ? col[2] : 0);
    fflush(stderr);
}

int qwen36_forward(const qwen3_model *m, const int32_t *tokens, int n_tok, float *logits) {
    const qwen3_config *c = &m->cfg;
    const int E   = (int)c->n_embd;          /* 2048 */
    const int V   = (int)c->n_vocab;
    const float eps = c->rms_eps;
    const int dbg = getenv("SP_Q36_DBG") != NULL;

    /* GDN geometry */
    const int Sd   = (int)c->q36_gdn_state;      /* 128 (head dim) */
    const int Hk   = (int)c->q36_gdn_n_k_heads;  /* 16 */
    const int Hv   = (int)c->q36_gdn_n_v_heads;  /* 32 */
    const int Vdim = (int)c->q36_gdn_inner;      /* 4096 = Hv*Sd (value_dim) */
    const int Kdim = Hk * Sd;                    /* 2048 (key_dim) */
    const int convC = Kdim * 2 + Vdim;           /* 8192 (conv channels) */
    const int CK   = (int)c->q36_gdn_conv_k;     /* 4 */

    int rc = 1;
    float *x   = (float *)malloc((size_t)n_tok * E * sizeof(float));   /* residual stream */
    float *nx  = (float *)malloc((size_t)n_tok * E * sizeof(float));   /* normed scratch */
    float *blk = (float *)malloc((size_t)n_tok * E * sizeof(float));   /* block output [E] */
    /* GDN scratch */
    float *qkv = (float *)malloc((size_t)n_tok * convC * sizeof(float));
    float *zg  = (float *)malloc((size_t)n_tok * Vdim * sizeof(float));
    float *cs  = (float *)malloc((size_t)n_tok * convC * sizeof(float));   /* conv+silu */
    float *bet = (float *)malloc((size_t)n_tok * Hv * sizeof(float));
    float *alp = (float *)malloc((size_t)n_tok * Hv * sizeof(float));
    float *od  = (float *)malloc((size_t)n_tok * Vdim * sizeof(float));    /* recurrence out */
    float *fno = (float *)malloc((size_t)n_tok * Vdim * sizeof(float));    /* gated-norm out */
    float *St  = (float *)malloc((size_t)Sd * Sd * sizeof(float));         /* per-head state */
    float *skv = (float *)malloc((size_t)Sd * sizeof(float));
    float *dlt = (float *)malloc((size_t)Sd * sizeof(float));
    if (!x || !nx || !blk || !qkv || !zg || !cs || !bet || !alp || !od || !fno || !St || !skv || !dlt)
        goto done;

    sp_kernels_read_env();

    for (int t = 0; t < n_tok; t++)
        if (sp_embed_row(m, tokens[t], E, x + (size_t)t * E)) goto done;   /* no embed scale */

    for (uint32_t il = 0; il < c->n_layers; il++) {
        const qwen3_layer *L = &m->layers[il];

        /* pre-norm */
        for (int t = 0; t < n_tok; t++)
            sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, L->attn_norm), E, eps, nx + (size_t)t * E);

        if (L->q36_is_recurrent) {
            /* ── Gated DeltaNet block ── */
            if (sp_matmul(m, L->gdn_qkv,  nx, n_tok, E, convC, qkv)) goto done;
            if (sp_matmul(m, L->gdn_gate, nx, n_tok, E, Vdim,  zg))  goto done;
            if (sp_matmul(m, L->gdn_beta, nx, n_tok, E, Hv,    bet)) goto done;
            if (sp_matmul(m, L->gdn_alpha,nx, n_tok, E, Hv,    alp)) goto done;
            const float *cw = sp_as_f32(m, L->gdn_conv1d);   /* [CK, convC]; idx(j,ch)=ch*CK+j */
            const float *aA = sp_as_f32(m, L->gdn_a);        /* [Hv] */
            const float *dB = sp_as_f32(m, L->gdn_dt_bias);  /* [Hv] */

            /* causal depthwise conv (kernel CK) + SiLU, per channel */
            for (int t = 0; t < n_tok; t++) {
                for (int ch = 0; ch < convC; ch++) {
                    float acc = 0.0f;
                    for (int j = 0; j < CK; j++) {
                        int tt = t - (CK - 1) + j;
                        if (tt >= 0) acc += qkv[(size_t)tt * convC + ch] * cw[ch * CK + j];
                    }
                    cs[(size_t)t * convC + ch] = silu_f(acc);
                }
            }
            /* split + L2-norm q,k per head (v unnormed). layout in cs per token:
             *   q: [0, Kdim)  -> Hk heads x Sd ; k: [Kdim, 2Kdim) ; v: [2Kdim, convC) */
            for (int t = 0; t < n_tok; t++) {
                float *qb = cs + (size_t)t * convC;            /* q base   */
                float *kb = qb + Kdim;                         /* k base   */
                for (int h = 0; h < Hk; h++) {
                    l2norm(qb + h * Sd, Sd, eps);
                    l2norm(kb + h * Sd, Sd, eps);
                }
            }
            /* per v-head gated delta-rule recurrence (sequential over tokens).
             * kq head for v-head h is (h % Hk) (ggml_repeat tiling). q scaled 1/sqrt(Sd). */
            const float qscale = 1.0f / sqrtf((float)Sd);
            for (int h = 0; h < Hv; h++) {
                int hk = h % Hk;
                memset(St, 0, (size_t)Sd * Sd * sizeof(float));
                for (int t = 0; t < n_tok; t++) {
                    const float *qh = cs + (size_t)t * convC + (size_t)hk * Sd;           /* q (normed) */
                    const float *kh = cs + (size_t)t * convC + Kdim + (size_t)hk * Sd;    /* k (normed) */
                    const float *vh = cs + (size_t)t * convC + 2 * Kdim + (size_t)h * Sd; /* v */
                    float gh = expf(aA[h] * softplus_f(alp[(size_t)t * Hv + h] + dB[h])); /* decay */
                    float bh = bet[(size_t)t * Hv + h];
                    /* S *= gh */
                    for (int i = 0; i < Sd * Sd; i++) St[i] *= gh;
                    /* sk[j] = sum_i S[i,j]*k[i] ; d[j] = bh*(v[j]-sk[j]) */
                    for (int j = 0; j < Sd; j++) {
                        float s = 0.0f;
                        for (int i = 0; i < Sd; i++) s += St[(size_t)i * Sd + j] * kh[i];
                        skv[j] = s;
                        dlt[j] = bh * (vh[j] - s);
                    }
                    /* S[i,j] += k[i]*d[j] */
                    for (int i = 0; i < Sd; i++) {
                        float ki = kh[i];
                        float *Si = St + (size_t)i * Sd;
                        for (int j = 0; j < Sd; j++) Si[j] += ki * dlt[j];
                    }
                    /* o[j] = sum_i S[i,j]*(q[i]*qscale) */
                    float *oh = od + (size_t)t * Vdim + (size_t)h * Sd;
                    for (int j = 0; j < Sd; j++) {
                        float s = 0.0f;
                        for (int i = 0; i < Sd; i++) s += St[(size_t)i * Sd + j] * (qh[i] * qscale);
                        oh[j] = s;
                    }
                }
            }
            /* gated output norm: RMSNorm(o_head, ssm_norm) * silu(z_head), per head */
            {
                const float *gn = sp_as_f32(m, L->gdn_norm);   /* [Sd] */
                for (int t = 0; t < n_tok; t++) {
                    for (int h = 0; h < Hv; h++) {
                        float *o_in  = od  + (size_t)t * Vdim + (size_t)h * Sd;
                        float *o_out = fno + (size_t)t * Vdim + (size_t)h * Sd;
                        float tmp[256];
                        memcpy(tmp, o_in, (size_t)Sd * sizeof(float));
                        sp_rmsnorm_head(tmp, gn, Sd, eps);     /* in-place RMSNorm w/ weight */
                        const float *zh = zg + (size_t)t * Vdim + (size_t)h * Sd;
                        for (int d = 0; d < Sd; d++) o_out[d] = tmp[d] * silu_f(zh[d]);
                    }
                }
            }
            if (sp_matmul(m, L->gdn_out, fno, n_tok, Vdim, E, blk)) goto done;
            if (dbg) dbg_fp("linear_attn_out", (int)il, blk + (size_t)(n_tok - 1) * E, E);
        } else {
            /* ── full-attention block (Stage 2b: STUB pass-through) ── */
            memset(blk, 0, (size_t)n_tok * E * sizeof(float));
            if (dbg) dbg_fp("attn_output(stub)", (int)il, blk + (size_t)(n_tok - 1) * E, E);
        }

        /* attn residual */
        for (int i = 0; i < n_tok * E; i++) x[i] += blk[i];

        /* post-attn norm -> MoE FFN (Stage 2c: STUB, adds 0) -> ffn residual */
        /* (no-op this revision; residual stream carries attn output only) */
        if (dbg) dbg_fp("l_out", (int)il, x + (size_t)(n_tok - 1) * E, E);
    }

    /* final norm + LM head */
    for (int t = 0; t < n_tok; t++)
        sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, m->output_norm), E, eps, nx + (size_t)t * E);
    if (sp_matmul(m, m->output, nx, n_tok, E, V, logits)) goto done;
    rc = 0;

done:
    free(x); free(nx); free(blk); free(qkv); free(zg); free(cs); free(bet);
    free(alp); free(od); free(fno); free(St); free(skv); free(dlt);
    return rc;
}
