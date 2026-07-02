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
 * STATUS: COMPLETE — GDN + gated full-attn/IMRoPE + MoE FFN all implemented;
 * M_QWEN36 gate = top-1 bit-exact to the llama.cpp oracle (3/3; re-verified
 * 2026-07-02, G-MOE-FORWARD-35B). Persistent-STATE decode (qwen36_state /
 * qwen36_step, end of file) added 2026-07-02 for NORTHSTAR brick 3: the GDN
 * recurrence is O(1)-state by construction; only the 10 full-attn layers carry
 * a KV window. (The former "Stage 2b/2c stubs" note here was stale.) */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/model.h"
#include "sp/forward_dispatch.h"   /* sp_matmul / sp_embed_row / sp_as_f32 / sp_kernels_read_env */
#include "sp/forward_kernels.h"    /* sp_rmsnorm / sp_rmsnorm_head */
#include "sp/weight_dtype.h"       /* sp_dequant_row (rank-3 expert slices) */
#include "sp/arena.h"              /* sp_arena_find / sp_arena_tensor (.sp-model path) */
#include "sp/frobenius_lift.h"     /* sp_frob_packed_dequant_row (arena expert-slice) */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>   /* timespec_get (SP_Q36_PROF section timers) */

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

/* bytes for `n` contiguous weight elements (F32/F16/Q8_0/Q4_K/Q6_K). */
static size_t q36_rb(uint32_t type, int n) {
    switch (type) {
        case 0:  return (size_t)n * 4;          /* F32  */
        case 1:  return (size_t)n * 2;          /* F16  */
        case 8:  return (size_t)(n / 32) * 34;  /* Q8_0 */
        case 12: return (size_t)(n / 256) * 144;/* Q4_K */
        case 14: return (size_t)(n / 256) * 210;/* Q6_K */
        default: return 0;
    }
}

/* single-token matmul with expert slice `e` of a rank-3 tensor [in, out, n_expert]:
 * row o of expert e is the `in`-vector at element ((e*out)+o)*in. f32 reference dot. */
static int expert_mm(const qwen3_model *m, const gguf_tensor *W, int e,
                     const float *X, int in, int out, float *Y) {
    /* .sp-model path: no GGUF — read expert e's slice from the packed arena tensor
     * (rank-3 packed as (rows*n_expert) rows; expert e = rows [e*out, (e+1)*out)). */
    /* NORTHSTAR brick 4: rows independent -> OpenMP over o, per-thread wrow scratch;
     * per-row math untouched = bit-identical to serial (parity re-gated). Serial null
     * floor when compiled without /openmp. */
    if (m->arena) {
        const sp_arena_tensor *at = sp_arena_find(m->arena, W->name);
        if (at) {
            int fail = 0;
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
                float *wrow = (float *)malloc((size_t)in * sizeof(float));
                if (!wrow) {
                    fail = 1;
                } else {
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 8)
#endif
                    for (int o = 0; o < out; o++) {
                        if (fail) continue;
                        if (sp_frob_packed_dequant_row(&at->pt, e * out + o, wrow)) { fail = 1; continue; }
                        float acc = 0.0f;
                        for (int i = 0; i < in; i++) acc += wrow[i] * X[i];
                        Y[o] = acc;
                    }
                    free(wrow);
                }
            }
            return fail;
        }
    }
    /* GGUF-direct path */
    const uint8_t *base = (const uint8_t *)gguf_tensor_data(m->gguf, W);
    size_t rb = q36_rb(W->type, in);
    if (!base || rb == 0) return 1;
    {
        int fail = 0;
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            float *wrow = (float *)malloc((size_t)in * sizeof(float));
            if (!wrow) {
                fail = 1;
            } else {
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 8)
#endif
                for (int o = 0; o < out; o++) {
                    if (fail) continue;
                    if (sp_dequant_row(base + ((size_t)e * out + o) * rb, W->type, in, wrow)) { fail = 1; continue; }
                    float acc = 0.0f;
                    for (int i = 0; i < in; i++) acc += wrow[i] * X[i];
                    Y[o] = acc;
                }
                free(wrow);
            }
        }
        return fail;
    }
}

/* MoE FFN: f32 softmax/top-k/renorm router (NEVER quantized — top-k is a discrete
 * cliff) + Z_q/f32 routed experts (SwiGLU) + sigmoid-gated shared expert. */
static int moe_ffn(const qwen3_model *m, const qwen3_layer *L, const qwen3_config *c,
                   const float *nx, int n_tok, float *out) {
    const int E = (int)c->n_embd;
    const int NE = (int)c->q36_n_expert, NU = (int)c->q36_n_expert_used;
    const int FF = (int)c->q36_n_ff_exp, FS = (int)c->q36_n_ff_shexp;
    const int FM = FF > FS ? FF : FS;
    const float wscale = c->q36_expert_weights_scale;
    int rc = 1;
    float *lg = (float *)malloc((size_t)NE * sizeof(float));
    float *g  = (float *)malloc((size_t)FM * sizeof(float));
    float *u  = (float *)malloc((size_t)FM * sizeof(float));
    float *hh = (float *)malloc((size_t)FM * sizeof(float));
    float *de = (float *)malloc((size_t)E  * sizeof(float));
    float *sh = (float *)malloc((size_t)E  * sizeof(float));
    int   *idx = (int *)malloc((size_t)NU * sizeof(int));
    float *wt  = (float *)malloc((size_t)NU * sizeof(float));
    char  *used = (char *)malloc((size_t)NE);
    if (!lg || !g || !u || !hh || !de || !sh || !idx || !wt || !used) goto done;

    for (int t = 0; t < n_tok; t++) {
        const float *x = nx + (size_t)t * E;
        float *yo = out + (size_t)t * E;
        /* router: f32 softmax over all NE experts. ffn_gate_inp is F32 [E,NE] — read via
         * sp_as_f32 (works GGUF-direct AND .sp-model owned-f32; not arena/sp_matmul). */
        const float *gi = sp_as_f32(m, L->ffn_gate_inp);
        if (!gi) goto done;
        for (int o = 0; o < NE; o++) {
            const float *wo = gi + (size_t)o * E;
            float a = 0.0f; for (int i = 0; i < E; i++) a += x[i] * wo[i];
            lg[o] = a;
        }
        float mx = lg[0];
        for (int i = 1; i < NE; i++) if (lg[i] > mx) mx = lg[i];
        double se = 0.0;
        for (int i = 0; i < NE; i++) { lg[i] = expf(lg[i] - mx); se += lg[i]; }
        for (int i = 0; i < NE; i++) lg[i] = (float)(lg[i] / se);
        /* top-NU by prob, then renormalize the chosen weights and scale */
        memset(used, 0, (size_t)NE);
        float wsum = 0.0f;
        for (int k = 0; k < NU; k++) {
            int best = -1; float bv = -1.0f;
            for (int i = 0; i < NE; i++) if (!used[i] && lg[i] > bv) { bv = lg[i]; best = i; }
            used[best] = 1; idx[k] = best; wt[k] = bv; wsum += bv;
        }
        for (int k = 0; k < NU; k++) wt[k] = (wt[k] / wsum) * wscale;
        /* routed experts (SwiGLU), accumulate weighted */
        memset(yo, 0, (size_t)E * sizeof(float));
        for (int k = 0; k < NU; k++) {
            int e = idx[k];
            if (expert_mm(m, L->ffn_gate_exps, e, x, E, FF, g)) goto done;
            if (expert_mm(m, L->ffn_up_exps,   e, x, E, FF, u)) goto done;
            for (int i = 0; i < FF; i++) hh[i] = silu_f(g[i]) * u[i];
            if (expert_mm(m, L->ffn_down_exps, e, hh, FF, E, de)) goto done;
            float w = wt[k];
            for (int i = 0; i < E; i++) yo[i] += w * de[i];
        }
        /* shared expert (always on), sigmoid-gated */
        if (sp_matmul(m, L->ffn_gate_shexp, x, 1, E, FS, g)) goto done;
        if (sp_matmul(m, L->ffn_up_shexp,   x, 1, E, FS, u)) goto done;
        for (int i = 0; i < FS; i++) hh[i] = silu_f(g[i]) * u[i];
        if (sp_matmul(m, L->ffn_down_shexp, hh, 1, FS, E, sh)) goto done;
        const float *gs = sp_as_f32(m, L->ffn_gate_inp_shexp);   /* F32 [E] shared-expert gate */
        if (!gs) goto done;
        float sg = 0.0f;
        for (int i = 0; i < E; i++) sg += x[i] * gs[i];
        sg = sigmoid_f(sg);
        for (int i = 0; i < E; i++) yo[i] += sg * sh[i];
    }
    rc = 0;
done:
    free(lg); free(g); free(u); free(hh); free(de); free(sh); free(idx); free(wt); free(used);
    return rc;
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
    float *moe = (float *)malloc((size_t)n_tok * E * sizeof(float));   /* MoE FFN output [E] */
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
    if (!x || !nx || !blk || !moe || !qkv || !zg || !cs || !bet || !alp || !od || !fno || !St || !skv || !dlt)
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
            if (dbg) dbg_fp("conv_output_silu", (int)il, cs + (size_t)(n_tok - 1) * convC, convC);
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
                    float bh = sigmoid_f(bet[(size_t)t * Hv + h]);   /* beta = sigmoid(ssm_beta.x) */
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
            /* ── gated full-attention + IMRoPE block ──
             * wq outputs [query|gate] per head (stride 2*HD); Q/K RMSNorm over HD;
             * RoPE (NEOX on first n_rot dims; IMRoPE reduces to NEOX for text — all
             * mrope position components equal the token position); GQA causal attn;
             * output *= sigmoid(gate); wo. */
            const int NH = (int)c->n_head, NKV = (int)c->n_head_kv, HD = (int)c->head_dim;
            const int QD = NH * HD, KVD = NKV * HD, group = NH / NKV;
            const int nrot = (int)c->q36_rope_dim;
            const float rbase = c->q36_rope_base;
            const float ascale = 1.0f / sqrtf((float)HD);
            const int QD2 = NH * HD * 2;                 /* == convC; reuse qkv for qf */
            float *qf = qkv;
            float *qq = (float *)malloc((size_t)n_tok * QD * sizeof(float));
            float *gt = (float *)malloc((size_t)n_tok * QD * sizeof(float));
            float *kk = (float *)malloc((size_t)n_tok * KVD * sizeof(float));
            float *vv = (float *)malloc((size_t)n_tok * KVD * sizeof(float));
            float *sc = (float *)malloc((size_t)n_tok * sizeof(float));
            if (!qq || !gt || !kk || !vv || !sc) { free(qq); free(gt); free(kk); free(vv); free(sc); goto done; }
            int af = (sp_matmul(m, L->attn_q, nx, n_tok, E, QD2, qf) ||
                      sp_matmul(m, L->attn_k, nx, n_tok, E, KVD, kk) ||
                      sp_matmul(m, L->attn_v, nx, n_tok, E, KVD, vv));
            const float *qnw = sp_as_f32(m, L->attn_q_norm);
            const float *knw = sp_as_f32(m, L->attn_k_norm);
            for (int t = 0; t < n_tok && !af; t++) {
                for (int h = 0; h < NH; h++) {
                    const float *src = qf + (size_t)t * QD2 + (size_t)h * HD * 2;
                    float *qh = qq + (size_t)t * QD + (size_t)h * HD;
                    float *gh = gt + (size_t)t * QD + (size_t)h * HD;
                    memcpy(qh, src,      (size_t)HD * sizeof(float));   /* query half */
                    memcpy(gh, src + HD, (size_t)HD * sizeof(float));   /* gate half  */
                    sp_rmsnorm_head(qh, qnw, HD, eps);
                    sp_rope_neox(qh, nrot, t, rbase);
                }
                for (int h = 0; h < NKV; h++) {
                    float *kh = kk + (size_t)t * KVD + (size_t)h * HD;
                    sp_rmsnorm_head(kh, knw, HD, eps);
                    sp_rope_neox(kh, nrot, t, rbase);
                }
            }
            for (int t = 0; t < n_tok && !af; t++) {
                for (int h = 0; h < NH; h++) {
                    int kvh = h / group;
                    const float *qh = qq + (size_t)t * QD + (size_t)h * HD;
                    float maxs = -INFINITY;
                    for (int s = 0; s <= t; s++) {
                        const float *kh = kk + (size_t)s * KVD + (size_t)kvh * HD;
                        float acc = 0.0f;
                        for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
                        float d = acc * ascale; sc[s] = d; if (d > maxs) maxs = d;
                    }
                    float sum = 0.0f;
                    for (int s = 0; s <= t; s++) { sc[s] = expf(sc[s] - maxs); sum += sc[s]; }
                    float inv = 1.0f / sum;
                    const float *gh = gt + (size_t)t * QD + (size_t)h * HD;
                    float *aout = od + (size_t)t * QD + (size_t)h * HD;   /* reuse od as attn concat */
                    for (int i = 0; i < HD; i++) aout[i] = 0.0f;
                    for (int s = 0; s <= t; s++) {
                        float w = sc[s] * inv;
                        const float *vh = vv + (size_t)s * KVD + (size_t)kvh * HD;
                        for (int i = 0; i < HD; i++) aout[i] += w * vh[i];
                    }
                    for (int i = 0; i < HD; i++) aout[i] *= sigmoid_f(gh[i]);   /* output gate */
                }
            }
            if (!af) af = sp_matmul(m, L->attn_output, od, n_tok, QD, E, blk);
            free(qq); free(gt); free(kk); free(vv); free(sc);
            if (af) goto done;
            if (dbg) dbg_fp("attn_output", (int)il, blk + (size_t)(n_tok - 1) * E, E);
        }

        /* attn residual */
        for (int i = 0; i < n_tok * E; i++) x[i] += blk[i];

        /* post-attn norm -> MoE FFN -> ffn residual */
        for (int t = 0; t < n_tok; t++)
            sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, L->post_attn_norm), E, eps, nx + (size_t)t * E);
        if (moe_ffn(m, L, c, nx, n_tok, moe)) goto done;
        if (dbg) dbg_fp("ffn_out", (int)il, moe + (size_t)(n_tok - 1) * E, E);
        for (int i = 0; i < n_tok * E; i++) x[i] += moe[i];
        if (dbg) dbg_fp("l_out", (int)il, x + (size_t)(n_tok - 1) * E, E);
    }

    /* final norm + LM head */
    for (int t = 0; t < n_tok; t++)
        sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, m->output_norm), E, eps, nx + (size_t)t * E);
    if (sp_matmul(m, m->output, nx, n_tok, E, V, logits)) goto done;
    rc = 0;

done:
    free(x); free(nx); free(blk); free(moe); free(qkv); free(zg); free(cs); free(bet);
    free(alp); free(od); free(fno); free(St); free(skv); free(dlt);
    return rc;
}

/* ===================== persistent-STATE decode (NORTHSTAR brick 3) =====================
 * What the stateless forward recomputes per call, carried across calls instead:
 *   - per GDN layer:   the gated delta-rule recurrent state S [Hv][Sd*Sd] (O(1) by
 *                      construction) + the (CK-1) most-recent PRE-conv qkv rows (the
 *                      causal depthwise conv tail);
 *   - per full-attn layer: the post-RoPE K and V rows [pos][KVD];
 *   - the position counter (RoPE + causal length).
 * The inner math below is copied VERBATIM from qwen36_forward (same op order, same
 * accumulation order) so a greedy step-decode reproduces the stateless self-feed
 * sequence exactly — that identity is the gate (G-MOE-STATE-PARITY). */

struct qwen36_state {
    int pos, max_pos, n_layers;
    int *ord;          /* [n_layers] ordinal within its class (gdn / attn) */
    float **S;         /* [n_gdn]  -> Hv*Sd*Sd   recurrent state          */
    float **tail;      /* [n_gdn]  -> (CK-1)*convC  pre-conv rows, oldest first */
    float **K, **Vv;   /* [n_attn] -> max_pos*KVD  post-RoPE K / V         */
    int n_gdn, n_attn;
};

void qwen36_state_free(qwen36_state *st) {
    if (!st) return;
    if (st->S)    { for (int i = 0; i < st->n_gdn; i++)  free(st->S[i]);    free(st->S); }
    if (st->tail) { for (int i = 0; i < st->n_gdn; i++)  free(st->tail[i]); free(st->tail); }
    if (st->K)    { for (int i = 0; i < st->n_attn; i++) free(st->K[i]);    free(st->K); }
    if (st->Vv)   { for (int i = 0; i < st->n_attn; i++) free(st->Vv[i]);   free(st->Vv); }
    free(st->ord); free(st);
}

qwen36_state *qwen36_state_new(const qwen3_model *m, int max_pos) {
    const qwen3_config *c = &m->cfg;
    const int Sd = (int)c->q36_gdn_state, Hk = (int)c->q36_gdn_n_k_heads,
              Hv = (int)c->q36_gdn_n_v_heads, Vdim = (int)c->q36_gdn_inner;
    const int Kdim = Hk * Sd, convC = Kdim * 2 + Vdim, CK = (int)c->q36_gdn_conv_k;
    const int KVD = (int)c->n_head_kv * (int)c->head_dim;
    qwen36_state *st = (qwen36_state *)calloc(1, sizeof(*st));
    if (!st) return NULL;
    st->max_pos = max_pos; st->n_layers = (int)c->n_layers;
    st->ord = (int *)malloc((size_t)st->n_layers * sizeof(int));
    if (!st->ord) { qwen36_state_free(st); return NULL; }
    for (int il = 0; il < st->n_layers; il++)
        st->ord[il] = m->layers[il].q36_is_recurrent ? st->n_gdn++ : st->n_attn++;
    st->S    = (float **)calloc((size_t)st->n_gdn, sizeof(float *));
    st->tail = (float **)calloc((size_t)st->n_gdn, sizeof(float *));
    st->K    = (float **)calloc((size_t)st->n_attn, sizeof(float *));
    st->Vv   = (float **)calloc((size_t)st->n_attn, sizeof(float *));
    if (!st->S || !st->tail || !st->K || !st->Vv) { qwen36_state_free(st); return NULL; }
    for (int i = 0; i < st->n_gdn; i++) {
        st->S[i]    = (float *)calloc((size_t)Hv * Sd * Sd, sizeof(float));
        st->tail[i] = (float *)calloc((size_t)(CK - 1) * convC, sizeof(float));
        if (!st->S[i] || !st->tail[i]) { qwen36_state_free(st); return NULL; }
    }
    for (int i = 0; i < st->n_attn; i++) {
        st->K[i]  = (float *)malloc((size_t)max_pos * KVD * sizeof(float));
        st->Vv[i] = (float *)malloc((size_t)max_pos * KVD * sizeof(float));
        if (!st->K[i] || !st->Vv[i]) { qwen36_state_free(st); return NULL; }
    }
    return st;
}

/* SP_Q36_PROF=1: per-section wall-time accumulators printed each step (brick-4
 * profile-before-believing). Sections: gdn_mm (qkv/gate/beta/alpha/out matmuls),
 * conv+recurrence, attn (full-attn layers), moe (routed+shared experts), head. */
static double q36_prof[5];
static double q36_now(void) {
    struct timespec t; timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int qwen36_step(const qwen3_model *m, qwen36_state *st, int32_t token, float *logits) {
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, V = (int)c->n_vocab;
    const int prof = getenv("SP_Q36_PROF") != NULL;
    double tp = 0.0;
    const float eps = c->rms_eps;
    const int Sd = (int)c->q36_gdn_state, Hk = (int)c->q36_gdn_n_k_heads,
              Hv = (int)c->q36_gdn_n_v_heads, Vdim = (int)c->q36_gdn_inner;
    const int Kdim = Hk * Sd, convC = Kdim * 2 + Vdim, CK = (int)c->q36_gdn_conv_k;
    const int pos = st->pos;
    if (pos >= st->max_pos) return 1;

    int rc = 1;
    float *x   = (float *)malloc((size_t)E * sizeof(float));
    float *nx  = (float *)malloc((size_t)E * sizeof(float));
    float *blk = (float *)malloc((size_t)E * sizeof(float));
    float *moe = (float *)malloc((size_t)E * sizeof(float));
    float *qkv = (float *)malloc((size_t)convC * sizeof(float));   /* pre-conv row */
    float *zg  = (float *)malloc((size_t)Vdim * sizeof(float));
    float *cs  = (float *)malloc((size_t)convC * sizeof(float));   /* conv+silu row */
    float *bet = (float *)malloc((size_t)Hv * sizeof(float));
    float *alp = (float *)malloc((size_t)Hv * sizeof(float));
    float *od  = (float *)malloc((size_t)(Vdim > (int)c->n_head * (int)c->head_dim
                                          ? Vdim : (int)c->n_head * (int)c->head_dim) * sizeof(float));
    float *fno = (float *)malloc((size_t)Vdim * sizeof(float));
    float *skv = (float *)malloc((size_t)Sd * sizeof(float));
    float *dlt = (float *)malloc((size_t)Sd * sizeof(float));
    if (!x || !nx || !blk || !moe || !qkv || !zg || !cs || !bet || !alp || !od || !fno || !skv || !dlt)
        goto done;

    sp_kernels_read_env();
    if (sp_embed_row(m, token, E, x)) goto done;

    for (uint32_t il = 0; il < c->n_layers; il++) {
        const qwen3_layer *L = &m->layers[il];
        sp_rmsnorm(x, sp_as_f32(m, L->attn_norm), E, eps, nx);

        if (L->q36_is_recurrent) {
            const int g = st->ord[il];
            float *S    = st->S[g];
            float *tail = st->tail[g];
            if (prof) tp = q36_now();
            if (sp_matmul(m, L->gdn_qkv,  nx, 1, E, convC, qkv)) goto done;
            if (sp_matmul(m, L->gdn_gate, nx, 1, E, Vdim,  zg))  goto done;
            if (sp_matmul(m, L->gdn_beta, nx, 1, E, Hv,    bet)) goto done;
            if (sp_matmul(m, L->gdn_alpha,nx, 1, E, Hv,    alp)) goto done;
            if (prof) { q36_prof[0] += q36_now() - tp; tp = q36_now(); }
            const float *cw = sp_as_f32(m, L->gdn_conv1d);
            const float *aA = sp_as_f32(m, L->gdn_a);
            const float *dB = sp_as_f32(m, L->gdn_dt_bias);

            /* causal depthwise conv over [tail(CK-1 rows), current]; row j<CK-1 is
             * global time pos-(CK-1)+j -> tail[j] (zero-initialized covers pos<CK-1,
             * matching the stateless zero-padding). Same j-ascending accumulation. */
            for (int ch = 0; ch < convC; ch++) {
                float acc = 0.0f;
                for (int j = 0; j < CK; j++) {
                    int tt = pos - (CK - 1) + j;
                    if (tt < 0) continue;
                    const float *row = (j == CK - 1) ? qkv : tail + (size_t)j * convC;
                    acc += row[ch] * cw[ch * CK + j];
                }
                cs[ch] = silu_f(acc);
            }
            /* advance the pre-conv tail (oldest-first) */
            memmove(tail, tail + convC, (size_t)(CK - 2) * convC * sizeof(float));
            memcpy(tail + (size_t)(CK - 2) * convC, qkv, (size_t)convC * sizeof(float));

            float *qb = cs, *kb = cs + Kdim;
            for (int h = 0; h < Hk; h++) { l2norm(qb + h * Sd, Sd, eps); l2norm(kb + h * Sd, Sd, eps); }

            const float qscale = 1.0f / sqrtf((float)Sd);
            for (int h = 0; h < Hv; h++) {
                int hk = h % Hk;
                float *St_ = S + (size_t)h * Sd * Sd;
                const float *qh = cs + (size_t)hk * Sd;
                const float *kh = cs + Kdim + (size_t)hk * Sd;
                const float *vh = cs + 2 * Kdim + (size_t)h * Sd;
                float gh = expf(aA[h] * softplus_f(alp[h] + dB[h]));
                float bh = sigmoid_f(bet[h]);
                for (int i = 0; i < Sd * Sd; i++) St_[i] *= gh;
                for (int j = 0; j < Sd; j++) {
                    float s = 0.0f;
                    for (int i = 0; i < Sd; i++) s += St_[(size_t)i * Sd + j] * kh[i];
                    skv[j] = s;
                    dlt[j] = bh * (vh[j] - s);
                }
                for (int i = 0; i < Sd; i++) {
                    float ki = kh[i];
                    float *Si = St_ + (size_t)i * Sd;
                    for (int j = 0; j < Sd; j++) Si[j] += ki * dlt[j];
                }
                float *oh = od + (size_t)h * Sd;
                for (int j = 0; j < Sd; j++) {
                    float s = 0.0f;
                    for (int i = 0; i < Sd; i++) s += St_[(size_t)i * Sd + j] * (qh[i] * qscale);
                    oh[j] = s;
                }
            }
            {
                const float *gn = sp_as_f32(m, L->gdn_norm);
                for (int h = 0; h < Hv; h++) {
                    float *o_in  = od  + (size_t)h * Sd;
                    float *o_out = fno + (size_t)h * Sd;
                    float tmp[256];
                    memcpy(tmp, o_in, (size_t)Sd * sizeof(float));
                    sp_rmsnorm_head(tmp, gn, Sd, eps);
                    const float *zh = zg + (size_t)h * Sd;
                    for (int d = 0; d < Sd; d++) o_out[d] = tmp[d] * silu_f(zh[d]);
                }
            }
            if (prof) { q36_prof[1] += q36_now() - tp; tp = q36_now(); }
            if (sp_matmul(m, L->gdn_out, fno, 1, Vdim, E, blk)) goto done;
            if (prof) q36_prof[0] += q36_now() - tp;
        } else {
            if (prof) tp = q36_now();
            const int a = st->ord[il];
            const int NH = (int)c->n_head, NKV = (int)c->n_head_kv, HD = (int)c->head_dim;
            const int QD = NH * HD, KVD = NKV * HD, group = NH / NKV;
            const int nrot = (int)c->q36_rope_dim;
            const float rbase = c->q36_rope_base;
            const float ascale = 1.0f / sqrtf((float)HD);
            const int QD2 = NH * HD * 2;
            float *qf = qkv;                                    /* QD2 == convC */
            float *kk = st->K[a]  + (size_t)pos * KVD;
            float *vv = st->Vv[a] + (size_t)pos * KVD;
            float *qq = fno;                                    /* reuse: QD <= Vdim */
            float *gt = zg;                                     /* reuse: QD <= Vdim */
            float *sc = (float *)malloc((size_t)(pos + 1) * sizeof(float));
            if (!sc) goto done;
            if (sp_matmul(m, L->attn_q, nx, 1, E, QD2, qf) ||
                sp_matmul(m, L->attn_k, nx, 1, E, KVD, kk) ||
                sp_matmul(m, L->attn_v, nx, 1, E, KVD, vv)) { free(sc); goto done; }
            const float *qnw = sp_as_f32(m, L->attn_q_norm);
            const float *knw = sp_as_f32(m, L->attn_k_norm);
            for (int h = 0; h < NH; h++) {
                const float *src = qf + (size_t)h * HD * 2;
                float *qh = qq + (size_t)h * HD;
                float *gh = gt + (size_t)h * HD;
                memcpy(qh, src,      (size_t)HD * sizeof(float));
                memcpy(gh, src + HD, (size_t)HD * sizeof(float));
                sp_rmsnorm_head(qh, qnw, HD, eps);
                sp_rope_neox(qh, nrot, pos, rbase);
            }
            for (int h = 0; h < NKV; h++) {
                float *kh = kk + (size_t)h * HD;
                sp_rmsnorm_head(kh, knw, HD, eps);
                sp_rope_neox(kh, nrot, pos, rbase);
            }
            for (int h = 0; h < NH; h++) {
                int kvh = h / group;
                const float *qh = qq + (size_t)h * HD;
                float maxs = -INFINITY;
                for (int s = 0; s <= pos; s++) {
                    const float *kh = st->K[a] + (size_t)s * KVD + (size_t)kvh * HD;
                    float acc = 0.0f;
                    for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
                    float d = acc * ascale; sc[s] = d; if (d > maxs) maxs = d;
                }
                float sum = 0.0f;
                for (int s = 0; s <= pos; s++) { sc[s] = expf(sc[s] - maxs); sum += sc[s]; }
                float inv = 1.0f / sum;
                const float *gh = gt + (size_t)h * HD;
                float *aout = od + (size_t)h * HD;
                for (int i = 0; i < HD; i++) aout[i] = 0.0f;
                for (int s = 0; s <= pos; s++) {
                    float w = sc[s] * inv;
                    const float *vh = st->Vv[a] + (size_t)s * KVD + (size_t)kvh * HD;
                    for (int i = 0; i < HD; i++) aout[i] += w * vh[i];
                }
                for (int i = 0; i < HD; i++) aout[i] *= sigmoid_f(gh[i]);
            }
            free(sc);
            if (sp_matmul(m, L->attn_output, od, 1, QD, E, blk)) goto done;
            if (prof) q36_prof[2] += q36_now() - tp;
        }

        for (int i = 0; i < E; i++) x[i] += blk[i];
        sp_rmsnorm(x, sp_as_f32(m, L->post_attn_norm), E, eps, nx);
        if (prof) tp = q36_now();
        if (moe_ffn(m, L, c, nx, 1, moe)) goto done;
        if (prof) q36_prof[3] += q36_now() - tp;
        for (int i = 0; i < E; i++) x[i] += moe[i];
    }

    sp_rmsnorm(x, sp_as_f32(m, m->output_norm), E, eps, nx);
    if (prof) tp = q36_now();
    if (sp_matmul(m, m->output, nx, 1, E, V, logits)) goto done;
    if (prof) {
        q36_prof[4] += q36_now() - tp;
        fprintf(stderr, "[q36_prof cum] gdn_mm=%.2fs conv+recur=%.2fs attn=%.2fs moe=%.2fs head=%.2fs\n",
                q36_prof[0], q36_prof[1], q36_prof[2], q36_prof[3], q36_prof[4]);
    }
    st->pos++;
    rc = 0;
done:
    free(x); free(nx); free(blk); free(moe); free(qkv); free(zg); free(cs);
    free(bet); free(alp); free(od); free(fno); free(skv); free(dlt);
    return rc;
}
