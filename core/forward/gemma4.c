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

/* ADR-011 CPU LAYER OFFLOAD — one layer's FFN residual block on the CPU for a SINGLE token,
 * in place on x[E]. The FFN is ~90% of a layer's weight and is STATELESS (touches no KV), so a
 * contiguous TAIL of FFNs can run on the CPU (weights resident in 40 GB/s host DRAM) while the
 * attention (which shares KV with early GPU layers) stays on the GPU — freeing the FFN weight
 * VRAM (~150 MB/layer) with only an E-float activation crossing at each boundary. Arithmetic is
 * IDENTICAL to gemma4_forward_impl's FFN block (lines: ffn_norm → gate/up → gelu·up → down →
 * post_ffw_norm → residual add), reusing sp_matmul (reads the OK_Q4B arena weights on CPU). Under
 * SP_BYTEEXACT the CPU sp_matmul and the GPU dp4a agree bit-for-bit (the CRT gift); in float chat
 * mode they agree to coherence (same argmax). Caller passes the residual x (E) IN/OUT. Returns 0.*/
int gemma4_ffn_block_cpu(const qwen3_model *m, int L, float *x) {
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff;
    const float eps = c->rms_eps;
    if (L < 0 || L >= (int)c->n_layers) return 1;
    const qwen3_layer *ly = &m->layers[L];
    const int FF_L = (ly->ffn_gate && ly->ffn_gate->n_dims >= 2 && ly->ffn_gate->dims[1] > 0)
                     ? (int)ly->ffn_gate->dims[1] : FF;
    float *nx = (float *)malloc((size_t)E * sizeof(float));
    float *g  = (float *)malloc((size_t)FF_L * sizeof(float));
    float *up = (float *)malloc((size_t)FF_L * sizeof(float));
    float *dn = (float *)malloc((size_t)E * sizeof(float));
    int rc = 1;
    if (!nx || !g || !up || !dn) goto done;
    sp_rmsnorm(x, sp_as_f32(m, ly->ffn_norm), E, eps, nx);
    if (sp_matmul(m, ly->ffn_gate, nx, 1, E, FF_L, g))  goto done;
    if (sp_matmul(m, ly->ffn_up,   nx, 1, E, FF_L, up)) goto done;
    for (int i = 0; i < FF_L; i++) g[i] = g4_gelu(g[i]) * up[i];
    if (sp_matmul(m, ly->ffn_down, g, 1, FF_L, E, dn))  goto done;
    /* post_ffw_norm on the FFN output, then residual add (gemma sandwich norm). */
    if (ly->post_ffw_norm) { sp_rmsnorm(dn, sp_as_f32(m, ly->post_ffw_norm), E, eps, nx);
        for (int i = 0; i < E; i++) x[i] += nx[i]; }
    else for (int i = 0; i < E; i++) x[i] += dn[i];
    rc = 0;
done:
    free(nx); free(g); free(up); free(dn);
    return rc;
}

/* ── ADR-012 — CONTIGUOUS full-layer CPU tail (the sync-killer) ───────────────────────────────
 * Runs layers [L_start, NL) of the gemma4 forward for ONE token at absolute position `pos`
 * ENTIRELY on the CPU. Where gemma4_ffn_block_cpu (ADR-011) offloads only the FFN and stays
 * INTERLEAVED with GPU attention — forcing a GPU<->CPU round-trip (D2H + 2x sync + H2D) per
 * offloaded layer, ~2K syncs/token, the measured sync-bound wall — this severs the WHOLE tail:
 * the caller hands over the residual + the two shared-KV owner caches ONCE, the CPU runs every
 * tail layer (attention over the host KV + FFN + AltUp) with NO GPU interaction, and hands the
 * residual back ONCE. Per-token boundary crossings collapse from ~2K to ~2 => the slope
 * approaches the memory-bound floor (~4 ms/layer) instead of ~33 ms/layer.
 *
 * gemma4-12b-b1 has NO KV sharing (kvfs == n_layers): EVERY layer OWNS its K/V and reads only its
 * OWN cache. So each tail layer keeps its own KV cache ENTIRELY on host (hK[L]/hV[L], linear
 * [Pmax*kvd]); it computes Wk/Wv, k/v-norm and RoPE, STORES the new position, and attends over its
 * own [0..pos]. There is NO cross-boundary KV dependency (no GPU layer reads a tail layer's KV and
 * vice-versa), so the ONLY thing crossing the boundary is the residual (+ this token's AltUp
 * inputs `ipl`). Because the tail runs on CPU for EVERY token (prefill + decode via g4_kv_step),
 * the host KV is built from position 0 — no seeding. V-less global layers set V = the raw K
 * projection (pre-norm/RoPE), exactly like the reference. Weights come from the host OK_Q4B arena
 * via sp_matmul (host-resident; the GPU upload is the copy the caller SKIPS to free the VRAM).
 * `ipl` = the per-layer AltUp inputs for this token ([NL*PL], = the device dipl). CRT makes the
 * CPU legs bit-identical to the GPU under SP_BYTEEXACT. x[E] in/out. Returns 0. Default never calls. */
int gemma4_tail_cpu(const qwen3_model *m, int L_start, int pos, float *x, const float *ipl,
                    float **hK, float **hV) {
    const qwen3_config *c = &m->cfg;
    const int   E  = (int)c->n_embd, NL = (int)c->n_layers;
    const float eps = c->rms_eps;
    const int   PL = (int)c->g4_n_embd_per_layer;
    const int   period = (int)c->g4_swa_period ? (int)c->g4_swa_period : 6;
    const int   g_nh=(int)c->n_head, g_nkv=(int)c->n_head_kv, g_hd=(int)c->head_dim;
    const int   s_nh=(int)c->g4_nh_swa, s_nkv=(int)c->g4_nkv_swa, s_hd=(int)c->g4_hd_swa;
    const float g_base=c->rope_freq_base, s_base=c->g4_rope_base_swa;
    const int   SW=(int)c->sliding_window;
    const int   QD_g=g_nh*g_hd, QD_s=s_nh*s_hd, QD_max=QD_g>QD_s?QD_g:QD_s;
    int rc = 1;
    float *nx = (float*)malloc((size_t)E*sizeof(float));
    float *q  = (float*)malloc((size_t)QD_max*sizeof(float));
    float *ao = (float*)malloc((size_t)QD_max*sizeof(float));
    float *ap = (float*)malloc((size_t)E*sizeof(float));
    float *sc = (float*)malloc((size_t)(pos+1)*sizeof(float));
    float *pgate = PL ? (float*)malloc((size_t)PL*sizeof(float)) : NULL;
    float *pproj = PL ? (float*)malloc((size_t)E*sizeof(float)) : NULL;
    if (!nx || !q || !ao || !ap || !sc || (PL && (!pgate || !pproj))) goto done;
    for (int L = L_start; L < NL; L++) {
        const qwen3_layer *ly = &m->layers[L];
        const int global = ((L % period) == period - 1);
        const int nh  = global ? g_nh  : s_nh;
        const int nkv = global ? g_nkv : s_nkv;
        const int hd  = global ? g_hd  : s_hd;
        const int grp = nh / nkv, qd = nh * hd, kvd = nkv * hd;
        const float rbase = global ? g_base : s_base;
        const float *ff   = global ? sp_as_f32(m, m->rope_freqs) : NULL;
        const int   win   = global ? -1 : SW;
        /* ── q ── */
        sp_rmsnorm(x, sp_as_f32(m, ly->attn_norm), E, eps, nx);
        if (sp_matmul(m, ly->attn_q, nx, 1, E, qd, q)) goto done;
        const float *qn = sp_as_f32(m, ly->attn_q_norm);
        for (int h = 0; h < nh; h++) {
            float *qh = q + (size_t)h * hd;
            sp_rmsnorm_head(qh, qn, hd, eps);
            sp_rope_neox_freqs(qh, hd, pos, rbase, ff);
        }
        /* ── k/v (OWNER): compute + norm + RoPE, STORE into this layer's OWN host cache @ pos.
         * gemma4-12b has NO KV sharing (kvfs==NL) so each layer keeps its own KV entirely on host;
         * built from position 0 since the tail runs on CPU every token. V-less globals: V = raw K. */
        float *kh = hK[L] + (size_t)pos * kvd;
        float *vh = hV[L] + (size_t)pos * kvd;
        if (sp_matmul(m, ly->attn_k, nx, 1, E, kvd, kh)) goto done;
        if (ly->attn_v) { if (sp_matmul(m, ly->attn_v, nx, 1, E, kvd, vh)) goto done; }
        else memcpy(vh, kh, (size_t)kvd * sizeof(float));
        const float *kn = sp_as_f32(m, ly->attn_k_norm);
        for (int h = 0; h < nkv; h++) {
            sp_rmsnorm_head(kh + (size_t)h*hd, kn, hd, eps);
            sp_rope_neox_freqs(kh + (size_t)h*hd, hd, pos, rbase, ff);
            g4_rmsnorm_noweight(vh + (size_t)h*hd, hd, eps);
        }
        /* ── attention over this layer's own host KV [0..pos] (windowed for SWA) ── */
        for (int h = 0; h < nh; h++)
            sp_attn_head(q + (size_t)h * hd, hK[L], hV[L], pos, kvd,
                         h / grp, hd, 1.0f, win, sc, ao + (size_t)h * hd);
        if (sp_matmul(m, ly->attn_output, ao, 1, qd, E, ap)) goto done;
        sp_rmsnorm(ap, sp_as_f32(m, ly->post_attn_norm), E, eps, nx);
        for (int i = 0; i < E; i++) x[i] += nx[i];
        /* ── FFN (GeGLU + post_ffw sandwich norm) — the ADR-011 block, reused ── */
        if (gemma4_ffn_block_cpu(m, L, x)) goto done;
        /* ── AltUp per-layer-input injection ── */
        if (PL) {
            if (sp_matmul(m, ly->per_layer_inp_gate, x, 1, E, PL, pgate)) goto done;
            const float *iplL = ipl + (size_t)L * PL;
            for (int i = 0; i < PL; i++) pgate[i] = g4_gelu(pgate[i]) * iplL[i];
            if (sp_matmul(m, ly->per_layer_proj, pgate, 1, PL, E, pproj)) goto done;
            sp_rmsnorm(pproj, sp_as_f32(m, ly->per_layer_post_norm), E, eps, nx);
            for (int i = 0; i < E; i++) x[i] += nx[i];
        }
        /* ── per-layer output scale (scalar) ── */
        if (ly->out_scale) { const float *os = sp_as_f32(m, ly->out_scale);
            if (os) { float sv = os[0]; for (int i = 0; i < E; i++) x[i] *= sv; } }
    }
    rc = 0;
done:
    free(nx); free(q); free(ao); free(ap); free(sc); free(pgate); free(pproj);
    return rc;
}

static int gemma4_forward_impl(const qwen3_model *m, const int32_t *tokens, int n_tok,
                               float *logits, float *feat_out, g4_kv_tap *kv) {
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
    /* Real Gemma4 has CONSTANT n_head/n_head_kv but PER-LAYER head_dim (global 512,
     * SWA 256), so the Q/K/V projection widths differ per layer type. Buffers are
     * sized to the max; per-layer QD/KVD are computed inside the loop. (Stage-1
     * assumed constant widths — corrected after GGUF dim inspection.) */
    const int   QD_g = g_nh * g_hd, QD_s = s_nh * s_hd, QD_max = QD_g > QD_s ? QD_g : QD_s;
    const int   KVD_g = g_nkv * g_hd, KVD_s = s_nkv * s_hd, KVD_max = KVD_g > KVD_s ? KVD_g : KVD_s;
    /* Per-layer FFN width (Gemma4 E-series is MatFormer/elastic: the FFN doubles in
     * the back half — E2B layers 0-14 n_ff=6144, layers 15-34 n_ff=12288). Each
     * layer's width = its bound ffn_gate out-dim (== llama.cpp hparams.n_ff(il));
     * fall back to c->n_ff when tensor dims are unavailable (synthetic fixture).
     * g/up buffers are sized to the per-layer max. */
    int FF_max = FF;
    for (int L = 0; L < NL; L++) {
        const gguf_tensor *fg = m->layers[L].ffn_gate;
        int ffl = (fg && fg->n_dims >= 2 && fg->dims[1] > 0) ? (int)fg->dims[1] : FF;
        if (ffl > FF_max) FF_max = ffl;
    }

    sp_kernels_read_env();
    int rc = 1;

    float  *x   = (float *)malloc((size_t)n_tok * E   * sizeof(float)); /* residual stream */
    float  *nx  = (float *)malloc((size_t)n_tok * E   * sizeof(float)); /* norm scratch */
    float  *q   = (float *)malloc((size_t)n_tok * QD_max * sizeof(float));
    float  *ao  = (float *)malloc((size_t)n_tok * QD_max * sizeof(float));
    float  *ap  = (float *)malloc((size_t)n_tok * E   * sizeof(float));
    float  *g   = (float *)malloc((size_t)n_tok * FF_max * sizeof(float));
    float  *up  = (float *)malloc((size_t)n_tok * FF_max * sizeof(float));
    float  *dn  = (float *)malloc((size_t)n_tok * E   * sizeof(float));
    float  *sc  = (float *)malloc((size_t)n_tok       * sizeof(float));
    float **Kst = (float **)calloc((size_t)NL, sizeof(float *)); /* per-owner K (shared idx stay NULL) */
    float **Vst = (float **)calloc((size_t)NL, sizeof(float *));
    /* EAGLE/MTP KV-tap (serving piece 1): capture the actual Kuse/Vuse for the trunk's
     * last full (NL-1) and last SWA (NL-2) layers during the layer loop; copied out below. */
    float *cap_kf = NULL, *cap_vf = NULL, *cap_ks = NULL, *cap_vs = NULL;
    int    cap_kvdf = 0, cap_kvds = 0;
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
        const int   FF_L = (ly->ffn_gate && ly->ffn_gate->n_dims >= 2 && ly->ffn_gate->dims[1] > 0)
                           ? (int)ly->ffn_gate->dims[1] : FF;   /* per-layer FFN width */
        const int   global = ((L % period) == period - 1);
        const int   nh  = global ? g_nh  : s_nh;
        const int   nkv = global ? g_nkv : s_nkv;
        const int   hd  = global ? g_hd  : s_hd;
        const int   grp = nh / nkv;
        const int   qd  = nh * hd;     /* per-layer Q projection width (2048 SWA / 4096 global) */
        const int   kvd = nkv * hd;    /* per-layer KV width (256 SWA / 512 global) */
        const float rbase = global ? g_base : s_base;
        const float *ff = global ? sp_as_f32(m, m->rope_freqs) : NULL; /* [hd/2] proportional factors */
        const int   win = global ? -1 : SW;
        const float ascale = 1.0f;   /* Gemma4: self.scaling = 1.0 */

        /* ── attention ── */
        for (int t = 0; t < n_tok; t++)
            sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, ly->attn_norm), E, eps, nx + (size_t)t * E);

        if (sp_matmul(m, ly->attn_q, nx, n_tok, E, qd, q)) goto done;
        const float *qn = sp_as_f32(m, ly->attn_q_norm);
        for (int t = 0; t < n_tok; t++)
            for (int h = 0; h < nh; h++) {
                float *qh = q + (size_t)t * qd + (size_t)h * hd;
                sp_rmsnorm_head(qh, qn, hd, eps);
                sp_rope_neox_freqs(qh, hd, t, rbase, ff);
            }

        float *Kuse, *Vuse;
        if (L < kvfs) {
            float *K  = (float *)malloc((size_t)n_tok * kvd * sizeof(float));
            float *Vb = (float *)malloc((size_t)n_tok * kvd * sizeof(float));
            if (!K || !Vb) { free(K); free(Vb); goto done; }
            if (sp_matmul(m, ly->attn_k, nx, n_tok, E, kvd, K))  { free(K); free(Vb); goto done; }
            /* V-less layers (dense gemma-4 globals): attn_v is ABSENT — V is the
             * RAW K projection (llama.cpp gemma4-iswa.cpp: "if v_proj is not
             * present, use Kcur"). Copy BEFORE k_norm/rope; V then takes only
             * the weightless norm below (and never the rope). */
            if (ly->attn_v) {
                if (sp_matmul(m, ly->attn_v, nx, n_tok, E, kvd, Vb)) { free(K); free(Vb); goto done; }
            } else {
                memcpy(Vb, K, (size_t)n_tok * kvd * sizeof(float));
            }
            const float *kn = sp_as_f32(m, ly->attn_k_norm);
            for (int t = 0; t < n_tok; t++)
                for (int h = 0; h < nkv; h++) {
                    float *kh = K  + (size_t)t * kvd + (size_t)h * hd;
                    sp_rmsnorm_head(kh, kn, hd, eps);
                    sp_rope_neox_freqs(kh, hd, t, rbase, ff);
                    g4_rmsnorm_noweight(Vb + (size_t)t * kvd + (size_t)h * hd, hd, eps);
                }
            Kst[L] = K; Vst[L] = Vb; Kuse = K; Vuse = Vb;
        } else {
            const int src = kvfs - (global ? 1 : 2);
            /* shared-KV owner must be a real owner layer: a malformed arch
             * struct (kvfs < 2) would index Kst[-1] — fail loudly, never OOB.
             * (P3 pre-flight audit finding, 2026-06-10.) */
            if (src < 0 || src >= kvfs) goto done;
            Kuse = Kst[src]; Vuse = Vst[src];
            if (!Kuse || !Vuse) goto done;
        }

        if (kv) {   /* EAGLE/MTP KV-tap: grab the actual KV the last full/SWA layers attend */
            if (L == NL - 1) { cap_kf = Kuse; cap_vf = Vuse; cap_kvdf = kvd; }
            if (L == NL - 2) { cap_ks = Kuse; cap_vs = Vuse; cap_kvds = kvd; }
        }

        for (int t = 0; t < n_tok; t++)
            for (int h = 0; h < nh; h++)
                sp_attn_head(q + (size_t)t * qd + (size_t)h * hd, Kuse, Vuse, t, kvd,
                             h / grp, hd, ascale, win, sc, ao + (size_t)t * qd + (size_t)h * hd);

        if (sp_matmul(m, ly->attn_output, ao, n_tok, qd, E, ap)) goto done;
        for (int t = 0; t < n_tok; t++) {
            sp_rmsnorm(ap + (size_t)t * E, sp_as_f32(m, ly->post_attn_norm), E, eps, nx + (size_t)t * E);
            float *xt = x + (size_t)t * E; const float *pt = nx + (size_t)t * E;
            for (int i = 0; i < E; i++) xt[i] += pt[i];
        }

        /* ── FFN (GeGLU, per-layer width FF_L) + post_ffw_norm residual ── */
        for (int t = 0; t < n_tok; t++)
            sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, ly->ffn_norm), E, eps, nx + (size_t)t * E);
        if (sp_matmul(m, ly->ffn_gate, nx, n_tok, E, FF_L, g)) goto done;
        if (sp_matmul(m, ly->ffn_up,   nx, n_tok, E, FF_L, up)) goto done;
        for (size_t i = 0; i < (size_t)n_tok * FF_L; i++) g[i] = g4_gelu(g[i]) * up[i];
        if (sp_matmul(m, ly->ffn_down, g, n_tok, FF_L, E, dn)) goto done;
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

    /* EAGLE/MTP feature tap (step 2b): nx is the post-output_norm hidden that produced
     * `logits` (the LM head consumes it). Copy it out byte-identically when requested. */
    if (feat_out) memcpy(feat_out, nx, (size_t)n_tok * (size_t)E * sizeof(float));

    if (kv) {   /* copy the captured target KV out BEFORE the Kst/Vst free at done: */
        kv->n_pos = n_tok; kv->kvd_full = cap_kvdf; kv->kvd_swa = cap_kvds;
        size_t bf = (size_t)n_tok * (size_t)cap_kvdf, bs = (size_t)n_tok * (size_t)cap_kvds;
        kv->k_full = (float *)malloc(bf * sizeof(float)); kv->v_full = (float *)malloc(bf * sizeof(float));
        kv->k_swa  = (float *)malloc(bs * sizeof(float)); kv->v_swa  = (float *)malloc(bs * sizeof(float));
        if (cap_kf && kv->k_full) memcpy(kv->k_full, cap_kf, bf * sizeof(float));
        if (cap_vf && kv->v_full) memcpy(kv->v_full, cap_vf, bf * sizeof(float));
        if (cap_ks && kv->k_swa)  memcpy(kv->k_swa,  cap_ks, bs * sizeof(float));
        if (cap_vs && kv->v_swa)  memcpy(kv->v_swa,  cap_vs, bs * sizeof(float));
    }

    rc = 0;
done:
    free(x); free(nx); free(q); free(ao); free(ap); free(g); free(up); free(dn); free(sc);
    if (Kst) { for (int L = 0; L < NL; L++) free(Kst[L]); free(Kst); }
    if (Vst) { for (int L = 0; L < NL; L++) free(Vst[L]); free(Vst); }
    free(ipl); free(pgate); free(pproj); free(ple);
    return rc;
}

/* Public entry: standard forward (logits only). */
int gemma4_forward(const qwen3_model *m, const int32_t *tokens, int n_tok, float *logits) {
    return gemma4_forward_impl(m, tokens, n_tok, logits, NULL, NULL);
}

/* EAGLE/MTP FEATURE TAP (step 2b): standard forward PLUS the post-output_norm hidden
 * (the "feature" the LM head consumes = embedding_length_out) for every token. The
 * gemma4-assistant draft seeds inp_h from the LAST token's row (feat_out + (n_tok-1)*E).
 * feat_out must hold n_tok * hidden_dim floats; NULL == gemma4_forward. The copied buffer
 * is byte-identical to the nx that produced `logits` => feature<->logits consistent by
 * construction (one compute path, no divergence). */
int gemma4_forward_feat(const qwen3_model *m, const int32_t *tokens, int n_tok,
                        float *logits, float *feat_out) {
    return gemma4_forward_impl(m, tokens, n_tok, logits, feat_out, NULL);
}

/* EAGLE/MTP KV tap (serving piece 1): forward + feature + the target KV (last full /
 * last SWA layer) the draft attends. Callee malloc's kv->{k,v}_{full,swa}; caller frees. */
int gemma4_forward_kvtap(const qwen3_model *m, const int32_t *tokens, int n_tok,
                         float *logits, float *feat_out, g4_kv_tap *kv) {
    return gemma4_forward_impl(m, tokens, n_tok, logits, feat_out, kv);
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
