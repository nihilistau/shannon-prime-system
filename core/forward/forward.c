/* forward.c — Qwen3 f32 reference forward pass + persistent-KV O(n) decode (the
 * math core's L1 forward orchestration). Relocated out of the engine's
 * src/forward/forward.c onto the migrated kernel stack: the elementary primitives
 * are forward_kernels (sp_dot/rmsnorm/rope/attn), the weight-lift matmul/embedding
 * is forward_dispatch (sp_matmul/sp_embed_row/sp_as_f32), and the overlay codecs are
 * the Phase-1 math modules (poly_ring sp_pr_* for NTT-attention, vht2 sp_spinor_* for
 * the Spinor KV cache, kste sp_kste_encode for the KSTE KV signatures). All gate knobs
 * default OFF = the pure-f32 reference (the engine's E_CPU_2 path).
 *
 * 13-step transformer prefill over a token-ID sequence, causal:
 *   embed -> per layer { RMSNorm -> Q/K/V proj -> per-head QK-RMSNorm ->
 *     RoPE(NEOX) -> GQA causal attention (fp32 softmax) -> O proj -> residual
 *     -> RMSNorm -> SwiGLU FFN -> residual } -> final RMSNorm -> LM head.
 * Weights are dequantized on demand (or inline-lifted from the packed arena).
 *
 * The single behavioral edit vs the engine original is the kernel-call renaming to
 * the math-core ABI names + the dropped #ifdef-AVX include (the L1 reference is
 * scalar; the CPU backend's vectorized path gates against this). The definitive
 * behavior-preservation gate is the engine's E_CPU_2 / GEN_KV regression, which
 * re-runs against this implementation when the engine consumes the migrated library.
 */
#define _CRT_SECURE_NO_WARNINGS   /* getenv is fine here (MSVC C4996) */
#include "sp/model.h"
#include "sp/forward_dispatch.h"   /* sp_matmul / sp_embed_row / sp_as_f32 / sp_kernels_read_env */
#include "sp/forward_kernels.h"    /* sp_rmsnorm / sp_rmsnorm_head / sp_rope_neox / sp_attn_head */
#include "sp/poly_ring.h"          /* sp_pr_init / sp_pr_inner / sp_pr_free (NTT-attention overlay) */
#include "sp/poly_ring_bluestein.h"/* NTT.5a Bluestein wrapper for HD ∈ {2..256} ∖ {512} (NTT.5c) */
#include "sp/spinor_block.h"       /* sp_spinor_* + the frozen 63-byte KV block contract */
#include "sp/arm.h"                /* ARM two-ring: ±1 recall router + r1slot + Ring-2 backend */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* Format-lock (Piece 3, roadmap §8.2.2): the persistent Spinor KV cache stores
 * NBLK = ceil(head_dim/SP_SPINOR_BODY_LEN) frozen 63-byte blocks per head — the
 * on-disk/on-wire §4.9 KV layout. The split arithmetic + codec live in the math
 * core (sp_spinor_blocks_for / sp_spinor_encode_vec / sp_spinor_decode_vec); these
 * asserts freeze the block contract every backend depends on, so a change is a
 * compile error until SP_SPINOR_LAYOUT_VERSION is bumped + migrated. */
_Static_assert(sizeof(sp_spinor_block_t) == 63, "Spinor KV block is 63 bytes (frozen)");
_Static_assert(SP_SPINOR_BODY_LEN == 55, "Spinor block carries 55 anchors; KV head-split assumes this");
_Static_assert(SP_SPINOR_LAYOUT_VERSION == 1u, "Spinor KV layout v1 frozen; bump + migrate to change");

/* NTT-attention (E_CPU_5): when SP_ENGINE_NTT_ATTN=1, each attention score <q,k>
 * is computed by the Phase-1C poly-ring kernel — quantize the (post-norm, post-
 * RoPE) head vectors to int32 (scale SP_NTT_ATTN_SCALE), recover <q,k> EXACTLY as
 * coefficient 0 of the negacyclic product (sp_pr_inner), then divide back out the
 * scale. Sieve OFF. Softmax + V-sum stay f32. Gated against the f32-dot baseline. */
static int g_ntt_attn = 0;
#define SP_NTT_ATTN_SCALE 65536.0   /* 2^16: |q_int| ~ 2^21, |<q,k>| ~ 2^49 << M/2 ~ 2^59 */
#define SP_KSTE_KV_SCALE  65536.0   /* fixed int32 quant for KSTE KV signatures (E_CPU_6) */

/* Inline VHT2+Spinor KV-cache compression (E_CPU_8). When SP_KV_SPINOR=1 each
 * post-norm/post-RoPE K and post-proj V head vector is stored as the frozen 63-byte
 * Spinor block(s) and decoded back (lossy) before attention reads it. Gate OFF skips
 * it entirely => bit-identical to the E_CPU_2 reference. */
static int g_kv_spinor = 0;
/* SP_KV_SPINOR_REF=1: in qwen3_generate_kv, store the cache as f32 + an in-place
 * round-trip (the §4.9 reference fp32 cache, parity tests only) instead of the
 * production Spinor-block cache. Decode-from-block is arithmetically identical to the
 * in-place round-trip, so the two paths must produce identical sequences. */
static int g_kv_spinor_ref = 0;

/* ── ARM two-ring knobs (C2.1, ported from the engine production impl). All OFF
 * => the exact full-context baseline (bit-identical parity). Same env names as
 * the engine so every existing gate/harness drives this path unchanged. */
static int g_recall_b = 0;          /* SP_RECALL_B: recall budget (0 = off = full attention) */
static int g_recall_r = 16;         /* SP_RECALL_R: projection rank (<= SP_ARM_R_MAX) */
static int g_recall_w = 64;         /* SP_RECALL_W: always-keep recent window */
static int g_recall_sink = 4;       /* SP_RECALL_SINK: pinned StreamingLLM sink anchors */
static int g_ring2 = 0;             /* SP_RING2=1: history spills to the mock RAM Ring-2 store */
static int g_ring2_store = 0;       /* SP_RING2_DISK=1: history spills through the Ring-2 BACKEND
                                     * (math-core reference = portable stdio store; platform stores
                                     * — Optane IOCP, QUIC peer — register the same interface) */
static const char *g_ring2_dir = 0; /* SP_RING2_DIR: stdio store directory (default ".") */
static int g_recall_decode_only = 0;/* SP_RECALL_DECODE_ONLY=1: dense exact prefill; router engages at decode */
static int g_recall_fuse = 0;       /* SP_RECALL_FUSE=1: compact-and-spill (dense full-P prefill buffer,
                                     * bulk-spilled + freed at the prefill->decode boundary) */

/* The multi-block KV head codec lives in the math core (sp_spinor_blocks_for /
 * encode_vec / decode_vec, sp/spinor_block.h). The only local helper is the in-place
 * round-trip used by the prefill KV path (E_CPU_8) and the generate_kv f32 parity ref. */
#define KV_HEAD_MAX_BLOCKS 16   /* stack temp; covers head_dim up to 16*55 = 880 */
static void kv_spinor_roundtrip(float *vec, int d) {
    sp_spinor_block_t blks[KV_HEAD_MAX_BLOCKS];
    if (sp_spinor_blocks_for(d) > KV_HEAD_MAX_BLOCKS) return;   /* head_dim beyond supported range */
    sp_spinor_encode_vec(vec, d, blks);
    (void)sp_spinor_decode_vec(blks, d, vec);   /* own freshly-encoded blocks: CRC valid */
}

/* Read the runtime gate knobs once per forward/decode entry (all default OFF = the
 * pure-f32 E_CPU_2 reference). Shared by the prefill and the decode loop. */
static void read_env_knobs(void) {
    sp_kernels_read_env();   /* SP_ENGINE_F16_ACT, SP_ENGINE_FROB, SP_Q4_PROMOTE */
    { const char *e = getenv("SP_ENGINE_NTT_ATTN"); g_ntt_attn = (e && e[0] == '1'); }
    { const char *e = getenv("SP_KV_SPINOR");       g_kv_spinor = (e && e[0] == '1'); }
    { const char *e = getenv("SP_KV_SPINOR_REF");   g_kv_spinor_ref = (e && e[0] == '1'); }
    { const char *e = getenv("SP_RECALL_B");        g_recall_b = e ? atoi(e) : 0; if (g_recall_b < 0) g_recall_b = 0; }
    { const char *e = getenv("SP_RECALL_R");        g_recall_r = e ? atoi(e) : 16; if (g_recall_r < 1 || g_recall_r > SP_ARM_R_MAX) g_recall_r = 16; }
    { const char *e = getenv("SP_RECALL_W");        g_recall_w = e ? atoi(e) : 64; if (g_recall_w < 0) g_recall_w = 0; }
    { const char *e = getenv("SP_RECALL_SINK");     g_recall_sink = e ? atoi(e) : 4; if (g_recall_sink < 0) g_recall_sink = 0; }
    { const char *e = getenv("SP_RING2");           g_ring2 = (e && e[0] == '1'); }
    { const char *e = getenv("SP_RING2_DISK");      g_ring2_store = (e && e[0] == '1'); }
    { const char *e = getenv("SP_RING2_DIR");       g_ring2_dir = (e && e[0]) ? e : "."; }
    { const char *e = getenv("SP_RECALL_DECODE_ONLY"); g_recall_decode_only = (e && e[0] == '1'); }
    { const char *e = getenv("SP_RECALL_FUSE");        g_recall_fuse = (e && e[0] == '1'); }
}

int qwen3_forward_ex2(const qwen3_model *m, const int32_t *tokens, int n_tok,
                      float *logits, sp_kste_tree_t *kv_trees,
                      void *backend_handle,
                      sp_compute_ntt_dispatch_fn backend_forward,
                      sp_compute_ntt_dispatch_fn backend_inverse) {
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv;
    const int QD = NH * HD;          /* q proj width  */
    const int KVD = NKV * HD;        /* kv proj width */
    const int group = NH / NKV;      /* q-heads per kv-head */
    const int V = (int)c->n_vocab;
    const float eps = c->rms_eps, base = c->rope_freq_base;
    const float ascale = 1.0f / sqrtf((float)HD);

    read_env_knobs();

    int rc = 1;
    /* NTT.5c: dispatch on HD between direct sp_pr (HD ∈ {128,256,512}) and
     * Bluestein-wrapped sp_pr_bluestein (HD ∈ {2,4,8,16,32,64,128,256}).
     * Direct is faster for HD ∈ {128,256,512} (no zero-pad); Bluestein is
     * required for HD ∈ {2..64}. HD with odd factors leaves both NULL →
     * overlay disabled (fp32 attention via sp_attn_head). */
    sp_pr_ctx           *pr   = NULL;
    sp_pr_bluestein_ctx *pr_b = NULL;
    int overlay_active = 0;        /* set in init block when at least one ctx alloc'd */
    int32_t *qi = NULL, *ki = NULL;
    int32_t *kq = NULL;            /* int32 scratch for KSTE KV encoding */
    float *x   = (float *)malloc((size_t)n_tok * E * sizeof(float));   /* residual stream */
    float *nx  = (float *)malloc((size_t)n_tok * E * sizeof(float));   /* normed */
    float *q   = (float *)malloc((size_t)n_tok * QD * sizeof(float));
    float *k   = (float *)malloc((size_t)n_tok * KVD * sizeof(float));
    float *v   = (float *)malloc((size_t)n_tok * KVD * sizeof(float));
    float *ao  = (float *)malloc((size_t)n_tok * QD * sizeof(float));  /* attn out (concat heads) */
    float *ap  = (float *)malloc((size_t)n_tok * E * sizeof(float));   /* attn out proj */
    float *g   = (float *)malloc((size_t)n_tok * FF * sizeof(float));
    float *up  = (float *)malloc((size_t)n_tok * FF * sizeof(float));
    float *dn  = (float *)malloc((size_t)n_tok * E * sizeof(float));
    float *sc  = (float *)malloc((size_t)n_tok * sizeof(float));       /* attn scores */
    if (!x || !nx || !q || !k || !v || !ao || !ap || !g || !up || !dn || !sc) goto done;

    if (g_ntt_attn) {
        /* HD dispatch — see comment above. */
        if (HD == 128 || HD == 256 || HD == 512) {
            pr = sp_pr_init((uint32_t)HD);
            if (pr) overlay_active = 1;
        } else if (HD >= 2 && HD <= 256 && (HD & (HD - 1)) == 0) {
            pr_b = sp_pr_bluestein_init((uint32_t)HD);
            if (pr_b) {
                overlay_active = 1;
                /* NTT.5c: if a compute backend triple was passed in
                 * (NTT.5b path; SP_ENGINE_NTT_ATTN_HEX=1 daemon side
                 * routes through here via sp_prefill_chunk extracting the
                 * session's registered backend via the L1 readback
                 * accessors), thread it through to the Bluestein ctx. The
                 * setter is a no-op when all three are NULL. */
                if (backend_handle || backend_forward || backend_inverse)
                    sp_pr_bluestein_set_backend(pr_b, backend_handle,
                                                backend_forward,
                                                backend_inverse);
            }
        }
        /* If HD has odd factors > 1 (e.g. 96, 192, 288, 384) both ctx stay
         * NULL — overlay_active stays 0; the attention loop below falls
         * through to the standard fp32 sp_attn_head path. Banned: do NOT
         * propose mixed-radix or zero-padding HD (see
         * reference-ntt-bluestein-arbitrary-n-escape). */
        if (overlay_active) {
            qi = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
            ki = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
            if (!qi || !ki) goto done;
        }
    }
    if (kv_trees) {
        kq = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
        if (!kq) goto done;
    }

    /* ── embedding lookup: token t's embedding is the contiguous E floats at t*E ── */
    for (int t = 0; t < n_tok; t++)
        if (sp_embed_row(m, tokens[t], E, x + (size_t)t * E)) goto done;

    for (uint32_t L = 0; L < c->n_layers; L++) {
        const qwen3_layer *ly = &m->layers[L];

        /* ── attention block ── */
        for (int t = 0; t < n_tok; t++)
            sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, ly->attn_norm), E, eps, nx + (size_t)t * E);

        if (sp_matmul(m, ly->attn_q, nx, n_tok, E, QD, q)) goto done;
        if (sp_matmul(m, ly->attn_k, nx, n_tok, E, KVD, k)) goto done;
        if (sp_matmul(m, ly->attn_v, nx, n_tok, E, KVD, v)) goto done;

        /* per-head QK-RMSNorm (over head_dim) then NEOX RoPE at position t */
        const float *qn = sp_as_f32(m, ly->attn_q_norm);
        const float *kn = sp_as_f32(m, ly->attn_k_norm);
        for (int t = 0; t < n_tok; t++) {
            for (int h = 0; h < NH; h++) {
                float *qh = q + (size_t)t * QD + (size_t)h * HD;
                sp_rmsnorm_head(qh, qn, HD, eps);
                sp_rope_neox(qh, HD, t, base);
            }
            for (int h = 0; h < NKV; h++) {
                float *kh = k + (size_t)t * KVD + (size_t)h * HD;
                sp_rmsnorm_head(kh, kn, HD, eps);
                sp_rope_neox(kh, HD, t, base);
            }
        }

        /* Inline VHT2+Spinor KV compression (E_CPU_8): store each cached K/V head
         * vector as Spinor block(s) and read back the lossy reconstruction. Applied
         * after QK-norm+RoPE so the cache holds position-finalized K (and post-proj V). */
        if (g_kv_spinor) {
            for (int t = 0; t < n_tok; t++)
                for (int h = 0; h < NKV; h++) {
                    kv_spinor_roundtrip(k + (size_t)t * KVD + (size_t)h * HD, HD);
                    kv_spinor_roundtrip(v + (size_t)t * KVD + (size_t)h * HD, HD);
                }
        }

        /* KSTE KV-cache overlay (E_CPU_6): encode each cached K head-vector to its
         * 64-byte signature. Deterministic int32 quantization -> byte-identical. */
        if (kv_trees) {
            for (int t = 0; t < n_tok; t++)
                for (int h = 0; h < NKV; h++) {
                    const float *kh = k + (size_t)t * KVD + (size_t)h * HD;
                    for (int i = 0; i < HD; i++)
                        kq[i] = (int32_t)lrintf(kh[i] * (float)SP_KSTE_KV_SCALE);
                    sp_kste_encode(kq, HD, &kv_trees[((size_t)L * n_tok + t) * NKV + h]);
                }
        }

        /* GQA causal attention. Plain f32 path uses the shared sp_attn_head (full
         * causal, win=-1); the NTT-attention overlay (E_CPU_5) stays inline since its
         * score is computed via the exact poly-ring inner product. */
        for (int t = 0; t < n_tok; t++) {
            for (int h = 0; h < NH; h++) {
                int kvh = h / group;
                const float *qh = q + (size_t)t * QD + (size_t)h * HD;
                float *out = ao + (size_t)t * QD + (size_t)h * HD;
                if (!g_ntt_attn || !overlay_active) {
                    /* NTT.5c: overlay_active=0 covers both (a) g_ntt_attn=0
                     * (env gate off) and (b) g_ntt_attn=1 but HD is not in
                     * the Bluestein/direct admissible set (HD with odd
                     * factors). Both cases route through fp32 sp_attn_head. */
                    sp_attn_head(qh, k, v, t, KVD, kvh, HD, ascale, -1, sc, out);
                    continue;
                }
                for (int i = 0; i < HD; i++) qi[i] = (int32_t)lrintf(qh[i] * (float)SP_NTT_ATTN_SCALE);
                float maxs = -INFINITY;
                for (int s = 0; s <= t; s++) {
                    const float *kh = k + (size_t)s * KVD + (size_t)kvh * HD;
                    for (int i = 0; i < HD; i++) ki[i] = (int32_t)lrintf(kh[i] * (float)SP_NTT_ATTN_SCALE);
                    /* NTT.5c dispatch: pr_b (Bluestein) for HD ∈ {2..64,128,256}
                     * when picked above; pr (direct) for HD ∈ {128,256,512}. At
                     * most one of the two is non-NULL per overlay_active=1. */
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
                    const float *vh = v + (size_t)s * KVD + (size_t)kvh * HD;
                    for (int i = 0; i < HD; i++) out[i] += w * vh[i];
                }
            }
        }

        if (sp_matmul(m, ly->attn_output, ao, n_tok, QD, E, ap)) goto done;
        for (size_t i = 0; i < (size_t)n_tok * E; i++) x[i] += ap[i];   /* residual */

        /* ── FFN block (SwiGLU) ── */
        for (int t = 0; t < n_tok; t++)
            sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, ly->ffn_norm), E, eps, nx + (size_t)t * E);
        if (sp_matmul(m, ly->ffn_gate, nx, n_tok, E, FF, g)) goto done;
        if (sp_matmul(m, ly->ffn_up,   nx, n_tok, E, FF, up)) goto done;
        for (size_t i = 0; i < (size_t)n_tok * FF; i++) {
            float gv = g[i];
            float silu = gv / (1.0f + expf(-gv));
            g[i] = silu * up[i];
        }
        if (sp_matmul(m, ly->ffn_down, g, n_tok, FF, E, dn)) goto done;
        for (size_t i = 0; i < (size_t)n_tok * E; i++) x[i] += dn[i];    /* residual */
    }

    /* ── final norm + LM head ── */
    for (int t = 0; t < n_tok; t++)
        sp_rmsnorm(x + (size_t)t * E, sp_as_f32(m, m->output_norm), E, eps, nx + (size_t)t * E);
    if (sp_matmul(m, m->output, nx, n_tok, E, V, logits)) goto done;

    rc = 0;
done:
    free(x); free(nx); free(q); free(k); free(v); free(ao); free(ap);
    free(g); free(up); free(dn); free(sc);
    free(qi); free(ki); free(kq);
    sp_pr_free(pr);
    sp_pr_bluestein_free(pr_b);                 /* NTT.5c: free the Bluestein ctx if used */
    return rc;
}

/* NTT.5c: legacy entry points become wrappers passing the all-NULL backend
 * triple (= host path). Existing callers (tools/probe, qwen3_generate,
 * qwen3_forward_test) keep working unchanged; only sp_prefill_chunk has been
 * updated to call _ex2 with the L1 session's registered backend. */
int qwen3_forward_ex(const qwen3_model *m, const int32_t *tokens, int n_tok,
                     float *logits, sp_kste_tree_t *kv_trees) {
    return qwen3_forward_ex2(m, tokens, n_tok, logits, kv_trees, NULL, NULL, NULL);
}

int qwen3_forward(const qwen3_model *m, const int32_t *tokens, int n_tok, float *logits) {
    return qwen3_forward_ex2(m, tokens, n_tok, logits, NULL, NULL, NULL, NULL);
}

/* Persistent-KV O(n) greedy decode (GEN_KV) with the ARM two-ring memory.
 *
 * Each token is processed once: per-layer K/V are computed for the single new
 * token, stored post-RoPE into a position-indexed cache, and attention reads
 * the cached K/V for earlier positions. Honors the same gates as the prefill
 * (SP_ENGINE_FROB, SP_KV_SPINOR) plus, decode-side, SP_ENGINE_NTT_ATTN (the
 * exact poly-ring score) and the ARM two-ring knobs (SP_RECALL_* / SP_RING2*).
 *
 * ARM (ported from the engine C2.1 production impl, serial L1 reference):
 *   - recall sidecar: each position's post-RoPE K is ±1-projected (sp/arm.h,
 *     frozen seed) into projk; at attention, sp_arm_select picks
 *     sinks ∪ top-(B-W-sink) ∪ recent-W (expected-O(N) quickselect).
 *   - Ring-1: when offloading, the f32 kc/vc cache holds ONLY sinks + the
 *     W-window (ring buffer, cap = sink+W, sp_arm_r1slot); older tokens are
 *     structurally evicted and served from Ring-2.
 *   - Ring-2: the spilled history. SP_RING2=1 alone = mock RAM byte store
 *     (spill/fetch parity); SP_RING2_DISK=1 = the abstract block-store backend
 *     (math-core reference = portable stdio under SP_RING2_DIR; the engine
 *     registers its Optane NO_BUFFERING+IOCP store through the same interface,
 *     a QUIC peer is the same interface over the mesh). v1 dedupe: per layer,
 *     the UNION of all heads' recalled blocks is fetched once into staging.
 *   - SP_RECALL_DECODE_ONLY=1: dense exact prefill (full cache, no offload),
 *     router engages only at decode. SP_RECALL_FUSE=1 (compact-and-spill):
 *     dense prefill in a full-P buffer; at the prefill->decode boundary the
 *     history is bulk-spilled to Ring-2, sinks+window copied into the small
 *     cache, and the buffer FREED.
 *
 * All knobs OFF => the exact full-context baseline path, bit-identical to the
 * pre-ARM decode. Greedy argmax must match the O(n^2) re-prefill generate up
 * to the float-reassociation floor; GEN_KV gates on sequence identity. */

/* Score one (q,k) pair: the exact poly-ring inner product when the NTT decode
 * overlay is active (qi pre-quantized per head), else the f32 dot — the same
 * arithmetic order as the pre-ARM decode, so gate-off parity is bit-exact. */
static float kv_pair_score(const float *qh, const float *kh, int HD, float ascale,
                           sp_pr_ctx *pr, sp_pr_bluestein_ctx *pr_b,
                           const int32_t *qi, int32_t *ki) {
    if (pr || pr_b) {
        for (int i = 0; i < HD; i++) ki[i] = (int32_t)lrintf(kh[i] * (float)SP_NTT_ATTN_SCALE);
        int64_t ip = pr_b ? sp_pr_bluestein_inner(pr_b, qi, ki) : sp_pr_inner(pr, qi, ki);
        return (float)((double)ip / (SP_NTT_ATTN_SCALE * SP_NTT_ATTN_SCALE)) * ascale;
    }
    float acc = 0.0f;
    for (int i = 0; i < HD; i++) acc += qh[i] * kh[i];
    return acc * ascale;
}

int qwen3_generate_kv(const qwen3_model *m, int32_t *seq, int n_prompt, int n_gen,
                      int eos_id) {
    if (!m || !seq || n_prompt <= 0 || n_gen < 0) return -1;
    read_env_knobs();
    const qwen3_config *c = &m->cfg;
    const int E = (int)c->n_embd, FF = (int)c->n_ff, HD = (int)c->head_dim;
    const int NH = (int)c->n_head, NKV = (int)c->n_head_kv;
    const int QD = NH * HD, KVD = NKV * HD, group = NH / NKV, V = (int)c->n_vocab;
    const float eps = c->rms_eps, base = c->rope_freq_base, ascale = 1.0f / sqrtf((float)HD);
    const int P = n_prompt + n_gen;

    const int NBLK = sp_spinor_blocks_for(HD);
    const int use_blocks = g_kv_spinor && !g_kv_spinor_ref;

    int rc = -1, n = n_prompt, produced = 0;
    /* All pointers NULL-first so any early `goto done` frees only NULLs. */
    sp_spinor_block_t *kcb = NULL, *vcb = NULL;   /* block KV cache (use_blocks) */
    float *kc = NULL, *vc = NULL;                 /* f32 KV cache (window-sized when offloading) */
    float *kpre = NULL, *vpre = NULL;             /* SP_RECALL_FUSE full-P dense prefill buffer */
    float *kdec = NULL, *vdec = NULL;             /* per-layer decode scratch (use_blocks) */
    float *x = NULL, *nx = NULL, *q = NULL, *knew = NULL, *vnew = NULL, *ao = NULL;
    float *ap = NULL, *gg = NULL, *up = NULL, *dn = NULL, *sc = NULL, *lg = NULL;
    signed char *recallR = NULL; float *projk = NULL;    /* recall sidecar (g_recall_b>0) */
    float *ring2k = NULL, *ring2v = NULL;                /* mock RAM Ring-2 store */
    float *stgK = NULL, *stgV = NULL;                    /* backend dedupe staging */
    int *stg_stamp = NULL, *stg_slot = NULL, *stg_pos = NULL;
    int *ri_all = NULL, *m_all = NULL;                   /* per-head recall sets (backend path) */
    int *rb_which = NULL; uint64_t *rb_off = NULL; void **rb_dst = NULL; /* optional read_batch reqs */
    int *ri = NULL; sp_arm_sidx *cand = NULL;            /* inline-path selection scratch */
    sp_arm_ring2_backend r2be; int r2be_on = 0; r2be.handle = NULL; r2be.close = NULL;
    int stg_gen = 0;
    /* NTT decode-attention overlay (SP_ENGINE_NTT_ATTN). */
    sp_pr_ctx *pr = NULL; sp_pr_bluestein_ctx *pr_b = NULL; int overlay_active = 0;
    int32_t *qi = NULL, *ki = NULL;

    /* ── ARM configuration (engine-faithful gating) ── */
    const int ring2_on = (g_ring2 && g_recall_b > 0 && !use_blocks && !g_recall_decode_only);
    r2be_on = ring2_on && g_ring2_store;
    const int fuse = (g_recall_fuse && r2be_on);
    const int r1W = (g_recall_w > 0) ? g_recall_w : 1;
    const int r1cap = ring2_on ? (g_recall_sink + r1W) : P;

    if (use_blocks) {
        size_t nb = (size_t)c->n_layers * P * NKV * NBLK;
        kcb  = (sp_spinor_block_t *)malloc(nb * sizeof(sp_spinor_block_t));
        vcb  = (sp_spinor_block_t *)malloc(nb * sizeof(sp_spinor_block_t));
        kdec = (float *)malloc((size_t)P * KVD * sizeof(float));
        vdec = (float *)malloc((size_t)P * KVD * sizeof(float));
        if (!kcb || !vcb || !kdec || !vdec) goto done;
        fprintf(stderr, "    [KV] Spinor-block cache: %zu B vs f32 %zu B (%.2fx) - %d blocks/head, %d B/block\n",
                2 * nb * sizeof(sp_spinor_block_t),
                2 * (size_t)c->n_layers * P * KVD * sizeof(float),
                (double)((size_t)c->n_layers * P * KVD * sizeof(float)) /
                (double)((size_t)c->n_layers * P * NKV * NBLK * sizeof(sp_spinor_block_t)),
                NBLK, (int)sizeof(sp_spinor_block_t));
    } else {
        kc = (float *)malloc((size_t)c->n_layers * r1cap * KVD * sizeof(float));
        vc = (float *)malloc((size_t)c->n_layers * r1cap * KVD * sizeof(float));
        if (!kc || !vc) goto done;
        if (ring2_on)
            fprintf(stderr, "    [ring1] f32 cache SHRUNK to window: %d slots/layer (sink %d + W %d) = %.1f MB vs full %.1f MB (%.0fx)\n",
                    r1cap, g_recall_sink, r1W,
                    2.0 * c->n_layers * r1cap * KVD * sizeof(float) / 1e6,
                    2.0 * c->n_layers * P * KVD * sizeof(float) / 1e6,
                    (double)P / (double)r1cap);
    }
    x    = (float *)malloc((size_t)E * sizeof(float));
    nx   = (float *)malloc((size_t)E * sizeof(float));
    q    = (float *)malloc((size_t)QD * sizeof(float));
    knew = (float *)malloc((size_t)KVD * sizeof(float));
    vnew = (float *)malloc((size_t)KVD * sizeof(float));
    ao   = (float *)malloc((size_t)QD * sizeof(float));
    ap   = (float *)malloc((size_t)E * sizeof(float));
    gg   = (float *)malloc((size_t)FF * sizeof(float));
    up   = (float *)malloc((size_t)FF * sizeof(float));
    dn   = (float *)malloc((size_t)E * sizeof(float));
    sc   = (float *)malloc((size_t)P * sizeof(float));
    lg   = (float *)malloc((size_t)V * sizeof(float));
    if (!x || !nx || !q || !knew || !vnew || !ao || !ap || !gg || !up || !dn || !sc || !lg)
        goto done;
    if (g_recall_b > 0) {
        recallR = (signed char *)malloc((size_t)g_recall_r * HD);
        projk   = (float *)malloc((size_t)c->n_layers * P * NKV * (size_t)g_recall_r * sizeof(float));
        ri      = (int *)malloc((size_t)P * sizeof(int));
        cand    = (sp_arm_sidx *)malloc((size_t)P * sizeof(sp_arm_sidx));
        if (!recallR || !projk || !ri || !cand) goto done;
        sp_arm_build_R(recallR, g_recall_r, HD);   /* frozen ±1 matrix, deterministic */
        fprintf(stderr, "    [recall] sidecar ON: r=%d B=%d W=%d sink=%d (post-RoPE ±1 projection router + sinks)\n",
                g_recall_r, g_recall_b, g_recall_w, g_recall_sink);
    }
    if (ring2_on && !r2be_on) {
        size_t kvn = (size_t)c->n_layers * P * KVD;
        ring2k = (float *)malloc(kvn * sizeof(float));
        ring2v = (float *)malloc(kvn * sizeof(float));
        if (!ring2k || !ring2v) goto done;
        fprintf(stderr, "    [ring2] mock RAM spill ON: Ring-1 holds only sink+W=%d slots\n",
                g_recall_sink + r1W);
    } else if (r2be_on) {
        if (sp_arm_ring2_stdio_open(g_ring2_dir, &r2be)) { r2be_on = 0; goto done; }
        stgK = (float *)malloc((size_t)P * KVD * sizeof(float));
        stgV = (float *)malloc((size_t)P * KVD * sizeof(float));
        stg_stamp = (int *)malloc((size_t)P * sizeof(int));
        stg_slot  = (int *)malloc((size_t)P * sizeof(int));
        stg_pos   = (int *)malloc((size_t)P * sizeof(int));
        ri_all = (int *)malloc((size_t)NH * P * sizeof(int));
        m_all  = (int *)malloc((size_t)NH * sizeof(int));
        if (r2be.read_batch) {
            rb_which = (int *)malloc((size_t)2 * P * sizeof(int));
            rb_off   = (uint64_t *)malloc((size_t)2 * P * sizeof(uint64_t));
            rb_dst   = (void **)malloc((size_t)2 * P * sizeof(void *));
            if (!rb_which || !rb_off || !rb_dst) goto done;
        }
        if (!stgK || !stgV || !stg_stamp || !stg_slot || !stg_pos || !ri_all || !m_all) goto done;
        for (int i = 0; i < P; i++) stg_stamp[i] = -1;
        if (fuse) {
            kpre = (float *)malloc((size_t)c->n_layers * P * KVD * sizeof(float));
            vpre = (float *)malloc((size_t)c->n_layers * P * KVD * sizeof(float));
            if (!kpre || !vpre) goto done;
            fprintf(stderr, "    [fuse] compact-and-spill: dense full-P prefill buffer (%.1f MB), freed at the boundary\n",
                    2.0 * c->n_layers * P * KVD * sizeof(float) / 1e6);
        }
        fprintf(stderr, "    [ring2] BACKEND spill ON (W=%d, sinks pinned; v1 per-layer dedupe staging%s)\n",
                g_recall_w, r2be.read_batch ? ", batched reads" : ", serial reference reads");
    }
    /* NTT decode overlay init — mirrors the prefill overlay (HD-dispatch). */
    if (g_ntt_attn) {
        if (HD == 128 || HD == 256 || HD == 512) {
            pr = sp_pr_init((uint32_t)HD); if (pr) overlay_active = 1;
        } else if (HD >= 2 && HD <= 256 && (HD & (HD - 1)) == 0) {
            pr_b = sp_pr_bluestein_init((uint32_t)HD); if (pr_b) overlay_active = 1;
        }
        if (overlay_active) {
            qi = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
            ki = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
            if (!qi || !ki) goto done;
        }
    }

    for (int pos = 0; pos < P; pos++) {
        int tok = seq[pos];
        if (sp_embed_row(m, tok, E, x)) goto done;

        if (fuse && pos == n_prompt) {
            /* FUSE boundary: bulk-spill the prefill history to Ring-2, copy sinks +
             * the recent window into the (sink+W) cache, FREE the full-P buffer. */
            int W = r1W, sink = g_recall_sink;
            int wlo = n_prompt - W; if (wlo < sink) wlo = sink;
            for (uint32_t L = 0; L < c->n_layers; L++) {
                const float *kpL = kpre + (size_t)L * P * KVD, *vpL = vpre + (size_t)L * P * KVD;
                for (int s = 0; s < n_prompt; s++) {
                    uint64_t boff = (uint64_t)((size_t)L * P + s) * (uint64_t)((size_t)KVD * sizeof(float));
                    if (r2be.write_block(r2be.handle, 0, boff, kpL + (size_t)s * KVD, (size_t)KVD * sizeof(float)) ||
                        r2be.write_block(r2be.handle, 1, boff, vpL + (size_t)s * KVD, (size_t)KVD * sizeof(float))) goto done;
                }
                float *kcL = kc + (size_t)L * r1cap * KVD, *vcL = vc + (size_t)L * r1cap * KVD;
                for (int s = 0; s < sink && s < n_prompt; s++) {
                    int sl = sp_arm_r1slot(s, 1, sink, W);
                    memcpy(kcL + (size_t)sl * KVD, kpL + (size_t)s * KVD, (size_t)KVD * sizeof(float));
                    memcpy(vcL + (size_t)sl * KVD, vpL + (size_t)s * KVD, (size_t)KVD * sizeof(float));
                }
                for (int s = wlo; s < n_prompt; s++) {
                    int sl = sp_arm_r1slot(s, 1, sink, W);
                    memcpy(kcL + (size_t)sl * KVD, kpL + (size_t)s * KVD, (size_t)KVD * sizeof(float));
                    memcpy(vcL + (size_t)sl * KVD, vpL + (size_t)s * KVD, (size_t)KVD * sizeof(float));
                }
            }
            free(kpre); free(vpre); kpre = NULL; vpre = NULL;
            fprintf(stderr, "    [fuse] boundary @ pos=%d: spilled %d tok/layer to Ring-2, freed the prefill buffer\n",
                    n_prompt, n_prompt);
        }

        for (uint32_t L = 0; L < c->n_layers; L++) {
            const qwen3_layer *ly = &m->layers[L];
            sp_rmsnorm(x, sp_as_f32(m, ly->attn_norm), E, eps, nx);
            if (sp_matmul(m, ly->attn_q, nx, 1, E, QD, q))   goto done;
            if (sp_matmul(m, ly->attn_k, nx, 1, E, KVD, knew)) goto done;
            if (sp_matmul(m, ly->attn_v, nx, 1, E, KVD, vnew)) goto done;

            const float *qn = sp_as_f32(m, ly->attn_q_norm), *kn = sp_as_f32(m, ly->attn_k_norm);
            for (int h = 0; h < NH;  h++) { float *qh = q    + (size_t)h * HD; sp_rmsnorm_head(qh, qn, HD, eps); sp_rope_neox(qh, HD, pos, base); }
            for (int h = 0; h < NKV; h++) { float *kh = knew + (size_t)h * HD; sp_rmsnorm_head(kh, kn, HD, eps); sp_rope_neox(kh, HD, pos, base); }

            /* recall sidecar: project the post-RoPE K of this (layer,pos) per kv-head. */
            if (g_recall_b > 0)
                for (int hh = 0; hh < NKV; hh++)
                    sp_arm_project(recallR, g_recall_r, HD, knew + (size_t)hh * HD,
                                   projk + (((size_t)L * P + pos) * NKV + hh) * (size_t)g_recall_r);

            /* Store the position-finalized K/V; KC/VC point at what attention reads. */
            const float *KC, *VC;
            if (use_blocks) {
                sp_spinor_block_t *kb = kcb + ((size_t)L * P + pos) * NKV * NBLK;
                sp_spinor_block_t *vb = vcb + ((size_t)L * P + pos) * NKV * NBLK;
                for (int h = 0; h < NKV; h++) {
                    sp_spinor_encode_vec(knew + (size_t)h * HD, HD, kb + (size_t)h * NBLK);
                    sp_spinor_encode_vec(vnew + (size_t)h * HD, HD, vb + (size_t)h * NBLK);
                }
                for (int s = 0; s <= pos; s++) {
                    const sp_spinor_block_t *ks = kcb + ((size_t)L * P + s) * NKV * NBLK;
                    const sp_spinor_block_t *vs = vcb + ((size_t)L * P + s) * NKV * NBLK;
                    for (int h = 0; h < NKV; h++) {
                        (void)sp_spinor_decode_vec(ks + (size_t)h * NBLK, HD, kdec + (size_t)s * KVD + (size_t)h * HD);
                        (void)sp_spinor_decode_vec(vs + (size_t)h * NBLK, HD, vdec + (size_t)s * KVD + (size_t)h * HD);
                    }
                }
                KC = kdec; VC = vdec;
            } else if (fuse && pos < n_prompt) {
                /* FUSE dense prefill: full-P buffer, no window write, no spill. */
                memcpy(kpre + ((size_t)L * P + pos) * KVD, knew, (size_t)KVD * sizeof(float));
                memcpy(vpre + ((size_t)L * P + pos) * KVD, vnew, (size_t)KVD * sizeof(float));
                KC = kpre + (size_t)L * P * KVD;
                VC = vpre + (size_t)L * P * KVD;
            } else {
                if (g_kv_spinor)   /* SP_KV_SPINOR_REF: f32 cache with the lossy round-trip */
                    for (int h = 0; h < NKV; h++) { kv_spinor_roundtrip(knew + (size_t)h * HD, HD); kv_spinor_roundtrip(vnew + (size_t)h * HD, HD); }
                int wslot = sp_arm_r1slot(pos, ring2_on, g_recall_sink, r1W);
                memcpy(kc + ((size_t)L * r1cap + wslot) * KVD, knew, (size_t)KVD * sizeof(float));
                memcpy(vc + ((size_t)L * r1cap + wslot) * KVD, vnew, (size_t)KVD * sizeof(float));
                KC = kc + (size_t)L * r1cap * KVD;
                VC = vc + (size_t)L * r1cap * KVD;
            }
            /* Ring-2 spill of the new position (skipped during the fuse dense prefill). */
            if (r2be_on && !(fuse && pos < n_prompt)) {
                uint64_t boff = (uint64_t)((size_t)L * P + pos) * (uint64_t)((size_t)KVD * sizeof(float));
                if (r2be.write_block(r2be.handle, 0, boff, knew, (size_t)KVD * sizeof(float)) ||
                    r2be.write_block(r2be.handle, 1, boff, vnew, (size_t)KVD * sizeof(float))) goto done;
            } else if (ring2_on && !r2be_on) {
                memcpy(ring2k + ((size_t)L * P + pos) * KVD, knew, (size_t)KVD * sizeof(float));
                memcpy(ring2v + ((size_t)L * P + pos) * KVD, vnew, (size_t)KVD * sizeof(float));
            }

            /* ── attention over the cached [0,pos] ── */
            int winlo = (pos + 1 > g_recall_w) ? (pos + 1 - g_recall_w) : 0;
            int eB = (g_recall_decode_only && pos < n_prompt) ? 0 : g_recall_b;
            if (fuse && pos < n_prompt) {
                /* dense exact prefill over the full-P buffer (absolute slots) */
                for (int h = 0; h < NH; h++) {
                    int kvh = h / group; const float *qh = q + (size_t)h * HD;
                    if (overlay_active) for (int i = 0; i < HD; i++) qi[i] = (int32_t)lrintf(qh[i] * (float)SP_NTT_ATTN_SCALE);
                    float maxs = -INFINITY;
                    for (int s = 0; s <= pos; s++) {
                        float d = kv_pair_score(qh, KC + (size_t)s * KVD + (size_t)kvh * HD, HD, ascale, pr, pr_b, qi, ki);
                        sc[s] = d; if (d > maxs) maxs = d;
                    }
                    float sum = 0.0f;
                    for (int s = 0; s <= pos; s++) { sc[s] = expf(sc[s] - maxs); sum += sc[s]; }
                    float inv = 1.0f / sum; float *out = ao + (size_t)h * HD;
                    for (int i = 0; i < HD; i++) out[i] = 0.0f;
                    for (int s = 0; s <= pos; s++) {
                        float w = sc[s] * inv; const float *vh = VC + (size_t)s * KVD + (size_t)kvh * HD;
                        for (int i = 0; i < HD; i++) out[i] += w * vh[i];
                    }
                }
            } else if (r2be_on) {
                /* 3-phase dedupe: select all heads -> fetch the union once -> attend. */
                for (int h = 0; h < NH; h++)
                    m_all[h] = sp_arm_select(recallR, g_recall_r, HD, q + (size_t)h * HD,
                                             projk, (size_t)L, P, NKV, h / group, eB, g_recall_w,
                                             g_recall_sink, pos, cand, ri_all + (size_t)h * P);
                stg_gen++;
                int nstage = 0;
                for (int h = 0; h < NH; h++) {
                    const int *rih = ri_all + (size_t)h * P;
                    for (int jj = 0; jj < m_all[h]; jj++) {
                        int s = rih[jj];
                        if (s < winlo && s >= g_recall_sink && stg_stamp[s] != stg_gen) {
                            stg_stamp[s] = stg_gen; stg_slot[s] = nstage; stg_pos[nstage] = s; nstage++;
                        }
                    }
                }
                if (nstage > 0 && r2be.read_batch) {
                    for (int i = 0; i < nstage; i++) {
                        uint64_t boff = (uint64_t)((size_t)L * P + stg_pos[i]) * (uint64_t)((size_t)KVD * sizeof(float));
                        rb_which[2 * i] = 0;     rb_off[2 * i] = boff;     rb_dst[2 * i] = stgK + (size_t)i * KVD;
                        rb_which[2 * i + 1] = 1; rb_off[2 * i + 1] = boff; rb_dst[2 * i + 1] = stgV + (size_t)i * KVD;
                    }
                    if (r2be.read_batch(r2be.handle, rb_which, rb_off, rb_dst,
                                        (size_t)KVD * sizeof(float), 2 * nstage)) goto done;
                } else {
                    for (int i = 0; i < nstage; i++) {
                        uint64_t boff = (uint64_t)((size_t)L * P + stg_pos[i]) * (uint64_t)((size_t)KVD * sizeof(float));
                        if (r2be.read_block(r2be.handle, 0, boff, stgK + (size_t)i * KVD, (size_t)KVD * sizeof(float)) ||
                            r2be.read_block(r2be.handle, 1, boff, stgV + (size_t)i * KVD, (size_t)KVD * sizeof(float))) goto done;
                    }
                }
                for (int h = 0; h < NH; h++) {
                    int kvh = h / group; const float *qh = q + (size_t)h * HD;
                    const int *rih = ri_all + (size_t)h * P; int mm = m_all[h];
                    if (overlay_active) for (int i = 0; i < HD; i++) qi[i] = (int32_t)lrintf(qh[i] * (float)SP_NTT_ATTN_SCALE);
                    float maxs = -INFINITY;
                    for (int jj = 0; jj < mm; jj++) {
                        int s = rih[jj];
                        const float *kbase = (s < winlo && s >= g_recall_sink)
                            ? stgK + (size_t)stg_slot[s] * KVD
                            : KC + (size_t)sp_arm_r1slot(s, ring2_on, g_recall_sink, r1W) * KVD;
                        float d = kv_pair_score(qh, kbase + (size_t)kvh * HD, HD, ascale, pr, pr_b, qi, ki);
                        sc[jj] = d; if (d > maxs) maxs = d;
                    }
                    float sum = 0.0f;
                    for (int jj = 0; jj < mm; jj++) { sc[jj] = expf(sc[jj] - maxs); sum += sc[jj]; }
                    float inv = 1.0f / sum; float *out = ao + (size_t)h * HD;
                    for (int i = 0; i < HD; i++) out[i] = 0.0f;
                    for (int jj = 0; jj < mm; jj++) {
                        int s = rih[jj]; float w = sc[jj] * inv;
                        const float *vbase = (s < winlo && s >= g_recall_sink)
                            ? stgV + (size_t)stg_slot[s] * KVD
                            : VC + (size_t)sp_arm_r1slot(s, ring2_on, g_recall_sink, r1W) * KVD;
                        const float *vh = vbase + (size_t)kvh * HD;
                        for (int i = 0; i < HD; i++) out[i] += w * vh[i];
                    }
                }
            } else if (g_recall_b > 0) {
                /* inline select + attend (mock Ring-2 / pure sparse / decode-only). */
                for (int h = 0; h < NH; h++) {
                    int kvh = h / group; const float *qh = q + (size_t)h * HD;
                    int mm = sp_arm_select(recallR, g_recall_r, HD, qh, projk, (size_t)L, P,
                                           NKV, kvh, eB, g_recall_w, g_recall_sink, pos, cand, ri);
                    if (overlay_active) for (int i = 0; i < HD; i++) qi[i] = (int32_t)lrintf(qh[i] * (float)SP_NTT_ATTN_SCALE);
                    float maxs = -INFINITY;
                    for (int jj = 0; jj < mm; jj++) {
                        int s = ri[jj];
                        const float *kbase = (ring2_on && s < winlo && s >= g_recall_sink)
                            ? ring2k + ((size_t)L * P + s) * KVD
                            : KC + (size_t)sp_arm_r1slot(s, ring2_on, g_recall_sink, r1W) * KVD;
                        float d = kv_pair_score(qh, kbase + (size_t)kvh * HD, HD, ascale, pr, pr_b, qi, ki);
                        sc[jj] = d; if (d > maxs) maxs = d;
                    }
                    float sum = 0.0f;
                    for (int jj = 0; jj < mm; jj++) { sc[jj] = expf(sc[jj] - maxs); sum += sc[jj]; }
                    float inv = 1.0f / sum; float *out = ao + (size_t)h * HD;
                    for (int i = 0; i < HD; i++) out[i] = 0.0f;
                    for (int jj = 0; jj < mm; jj++) {
                        int s = ri[jj]; float w = sc[jj] * inv;
                        const float *vbase = (ring2_on && s < winlo && s >= g_recall_sink)
                            ? ring2v + ((size_t)L * P + s) * KVD
                            : VC + (size_t)sp_arm_r1slot(s, ring2_on, g_recall_sink, r1W) * KVD;
                        const float *vh = vbase + (size_t)kvh * HD;
                        for (int i = 0; i < HD; i++) out[i] += w * vh[i];
                    }
                }
            } else {
                /* plain full-context path (ARM off) — the bit-identical baseline. */
                for (int h = 0; h < NH; h++) {
                    int kvh = h / group;
                    const float *qh = q + (size_t)h * HD;
                    if (overlay_active) for (int i = 0; i < HD; i++) qi[i] = (int32_t)lrintf(qh[i] * (float)SP_NTT_ATTN_SCALE);
                    float maxs = -INFINITY;
                    for (int s = 0; s <= pos; s++) {
                        float d = kv_pair_score(qh, KC + (size_t)s * KVD + (size_t)kvh * HD, HD, ascale, pr, pr_b, qi, ki);
                        sc[s] = d; if (d > maxs) maxs = d;
                    }
                    float sum = 0.0f;
                    for (int s = 0; s <= pos; s++) { sc[s] = expf(sc[s] - maxs); sum += sc[s]; }
                    float inv = 1.0f / sum;
                    float *out = ao + (size_t)h * HD;
                    for (int i = 0; i < HD; i++) out[i] = 0.0f;
                    for (int s = 0; s <= pos; s++) {
                        float w = sc[s] * inv;
                        const float *vh = VC + (size_t)s * KVD + (size_t)kvh * HD;
                        for (int i = 0; i < HD; i++) out[i] += w * vh[i];
                    }
                }
            }
            if (sp_matmul(m, ly->attn_output, ao, 1, QD, E, ap)) goto done;
            for (int i = 0; i < E; i++) x[i] += ap[i];

            sp_rmsnorm(x, sp_as_f32(m, ly->ffn_norm), E, eps, nx);
            if (sp_matmul(m, ly->ffn_gate, nx, 1, E, FF, gg)) goto done;
            if (sp_matmul(m, ly->ffn_up,   nx, 1, E, FF, up)) goto done;
            for (int i = 0; i < FF; i++) { float gv = gg[i]; gg[i] = gv / (1.0f + expf(-gv)) * up[i]; }
            if (sp_matmul(m, ly->ffn_down, gg, 1, FF, E, dn)) goto done;
            for (int i = 0; i < E; i++) x[i] += dn[i];
        }

        if (pos >= n_prompt - 1 && produced < n_gen) {         /* emit next token */
            sp_rmsnorm(x, sp_as_f32(m, m->output_norm), E, eps, nx);
            if (sp_matmul(m, m->output, nx, 1, E, V, lg)) goto done;
            int amax = 0;
            for (int j = 1; j < V; j++) if (lg[j] > lg[amax]) amax = j;
            seq[n_prompt + produced] = amax;
            produced++; n = n_prompt + produced;
            if ((eos_id >= 0 && amax == eos_id) || produced >= n_gen) break;
        }
    }
    rc = n;
done:
    free(kcb); free(vcb); free(kdec); free(vdec);
    free(kc); free(vc); free(kpre); free(vpre); free(x); free(nx); free(q); free(knew); free(vnew);
    free(ao); free(ap); free(gg); free(up); free(dn); free(sc); free(lg);
    free(recallR); free(projk); free(ri); free(cand);
    free(ring2k); free(ring2v);
    free(stgK); free(stgV); free(stg_stamp); free(stg_slot); free(stg_pos);
    free(ri_all); free(m_all); free(rb_which); free(rb_off); free(rb_dst);
    if (r2be.handle && r2be.close) r2be.close(r2be.handle);
    free(qi); free(ki); sp_pr_free(pr); sp_pr_bluestein_free(pr_b);
    return rc;
}
