/* decode.c — the canonical persistent-KV O(n) decode (GEN_KV) with the ARM
 * two-ring memory + the NTT decode-attention overlay. Split out of forward.c
 * (Stage C) so this translation unit is the ONLY archive member that defines
 * qwen3_generate_kv / qwen3_ppl_decode: the engine links libsp_forward and the
 * linker pulls exactly this object — the engine keeps its own fast prefill
 * (qwen3_forward*) without symbol collision, while the DECODE has one source
 * of truth: here. Knob statics are TU-local copies (re-read from env per
 * entry, so behaviorally identical to the prefill TU's).
 */
#define _CRT_SECURE_NO_WARNINGS
#include "sp/model.h"
#include "sp/forward_dispatch.h"
#include "sp/forward_kernels.h"
#include "sp/poly_ring.h"
#include "sp/poly_ring_bluestein.h"
#include "sp/spinor_block.h"
#include "sp/arm.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#define SP_NTT_ATTN_SCALE 65536.0   /* 2^16 — must match forward.c (frozen) */

static int g_ntt_attn = 0;
static int g_kv_spinor = 0;
static int g_kv_spinor_ref = 0;
static int g_recall_b = 0;
static int g_recall_r = 16;
static int g_recall_w = 64;
static int g_recall_sink = 4;
static int g_ring2 = 0;
static int g_ring2_store = 0;
static const char *g_ring2_dir = 0;
static int g_recall_decode_only = 0;
static int g_recall_fuse = 0;
/* SP_NTT_KV=1 — THE NTT FUSION: K is stored NATIVELY as its dual-prime residue
 * block (write-once transform after RoPE; sp_pr_kstore_encode) and attention
 * scores read the residues directly (query transformed once per head/step,
 * residue dot + Garner via sp_pr_score_kstore — exact <q,k>, bit-equal to the
 * per-pair overlay, gate T_PR_KSTORE). Stage-1 scope: the plain resident path
 * (ARM knobs off, Spinor-block cache off) and direct-N head dims {128,256,512}
 * (Qwen3 HD=128); inadmissible configs fall back to the baseline silently.
 * Residue-domain Ring-1/Ring-2 spill (block == QUIC ResidueBlock) composes next. */
static int g_ntt_kv = 0;
/* SP_RECALL_BITS=1 — bit-packed popcount router (SimHash overlay): the projk
 * sidecar shrinks from r floats to ONE u64 per (pos,kvh) (32x at r=32) and
 * candidate scoring becomes popcount(qsig ^ ksig). STRICTLY LOSSIER than the
 * f32 projection dot — gated by NIAH retrieval + PPL deflection per regime
 * (see sp/arm.h contract). Off => the proven f32 router, bit-identical. */
static int g_recall_bits = 0;
/* SP_RECALL_KVSEL=1 — GQA KV-head selection (the read-amplification lever):
 * route/select ONCE per KV-head using the GROUP-CENTROID query (sum of the
 * group's Q-heads — the Rademacher projection is linear, so the centroid's
 * signature is the group's majority direction), instead of one independent
 * selection per Q-head. Group members then attend over the SHARED recall
 * set, collapsing the per-layer fetch union by up to group-size (measured
 * ~1 GB/token at 32k from 16-way Q-head divergence; this halves it at the
 * source for NH/NKV=2). CHANGES THE RECALLED SET — regime-gated by NIAH +
 * PPL like every router-policy change; off = proven per-Q-head selection,
 * bit-identical. Applies to both the f32 and bits routers. */
static int g_recall_kvsel = 0;

#define KV_HEAD_MAX_BLOCKS 16
static void kv_spinor_roundtrip(float *vec, int d) {
    sp_spinor_block_t blks[KV_HEAD_MAX_BLOCKS];
    if (sp_spinor_blocks_for(d) > KV_HEAD_MAX_BLOCKS) return;
    sp_spinor_encode_vec(vec, d, blks);
    (void)sp_spinor_decode_vec(blks, d, vec);
}

static void read_env_knobs(void) {
    sp_kernels_read_env();
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
    { const char *e = getenv("SP_NTT_KV");             g_ntt_kv = (e && e[0] == '1'); }
    { const char *e = getenv("SP_RECALL_BITS");        g_recall_bits = (e && e[0] == '1'); }
    { const char *e = getenv("SP_RECALL_KVSEL");       g_recall_kvsel = (e && e[0] == '1'); }
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

/* fusion keystore dispatch: direct sp_pr for HD in {128,256,512}, Bluestein
 * for the remaining power-of-2 HDs (<=256). Same block contract either way:
 * [W residues mod q1][W residues mod q2], W = kstore_words. */
static size_t fz_kstore_words(sp_pr_ctx *pr, sp_pr_bluestein_ctx *pr_b) {
    return pr ? sp_pr_kstore_words(pr) : sp_pr_bluestein_kstore_words(pr_b);
}
/* batched dispatch (q-transform amortization): the per-token homologous
 * transforms (NH queries / NKV key encodes per layer) go through ONE
 * sp_ntt_fwd_batch call on the direct-N arm — the engine override runs
 * lanes = heads. Bit-equal to the single-call path (gate T_PR_BATCH). */
static void fz_kstore_encode_batch(sp_pr_ctx *pr, sp_pr_bluestein_ctx *pr_b,
                                   const int32_t *k, size_t kstride,
                                   uint32_t *out, size_t ostride, int nb) {
    if (pr) sp_pr_kstore_encode_batch(pr, k, kstride, out, ostride, nb);
    else    sp_pr_bluestein_kstore_encode_batch(pr_b, k, kstride, out, ostride, nb);
}
static void fz_query_begin_batch(sp_pr_ctx *pr, sp_pr_bluestein_ctx *pr_b,
                                 const int32_t *q, size_t qstride, int nb) {
    if (pr) sp_pr_query_begin_batch(pr, q, qstride, nb);
    else    sp_pr_bluestein_query_begin_batch(pr_b, q, qstride, nb);
}
static int64_t fz_score_kstore_b(sp_pr_ctx *pr, sp_pr_bluestein_ctx *pr_b,
                                 int i, const uint32_t *kres) {
    return pr ? sp_pr_score_kstore_b(pr, i, kres)
              : sp_pr_bluestein_score_kstore_b(pr_b, i, kres);
}


/* Shared decode body for qwen3_generate_kv (ppl_mode=0, argmax emit) and
 * qwen3_ppl_decode (ppl_mode=1, teacher-forced NLL of seq[pos+1] over
 * [n_warm, P-2]). ONE forward path — the recall router + two-ring are
 * identical in both modes. */
static int generate_kv_impl(const qwen3_model *m, int32_t *seq, int n_prompt, int n_gen,
                            int eos_id, int ppl_mode, int n_warm,
                            double *nll_out, long *nscored_out) {
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
    uint64_t *sigk = NULL;       /* SP_RECALL_BITS: 1-u64 sign-bit sidecar instead */
    float *qg = NULL;            /* SP_RECALL_KVSEL: group-centroid query scratch  */
    float *ring2k = NULL, *ring2v = NULL;                /* mock RAM Ring-2 store */
    float *stgK = NULL, *stgV = NULL;                    /* backend dedupe staging */
    int *stg_stamp = NULL, *stg_slot = NULL, *stg_pos = NULL;
    int *ri_all = NULL, *m_all = NULL;                   /* per-head recall sets (backend path) */
    int *rb_which = NULL; uint64_t *rb_off = NULL; void **rb_dst = NULL; /* optional read_batch reqs */
    int *ri = NULL; sp_arm_sidx *cand = NULL;            /* inline-path selection scratch */
    sp_arm_ring2_backend r2be; int r2be_on = 0; r2be.handle = NULL; r2be.close = NULL;
    int r2be_owned = 0;      /* stdio reference = owned (closed at done); registered = borrowed */
    int stg_hooked = 0;      /* staging buffers came from the backend's aligned allocator */
    int stg_gen = 0;
    /* NTT decode-attention overlay (SP_ENGINE_NTT_ATTN). */
    sp_pr_ctx *pr = NULL; sp_pr_bluestein_ctx *pr_b = NULL; int overlay_active = 0;
    int32_t *qi = NULL, *ki = NULL;
    int32_t *qib = NULL, *kib = NULL;  /* fusion batch staging: NH×HD / NKV×HD */
    uint32_t *kres = NULL; int fusion_on = 0; size_t krw = 0;  /* SP_NTT_KV keystore */
    uint32_t *kres_pre = NULL;     /* fusion x fuse: full-P residue prefill buffer */
    uint32_t *ring2k_res = NULL;   /* fusion x mock Ring-2: residue K byte store */
    uint32_t *stgKres = NULL;      /* fusion x backend: residue K dedupe staging */

    /* ── ARM configuration (engine-faithful gating) ── */
    const int ring2_on = (g_ring2 && g_recall_b > 0 && !use_blocks && !g_recall_decode_only);
    r2be_on = ring2_on && g_ring2_store;
    const int fuse = (g_recall_fuse && r2be_on);
    const int r1W = (g_recall_w > 0) ? g_recall_w : 1;
    const int r1cap = ring2_on ? (g_recall_sink + r1W) : P;

    /* NTT decode overlay + FUSION init — mirrors the prefill overlay HD-dispatch.
     * overlay (g_ntt_attn): per-pair poly-ring score, direct or Bluestein.
     * fusion (g_ntt_kv):    keystore residue cache, DIRECT-N only. */
    if (g_ntt_attn || g_ntt_kv) {
        if (HD == 128 || HD == 256 || HD == 512) {
            pr = sp_pr_init((uint32_t)HD);
        } else if ((g_ntt_attn || g_ntt_kv) && HD >= 2 && HD <= 256 && (HD & (HD - 1)) == 0) {
            pr_b = sp_pr_bluestein_init((uint32_t)HD);
        }
        overlay_active = g_ntt_attn && (pr || pr_b);
        /* stage-2: fusion composes with the ARM rings — the K stream is residue
         * blocks end-to-end (Ring-1 slots, mock store, backend spill/fetch, fuse
         * buffer). V stays f32 (never scored; fp is plumbing). */
        fusion_on = g_ntt_kv && (pr != NULL || pr_b != NULL) && !use_blocks;
        if (overlay_active || fusion_on) {
            qi = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
            ki = (int32_t *)malloc((size_t)HD * sizeof(int32_t));
            if (!qi || !ki) goto done;
        }
        if (fusion_on) {
            krw = fz_kstore_words(pr, pr_b);               /* 2N (direct) / 2M (Bluestein) u32 per head */
            kres = (uint32_t *)malloc((size_t)c->n_layers * r1cap * NKV * krw * sizeof(uint32_t));
            qib  = (int32_t *)malloc((size_t)NH  * HD * sizeof(int32_t));
            kib  = (int32_t *)malloc((size_t)NKV * HD * sizeof(int32_t));
            if (!kres || !qib || !kib) goto done;
            fprintf(stderr, "    [ntt-kv] FUSION ON: K cached natively as dual-prime residue blocks "
                    "(%zu u32/head, write-once transform; scores = residue dot + Garner, exact)\n", krw);
        }
        if (overlay_active && fusion_on) overlay_active = 0;   /* fusion branch supersedes */
    }


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
        ri      = (int *)malloc((size_t)P * sizeof(int));
        cand    = (sp_arm_sidx *)malloc((size_t)P * sizeof(sp_arm_sidx));
        if (g_recall_bits) {
            sigk = (uint64_t *)malloc((size_t)c->n_layers * P * NKV * sizeof(uint64_t));
            if (!sigk) goto done;
        } else {
            projk = (float *)malloc((size_t)c->n_layers * P * NKV * (size_t)g_recall_r * sizeof(float));
            if (!projk) goto done;
        }
        if (!recallR || !ri || !cand) goto done;
        if (g_recall_kvsel) {
            qg = (float *)malloc((size_t)HD * sizeof(float));
            if (!qg) goto done;
            fprintf(stderr, "    [recall] KV-HEAD selection ON: %d selections/layer (was %d) — "
                    "group-centroid routing, shared recall sets\n", NKV, NH);
        }
        sp_arm_build_R(recallR, g_recall_r, HD);   /* frozen ±1 matrix, deterministic */
        if (g_recall_bits)
            fprintf(stderr, "    [recall] sidecar ON: r=%d B=%d W=%d sink=%d BIT-PACKED popcount router "
                    "(%zu KB vs %zu KB f32, %dx)\n",
                    g_recall_r, g_recall_b, g_recall_w, g_recall_sink,
                    ((size_t)c->n_layers * P * NKV * sizeof(uint64_t)) >> 10,
                    ((size_t)c->n_layers * P * NKV * (size_t)g_recall_r * sizeof(float)) >> 10,
                    (int)((size_t)g_recall_r * sizeof(float) / sizeof(uint64_t)));
        else
            fprintf(stderr, "    [recall] sidecar ON: r=%d B=%d W=%d sink=%d (post-RoPE ±1 projection router + sinks)\n",
                    g_recall_r, g_recall_b, g_recall_w, g_recall_sink);
    }
    if (ring2_on && !r2be_on) {
        size_t kvn = (size_t)c->n_layers * P * KVD;
        if (!fusion_on) { ring2k = (float *)malloc(kvn * sizeof(float)); if (!ring2k) goto done; }
        else { ring2k_res = (uint32_t *)malloc((size_t)c->n_layers * P * NKV * krw * sizeof(uint32_t));
               if (!ring2k_res) goto done; }
        ring2v = (float *)malloc(kvn * sizeof(float));
        if (!ring2v) goto done;
        fprintf(stderr, "    [ring2] mock RAM spill ON: Ring-1 holds only sink+W=%d slots\n",
                g_recall_sink + r1W);
    } else if (r2be_on) {
        /* Stage C: a registered platform backend (engine Optane IOCP, QUIC peer)
         * takes precedence over the stdio reference. Registered = BORROWED — the
         * registrant owns the lifetime; we never close it. */
        if (sp_arm_ring2_registered(&r2be)) {
            fprintf(stderr, "    [ring2] using REGISTERED platform backend (%s reads%s)\n",
                    r2be.read_batch ? "batched" : "serial",
                    r2be.alloc_aligned ? ", direct-I/O aligned staging" : "");
        } else {
            if (sp_arm_ring2_stdio_open(g_ring2_dir, &r2be)) { r2be_on = 0; goto done; }
            r2be_owned = 1;
        }
        stg_hooked = (r2be.alloc_aligned != NULL);
        if (!fusion_on)
            stgK = stg_hooked ? (float *)r2be.alloc_aligned(r2be.handle, (size_t)P * KVD * sizeof(float))
                              : (float *)malloc((size_t)P * KVD * sizeof(float));
        else
            stgKres = stg_hooked ? (uint32_t *)r2be.alloc_aligned(r2be.handle, (size_t)P * NKV * krw * sizeof(uint32_t))
                                 : (uint32_t *)malloc((size_t)P * NKV * krw * sizeof(uint32_t));
        stgV = stg_hooked ? (float *)r2be.alloc_aligned(r2be.handle, (size_t)P * KVD * sizeof(float))
                          : (float *)malloc((size_t)P * KVD * sizeof(float));
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
        if ((!stgK && !stgKres) || !stgV || !stg_stamp || !stg_slot || !stg_pos || !ri_all || !m_all) goto done;
        for (int i = 0; i < P; i++) stg_stamp[i] = -1;
        if (fuse) {
            if (!fusion_on) { kpre = (float *)malloc((size_t)c->n_layers * P * KVD * sizeof(float)); if (!kpre) goto done; }
            else { kres_pre = (uint32_t *)malloc((size_t)c->n_layers * P * NKV * krw * sizeof(uint32_t));
                   if (!kres_pre) goto done; }
            vpre = (float *)malloc((size_t)c->n_layers * P * KVD * sizeof(float));
            if (!vpre) goto done;
            fprintf(stderr, "    [fuse] compact-and-spill: dense full-P prefill buffer (%.1f MB), freed at the boundary\n",
                    2.0 * c->n_layers * P * KVD * sizeof(float) / 1e6);
        }
        fprintf(stderr, "    [ring2] BACKEND spill ON (W=%d, sinks pinned; v1 per-layer dedupe staging%s)\n",
                g_recall_w, r2be.read_batch ? ", batched reads" : ", serial reference reads");
    }
    for (int pos = 0; pos < P; pos++) {
        int tok = seq[pos];
        if (sp_embed_row(m, tok, E, x)) goto done;

        if (fuse && pos == n_prompt) {
            /* FUSE boundary: bulk-spill the prefill history to Ring-2, copy sinks +
             * the recent window into the (sink+W) cache, FREE the full-P buffer. */
            int W = r1W, sink = g_recall_sink;
            int wlo = n_prompt - W; if (wlo < sink) wlo = sink;
            const size_t kblk = fusion_on ? (size_t)NKV * krw * sizeof(uint32_t)
                                          : (size_t)KVD * sizeof(float);
            for (uint32_t L = 0; L < c->n_layers; L++) {
                const float *kpL = kpre ? kpre + (size_t)L * P * KVD : NULL;
                const uint32_t *krL = kres_pre ? kres_pre + (size_t)L * P * NKV * krw : NULL;
                const float *vpL = vpre + (size_t)L * P * KVD;
                for (int s = 0; s < n_prompt; s++) {
                    uint64_t bK = (uint64_t)((size_t)L * P + s) * (uint64_t)kblk;
                    uint64_t bV = (uint64_t)((size_t)L * P + s) * (uint64_t)((size_t)KVD * sizeof(float));
                    const void *ksrc = fusion_on ? (const void *)(krL + (size_t)s * NKV * krw)
                                                 : (const void *)(kpL + (size_t)s * KVD);
                    if (r2be.write_block(r2be.handle, 0, bK, ksrc, kblk) ||
                        r2be.write_block(r2be.handle, 1, bV, vpL + (size_t)s * KVD, (size_t)KVD * sizeof(float))) goto done;
                }
                float *vcL = vc + (size_t)L * r1cap * KVD;
                float *kcL = kc + (size_t)L * r1cap * KVD;
                uint32_t *krcL = fusion_on ? kres + (size_t)L * r1cap * NKV * krw : NULL;
                for (int pass = 0; pass < 2; pass++) {
                    int lo = pass ? wlo : 0, hi = pass ? n_prompt : (sink < n_prompt ? sink : n_prompt);
                    for (int s = lo; s < hi; s++) {
                        int sl = sp_arm_r1slot(s, 1, sink, W);
                        if (fusion_on)
                            memcpy(krcL + (size_t)sl * NKV * krw, krL + (size_t)s * NKV * krw, kblk);
                        else
                            memcpy(kcL + (size_t)sl * KVD, kpL + (size_t)s * KVD, (size_t)KVD * sizeof(float));
                        memcpy(vcL + (size_t)sl * KVD, vpL + (size_t)s * KVD, (size_t)KVD * sizeof(float));
                    }
                }
            }
            free(kpre); free(vpre); free(kres_pre); kpre = NULL; vpre = NULL; kres_pre = NULL;
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
            if (g_recall_b > 0) {
                if (g_recall_bits)   /* head-major sidecar (arm.h layout v2) */
                    for (int hh = 0; hh < NKV; hh++)
                        sigk[((size_t)L * NKV + hh) * P + pos] =
                            sp_arm_project_sig(recallR, g_recall_r, HD, knew + (size_t)hh * HD);
                else
                    for (int hh = 0; hh < NKV; hh++)
                        sp_arm_project(recallR, g_recall_r, HD, knew + (size_t)hh * HD,
                                       projk + (((size_t)L * P + pos) * NKV + hh) * (size_t)g_recall_r);
            }

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
                if (fusion_on) {
                    for (int h = 0; h < NKV; h++) {
                        const float *kh = knew + (size_t)h * HD;
                        for (int i = 0; i < HD; i++)
                            kib[(size_t)h * HD + i] = (int32_t)lrintf(kh[i] * (float)SP_NTT_ATTN_SCALE);
                    }
                    fz_kstore_encode_batch(pr, pr_b, kib, (size_t)HD,
                                           kres_pre + ((size_t)L * P + pos) * NKV * krw, krw, NKV);
                } else
                    memcpy(kpre + ((size_t)L * P + pos) * KVD, knew, (size_t)KVD * sizeof(float));
                memcpy(vpre + ((size_t)L * P + pos) * KVD, vnew, (size_t)KVD * sizeof(float));
                KC = kpre ? kpre + (size_t)L * P * KVD : NULL;
                VC = vpre + (size_t)L * P * KVD;
            } else {
                if (g_kv_spinor)   /* SP_KV_SPINOR_REF: f32 cache with the lossy round-trip */
                    for (int h = 0; h < NKV; h++) { kv_spinor_roundtrip(knew + (size_t)h * HD, HD); kv_spinor_roundtrip(vnew + (size_t)h * HD, HD); }
                int wslot = sp_arm_r1slot(pos, ring2_on, g_recall_sink, r1W);
                memcpy(kc + ((size_t)L * r1cap + wslot) * KVD, knew, (size_t)KVD * sizeof(float));
                memcpy(vc + ((size_t)L * r1cap + wslot) * KVD, vnew, (size_t)KVD * sizeof(float));
                KC = kc + (size_t)L * r1cap * KVD;
                VC = vc + (size_t)L * r1cap * KVD;
                /* NTT FUSION write path: the position-finalized K head is quantized
                 * and forward-transformed ONCE; the residue block is the stored unit
                 * (ring slot when offloading — identical to (L*P+pos) when r1cap==P). */
                if (fusion_on) {
                    for (int h = 0; h < NKV; h++) {
                        const float *kh = knew + (size_t)h * HD;
                        for (int i = 0; i < HD; i++)
                            kib[(size_t)h * HD + i] = (int32_t)lrintf(kh[i] * (float)SP_NTT_ATTN_SCALE);
                    }
                    fz_kstore_encode_batch(pr, pr_b, kib, (size_t)HD,
                                           kres + ((size_t)L * r1cap + wslot) * NKV * krw, krw, NKV);
                }
            }
            /* Ring-2 spill of the new position (skipped during the fuse dense prefill). */
            if (r2be_on && !(fuse && pos < n_prompt)) {
                const size_t kblk = fusion_on ? (size_t)NKV * krw * sizeof(uint32_t)
                                              : (size_t)KVD * sizeof(float);
                uint64_t bK = (uint64_t)((size_t)L * P + pos) * (uint64_t)kblk;
                uint64_t bV = (uint64_t)((size_t)L * P + pos) * (uint64_t)((size_t)KVD * sizeof(float));
                const void *ksrc = fusion_on
                    ? (const void *)(kres + (((size_t)L * r1cap +
                          (size_t)sp_arm_r1slot(pos, ring2_on, g_recall_sink, r1W)) * NKV) * krw)
                    : (const void *)knew;
                if (r2be.write_block(r2be.handle, 0, bK, ksrc, kblk) ||
                    r2be.write_block(r2be.handle, 1, bV, vnew, (size_t)KVD * sizeof(float))) goto done;
            } else if (ring2_on && !r2be_on) {
                if (fusion_on)
                    memcpy(ring2k_res + ((size_t)L * P + pos) * NKV * krw,
                           kres + (((size_t)L * r1cap +
                               (size_t)sp_arm_r1slot(pos, ring2_on, g_recall_sink, r1W)) * NKV) * krw,
                           (size_t)NKV * krw * sizeof(uint32_t));
                else
                    memcpy(ring2k + ((size_t)L * P + pos) * KVD, knew, (size_t)KVD * sizeof(float));
                memcpy(ring2v + ((size_t)L * P + pos) * KVD, vnew, (size_t)KVD * sizeof(float));
            }

            /* ── attention over the cached [0,pos] ── */
            int winlo = (pos + 1 > g_recall_w) ? (pos + 1 - g_recall_w) : 0;
            int eB = (g_recall_decode_only && pos < n_prompt) ? 0 : g_recall_b;
            if (fuse && pos < n_prompt) {
                /* dense exact prefill over the full-P buffer (absolute slots) */
                if (fusion_on) {           /* all NH query transforms in ONE batch */
                    for (int h = 0; h < NH; h++) {
                        const float *qh = q + (size_t)h * HD;
                        for (int i = 0; i < HD; i++)
                            qib[(size_t)h * HD + i] = (int32_t)lrintf(qh[i] * (float)SP_NTT_ATTN_SCALE);
                    }
                    fz_query_begin_batch(pr, pr_b, qib, (size_t)HD, NH);
                }
                for (int h = 0; h < NH; h++) {
                    int kvh = h / group; const float *qh = q + (size_t)h * HD;
                    if (overlay_active) for (int i = 0; i < HD; i++) qi[i] = (int32_t)lrintf(qh[i] * (float)SP_NTT_ATTN_SCALE);
                    float maxs = -INFINITY;
                    for (int s = 0; s <= pos; s++) {
                        float d;
                        if (fusion_on) {
                            int64_t ip = fz_score_kstore_b(pr, pr_b, h, kres_pre + (((size_t)L * P + s) * NKV + kvh) * krw);
                            d = (float)((double)ip / (SP_NTT_ATTN_SCALE * SP_NTT_ATTN_SCALE)) * ascale;
                        } else
                            d = kv_pair_score(qh, KC + (size_t)s * KVD + (size_t)kvh * HD, HD, ascale, pr, pr_b, qi, ki);
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
                /* 3-phase dedupe: select -> fetch the union once -> attend.
                 * SP_RECALL_KVSEL: ONE selection per KV-head from the group-
                 * centroid query (projection is linear => centroid signature =
                 * the group's majority direction); group members share the set,
                 * collapsing the fetch union (the read-amplification lever). */
                if (g_recall_kvsel) {
                    for (int kvh = 0; kvh < NKV; kvh++) {
                        for (int i = 0; i < HD; i++) qg[i] = 0.0f;
                        for (int g2 = 0; g2 < group; g2++) {
                            const float *qx = q + (size_t)(kvh * group + g2) * HD;
                            for (int i = 0; i < HD; i++) qg[i] += qx[i];
                        }
                        int m = g_recall_bits
                            ? sp_arm_select_sig(recallR, g_recall_r, HD, qg, sigk, (size_t)L, P,
                                                NKV, kvh, eB, g_recall_w, g_recall_sink, pos,
                                                cand, ri_all + (size_t)kvh * P)
                            : sp_arm_select(recallR, g_recall_r, HD, qg, projk, (size_t)L, P,
                                            NKV, kvh, eB, g_recall_w, g_recall_sink, pos,
                                            cand, ri_all + (size_t)kvh * P);
                        for (int g2 = 0; g2 < group; g2++) m_all[kvh * group + g2] = m;
                    }
                } else {
                    for (int h = 0; h < NH; h++)
                        m_all[h] = g_recall_bits
                            ? sp_arm_select_sig(recallR, g_recall_r, HD, q + (size_t)h * HD,
                                                sigk, (size_t)L, P, NKV, h / group, eB, g_recall_w,
                                                g_recall_sink, pos, cand, ri_all + (size_t)h * P)
                            : sp_arm_select(recallR, g_recall_r, HD, q + (size_t)h * HD,
                                            projk, (size_t)L, P, NKV, h / group, eB, g_recall_w,
                                            g_recall_sink, pos, cand, ri_all + (size_t)h * P);
                }
                stg_gen++;
                int nstage = 0;
                for (int h = 0; h < NH; h++) {
                    const int *rih = ri_all + (size_t)(g_recall_kvsel ? (h / group) : h) * P;
                    for (int jj = 0; jj < m_all[h]; jj++) {
                        int s = rih[jj];
                        if (s < winlo && s >= g_recall_sink && stg_stamp[s] != stg_gen) {
                            stg_stamp[s] = stg_gen; stg_slot[s] = nstage; stg_pos[nstage] = s; nstage++;
                        }
                    }
                }
                {
                const size_t kblk = fusion_on ? (size_t)NKV * krw * sizeof(uint32_t)
                                              : (size_t)KVD * sizeof(float);
                const size_t vblk = (size_t)KVD * sizeof(float);
                if (nstage > 0 && r2be.read_batch2) {
                    /* mixed-stream batch: BOTH streams in one call so a split-
                     * device backend overlaps the two physical queues (the
                     * device-overlap fix — two sequential read_batch calls
                     * serialize the drives). rb_* arrays are sized 2*P. */
                    for (int i = 0; i < nstage; i++) {
                        rb_which[i] = 0;
                        rb_off[i]   = (uint64_t)((size_t)L * P + stg_pos[i]) * (uint64_t)kblk;
                        rb_dst[i]   = fusion_on ? (void *)(stgKres + (size_t)i * NKV * krw)
                                                : (void *)(stgK + (size_t)i * KVD);
                        rb_which[nstage + i] = 1;
                        rb_off[nstage + i]   = (uint64_t)((size_t)L * P + stg_pos[i]) * (uint64_t)vblk;
                        rb_dst[nstage + i]   = stgV + (size_t)i * KVD;
                    }
                    const size_t lens[2] = { kblk, vblk };
                    if (r2be.read_batch2(r2be.handle, rb_which, rb_off, rb_dst, lens, 2 * nstage)) goto done;
                } else if (nstage > 0 && r2be.read_batch) {
                    /* per-stream batches: K and V block sizes differ under fusion */
                    for (int i = 0; i < nstage; i++) {
                        rb_which[i] = 0;
                        rb_off[i]   = (uint64_t)((size_t)L * P + stg_pos[i]) * (uint64_t)kblk;
                        rb_dst[i]   = fusion_on ? (void *)(stgKres + (size_t)i * NKV * krw)
                                                : (void *)(stgK + (size_t)i * KVD);
                    }
                    if (r2be.read_batch(r2be.handle, rb_which, rb_off, rb_dst, kblk, nstage)) goto done;
                    for (int i = 0; i < nstage; i++) {
                        rb_which[i] = 1;
                        rb_off[i]   = (uint64_t)((size_t)L * P + stg_pos[i]) * (uint64_t)vblk;
                        rb_dst[i]   = stgV + (size_t)i * KVD;
                    }
                    if (r2be.read_batch(r2be.handle, rb_which, rb_off, rb_dst, vblk, nstage)) goto done;
                } else {
                    for (int i = 0; i < nstage; i++) {
                        uint64_t bK = (uint64_t)((size_t)L * P + stg_pos[i]) * (uint64_t)kblk;
                        uint64_t bV = (uint64_t)((size_t)L * P + stg_pos[i]) * (uint64_t)vblk;
                        void *kdst = fusion_on ? (void *)(stgKres + (size_t)i * NKV * krw)
                                               : (void *)(stgK + (size_t)i * KVD);
                        if (r2be.read_block(r2be.handle, 0, bK, kdst, kblk) ||
                            r2be.read_block(r2be.handle, 1, bV, stgV + (size_t)i * KVD, vblk)) goto done;
                    }
                }
                }
                if (fusion_on) {           /* all NH query transforms in ONE batch */
                    for (int h = 0; h < NH; h++) {
                        const float *qh = q + (size_t)h * HD;
                        for (int i = 0; i < HD; i++)
                            qib[(size_t)h * HD + i] = (int32_t)lrintf(qh[i] * (float)SP_NTT_ATTN_SCALE);
                    }
                    fz_query_begin_batch(pr, pr_b, qib, (size_t)HD, NH);
                }
                for (int h = 0; h < NH; h++) {
                    int kvh = h / group; const float *qh = q + (size_t)h * HD;
                    const int *rih = ri_all + (size_t)(g_recall_kvsel ? kvh : h) * P;
                    int mm = m_all[h];
                    if (overlay_active) for (int i = 0; i < HD; i++) qi[i] = (int32_t)lrintf(qh[i] * (float)SP_NTT_ATTN_SCALE);
                    float maxs = -INFINITY;
                    for (int jj = 0; jj < mm; jj++) {
                        int s = rih[jj];
                        float d;
                        if (fusion_on) {
                            const uint32_t *kblk = (s < winlo && s >= g_recall_sink)
                                ? stgKres + (size_t)stg_slot[s] * NKV * krw
                                : kres + ((size_t)L * r1cap +
                                      (size_t)sp_arm_r1slot(s, ring2_on, g_recall_sink, r1W)) * NKV * krw;
                            int64_t ip = fz_score_kstore_b(pr, pr_b, h, kblk + (size_t)kvh * krw);
                            d = (float)((double)ip / (SP_NTT_ATTN_SCALE * SP_NTT_ATTN_SCALE)) * ascale;
                        } else {
                            const float *kbase = (s < winlo && s >= g_recall_sink)
                                ? stgK + (size_t)stg_slot[s] * KVD
                                : KC + (size_t)sp_arm_r1slot(s, ring2_on, g_recall_sink, r1W) * KVD;
                            d = kv_pair_score(qh, kbase + (size_t)kvh * HD, HD, ascale, pr, pr_b, qi, ki);
                        }
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
                if (fusion_on) {           /* all NH query transforms in ONE batch */
                    for (int h = 0; h < NH; h++) {
                        const float *qh = q + (size_t)h * HD;
                        for (int i = 0; i < HD; i++)
                            qib[(size_t)h * HD + i] = (int32_t)lrintf(qh[i] * (float)SP_NTT_ATTN_SCALE);
                    }
                    fz_query_begin_batch(pr, pr_b, qib, (size_t)HD, NH);
                }
                int kv_mm = 0;   /* SP_RECALL_KVSEL: the group's shared set size */
                for (int h = 0; h < NH; h++) {
                    int kvh = h / group; const float *qh = q + (size_t)h * HD;
                    int mm;
                    if (g_recall_kvsel) {
                        if (h % group == 0) {   /* group leader selects with the centroid */
                            for (int i = 0; i < HD; i++) qg[i] = 0.0f;
                            for (int g2 = 0; g2 < group; g2++) {
                                const float *qx = q + (size_t)(kvh * group + g2) * HD;
                                for (int i = 0; i < HD; i++) qg[i] += qx[i];
                            }
                            kv_mm = g_recall_bits
                                ? sp_arm_select_sig(recallR, g_recall_r, HD, qg, sigk, (size_t)L, P,
                                                    NKV, kvh, eB, g_recall_w, g_recall_sink, pos, cand, ri)
                                : sp_arm_select(recallR, g_recall_r, HD, qg, projk, (size_t)L, P,
                                                NKV, kvh, eB, g_recall_w, g_recall_sink, pos, cand, ri);
                        }
                        mm = kv_mm;             /* ri persists across the group */
                    } else
                        mm = g_recall_bits
                            ? sp_arm_select_sig(recallR, g_recall_r, HD, qh, sigk, (size_t)L, P,
                                                NKV, kvh, eB, g_recall_w, g_recall_sink, pos, cand, ri)
                            : sp_arm_select(recallR, g_recall_r, HD, qh, projk, (size_t)L, P,
                                            NKV, kvh, eB, g_recall_w, g_recall_sink, pos, cand, ri);
                    if (overlay_active) for (int i = 0; i < HD; i++) qi[i] = (int32_t)lrintf(qh[i] * (float)SP_NTT_ATTN_SCALE);
                    float maxs = -INFINITY;
                    for (int jj = 0; jj < mm; jj++) {
                        int s = ri[jj];
                        float d;
                        if (fusion_on) {
                            const uint32_t *kblk = (ring2_on && s < winlo && s >= g_recall_sink)
                                ? ring2k_res + ((size_t)L * P + s) * NKV * krw
                                : kres + ((size_t)L * r1cap +
                                      (size_t)sp_arm_r1slot(s, ring2_on, g_recall_sink, r1W)) * NKV * krw;
                            int64_t ip = fz_score_kstore_b(pr, pr_b, h, kblk + (size_t)kvh * krw);
                            d = (float)((double)ip / (SP_NTT_ATTN_SCALE * SP_NTT_ATTN_SCALE)) * ascale;
                        } else {
                            const float *kbase = (ring2_on && s < winlo && s >= g_recall_sink)
                                ? ring2k + ((size_t)L * P + s) * KVD
                                : KC + (size_t)sp_arm_r1slot(s, ring2_on, g_recall_sink, r1W) * KVD;
                            d = kv_pair_score(qh, kbase + (size_t)kvh * HD, HD, ascale, pr, pr_b, qi, ki);
                        }
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
            } else if (fusion_on) {
                /* NTT FUSION read path: ALL NH query transforms in one batch
                 * (the engine override runs lanes = heads); each cached K costs
                 * one residue dot per prime + scalar Garner — the exact integer
                 * <q,k>, no per-pair forward, no inverse butterflies. */
                for (int h = 0; h < NH; h++) {
                    const float *qh = q + (size_t)h * HD;
                    for (int i = 0; i < HD; i++)
                        qib[(size_t)h * HD + i] = (int32_t)lrintf(qh[i] * (float)SP_NTT_ATTN_SCALE);
                }
                fz_query_begin_batch(pr, pr_b, qib, (size_t)HD, NH);
                for (int h = 0; h < NH; h++) {
                    int kvh = h / group;
                    float maxs = -INFINITY;
                    for (int s = 0; s <= pos; s++) {
                        int64_t ip = fz_score_kstore_b(pr, pr_b, h, kres + (((size_t)L * r1cap + s) * NKV + kvh) * krw);
                        float d = (float)((double)ip / (SP_NTT_ATTN_SCALE * SP_NTT_ATTN_SCALE)) * ascale;
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

        if (ppl_mode) {
            /* G2: teacher-forced autoregressive PPL — logits at every scored pos,
             * accumulate -log p(seq[pos+1]) for pos in [n_warm, P-2]. The two-ring
             * path above is exercised exactly as in production decode. */
            if (pos + 1 < P && pos >= n_warm) {
                sp_rmsnorm(x, sp_as_f32(m, m->output_norm), E, eps, nx);
                if (sp_matmul(m, m->output, nx, 1, E, V, lg)) goto done;
                int tgt = seq[pos + 1];
                if (tgt >= 0 && tgt < V) {
                    float maxl = lg[0];
                    for (int j = 1; j < V; j++) if (lg[j] > maxl) maxl = lg[j];
                    double sumexp = 0.0;
                    for (int j = 0; j < V; j++) sumexp += exp((double)lg[j] - (double)maxl);
                    double logp = (double)lg[tgt] - (double)maxl - log(sumexp);
                    if (nll_out)     *nll_out += -logp;
                    if (nscored_out) (*nscored_out)++;
                }
            }
        } else if (pos >= n_prompt - 1 && produced < n_gen) {  /* emit next token */
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
    free(recallR); free(projk); free(sigk); free(qg); free(ri); free(cand);
    free(ring2k); free(ring2v);
    if (stg_hooked) {                  /* backend-provided direct-I/O buffers */
        if (stgK) r2be.free_aligned(r2be.handle, stgK);
        if (stgV) r2be.free_aligned(r2be.handle, stgV);
    } else { free(stgK); free(stgV); }
    free(stg_stamp); free(stg_slot); free(stg_pos);
    free(ri_all); free(m_all); free(rb_which); free(rb_off); free(rb_dst);
    if (r2be_owned && r2be.handle && r2be.close) r2be.close(r2be.handle);
    free(kres_pre); free(ring2k_res);
    if (stgKres) { if (stg_hooked) r2be.free_aligned(r2be.handle, stgKres); else free(stgKres); }
    free(qi); free(ki); free(qib); free(kib); free(kres);
    sp_pr_free(pr); sp_pr_bluestein_free(pr_b);
    return rc;
}

int qwen3_generate_kv(const qwen3_model *m, int32_t *seq, int n_prompt, int n_gen,
                      int eos_id) {
    return generate_kv_impl(m, seq, n_prompt, n_gen, eos_id, /*ppl_mode=*/0, 0, NULL, NULL);
}

/* Teacher-forced autoregressive perplexity over the DECODE path (G2): the recall
 * router + two-ring (SP_RECALL_* / SP_RING2*) are exercised exactly as production
 * decode. toks[0,n_toks) is the corpus slice; positions [n_warm, n_toks-1) are
 * scored (predict toks[pos+1] from the logits at pos). Returns 0 on success and
 * sets *ppl = exp(mean NLL) (+ *n_scored if non-NULL). Shares generate_kv_impl —
 * one forward path, no divergence. */
int qwen3_ppl_decode(const qwen3_model *m, int32_t *toks, int n_toks, int n_warm,
                     double *ppl, long *n_scored) {
    if (!m || !toks || n_toks < 4 || !ppl) return 1;
    if (n_warm < 1) n_warm = 1;
    if (n_warm > n_toks - 2) n_warm = n_toks - 2;
    double nll = 0.0; long ns = 0;
    int rc = generate_kv_impl(m, toks, n_toks, /*n_gen=*/0, /*eos=*/-1,
                              /*ppl_mode=*/1, n_warm, &nll, &ns);
    if (rc < 0 || ns <= 0) return 1;
    *ppl = exp(nll / (double)ns);
    if (n_scored) *n_scored = ns;
    return 0;
}
