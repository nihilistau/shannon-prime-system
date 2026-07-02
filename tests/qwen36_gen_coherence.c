/* qwen36_gen_coherence.c — G-MOE-COHERENCE: greedy self-feed generation on the
 * Qwen3.6-35B-A3B via the STATELESS qwen36_forward (no persistent-KV path exists
 * for qwen36 yet — each step recomputes all positions; the timing printed here is
 * the stateless proxy, NOT a production tok/s).
 *
 * Modes:
 *   qwen36_gen_coherence gguf <model.gguf> [n_gen]
 *   qwen36_gen_coherence spm  <model.sp-model> <tok.sp-tokenizer> [n_gen]
 *
 * Prompt = the fixed 6-token M_QWEN36 oracle prompt; prints one token id per line
 * ("TOK <i> <id> <sec>s") + the full sequence at the end ("SEQ: ..."). Compare the
 * gguf and spm sequences offline for the agreement count. */
#include "sp/model.h"
#include "sp/sp_model.h"
#include "sp/sp_status.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* NORTHSTAR GPU-1 hybrid (compile-gated: /DQ36_GPU + link the CUDA backend lib +
 * cudart/cublas): upload the DENSE weight tensors (gdn projections, full-attn
 * projections, LM head) once to VRAM and register the dp4a GEMV service as the
 * sp_matmul external hook. GDN recurrence, conv, MoE routing/experts stay on CPU.
 * QWEN36_GPU=1 enables at runtime. */
#ifdef Q36_GPU
#include "sp/forward_dispatch.h"   /* sp_matmul_register_ext */
#include "sp/arena.h"              /* sp_arena_find */
extern void *sp_q36gpu_new(void);
extern int   sp_q36gpu_upload(void *h, const char *name, const sp_frob_packed_tensor *pt);
extern int   sp_q36gpu_matmul(void *h, const char *name, const float *X,
                              int n_tok, int in, int out, float *Y);
extern int   sp_q36gpu_upload_experts(void *h, const char *tag,
                                      const sp_frob_packed_tensor *ptg,
                                      const sp_frob_packed_tensor *ptu,
                                      const sp_frob_packed_tensor *ptd,
                                      int NE, int FF, int E);
extern int   sp_q36gpu_moe(void *h, const char *tag, const int *idx, const float *wt,
                           int NU, const float *x, int E, int FF, float *y);
static int q36_gpu_ext(void *ctx, const char *n, const float *X, int nt, int in, int out, float *Y) {
    return sp_q36gpu_matmul(ctx, n, X, nt, in, out, Y);
}
extern int   sp_q36gpu_moe_stream(void *h,
                                  const sp_frob_packed_tensor *ptg,
                                  const sp_frob_packed_tensor *ptu,
                                  const sp_frob_packed_tensor *ptd,
                                  const int *idx, const float *wt, int NU,
                                  const float *x, int E, int FF, float *y);
/* GPU-3: host-side table of every layer's expert tensors so non-resident layers can
 * STREAM their selected experts instead of falling back to CPU (QWEN36_GPU_STREAM=1). */
static struct { const char *gname; const sp_frob_packed_tensor *g, *u, *d; } q36_tbl[64];
static int q36_tbl_n = 0, q36_stream_on = 0;
/* GPU-4: routing-locality telemetry — decides streaming-vs-LRU with data. Counts how
 * many of a layer's selected experts repeat from the SAME layer's previous token. */
static int  q36_loc_prev[64][16], q36_loc_seen[64];
static long q36_loc_hits = 0, q36_loc_tot = 0;
static void q36_loc_track(int li, const int *idx, int NU) {
    if (li < 0 || li >= 64 || NU > 16) return;
    if (q36_loc_seen[li]) {
        for (int a = 0; a < NU; a++)
            for (int b = 0; b < NU; b++)
                if (idx[a] == q36_loc_prev[li][b]) { q36_loc_hits++; break; }
        q36_loc_tot += NU;
    }
    for (int a = 0; a < NU; a++) q36_loc_prev[li][a] = idx[a];
    q36_loc_seen[li] = 1;
}
static int q36_moe_ext(void *ctx, const char *gate_name, const int *idx, const float *wt,
                       int NU, const float *x, int E, int FF, float *y) {
    int li = -1;
    for (int i = 0; i < q36_tbl_n; i++)
        if (strcmp(q36_tbl[i].gname, gate_name) == 0) { li = i; break; }
    q36_loc_track(li, idx, NU);
    if (sp_q36gpu_moe(ctx, gate_name, idx, wt, NU, x, E, FF, y) == 0) return 0;
    if (!q36_stream_on || li < 0) return -1;
    return sp_q36gpu_moe_stream(ctx, q36_tbl[li].g, q36_tbl[li].u, q36_tbl[li].d,
                                idx, wt, NU, x, E, FF, y);
}
static int q36_gpu_up(void *h, const qwen3_model *m, const gguf_tensor *W, size_t *bytes) {
    if (!W || !m->arena) return 0;
    const sp_arena_tensor *at = sp_arena_find(m->arena, W->name);
    if (!at) return 0;
    if (sp_q36gpu_upload(h, W->name, &at->pt)) { printf("gpu upload FAIL: %s\n", W->name); return 1; }
    *bytes += at->pt.codes_bytes;
    return 0;
}
#endif

static double now_s(void) {
    struct timespec t; timespec_get(&t, TIME_UTC);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 3) { printf("usage: %s gguf <model.gguf> [n] | spm <model.sp-model> <tok> [n]\n", argv[0]); return 1; }
    const char *mode = argv[1];
    qwen3_model *m = NULL;
    sp_model *spm = NULL;
    int n_gen = 32;

    if (strcmp(mode, "gguf") == 0) {
        if (argc > 3) n_gen = atoi(argv[3]);
        m = qwen3_load(argv[2]);
        if (!m) { printf("qwen3_load FAIL\n"); return 1; }
    } else {
        if (argc < 4) { printf("spm mode needs model + tokenizer\n"); return 1; }
        if (argc > 4) n_gen = atoi(argv[4]);
        if (sp_model_load(argv[2], argv[3], &spm) != SP_OK) { printf("sp_model_load FAIL\n"); return 1; }
        m = sp_model_to_qwen36(spm);
        if (!m) { printf("sp_model_to_qwen36 FAIL: %s\n", sp_last_error()); return 1; }
    }
    if (m->cfg.arch != SP_ARCH_QWEN36) { printf("arch != QWEN36\n"); return 1; }
#ifdef Q36_GPU
    if (getenv("QWEN36_GPU") && *getenv("QWEN36_GPU") == '1') {
        void *gh = sp_q36gpu_new();
        if (!gh) { printf("sp_q36gpu_new FAIL\n"); return 1; }
        size_t bytes = 0;
        for (uint32_t il = 0; il < m->cfg.n_layers; il++) {
            const qwen3_layer *L = &m->layers[il];
            if (L->q36_is_recurrent) {
                if (q36_gpu_up(gh, m, L->gdn_qkv,  &bytes) ||
                    q36_gpu_up(gh, m, L->gdn_gate, &bytes) ||
                    q36_gpu_up(gh, m, L->gdn_out,  &bytes)) return 1;
            } else {
                if (q36_gpu_up(gh, m, L->attn_q,      &bytes) ||
                    q36_gpu_up(gh, m, L->attn_k,      &bytes) ||
                    q36_gpu_up(gh, m, L->attn_v,      &bytes) ||
                    q36_gpu_up(gh, m, L->attn_output, &bytes)) return 1;
            }
        }
        if (q36_gpu_up(gh, m, m->output, &bytes)) return 1;
        sp_matmul_register_ext(q36_gpu_ext, gh);
        printf("GPU-1: dense weights resident on device: %.1f MB packed\n", bytes / 1048576.0);
        /* GPU-2: resident EXPERTS for as many layers as the budget allows (default
         * 8.5 GB, QWEN36_GPU_MOE_GB overrides). Router stays on CPU; a non-resident
         * layer's hook returns not-mine -> CPU experts (partial residency). */
        {
            double gb = getenv("QWEN36_GPU_MOE_GB") ? atof(getenv("QWEN36_GPU_MOE_GB")) : 8.5;
            size_t budget = (size_t)(gb * 1073741824.0), mbytes = 0;
            int mlayers = 0;
            int NE = (int)m->cfg.q36_n_expert, FF = (int)m->cfg.q36_n_ff_exp,
                E2 = (int)m->cfg.n_embd;
            int budget_hit = 0;
            for (uint32_t il = 0; il < m->cfg.n_layers; il++) {
                const qwen3_layer *L = &m->layers[il];
                const sp_arena_tensor *ag = sp_arena_find(m->arena, L->ffn_gate_exps->name);
                const sp_arena_tensor *au = sp_arena_find(m->arena, L->ffn_up_exps->name);
                const sp_arena_tensor *ad = sp_arena_find(m->arena, L->ffn_down_exps->name);
                if (!ag || !au || !ad) continue;
                /* GPU-3 stream table covers EVERY layer (fallback for non-resident) */
                if (q36_tbl_n < 64) {
                    q36_tbl[q36_tbl_n].gname = L->ffn_gate_exps->name;
                    q36_tbl[q36_tbl_n].g = &ag->pt;
                    q36_tbl[q36_tbl_n].u = &au->pt;
                    q36_tbl[q36_tbl_n].d = &ad->pt;
                    q36_tbl_n++;
                }
                size_t lb = ag->pt.codes_bytes + au->pt.codes_bytes + ad->pt.codes_bytes;
                if (budget_hit || mbytes + lb > budget) { budget_hit = 1; continue; }
                if (sp_q36gpu_upload_experts(gh, L->ffn_gate_exps->name,
                                             &ag->pt, &au->pt, &ad->pt, NE, FF, E2)) {
                    printf("GPU-2: expert upload FAIL at layer %u — stopping residency there\n", il);
                    budget_hit = 1; continue;
                }
                mbytes += lb; mlayers++;
            }
            q36_stream_on = getenv("QWEN36_GPU_STREAM") && *getenv("QWEN36_GPU_STREAM") == '1';
            if (mlayers > 0 || q36_stream_on) {
                sp_moe_register_ext(q36_moe_ext, gh);
                printf("GPU-2: experts resident for %d/%u layers: %.1f MB packed%s\n",
                       mlayers, m->cfg.n_layers, mbytes / 1048576.0,
                       q36_stream_on ? " (+GPU-3 streaming for the rump)" : "");
            }
        }
    }
#endif
    const int V = (int)m->cfg.n_vocab;
    printf("mode=%s V=%d n_gen=%d (stateless forward — O(n^2) proxy, not production tok/s)\n",
           mode, V, n_gen);

    /* Prompt: QWEN36_PROMPT_IDS="id,id,..." overrides the fixed 6-token gate prompt
     * (which is GIBBERISH in the 248320 vocab — those ids were minted for the 151k
     * Qwen vocab; fine for bit-parity, wrong for a coherence eyeball). */
    int base = 6;
    int32_t pbuf[64];
    const char *pe = getenv("QWEN36_PROMPT_IDS");
    if (pe && *pe) {
        base = 0;
        const char *s = pe;
        while (*s && base < 64) {
            pbuf[base++] = (int32_t)strtol(s, (char **)&s, 10);
            while (*s == ',' || *s == ' ') s++;
        }
    } else {
        const int32_t fixed[6] = { 785, 3974, 13876, 38835, 35308, 916 };
        memcpy(pbuf, fixed, sizeof(fixed));
    }
    int cap = base + n_gen;
    int32_t *seq = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    memcpy(seq, pbuf, (size_t)base * sizeof(int32_t));
    float *lg = (float *)malloc((size_t)cap * (size_t)V * sizeof(float));
    if (!seq || !lg) { printf("oom\n"); return 1; }

    /* STEP mode (QWEN36_STEP=1): persistent-state decode via qwen36_step — the
     * NORTHSTAR brick-3 path. Prefill by stepping the prompt, then greedy
     * generate; the G-MOE-STATE-PARITY gate = this sequence must equal the
     * stateless self-feed sequence. */
    const int step_mode = getenv("QWEN36_STEP") && *getenv("QWEN36_STEP") == '1';
    double t_all = now_s();
    if (step_mode) {
        qwen36_state *st = qwen36_state_new(m, cap + 1);
        if (!st) { printf("state_new FAIL\n"); return 1; }
        double t_pf = now_s();
        for (int t = 0; t < base; t++)
            if (qwen36_step(m, st, seq[t], lg)) { printf("step FAIL (prefill %d)\n", t); return 1; }
        printf("PREFILL %d tokens in %.2fs\n", base, now_s() - t_pf);
        t_all = now_s();
        for (int k = 0; k < n_gen; k++) {
            int am = 0; float best = lg[0];
            for (int i = 1; i < V; i++) if (lg[i] > best) { best = lg[i]; am = i; }
            seq[base + k] = am;
            double t0 = now_s();
            if (k + 1 < n_gen || 1) {
                if (qwen36_step(m, st, am, lg)) { printf("step FAIL at %d\n", k); return 1; }
            }
            printf("TOK %d %d %.2fs\n", k, am, now_s() - t0);
            fflush(stdout);
        }
        qwen36_state_free(st);
    } else {
    for (int k = 0; k < n_gen; k++) {
        int len = base + k;
        double t0 = now_s();
        if (qwen36_forward(m, seq, len, lg)) { printf("forward FAIL at %d\n", k); return 1; }
        const float *last = lg + (size_t)(len - 1) * V;
        int am = 0; float best = last[0];
        for (int i = 1; i < V; i++) if (last[i] > best) { best = last[i]; am = i; }
        seq[len] = am;
        printf("TOK %d %d %.2fs\n", k, am, now_s() - t0);
        fflush(stdout);
    }
    }
    double dt = now_s() - t_all;
#ifdef Q36_GPU
    if (q36_loc_tot > 0)
        printf("LOCALITY: %ld/%ld = %.1f%% of selected experts repeat from the same layer's previous token\n",
               q36_loc_hits, q36_loc_tot, 100.0 * q36_loc_hits / q36_loc_tot);
#endif
    printf("SEQ:");
    for (int i = 0; i < base + n_gen; i++) printf(" %d", seq[i]);
    printf("\nTOTAL %d tokens in %.1fs = %.3f tok/s (stateless proxy)\n", n_gen, dt, n_gen / dt);
    return 0;
}
