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
static int q36_gpu_ext(void *ctx, const char *n, const float *X, int nt, int in, int out, float *Y) {
    return sp_q36gpu_matmul(ctx, n, X, nt, in, out, Y);
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
    printf("SEQ:");
    for (int i = 0; i < base + n_gen; i++) printf(" %d", seq[i]);
    printf("\nTOTAL %d tokens in %.1fs = %.3f tok/s (stateless proxy)\n", n_gen, dt, n_gen / dt);
    return 0;
}
