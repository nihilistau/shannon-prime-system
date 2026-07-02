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

    double t_all = now_s();
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
    double dt = now_s() - t_all;
    printf("SEQ:");
    for (int i = 0; i < base + n_gen; i++) printf(" %d", seq[i]);
    printf("\nTOTAL %d tokens in %.1fs = %.3f tok/s (stateless proxy)\n", n_gen, dt, n_gen / dt);
    return 0;
}
