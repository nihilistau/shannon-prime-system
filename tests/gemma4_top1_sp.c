/* gemma4_top1_sp.c — Phase 3-G4 Stage 2 oracle top-1 validation (SP side).
 * Greedy-decodes N tokens from a FIXED token-ID prompt through gemma4_forward
 * (O(n^2) re-prefill) and prints the argmax token IDs, one per line as
 * "SP <i>: <id>". The same fixed IDs are fed to g4_oracle (libllama greedy); a
 * matching argmax sequence over N steps proves gemma4_forward is bit-faithful
 * top-1 to real Gemma4. No tokenizer needed — identical IDs to both sides.
 *
 * Build: like gemma4_gguf_forward_harness.c (link the math-core static libs).
 * Run:   gemma4_top1_sp.exe <gemma-4-E2B-it-...Q8_0.gguf> [n_predict] */
#include "sp/model.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: gemma4_top1_sp model.gguf [n_predict]\n"); return 1; }
    int N = argc > 2 ? atoi(argv[2]) : 16;
    qwen3_model *m = qwen3_load(argv[1]);
    if (!m) { printf("load NULL\n"); return 1; }
    const int V = (int)m->cfg.n_vocab;

    int32_t seq[6 + 64] = { 2, 818, 5279, 529, 8398, 563 };  /* identical to g4_oracle */
    int np = 6;
    printf("prompt:"); for (int i = 0; i < np; i++) printf(" %d", seq[i]); printf("\n"); fflush(stdout);

    float *lg = (float *)malloc((size_t)(np + N) * (size_t)V * sizeof(float));
    if (!lg) { printf("oom\n"); return 1; }
    for (int k = 0; k < N; k++) {
        int len = np + k;
        if (gemma4_forward(m, seq, len, lg)) { printf("forward fail at %d\n", k); return 1; }
        const float *last = lg + (size_t)(len - 1) * V;
        int am = 0; float best = last[0];
        for (int i = 1; i < V; i++) if (last[i] > best) { best = last[i]; am = i; }
        printf("SP %d: %d\n", k, am); fflush(stdout);
        seq[len] = am;
    }
    free(lg);
    qwen3_free(m);
    return 0;
}
