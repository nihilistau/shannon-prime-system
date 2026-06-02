/* qwen36_gate.c — M_QWEN36: Qwen3.6-35B-A3B (qwen35moe) forward correctness gate.
 *
 * Loads the Q4_K_M GGUF (qwen3_load), greedy-decodes from the fixed oracle prompt
 * through qwen36_forward, and asserts the top-1 sequence is BIT-EXACT to the
 * llama.cpp oracle (`5444 8 198`). This is the discrete-decision gate (precision-
 * robust); the f32 reference is the dev scaffold, not a bit-exactness claim.
 *
 * SLOW / model-gated: the 19.7 GB model is out-of-tree; skips cleanly (exit 0)
 * when absent. Override the model path via SP_QWEN36_GGUF. */
#include "sp/model.h"
#include <stdio.h>
#include <stdlib.h>

#ifndef SP_QWEN36_GGUF
#define SP_QWEN36_GGUF "D:/Files/Models/lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf"
#endif

int main(void) {
    const char *path = getenv("SP_QWEN36_GGUF"); if (!path) path = SP_QWEN36_GGUF;
    FILE *probe = fopen(path, "rb");
    if (!probe) { fprintf(stderr, "[M_QWEN36] model absent (%s) — SKIP\n", path); return 0; }
    fclose(probe);

    qwen3_model *m = qwen3_load(path);
    if (!m) { fprintf(stderr, "[M_QWEN36] qwen3_load FAIL\n"); return 1; }
    if (m->cfg.arch != SP_ARCH_QWEN36) { fprintf(stderr, "[M_QWEN36] arch != QWEN36\n"); qwen3_free(m); return 1; }

    const int32_t oracle[3] = { 5444, 8, 198 };          /* llama.cpp greedy, non-trivial predictions */
    int32_t seq[6 + 8] = { 785, 3974, 13876, 38835, 35308, 916 };
    int base = 6, V = (int)m->cfg.n_vocab, np = 3, fails = 0;
    float *lg = (float *)malloc((size_t)(base + np) * (size_t)V * sizeof(float));
    if (!lg) { fprintf(stderr, "[M_QWEN36] oom\n"); qwen3_free(m); return 1; }
    for (int k = 0; k < np; k++) {
        int len = base + k;
        if (qwen36_forward(m, seq, len, lg)) { fprintf(stderr, "[M_QWEN36] forward FAIL at %d\n", k); free(lg); qwen3_free(m); return 1; }
        const float *last = lg + (size_t)(len - 1) * V;
        int am = 0; float best = last[0];
        for (int i = 1; i < V; i++) if (last[i] > best) { best = last[i]; am = i; }
        fprintf(stderr, "[M_QWEN36] tok %d: SP=%d oracle=%d %s\n", k, am, oracle[k], am == oracle[k] ? "OK" : "MISMATCH");
        if (am != oracle[k]) fails++;
        seq[len] = am;
    }
    free(lg); qwen3_free(m);
    if (fails) { fprintf(stderr, "[M_QWEN36] FAIL (%d/%d mismatch)\n", fails, np); return 1; }
    fprintf(stderr, "[M_QWEN36] PASS — qwen35moe forward top-1 bit-exact to oracle (%d/%d)\n", np, np);
    return 0;
}
