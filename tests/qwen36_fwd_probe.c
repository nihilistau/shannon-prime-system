/* qwen36_fwd_probe.c — Stage 2a: run qwen36_forward on the fixed oracle prompt and
 * dump per-layer GDN fingerprints (SP_Q36_DBG) for comparison against
 * papers/qwen35moe-oracle-fingerprints.txt. Prints top-1 of the last position. */
#include "sp/model.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: qwen36_fwd_probe model.gguf\n"); return 1; }
    qwen3_model *m = qwen3_load(argv[1]);
    if (!m) { printf("load FAIL\n"); return 1; }
    int np = argc > 2 ? atoi(argv[2]) : 8;
    int32_t seq[6 + 64] = { 785, 3974, 13876, 38835, 35308, 916 };
    int base = 6, V = (int)m->cfg.n_vocab;
    float *lg = (float *)malloc((size_t)(base + np) * (size_t)V * sizeof(float));
    if (!lg) { printf("oom\n"); return 1; }
    /* greedy decode np tokens; compare to oracle 5444 8 198 785 3974 13876 38835 35308 */
    for (int k = 0; k < np; k++) {
        int len = base + k;
        if (qwen36_forward(m, seq, len, lg)) { printf("forward FAIL at %d\n", k); return 1; }
        const float *last = lg + (size_t)(len - 1) * V;
        int am = 0; float best = last[0];
        for (int i = 1; i < V; i++) if (last[i] > best) { best = last[i]; am = i; }
        printf("SP %d: %d\n", k, am); fflush(stdout);
        seq[len] = am;
    }
    free(lg); qwen3_free(m);
    return 0;
}
