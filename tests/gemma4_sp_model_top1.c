/* gemma4_sp_model_top1.c — Phase 3-G4 Task C: end-to-end production-path validation.
 * Loads a transcoded .sp-model + .sp-tokenizer (sp_model_load), reconstructs via
 * sp_model_to_gemma4, and greedy-decodes from FIXED token IDs through gemma4_forward,
 * printing argmax IDs as "SPM <i>: <id>". Compared against g4_oracle (same fixed IDs):
 * a match proves the full transcode -> load -> bridge -> forward path is bit-faithful.
 *
 * Build: link the math-core static libs (io_format + session + forward + deps).
 * Run:   gemma4_sp_model_top1.exe <model.sp-model> <tok.sp-tokenizer> [n_predict] */
#include "sp/sp_model.h"   /* sp_model_load / sp_model_unload / sp_model_to_gemma4 */
#include "sp/model.h"      /* qwen3_model, gemma4_forward, qwen3_free */
#include "sp/sp_status.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 3) { printf("usage: gemma4_sp_model_top1 model.sp-model tok.sp-tokenizer [n_predict]\n"); return 1; }
    int N = argc > 3 ? atoi(argv[3]) : 16;
    sp_model *m = NULL;
    if (sp_model_load(argv[1], argv[2], &m) != SP_OK) { printf("sp_model_load FAIL\n"); return 1; }
    qwen3_model *qm = sp_model_to_gemma4(m);
    if (!qm) { printf("sp_model_to_gemma4 FAIL\n"); return 1; }
    const int V = (int)qm->cfg.n_vocab;
    printf("loaded: NL=%u n_ff(0)=%u V=%d kvfs=%u period=%u\n",
           qm->cfg.n_layers, qm->cfg.n_ff, V, qm->cfg.g4_n_kv_from_start, qm->cfg.g4_swa_period);
    fflush(stdout);

    int32_t seq[6 + 64] = { 2, 818, 5279, 529, 8398, 563 };   /* identical to g4_oracle */
    int np = 6;
    float *lg = (float *)malloc((size_t)(np + N) * (size_t)V * sizeof(float));
    if (!lg) { printf("oom\n"); return 1; }
    for (int k = 0; k < N; k++) {
        int len = np + k;
        if (gemma4_forward(qm, seq, len, lg)) { printf("forward fail at %d\n", k); return 1; }
        const float *last = lg + (size_t)(len - 1) * V;
        int am = 0; float best = last[0];
        for (int i = 1; i < V; i++) if (last[i] > best) { best = last[i]; am = i; }
        printf("SPM %d: %d\n", k, am); fflush(stdout);
        seq[len] = am;
    }
    free(lg);
    qwen3_free(qm);
    sp_model_unload(m);
    return 0;
}
