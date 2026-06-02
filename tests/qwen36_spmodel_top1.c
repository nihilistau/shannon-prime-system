/* qwen36_spmodel_top1.c — C1 round-trip gate: load the OK_Q4 .sp-model via the
 * production path (sp_model_load -> sp_model_to_qwen36 -> qwen36_forward) and assert
 * the greedy top-1 sequence is bit-exact to the GGUF-direct/oracle (5444 8 198).
 * Run: qwen36_spmodel_top1 <model.sp-model> <tok.sp-tokenizer> [n_predict] */
#include "sp/sp_model.h"   /* sp_model_load / sp_model_unload / sp_model_to_qwen36 */
#include "sp/model.h"      /* qwen3_model, qwen36_forward, qwen3_free */
#include "sp/sp_status.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 3) { printf("usage: qwen36_spmodel_top1 model.sp-model tok.sp-tokenizer [n_predict]\n"); return 1; }
    int np = argc > 3 ? atoi(argv[3]) : 3;
    sp_model *m = NULL;
    if (sp_model_load(argv[1], argv[2], &m) != SP_OK) { printf("sp_model_load FAIL\n"); return 1; }
    { sp_arch_info ai; if (sp_model_arch(m, &ai) == SP_OK)
        printf("arch_id=%u interval=%u n_expert=%u/%u n_ff_exp=%u gdn_state=%u/%u/%u\n",
               ai.arch_id, ai.q36_full_attn_interval, ai.q36_n_expert, ai.q36_n_expert_used,
               ai.q36_n_ff_exp, ai.q36_gdn_state, ai.q36_gdn_n_k_heads, ai.q36_gdn_n_v_heads); }
    qwen3_model *qm = sp_model_to_qwen36(m);
    if (!qm) { printf("sp_model_to_qwen36 FAIL: %s\n", sp_last_error()); return 1; }
    const int V = (int)qm->cfg.n_vocab;
    printf("loaded: arch=%d NL=%u n_expert=%u/%u n_ff_exp=%u gdn_state=%u interval=%u V=%d\n",
           qm->cfg.arch, qm->cfg.n_layers, qm->cfg.q36_n_expert, qm->cfg.q36_n_expert_used,
           qm->cfg.q36_n_ff_exp, qm->cfg.q36_gdn_state, qm->cfg.q36_full_attn_interval, V);
    fflush(stdout);

    const int oracle[3] = { 5444, 8, 198 };
    int32_t seq[6 + 8] = { 785, 3974, 13876, 38835, 35308, 916 };
    int base = 6, fails = 0;
    float *lg = (float *)malloc((size_t)(base + np) * (size_t)V * sizeof(float));
    if (!lg) { printf("oom\n"); return 1; }
    for (int k = 0; k < np; k++) {
        int len = base + k;
        if (qwen36_forward(qm, seq, len, lg)) { printf("forward FAIL at %d\n", k); return 1; }
        const float *last = lg + (size_t)(len - 1) * V;
        int am = 0; float best = last[0];
        for (int i = 1; i < V; i++) if (last[i] > best) { best = last[i]; am = i; }
        printf("SPM %d: %d  oracle=%d  %s\n", k, am, oracle[k], am == oracle[k] ? "OK" : "MISMATCH");
        if (am != oracle[k]) fails++;
        seq[len] = am;
    }
    free(lg); qwen3_free(qm); sp_model_unload(m);
    printf("%s\n", fails ? "C1_ROUNDTRIP_FAIL" : "C1_ROUNDTRIP_OK (.sp-model top-1 bit-exact)");
    return fails ? 1 : 0;
}
