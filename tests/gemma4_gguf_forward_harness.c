/* gemma4_gguf_forward_harness.c — Phase 3-G4 Stage 2 TASK B oracle/forward
 * validation harness. Loads a real Gemma4 GGUF (E2B-Q8_0) via qwen3_load (the
 * gemma4 branch added in core/model/model.c), prints the derived config, runs
 * gemma4_forward to completion, and prints the last-position argmax + a 1-step
 * greedy continuation. Confirms the loader + per-layer-geometry forward run
 * crash-free + finite + softcap-bounded on the real weights.
 *
 * Build (MinGW, from repo root, after a normal cmake build):
 *   gcc -I include gemma4_gguf_forward_harness.c \
 *       build/core/forward/libsp_forward.a build/core/model/libsp_model.a \
 *       build/core/forward_dispatch/libsp_forward_dispatch.a \
 *       build/core/forward_kernels/libsp_forward_kernels.a \
 *       build/core/arena/libsp_arena.a build/core/frobenius/libsp_frobenius.a \
 *       build/core/ntt_crt/libsp_ntt_crt.a build/core/poly_ring/libsp_poly_ring.a \
 *       build/core/weight_dtype/libsp_weight_dtype.a build/core/gguf/libsp_gguf.a \
 *       build/core/ok_arith/libsp_ok_arith.a -lm -o g4_fwd.exe
 * Run: g4_fwd.exe "<path-to>/gemma-4-E2B-it-uncensored-Q8_0.gguf"
 *
 * Confirmed result (E2B-Q8_0, 2026-06-02): NL=35 nh=8 nkv=1 hd=512/256 PL=256
 * kvfs=15 period=5 softcap=30; forward rc=0; last argmax=16058 (val 25.91). */
#include "sp/model.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage\n"); return 1; }
    qwen3_model *m = qwen3_load(argv[1]);
    if (!m) { printf("load NULL\n"); return 1; }
    const qwen3_config *c = &m->cfg;
    printf("loaded: NL=%u E=%u FF=%u V=%u nh=%u nkv=%u hd=%u | swa nh=%u nkv=%u hd=%u | PL=%u kvfs=%u period=%u softcap=%.1f\n",
           c->n_layers, c->n_embd, c->n_ff, c->n_vocab, c->n_head, c->n_head_kv, c->head_dim,
           c->g4_nh_swa, c->g4_nkv_swa, c->g4_hd_swa, c->g4_n_embd_per_layer,
           c->g4_n_kv_from_start, c->g4_swa_period, (double)c->g4_logit_softcap);
    fflush(stdout);

    int GP = 6;
    int32_t toks[6] = { 2, 818, 5279, 529, 8398, 563 };
    float *lg = (float *)malloc((size_t)GP * c->n_vocab * sizeof(float));
    if (!lg) { printf("oom\n"); return 1; }
    printf("calling gemma4_forward...\n"); fflush(stdout);
    int rc = gemma4_forward(m, toks, GP, lg);
    printf("gemma4_forward rc=%d\n", rc); fflush(stdout);
    if (rc == 0) {
        const float *last = lg + (size_t)(GP - 1) * c->n_vocab;
        int am = 0; float best = last[0];
        for (uint32_t i = 1; i < c->n_vocab; i++) if (last[i] > best) { best = last[i]; am = (int)i; }
        printf("last argmax=%d val=%.4f  (lg[0]=%.4f lg[100]=%.4f)\n", am, (double)best,
               (double)last[0], (double)last[100]);
    }
    /* replicate the test's 1-step self-consistency chain */
    if (rc == 0) {
        const float *last = lg + (size_t)(GP - 1) * c->n_vocab;
        int am = 0; { float best = last[0]; for (uint32_t i=1;i<c->n_vocab;i++) if(last[i]>best){best=last[i];am=(int)i;} }
        int32_t seq[8]; for (int i=0;i<GP;i++) seq[i]=toks[i]; seq[GP]=am;
        float *step = (float*)malloc((size_t)(GP+1)*c->n_vocab*sizeof(float));
        printf("chain step forward...\n"); fflush(stdout);
        int rc2 = gemma4_forward(m, seq, GP+1, step);
        printf("chain rc=%d\n", rc2); fflush(stdout);
        if (rc2==0){ const float *sl=step+(size_t)GP*c->n_vocab; int a2=0; float b=sl[0]; for(uint32_t i=1;i<c->n_vocab;i++) if(sl[i]>b){b=sl[i];a2=(int)i;} printf("chain argmax=%d\n",a2);}
        free(step);
    }
    free(lg);
    qwen3_free(m);
    printf("DONE\n");
    return 0;
}
