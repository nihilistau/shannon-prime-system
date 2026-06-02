/* qwen36_load_probe.c — Phase 3-MoE Stage 1 validation: qwen3_load binds the
 * qwen35moe (Qwen3.6-35B-A3B) GGUF — arch detected, config read, per-layer GDN/
 * full-attn bifurcation + MoE tensors all bound. No forward yet. */
#include "sp/model.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: qwen36_load_probe model.gguf\n"); return 1; }
    qwen3_model *m = qwen3_load(argv[1]);
    if (!m) { printf("qwen3_load FAIL\n"); return 1; }
    const qwen3_config *c = &m->cfg;
    printf("arch=%d (QWEN36=%d) NL=%u n_embd=%u head=%u/%u hd=%u vocab=%u tied=%d\n",
           c->arch, SP_ARCH_QWEN36, c->n_layers, c->n_embd, c->n_head, c->n_head_kv,
           c->head_dim, c->n_vocab, c->tied_embedding);
    printf("MoE: n_expert=%u used=%u n_ff_exp=%u n_ff_shexp=%u wscale=%.4f\n",
           c->q36_n_expert, c->q36_n_expert_used, c->q36_n_ff_exp, c->q36_n_ff_shexp,
           c->q36_expert_weights_scale);
    printf("GDN: conv_k=%u state=%u H_k=%u H_v=%u inner=%u | full_attn_interval=%u\n",
           c->q36_gdn_conv_k, c->q36_gdn_state, c->q36_gdn_n_k_heads, c->q36_gdn_n_v_heads,
           c->q36_gdn_inner, c->q36_full_attn_interval);
    printf("IMRoPE: dim=%u base=%.0f sections=[%d,%d,%d,%d] nextn=%u\n",
           c->q36_rope_dim, c->q36_rope_base, c->q36_rope_sections[0], c->q36_rope_sections[1],
           c->q36_rope_sections[2], c->q36_rope_sections[3], c->q36_nextn_predict_layers);

    int n_gdn = 0, n_attn = 0, moe_all = 1, blk_all = 1;
    for (uint32_t i = 0; i < c->n_layers; i++) {
        const qwen3_layer *L = &m->layers[i];
        if (L->q36_is_recurrent) {
            n_gdn++;
            if (!(L->gdn_qkv && L->gdn_gate && L->gdn_conv1d && L->gdn_dt_bias && L->gdn_a &&
                  L->gdn_alpha && L->gdn_beta && L->gdn_norm && L->gdn_out)) blk_all = 0;
        } else {
            n_attn++;
            if (!(L->attn_q && L->attn_k && L->attn_v && L->attn_output &&
                  L->attn_q_norm && L->attn_k_norm)) blk_all = 0;
        }
        if (!(L->ffn_gate_inp && L->ffn_gate_exps && L->ffn_up_exps && L->ffn_down_exps &&
              L->ffn_gate_inp_shexp && L->ffn_gate_shexp && L->ffn_up_shexp && L->ffn_down_shexp))
            moe_all = 0;
    }
    printf("layers: GDN=%d full_attn=%d | all_block_tensors=%d all_moe_tensors=%d\n",
           n_gdn, n_attn, blk_all, moe_all);
    /* expected: GDN=30 full_attn=10, both all=1 */
    int ok = (c->arch == SP_ARCH_QWEN36) && blk_all && moe_all && n_attn == 10 && n_gdn == 30;
    printf("%s\n", ok ? "STAGE1_OK" : "STAGE1_FAIL");
    qwen3_free(m);
    return ok ? 0 : 1;
}
