/* qwen25_fixture.h — test-only synthetic Qwen2.5-shaped .sp-model builder.
 *
 * Builds a complete spec-conformant .sp-model + .sp-tokenizer for a tiny Qwen2.5
 * architecture. Weight set mirrors sp_model_to_qwen25: token_embd (tied LM head),
 * per-layer attn_norm / attn_q / attn_k / attn_v / attn_output / attn_q.bias /
 * attn_k.bias / attn_v.bias / ffn_norm / ffn_gate / ffn_up / ffn_down, output_norm.
 * Matmul weights are OK_Q8 + paired .scale; norms and biases are F32.
 *
 * arch_struct: SP_ARCH_ID_QWEN25, ffn_variant=0 (SwiGLU), norm_variant=0,
 * tied_embeddings=1, has_qk_norm=0. */
#ifndef SP_QWEN25_FIXTURE_H
#define SP_QWEN25_FIXTURE_H

#include <stdint.h>
#include <stddef.h>
#include "sp/sp_l1.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t n_layers, n_embd, n_ff, n_head, n_head_kv, head_dim, n_vocab;
    float    rope_freq_base;
    int      tied;
    size_t   model_len, tok_len;
    sp_arch_info arch;
} sp_qwen25_fixture_info;

int sp_qwen25_fixture_build(uint8_t **model_buf, uint8_t **tok_buf,
                            sp_qwen25_fixture_info *info);

int sp_qwen25_fixture_write(const char *path, const uint8_t *buf, size_t len);

#ifdef __cplusplus
}
#endif
#endif /* SP_QWEN25_FIXTURE_H */
