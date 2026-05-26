/* gemma3_fixture.h — test-only synthetic Gemma3-shaped .sp-model builder.
 *
 * Builds a complete spec-conformant .sp-model + .sp-tokenizer for a tiny Gemma3
 * architecture. The weight set mirrors what sp_model_to_gemma3 reconstructs:
 * token_embd (tied LM head), per-layer attn_norm / attn_q / attn_k / attn_v /
 * attn_output / attn_q_norm / attn_k_norm / post_attention_norm / ffn_norm /
 * ffn_gate / ffn_up / ffn_down / post_ffw_norm, and output_norm. Matmul weights
 * are OK_Q8 + paired .scale; norms are F32.
 *
 * arch_struct: SP_ARCH_ID_GEMMA3, ffn_variant=1 (GeGLU), norm_variant=1 (sandwich),
 * swa_window=512, tied_embeddings=1, has_qk_norm=1.
 *
 * Compiled into the session test executable via TEST_SOURCES; never into a library.
 */
#ifndef SP_GEMMA3_FIXTURE_H
#define SP_GEMMA3_FIXTURE_H

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
} sp_gemma3_fixture_info;

/* Build a tiny Gemma3-shaped .sp-model + .sp-tokenizer into fresh malloc'd buffers.
 * Returns 0 on success. Caller free()s *model_buf and *tok_buf. */
int sp_gemma3_fixture_build(uint8_t **model_buf, uint8_t **tok_buf,
                            sp_gemma3_fixture_info *info);

/* Write len bytes to path (binary). Returns 0 on success. */
int sp_gemma3_fixture_write(const char *path, const uint8_t *buf, size_t len);

#ifdef __cplusplus
}
#endif
#endif /* SP_GEMMA3_FIXTURE_H */
