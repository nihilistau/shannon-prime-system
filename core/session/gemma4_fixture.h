/* gemma4_fixture.h — test-only synthetic Gemma4-shaped .sp-model builder.
 *
 * Builds a complete spec-conformant tiny Gemma4 .sp-model + .sp-tokenizer whose
 * weight set mirrors what sp_model_to_gemma4 reconstructs: token_embd (tied LM
 * head), the AltUp globals (per_layer_token_embd / per_layer_model_proj /
 * per_layer_proj_norm / rope_freqs), and per layer attn_norm / attn_q / attn_k /
 * attn_v / attn_output / attn_q_norm / attn_k_norm / post_attention_norm /
 * ffn_norm / ffn_gate / ffn_up / ffn_down / post_ffw_norm + the per-layer-input
 * block (inp_gate / proj / post_norm / layer_output_scale), plus output_norm.
 *
 * The fixture deliberately uses period=3 (global layer when L%3==2) and
 * n_kv_from_start=3 so it exercises BOTH the per-layer head-geometry split
 * (SWA 8/4/2 vs global 16/2/1; QD=32, KVD=16 constant) AND the shared-KV reuse
 * (layers >= 3 reuse an owner's K/V: shared SWA -> owner 1, shared global ->
 * owner 2). Matmul weights are OK_Q8 + paired .scale; norms are F32.
 *
 * arch_struct: SP_ARCH_ID_GEMMA4 with the g4_* tail (SWA geometry, AltUp width,
 * shared-KV count, logit softcap, swa period). Compiled into the session test
 * executable via TEST_SOURCES; never into a library.
 */
#ifndef SP_GEMMA4_FIXTURE_H
#define SP_GEMMA4_FIXTURE_H

#include <stdint.h>
#include <stddef.h>
#include "sp/sp_l1.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t n_layers, n_embd, n_ff, n_vocab, n_embd_per_layer;
    uint32_t hd_swa, nh_swa, nkv_swa, hd_global, nh_global, nkv_global;
    uint32_t swa_period, n_kv_from_start, sliding_window;
    float    rope_base_global, rope_base_swa, logit_softcap;
    size_t   model_len, tok_len;
    sp_arch_info arch;
} sp_gemma4_fixture_info;

/* Build a tiny Gemma4-shaped .sp-model + .sp-tokenizer into fresh malloc'd
 * buffers. Returns 0 on success. Caller free()s *model_buf and *tok_buf. */
int sp_gemma4_fixture_build(uint8_t **model_buf, uint8_t **tok_buf,
                            sp_gemma4_fixture_info *info);

/* Write len bytes to path (binary). Returns 0 on success. */
int sp_gemma4_fixture_write(const char *path, const uint8_t *buf, size_t len);

#ifdef __cplusplus
}
#endif
#endif /* SP_GEMMA4_FIXTURE_H */
