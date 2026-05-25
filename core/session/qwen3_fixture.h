/* qwen3_fixture.h — test-only synthetic, qwen3-shaped .sp-model builder.
 *
 * Builds a complete, spec-conformant (PPT-LAT-SP-MODEL-v0) .sp-model + paired
 * .sp-tokenizer for a *tiny* qwen3 architecture: the full weight set the reference
 * forward dereferences — token_embd (tied LM head), per-layer attn_norm / q / k / v
 * / output (+ q_norm / k_norm) / ffn_norm / gate / up / down, and output_norm. The
 * matmul weights are stored OK_Q8 (int8 codes + a paired <name>.scale
 * FROBENIUS_SCALE_FP32 per-row scale); the norms are F32. This is exactly what
 * sp_model_to_qwen3 reconstructs into a runnable qwen3_model, so the session can be
 * gated bit-exactly against qwen3_forward without a multi-GB real model.
 *
 * Weight VALUES are arbitrary-but-deterministic (a fixed pattern); the parity gate
 * compares the session path against the reference forward on the *same*
 * reconstruction, so only determinism + finiteness matter, not realism.
 *
 * Compiled into the session test executable via TEST_SOURCES; never into a library.
 */
#ifndef SP_QWEN3_FIXTURE_H
#define SP_QWEN3_FIXTURE_H

#include <stdint.h>
#include <stddef.h>
#include "sp/sp_l1.h"   /* sp_arch_info embedded in the header */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t n_layers, n_embd, n_ff, n_head, n_head_kv, head_dim, n_vocab;
    float    rope_freq_base;
    int      tied;                 /* nonzero: output == token_embd */
    size_t   model_len, tok_len;
    sp_arch_info arch;             /* exact payload embedded (== what sp_model_arch returns) */
} sp_qwen3_fixture_info;

/* Overrides for the arch_struct payload + header sizing (2-L1.FP16 growth/precision
 * tests). All zero = the default build (full sizeof, new fields left 0). */
typedef struct {
    uint32_t arch_struct_size;     /* header arch_struct_size to write; 0 -> sizeof(sp_arch_info) */
    uint32_t preferred_precision;  /* sp_arch_info.preferred_precision payload value              */
    uint32_t n_ff_field;           /* sp_arch_info.n_ff payload value (0 -> leave unspecified)    */
    float    rms_eps_field;        /* sp_arch_info.rms_eps payload value (0 -> leave unspecified) */
} sp_qwen3_fixture_opts;

/* Build a tiny qwen3-shaped .sp-model + .sp-tokenizer into fresh malloc'd buffers
 * (*model_buf / *tok_buf — caller free()s both). Returns 0 on success. */
int sp_qwen3_fixture_build(uint8_t **model_buf, uint8_t **tok_buf,
                           sp_qwen3_fixture_info *info);

/* As sp_qwen3_fixture_build, with arch_struct overrides (opts == NULL -> defaults). */
int sp_qwen3_fixture_build_ex(uint8_t **model_buf, uint8_t **tok_buf,
                              sp_qwen3_fixture_info *info, const sp_qwen3_fixture_opts *opts);

/* Write len bytes to path (binary). Returns 0 on success. */
int sp_qwen3_fixture_write(const char *path, const uint8_t *buf, size_t len);

#ifdef __cplusplus
}
#endif
#endif /* SP_QWEN3_FIXTURE_H */
