/* forward_dispatch.h — the model-coupled weight-access layer of the L1 forward: the
 * matmul that reads a model's weight tensor (inline-lifting the packed arena or
 * dequantizing the GGUF source on demand), the token-embedding lookup, and the
 * norm/scale tensor accessor. Distinct from forward_kernels.h — those are the pure,
 * model-free reference primitives (dot/rmsnorm/rope/attn); this layer bridges the
 * model's quantized weight STORAGE to the dot, and is where the runtime weight-path
 * gate knobs (SP_CPU_SCALAR, SP_ENGINE_F16_ACT, SP_ENGINE_FROB, SP_Q4_PROMOTE) live.
 *
 * Relocated out of the engine's kernels.c. The reference accumulation order is the
 * scalar sp_dot_f32 (forward_kernels) — the engine's AVX dot is a CPU-backend
 * variant that gates against this reference, so it does not live here. The frozen
 * pure-f32 reference path (no arena, knobs off) reduces to "dequant a weight row,
 * sp_dot_f32 with the activation"; the arena inline-lift and Frob/Q4/F16-activation
 * paths are carried verbatim and validated end-to-end by the forward round-trip.
 */
#ifndef SP_FORWARD_DISPATCH_H
#define SP_FORWARD_DISPATCH_H

#include "sp/model.h"   /* qwen3_model, gguf_tensor — the storage these kernels read */

#ifdef __cplusplus
extern "C" {
#endif

/* Y[t,j] = sum_i W[i,j] * X[t,i]; W is the GGUF weight tensor [in,out] (ne0=in).
 * Honors the packed-weight arena when built (inline lift) and the SP_ENGINE_FROB /
 * SP_ENGINE_F16_ACT knobs; the default (no arena, knobs off) is the pure-f32 scalar
 * reference. `X` is n_tok rows of `in`; `Y` is n_tok rows of `out`. Returns 0 on success. */
int sp_matmul(const qwen3_model *m, const gguf_tensor *W,
              const float *X, int n_tok, int in, int out, float *Y);

/* Embedding lookup for token `tok` -> dst[E] (from the arena if the embedding is
 * packed, else dequantized from the GGUF mapping). Returns 0 on success. */
int sp_embed_row(const qwen3_model *m, int32_t tok, int E, float *dst);

/* Read a norm/scale weight tensor as f32: the owned f32 copy after source release,
 * else directly from the GGUF mapping. NULL if not found post-release. */
const float *sp_as_f32(const qwen3_model *m, const gguf_tensor *t);

/* Refresh the weight-path gate knobs from the environment (call once per forward
 * entry, before any sp_matmul). Q4 promotion calibration stats from the most recent
 * forward are read back via qwen3_q4_stats (declared in sp/model.h). */
void sp_kernels_read_env(void);

#ifdef __cplusplus
}
#endif
#endif /* SP_FORWARD_DISPATCH_H */
