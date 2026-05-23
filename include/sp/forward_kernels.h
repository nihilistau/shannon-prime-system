/* forward_kernels.h -- the portable scalar reference forward-pass kernels of the math
 * core: the pure-math operations a transformer forward composes, with no model,
 * arena, GGUF, or backend-intrinsic coupling. The four backends (AVX2 / HVX / CUDA /
 * Vulkan) provide accelerated variants the L1 forward dispatches to; these scalar
 * versions are the correctness reference each accelerated path gates against (the
 * §8.6.1 precision-floor reference). Moved out of the engine forward path so one
 * definition lives in libshannonprime and every consumer runs it.
 *
 * The engine's matmul / arena / embedding lift kernels stay behind for now -- they
 * are coupled to the model representation (GGUF tensors, the packed arena), which
 * migrates into the math core as its own later increment.
 */
#ifndef SP_FORWARD_KERNELS_H
#define SP_FORWARD_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Inner product of two length-n f32 vectors -- scalar, sequential: the reference
 * accumulation order. Vectorised backends reorder the reduction and gate to this. */
float sp_dot_f32(const float *a, const float *b, int n);

/* RMSNorm: out[i] = x[i] * w[i] / sqrt(mean(x^2) + eps). Sum of squares accumulated
 * in double (the reference precision behaviour). `rmsnorm_head` normalises a single
 * head-dim vector in place (per-head Q/K RMSNorm, before RoPE). */
void sp_rmsnorm(const float *x, const float *w, int n, float eps, float *out);
void sp_rmsnorm_head(float *v, const float *w, int d, float eps);

/* NEOX-style RoPE: rotate the (i, i+d/2) coordinate pairs of a head-dim vector `v`
 * at sequence position `p` with frequency base `base`. */
void sp_rope_neox(float *v, int d, int p, float base);

/* One GQA query head: causal (or windowed, win>=0) scaled-dot-product softmax
 * attention over the cached K/V. KC/VC are position-major caches of width
 * KVD = n_kv_heads*head_dim; `kvh` is the kv head this query maps to; `ascale` the
 * 1/sqrt(head_dim) softmax scale; `sc` is caller scratch (length >= pos+1); `out`
 * receives the HD-wide weighted-value result. */
void sp_attn_head(const float *qh, const float *KC, const float *VC,
                  int pos, int KVD, int kvh, int HD, float ascale, int win,
                  float *sc, float *out);

#ifdef __cplusplus
}
#endif
#endif /* SP_FORWARD_KERNELS_H */
