/* exact_islands.h -- the three exact-integer fp32-island replacements for the
 * BYTE-EXACT forward (CONTRACT-BYTEEXACT-forward.md). The engine runs the linear
 * algebra in O_K exactly (dp4a int4xint8->int32 is reduction-order-immune), but
 * RMSNorm / softmax / GELU are float "fp32 islands" (1/sqrt, exp, tanh in float):
 * the only non-byte-exact parts of the forward, and exactly the 1B validation's
 * nonzero deltas. These three functions replace them with fixed-point integer
 * arithmetic so that
 *   - every reduction (sum x^2, sum exp) is EXACT integer  => reduction-order-immune,
 *   - every transcendental (1/sqrt via integer isqrt, exp via 2^x integer poly,
 *     tanh via that exp) is a DETERMINISTIC integer function => machine-independent,
 * i.e. bit-identical logits across reduction order and machine, at gold fidelity.
 *
 * Ports of the proven offline prototypes (engine tools/ring3/g_norm_integer.py +
 * g_islands_integer.py; receipts G-NORM-INTEGER.log + G-BYTEEXACT-ISLANDS.log):
 * RMS relerr 7.6e-6 / softmax KL 8.8e-8 / GELU relerr 1.5e-6, all order-immune.
 *
 * Fixed-point layout (frozen, == the prototypes):
 *   - RMSNorm:  Q=16 (input), IB=20 (inverse-rms), Qw=16 (weight)
 *   - softmax:  Z=2^14 (logit grid),  exp in FB=30 fixed-point
 *   - GELU:     Z=2^16 (input grid),  tanh/exp in FB=30 fixed-point
 */
#ifndef SP_EXACT_ISLANDS_H
#define SP_EXACT_ISLANDS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Island 1 (keystone) -- exact-integer RMSNorm. out[i] = x[i]*sqrt(n/sum x^2)*w[i].
 * sum x^2 is accumulated EXACTLY in int64 (reduction-order-immune); 1/sqrt is the
 * exact integer isqrt of (n << 2*(Q+IB)) / sum_x2; the (Q+IB+Qw) fixed-point product
 * is fully integer. `w` is the direct per-channel multiplier (Gemma's residual +1 is
 * pre-absorbed at transcode, matching sp_rmsnorm). Drop-in for sp_rmsnorm; eps is
 * omitted (the exact route does not need the float stabiliser -- see contract). */
void sp_rmsnorm_exact(const float *x, const float *w, int n, float *out);

/* Island 2 -- exact-integer softmax. p = exp(z - max z) / sum, with the logits put on
 * a Z=2^14 grid, exp via the shared 2^x integer poly (FB=30), the denominator summed
 * EXACTLY in int64 (reduction-order-immune), and the final divide in double. `p`
 * receives m probabilities (length-m caller buffer). */
void sp_softmax_exact(const float *z, int m, double *p);

/* Island 3 -- exact-integer GELU-tanh (Gemma FFN/AltUp gate = ggml_gelu):
 * out = 0.5*x*(1 + tanh( sqrt(2/pi)*(x + 0.044715 x^3) )), the cubic + tanh evaluated
 * in FB=30 fixed-point (tanh via the shared exp primitive). Deterministic per-element
 * integer function. Drop-in for the elementwise GELU before the up-gate multiply. */
void sp_gelu_exact(const float *x, int n, float *out);

#ifdef __cplusplus
}
#endif
#endif /* SP_EXACT_ISLANDS_H */
