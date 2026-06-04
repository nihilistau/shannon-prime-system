/* ntt_batch.c — portable reference for the batched forward transform.
 *
 * THIS SYMBOL'S OWN ARCHIVE MEMBER, deliberately (same linker seam as
 * core/poly_ring/resdot.c): the engine overrides sp_ntt_fwd_batch with an
 * AVX2 lane-parallel kernel by defining the symbol in an always-pulled
 * object; binaries that don't get the canonical loop below.
 *
 * EXACTNESS CONTRACT: bit-equal to nb independent ntt_forward calls for every
 * input/stride combination (gate T_NTT_BATCH). Any override must preserve
 * this to the bit — the keystore parity gates (T_PR_KSTORE, T_GENKV_NTT_TOP1,
 * fusion run-gates) all sit on top of it.
 */
#include "sp/ntt_crt.h"

void sp_ntt_fwd_batch(const ntt_ctx *ctx, const int32_t *in, size_t in_stride,
                      uint32_t *out1, size_t out1_stride,
                      uint32_t *out2, size_t out2_stride, int nb) {
    for (int i = 0; i < nb; i++)
        ntt_forward(ctx, in + (size_t)i * in_stride,
                    out1 + (size_t)i * out1_stride,
                    out2 + (size_t)i * out2_stride);
}
