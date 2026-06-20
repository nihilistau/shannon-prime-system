/* weight_dtype.h -- portable weight-dtype dequantization for the math core: the
 * GGUF/GGML on-disk weight encodings a transformer forward reads, decoded to f32.
 * F16<->F32 conversion plus per-row dequant (F32 / F16 / Q8_0), with no model, arena,
 * or GGUF-loader coupling -- the leaf the matmul/embed weight lift sits on. Lifted out
 * of the engine forward path. The dtype-tag values are the standard on-disk GGML type
 * IDs (sp_dequant_row receives the raw GGUF tensor type tag); mirroring them here keeps
 * this decoupled from the GGUF loader's header. */
#ifndef SP_WEIGHT_DTYPE_H
#define SP_WEIGHT_DTYPE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* On-disk GGUF/GGML weight type tags this leaf dequantizes (the standard GGML IDs). */
typedef enum { SP_WDT_F32 = 0, SP_WDT_F16 = 1, SP_WDT_Q4_0 = 2, SP_WDT_Q5_0 = 6,
               SP_WDT_Q8_0 = 8, SP_WDT_Q4_K = 12, SP_WDT_Q6_K = 14 } sp_weight_dtype;

float    sp_f16_to_f32(uint16_t h);   /* IEEE half -> single precision. */
uint16_t sp_f32_to_f16(float f);      /* single -> IEEE half, round-to-nearest-even. */

/* Dequantize `n` on-disk weight elements of GGUF type `type` (an sp_weight_dtype /
 * GGML tag) into the caller's f32 buffer `dst`. Returns 0, or 1 on an unsupported
 * type or invalid length. */
int sp_dequant_row(const void *src, uint32_t type, int n, float *dst);

#ifdef __cplusplus
}
#endif
#endif /* SP_WEIGHT_DTYPE_H */
