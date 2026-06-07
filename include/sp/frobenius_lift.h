/* sp/frobenius_lift.h — Frobenius lift for Q8 weight storage.  (Phase 1E.)
 *
 * A weight tensor (rows x cols, fp32) is compressed to packed signed int8 plus
 * a PER-ROW fp32 scale array, then decompressed inline at matmul time.
 *
 * Why per-row ("Frobenius") scaling.  A single per-tensor scale collapses the
 * dynamic range of every row onto the same int8 grid; rows whose magnitudes
 * are small relative to the tensor max lose almost all their bits and the
 * quantised tensor degrades to noise.  Normalising each row by its own max
 * absolute value (the "Frobenius scale" of that row) keeps every row using the
 * full [-127,127] code range.  Per-row is the load-bearing choice.
 *
 * Storage layout (the lift):
 *   - packed[r*cols + c] : the int8 code q for weight w[r][c]
 *   - row_scale[r]       : fp32 scale s for row r  (s = max_c |w[r][c]|)
 * Memory: 1 byte per coefficient + 1 fp32 (4 bytes) per row.  For an R x C
 * tensor that is  R*C + 4*R  bytes, versus  4*R*C  for fp32 (~4x smaller) and
 * 8*R*C for fp64 (~8x smaller).  See sp_frob_packed_bytes / sp_frob_ratio.
 *
 * Quantisation (encode):
 *   x = v / s * 127                       (s = sp_frob_row_scale(row))
 *   q = ceil-shift-round(x), clamped to [-127, 127], stored as int8_t
 * Ceiling-shift rounding rule (round-half-AWAY-from-zero), deterministic and
 * independent of the hardware FP rounding mode:
 *   q = (x >= 0) ? floorf(x + 0.5f) : ceilf(x - 0.5f)
 *      == copysignf(floorf(fabsf(x) + 0.5f), x)
 * So +0.5 -> +1, -0.5 -> -1, +1.5 -> +2, -1.5 -> -2.  This is symmetric about
 * zero, which removes the systematic +1/2-LSB bias that plain round-half-up
 * (floorf(x + 0.5f) for all x) introduces on signed data.  We use floorf/ceilf
 * (not lrintf) precisely so the result never depends on fesetround().
 *
 * The code range is the SYMMETRIC interval [-127, 127] (note: NOT -128).  A
 * symmetric range is required for the dequant identity v_hat = q*(s/127) to be
 * able to reproduce v = +s exactly (code +127); allowing -128 would break that
 * symmetry without buying any extra precision for normalised data.
 *
 * Dequantisation (inline lift):
 *   v_hat = q * (s / 127)
 * In a real matmul this is: read packed byte -> sign-extend to int16 ->
 * multiply by the broadcast row scale (fp32) -> accumulate.  sp_frob_dequantize
 * mirrors that arithmetic exactly.
 *
 * A zero row (every element 0, hence s == 0) is encoded as all-zero codes with
 * row_scale 0, and dequantises back to all zeros.
 *
 * Part of the shared public API root (include/).  Include as
 *   #include "sp/frobenius_lift.h"
 */
#ifndef SP_FROBENIUS_LIFT_H
#define SP_FROBENIUS_LIFT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The symmetric int8 code limit.  Codes live in [-SP_FROB_QMAX, +SP_FROB_QMAX]. */
#define SP_FROB_QMAX 127

/* A packed Q8 weight tensor: int8 codes plus one fp32 scale per row.
 * `packed` points at rows*cols int8 codes in row-major order; `row_scale`
 * points at `rows` fp32 scales.  The descriptor borrows the buffers (it does
 * not own them); the caller manages their lifetime. */
typedef struct {
    int      rows;
    int      cols;
    int8_t  *packed;     /* rows*cols codes, row-major          */
    float   *row_scale;  /* rows scales: row_scale[r] for row r */
} sp_frob_tensor;

/* Per-row Frobenius scale: the maximum absolute value over `cols` elements of
 * `row`.  Returns 0.0f for an all-zero row (and is well-defined for cols <= 0,
 * returning 0).  This is the s used by quantise/dequantise for that row. */
float sp_frob_row_scale(const float *row, int cols);

/* Quantise a single value v in a row of scale s to its int8 code, using the
 * ceiling-shift (round-half-away-from-zero) rule documented above, clamped to
 * [-SP_FROB_QMAX, +SP_FROB_QMAX].  If s == 0 the code is 0 (zero row). */
int8_t sp_frob_quant1(float v, float scale);

/* Dequantise a single int8 code q in a row of scale s: v_hat = q * (s / 127). */
float sp_frob_dequant1(int8_t q, float scale);

/* Quantise an fp32 weight tensor `w` (rows x cols, row-major) into `packed`
 * (rows*cols int8 codes) and `row_scale` (rows fp32 scales).  For each row it
 * picks the per-row scale via sp_frob_row_scale, then quantises every element
 * with sp_frob_quant1.  `packed` and `row_scale` must be caller-allocated with
 * room for rows*cols and rows entries respectively. */
void sp_frob_quantize(const float *w, int rows, int cols,
                      int8_t *packed, float *row_scale);

/* Dequantise a packed Q8 tensor back to fp32 (rows x cols, row-major) into
 * `out`, mirroring the inline matmul lift: out[r][c] = packed[r][c]*(s_r/127).
 * `out` must be caller-allocated with room for rows*cols floats. */
void sp_frob_dequantize(const int8_t *packed, const float *row_scale,
                        int rows, int cols, float *out);

/* Packed byte size of an `rows` x `cols` Q8 tensor:
 *   rows*cols (one int8 code per coefficient) + rows*sizeof(float) (row scales).
 */
size_t sp_frob_packed_bytes(int rows, int cols);

/* Compression ratio of the packed Q8 form versus an unquantised source tensor
 * whose element width is `src_dtype_bytes` (e.g. 4 for fp32, 8 for fp64):
 *   (src_dtype_bytes * rows * cols) / sp_frob_packed_bytes(rows, cols).
 * Returns 0.0 if the packed size is 0 (degenerate rows/cols). */
double sp_frob_ratio(int rows, int cols, int src_dtype_bytes);

/* ── Q4 (4-bit) variant ──────────────────────────────────────────────────────
 * Same per-row "Frobenius" scale and round-half-away rule as Q8, but symmetric
 * 4-bit codes in [-SP_FROB_QMAX4, +SP_FROB_QMAX4] = [-7,7], packed two per byte.
 * 4x more aggressive than Q8 (~16x vs fp32) and correspondingly lossier, so it
 * is used as a mixed-precision path: the caller calibrates per row (see
 * sp_frob_q4_row_relerr) and promotes high-error rows back to Q8. The codec is
 * shared by every engine backend (CPU/CUDA/Vulkan/Hexagon). */

/* The symmetric int4 code limit. Q4 codes live in [-SP_FROB_QMAX4, +SP_FROB_QMAX4]. */
#define SP_FROB_QMAX4 7

/* Quantise v in a row of scale s to its 4-bit code (round-half-away, clamped to
 * [-7,7]); s == 0 -> code 0. The code is returned in an int8_t but is always in
 * [-7,7] (pack two per byte with sp_frob_q4_pack). */
int8_t sp_frob_quant1_q4(float v, float scale);

/* Dequantise a 4-bit code q in a row of scale s: v_hat = q * (s / 7). */
float sp_frob_dequant1_q4(int8_t q, float scale);

/* Pack/unpack symmetric 4-bit codes two-per-byte (low nibble = even index).
 * Round-trips exactly for the [-8,7] two's-complement range, so it is lossless
 * on valid Q4 codes — the only loss is the quantisation itself. `nib` must hold
 * (n + 1) / 2 bytes; unpack sign-extends each nibble. */
void sp_frob_q4_pack(const int8_t *codes, int n, uint8_t *nib);
void sp_frob_q4_unpack(const uint8_t *nib, int n, int8_t *codes);

/* Per-row Q4 round-trip relative error:
 *   || row - dequant_q4(quant_q4(row)) ||_2 / || row ||_2     (0 for a zero row).
 * The reusable calibration metric: the caller promotes a row from Q4 to Q8 when
 * this exceeds its threshold. (The promotion policy stays with the caller; this
 * is just the per-row sensitivity primitive.) */
float sp_frob_q4_row_relerr(const float *row, int cols);

/* Packed byte size of an `rows` x `cols` Q4 tensor:
 *   rows*ceil(cols/2) (two 4-bit codes per byte) + rows*sizeof(float) scales. */
size_t sp_frob_q4_packed_bytes(int rows, int cols);

/* ── Mixed-precision packed tensor (the load-bearing "arena" layout) ──────────
 * The packed-WEIGHT memory layout of roadmap §4.8: one weight matrix stored
 * per-row as EITHER Q8 or Q4 codes (Q4 mixed-precision promotes high-error rows
 * to Q8), with a per-row scale and a per-row byte offset into one codes buffer.
 * This is the byte format EVERY backend (CPU/CUDA/Vulkan/Hexagon) reads inline at
 * matmul time, so it is versioned and frozen here. Per-ROW Frobenius — NOT ggml's
 * per-32-block Q8_0; the two are not interchangeable. A change to the byte layout
 * or dequant convention REQUIRES bumping SP_FROB_ARENA_LAYOUT_VERSION + migration.
 * v2 (2026-06-08, SPEC OK_Q4B): optional per-32-BLOCK f16 scales via `bscale`.
 * bscale == NULL => v1 per-row semantics exactly (row_scale governs); bscale != NULL
 * => Q4B rows: dequant = code * f16_to_f32(bscale[r*bs_nblk + c/32]) and row_scale
 * MAY be NULL. alias bit 2 covers bscale. */
#define SP_FROB_ARENA_LAYOUT_VERSION 2u

typedef struct {
    int      rows;          /* weight rows (= out features) */
    int      cols;          /* elems per row (= in features) */
    uint8_t *row_prec;      /* [rows] per-row precision: 8 or 4 */
    float   *row_scale;     /* [rows] per-row Frobenius scale (max abs); may be NULL when bscale set */
    size_t  *row_off;       /* [rows] byte offset of the row's codes in `codes` */
    uint8_t *codes;         /* packed codes: Q8 row = cols int8; Q4 row = ceil(cols/2) bytes */
    size_t   codes_bytes;   /* used bytes in `codes` */
    uint8_t  alias_mask;    /* bit 0: codes aliased; bit 1: row_scale aliased; bit 2: bscale aliased */
    const uint16_t *bscale; /* OK_Q4B: [rows * bs_nblk] per-32-block f16 scales (NULL = per-row) */
    int      bs_nblk;       /* blocks per row = ceil(cols/32); 0 when bscale == NULL */
} sp_frob_packed_tensor;

/* Source-row reader: writes row `j` of the weight matrix as `cols` f32 into `dst`.
 * Returns 0 on success. The packer calls this once per row, so the caller (e.g. the
 * engine reading a GGUF tensor) never has to materialize the whole matrix as f32. */
typedef int (*sp_frob_row_fn)(void *ctx, int j, float *dst);

/* Pack a rows x cols weight matrix (rows fetched via `get_row(ctx, j, dst)`) into
 * `out` in the mixed-precision arena layout. `precision` 8 => every row Q8; 4 => Q4
 * with rows whose Q4 round-trip rel-error (sp_frob_q4_row_relerr) exceeds `promote`
 * stored Q8. Allocates and owns out->{row_prec,row_scale,row_off,codes} (release with
 * sp_frob_packed_free). If `promoted` is non-NULL it is incremented per promoted row.
 * Returns 0 on success; nonzero on a bad arg, alloc failure, or get_row failure (in
 * which case `out` is freed and zeroed). */
int sp_frob_pack_tensor(int rows, int cols, int precision, float promote,
                        sp_frob_row_fn get_row, void *ctx,
                        sp_frob_packed_tensor *out, long *promoted);

/* Free the buffers owned by a packed tensor and zero the descriptor. */
void sp_frob_packed_free(sp_frob_packed_tensor *t);

/* Reconstruct row `r` to `cols` f32 (inline lift: code * scale / qmax, qmax 127 for
 * a Q8 row or 7 for a Q4 row). `dst` holds `cols` floats. Returns 0 on success. */
int sp_frob_packed_dequant_row(const sp_frob_packed_tensor *t, int r, float *dst);

/* Total bytes the packed tensor occupies: codes + per-row (scale + offset + prec). */
size_t sp_frob_packed_tensor_bytes(const sp_frob_packed_tensor *t);

#ifdef __cplusplus
}
#endif

#endif /* SP_FROBENIUS_LIFT_H */
