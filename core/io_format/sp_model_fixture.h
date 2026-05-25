/* sp_model_fixture.h — test-only synthetic .sp-model / .sp-tokenizer builder.
 *
 * Builds a small but spec-conformant (PPT-LAT-SP-MODEL-v0) .sp-model + paired
 * .sp-tokenizer image entirely in memory, so the runtime loader tests are
 * deterministic and dependency-free (no multi-GB real model artifact, CI-safe
 * on Tier-2 Linux gcc). The negative tests mutate a copy of these buffers.
 *
 * Compiled into the io_format test executable via TEST_SOURCES; never linked
 * into the sp_io_format library itself.
 */
#ifndef SP_MODEL_FIXTURE_H
#define SP_MODEL_FIXTURE_H

#include <stdint.h>
#include <stddef.h>
#include "sp/sp_l1.h"   /* sp_arch_info — the arch payload embedded in the header */

#ifdef __cplusplus
extern "C" {
#endif

/* Layout facts the test needs to mutate / verify the built image without
 * re-deriving the byte offsets the builder chose. */
typedef struct {
    size_t       model_len;            /* total .sp-model byte length            */
    size_t       tok_len;              /* total .sp-tokenizer byte length        */
    size_t       tok_blob_offset;      /* absolute offset of the tokenizer blob  */
    size_t       spinor_data_offset;   /* absolute offset of Spinor block 0      */
    uint32_t     spinor_block_size;    /* on-disk bytes per Spinor block (== 64) */
    uint32_t     spinor_block_count;   /* number of Spinor blocks                */
    sp_arch_info arch;                 /* exact arch payload embedded (for arch test) */
    const char  *f32_tensor_name;      /* a resolvable tensor name (FIND_HIT)    */
    const char  *spinor_tensor_name;
} sp_fixture_info;

/* Build a valid .sp-model + .sp-tokenizer into freshly malloc'd buffers
 * (*model_buf / *tok_buf — caller free()s both). spinor_blocks is the number
 * of 64-byte Spinor blocks in the single Spinor tensor (use >= 3 to exercise a
 * mid-block full-sweep). Returns 0 on success, nonzero on allocation failure. */
int sp_fixture_build(uint8_t **model_buf, uint8_t **tok_buf,
                     uint32_t spinor_blocks, sp_fixture_info *info);

/* Write len bytes to path (binary). Returns 0 on success. */
int sp_fixture_write_file(const char *path, const uint8_t *buf, size_t len);

/* Absolute file offset of the sentinel byte (byte 63) of Spinor block `blk`. */
size_t sp_fixture_spinor_sentinel_at(const sp_fixture_info *info, uint32_t blk);

#ifdef __cplusplus
}
#endif
#endif /* SP_MODEL_FIXTURE_H */
