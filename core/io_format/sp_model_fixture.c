/* sp_model_fixture.c — see sp_model_fixture.h. Builds a spec-conformant
 * .sp-model + .sp-tokenizer image (PPT-LAT-SP-MODEL-v0 §3/§4/§5/§6/§7) using the
 * frozen packed structs from sp/sp_model.h, so the byte offsets are correct by
 * construction. Hash fields are filled with the real math-core primitives
 * (CRC-32 header, SHA-256 tokenizer, xxh64 name_hash) so the produced file
 * passes the loader's default-load checks. */
#include "sp_model_fixture.h"
#include "sp/sp_model.h"
#include "sp/sp_hash.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FX_TOK_BLOB    64u
#define FX_VOCAB      256u
#define FX_F32_BYTES  128u                 /* F32 tensor on-disk length          */
#define FX_TENSOR_CNT   2u

/* round n up to the next multiple of a (a a power of two). */
static uint64_t round_up_u64(uint64_t n, uint64_t a) { return (n + (a - 1)) & ~(a - 1); }

static void fill_arch(sp_arch_info *a) {
    memset(a, 0, sizeof *a);
    a->arch_id         = (uint32_t)SP_ARCH_ID_GEMMA3;
    a->vocab_size      = FX_VOCAB;
    a->hidden_dim      = 128u;
    a->n_layers        = 4u;
    a->n_heads         = 8u;
    a->n_kv_heads      = 2u;
    a->head_dim        = 16u;
    a->max_context     = 1024u;
    a->swa_window      = 256u;
    a->rope_freq_base  = 10000.0f;
    a->ffn_variant     = 1u;   /* GeGLU (gemma3) */
    a->norm_variant    = 1u;   /* sandwich       */
    a->tied_embeddings = 1u;
    a->has_qk_norm     = 1u;
}

int sp_fixture_build(uint8_t **model_buf, uint8_t **tok_buf,
                     uint32_t spinor_blocks, sp_fixture_info *info) {
    if (!model_buf || !tok_buf || !info || spinor_blocks == 0) return 1;

    /* ── tokenizer file: 128-byte header + blob ── */
    const size_t tok_len = (size_t)SP_TOK_HEADER_SIZE + FX_TOK_BLOB;
    uint8_t *tk = (uint8_t *)calloc(1, tok_len);
    if (!tk) return 1;
    {
        sp_tok_header th; memset(&th, 0, sizeof th);
        th.magic        = SP_TOK_MAGIC;
        th.version_major = SP_MODEL_VER_MAJOR;
        th.version_minor = SP_MODEL_VER_MINOR;
        th.header_size  = SP_TOK_HEADER_SIZE;
        th.type_id      = (uint32_t)SP_TOK_SENTENCEPIECE;
        th.vocab_size   = FX_VOCAB;
        th.bos_token    = 1u;
        th.eos_token    = 2u;
        th.pad_token    = 0u;
        th.unk_token    = 3u;
        th.blob_offset  = SP_TOK_HEADER_SIZE;
        th.blob_size    = FX_TOK_BLOB;
        memcpy(tk, &th, sizeof th);
        for (uint32_t i = 0; i < FX_TOK_BLOB; i++)
            tk[SP_TOK_HEADER_SIZE + i] = (uint8_t)(i & 0xFFu);
        /* header CRC over [0,52) written back at offset 52 */
        uint32_t tcrc = sp_crc32(tk, SP_TOK_CRC_COVER);
        memcpy(tk + SP_TOK_CRC_COVER, &tcrc, sizeof tcrc);
    }

    /* ── model geometry ── */
    const uint64_t table_off = SP_HEADER_SIZE;                              /* 512    */
    const uint64_t data_off  = round_up_u64(table_off + (uint64_t)FX_TENSOR_CNT * SP_TENSOR_ENTRY_SIZE,
                                            SP_DATA_REGION_ALIGN);          /* 65536  */
    const uint64_t f32_in    = 0u;
    const uint64_t spinor_in = round_up_u64(f32_in + FX_F32_BYTES, SP_TENSOR_ALIGN); /* 128 */
    const uint64_t spinor_sz = (uint64_t)spinor_blocks * 64u;
    const uint64_t data_len  = spinor_in + spinor_sz;
    const uint64_t file_size = data_off + data_len;

    uint8_t *mb = (uint8_t *)calloc(1, (size_t)file_size);
    if (!mb) { free(tk); return 1; }

    sp_arch_info arch; fill_arch(&arch);

    /* ── tensor table (sorted by name_hash ascending) ── */
    const char *nm_f32    = "tok_embd.weight";
    const char *nm_spinor = "spinor.kv";

    sp_tensor_entry ef, es; memset(&ef, 0, sizeof ef); memset(&es, 0, sizeof es);
    strncpy(ef.name, nm_f32, sizeof ef.name - 1);
    ef.dtype_id = (uint32_t)SP_DT_F32; ef.n_dims = 2u; ef.dims[0] = 4u; ef.dims[1] = 8u;
    ef.offset_in_data = f32_in; ef.size_bytes = FX_F32_BYTES;
    ef.block_size = 4u; ef.block_count = FX_F32_BYTES / 4u;
    ef.name_hash = sp_xxh64(nm_f32, strlen(nm_f32), 0);

    strncpy(es.name, nm_spinor, sizeof es.name - 1);
    es.dtype_id = (uint32_t)SP_DT_SPINOR63; es.n_dims = 1u; es.dims[0] = spinor_blocks;
    es.offset_in_data = spinor_in; es.size_bytes = spinor_sz;
    es.block_size = 64u; es.block_count = spinor_blocks;
    es.name_hash = sp_xxh64(nm_spinor, strlen(nm_spinor), 0);

    /* per-tensor blake3 over the on-disk bytes (placeholder digest; opt-in verify) */
    /* (data bytes are written below; compute after we lay them down) */

    sp_tensor_entry *e0, *e1;
    if (ef.name_hash <= es.name_hash) { e0 = &ef; e1 = &es; }
    else                              { e0 = &es; e1 = &ef; }

    /* ── data region ── */
    uint8_t *data = mb + data_off;
    for (uint32_t i = 0; i < FX_F32_BYTES; i++) data[f32_in + i] = 0u;
    for (uint32_t b = 0; b < spinor_blocks; b++) {
        uint8_t *blk = data + spinor_in + (uint64_t)b * 64u;
        for (uint32_t j = 0; j < 63u; j++) blk[j] = (uint8_t)((b + j) & 0xFFu);
        blk[63] = SP_SPINOR_SENTINEL;          /* 0xA5 pad/sentinel */
    }
    sp_blake3_256(data + ef.offset_in_data, ef.size_bytes, ef.blake3);
    sp_blake3_256(data + es.offset_in_data, es.size_bytes, es.blake3);

    memcpy(mb + table_off,                          e0, SP_TENSOR_ENTRY_SIZE);
    memcpy(mb + table_off + SP_TENSOR_ENTRY_SIZE,   e1, SP_TENSOR_ENTRY_SIZE);

    /* ── header ── */
    {
        sp_model_header h; memset(&h, 0, sizeof h);
        h.magic               = SP_MODEL_MAGIC;
        h.version_major       = SP_MODEL_VER_MAJOR;
        h.version_minor       = SP_MODEL_VER_MINOR;
        h.header_size         = SP_HEADER_SIZE;
        h.arch_id             = arch.arch_id;
        h.arch_struct_size    = (uint32_t)sizeof(sp_arch_info);
        h.arch_struct_capacity = 256u;
        memcpy(h.arch_struct, &arch, sizeof arch);
        sp_sha256(tk, tok_len, h.tokenizer_hash);
        h.vocab_size          = FX_VOCAB;
        h.tensor_count        = FX_TENSOR_CNT;
        h.tensor_table_offset = table_off;
        h.tensor_data_offset  = data_off;
        h.file_size           = file_size;
        h.created_unix_seconds = 1700000000ull;
        h.transcoded_from     = 0ull;
        memcpy(mb, &h, sizeof h);
        uint32_t hcrc = sp_crc32(mb, SP_HEADER_CRC_COVER);   /* [0,360) */
        memcpy(mb + SP_HEADER_CRC_COVER, &hcrc, sizeof hcrc);
    }

    info->model_len          = (size_t)file_size;
    info->tok_len            = tok_len;
    info->tok_blob_offset    = SP_TOK_HEADER_SIZE;
    info->spinor_data_offset = (size_t)(data_off + spinor_in);
    info->spinor_block_size  = 64u;
    info->spinor_block_count = spinor_blocks;
    info->arch               = arch;
    info->f32_tensor_name    = nm_f32;
    info->spinor_tensor_name = nm_spinor;

    *model_buf = mb;
    *tok_buf   = tk;
    return 0;
}

int sp_fixture_write_file(const char *path, const uint8_t *buf, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f) return 1;
    size_t w = fwrite(buf, 1, len, f);
    int rc = (fclose(f) != 0 || w != len) ? 1 : 0;
    return rc;
}

size_t sp_fixture_spinor_sentinel_at(const sp_fixture_info *info, uint32_t blk) {
    return info->spinor_data_offset + (size_t)blk * info->spinor_block_size
         + (info->spinor_block_size - 1u);
}
