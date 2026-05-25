/* qwen3_fixture.c — see qwen3_fixture.h. Assembles a tiny qwen3-shaped .sp-model
 * from the frozen packed structs (sp/sp_model.h), so byte offsets are correct by
 * construction. Real hash primitives (sp/sp_hash.h) fill the header CRC-32,
 * tokenizer SHA-256, and per-tensor xxh64 name hashes, so the file passes the
 * loader's default-load checks and sp_model_find_tensor resolves every tensor. */
#include "qwen3_fixture.h"
#include "sp/sp_model.h"
#include "sp/sp_hash.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* tiny architecture */
#define FX_NL    2u
#define FX_E     32u     /* n_embd  */
#define FX_FF    64u     /* n_ff    */
#define FX_NH    4u      /* n_head  */
#define FX_NKV   2u      /* n_head_kv (GQA) */
#define FX_HD    8u      /* head_dim */
#define FX_V     48u     /* n_vocab */
#define FX_TOKBLOB 64u
#define FX_MAXT  64      /* tensor-spec capacity */

typedef struct {
    char     name[80];
    uint32_t dtype;
    uint32_t n_dims;
    uint64_t dims[8];
    uint32_t block_size;
    uint64_t size_bytes;
    uint8_t *data;        /* owned during build */
    uint64_t name_hash;
    uint64_t offset_in_data;
} tspec;

static uint64_t round_up_u64(uint64_t n, uint64_t a) { return (n + (a - 1)) & ~(a - 1); }

/* Append a tensor spec; allocates + fills its data buffer deterministically. */
static int add_q8(tspec *T, int *n, const char *name, uint32_t rows, uint32_t cols, uint32_t salt) {
    if (*n >= FX_MAXT) return 1;
    tspec *s = &T[*n];
    memset(s, 0, sizeof *s);
    snprintf(s->name, sizeof s->name, "%s", name);
    s->dtype = (uint32_t)SP_DT_OK_Q8; s->n_dims = 2; s->dims[0] = cols; s->dims[1] = rows;
    s->block_size = 1u; s->size_bytes = (uint64_t)rows * cols;
    s->data = (uint8_t *)malloc((size_t)s->size_bytes);
    if (!s->data) return 1;
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < cols; c++) {
            int v = (int)((r * 7u + c * 13u + salt) % 251u) - 125;   /* int8 in [-125,125] */
            s->data[(size_t)r * cols + c] = (uint8_t)(int8_t)v;
        }
    (*n)++;

    /* paired per-row Frobenius scale */
    if (*n >= FX_MAXT) return 1;
    tspec *sc = &T[*n];
    memset(sc, 0, sizeof *sc);
    snprintf(sc->name, sizeof sc->name, "%s.scale", name);
    sc->dtype = (uint32_t)SP_DT_FROBENIUS_SCALE_FP32; sc->n_dims = 1; sc->dims[0] = rows;
    sc->block_size = 4u; sc->size_bytes = (uint64_t)rows * 4u;
    sc->data = (uint8_t *)malloc((size_t)sc->size_bytes);
    if (!sc->data) return 1;
    float *sf = (float *)sc->data;
    for (uint32_t r = 0; r < rows; r++) sf[r] = 0.5f + (float)(r % 7u) * 0.05f;   /* positive */
    (*n)++;
    return 0;
}

static int add_f32(tspec *T, int *n, const char *name, uint32_t len) {
    if (*n >= FX_MAXT) return 1;
    tspec *s = &T[*n];
    memset(s, 0, sizeof *s);
    snprintf(s->name, sizeof s->name, "%s", name);
    s->dtype = (uint32_t)SP_DT_F32; s->n_dims = 1; s->dims[0] = len;
    s->block_size = 4u; s->size_bytes = (uint64_t)len * 4u;
    s->data = (uint8_t *)malloc((size_t)s->size_bytes);
    if (!s->data) return 1;
    float *f = (float *)s->data;
    for (uint32_t i = 0; i < len; i++) f[i] = 0.9f + (float)(i % 5u) * 0.02f;   /* ~1 (norm weight) */
    (*n)++;
    return 0;
}

static int cmp_hash(const void *a, const void *b) {
    uint64_t ha = ((const tspec *)a)->name_hash, hb = ((const tspec *)b)->name_hash;
    return (ha < hb) ? -1 : (ha > hb) ? 1 : 0;
}

int sp_qwen3_fixture_build(uint8_t **model_buf, uint8_t **tok_buf, sp_qwen3_fixture_info *info) {
    if (!model_buf || !tok_buf || !info) return 1;
    memset(info, 0, sizeof *info);
    info->n_layers = FX_NL; info->n_embd = FX_E; info->n_ff = FX_FF;
    info->n_head = FX_NH; info->n_head_kv = FX_NKV; info->head_dim = FX_HD;
    info->n_vocab = FX_V; info->rope_freq_base = 1e6f; info->tied = 1;

    sp_arch_info *a = &info->arch;
    a->arch_id = (uint32_t)SP_ARCH_ID_QWEN3; a->vocab_size = FX_V; a->hidden_dim = FX_E;
    a->n_layers = FX_NL; a->n_heads = FX_NH; a->n_kv_heads = FX_NKV; a->head_dim = FX_HD;
    a->max_context = 256u; a->swa_window = 0u; a->rope_freq_base = 1e6f;
    a->ffn_variant = 0u; a->norm_variant = 0u; a->tied_embeddings = 1u; a->has_qk_norm = 1u;

    /* ── tokenizer ── */
    const size_t tok_len = (size_t)SP_TOK_HEADER_SIZE + FX_TOKBLOB;
    uint8_t *tk = (uint8_t *)calloc(1, tok_len);
    if (!tk) return 1;
    {
        sp_tok_header th; memset(&th, 0, sizeof th);
        th.magic = SP_TOK_MAGIC; th.version_major = SP_MODEL_VER_MAJOR;
        th.version_minor = SP_MODEL_VER_MINOR; th.header_size = SP_TOK_HEADER_SIZE;
        th.type_id = (uint32_t)SP_TOK_BPE_GPT2; th.vocab_size = FX_V;
        th.bos_token = 1u; th.eos_token = 2u; th.pad_token = 0u; th.unk_token = 3u;
        th.blob_offset = SP_TOK_HEADER_SIZE; th.blob_size = FX_TOKBLOB;
        memcpy(tk, &th, sizeof th);
        for (uint32_t i = 0; i < FX_TOKBLOB; i++) tk[SP_TOK_HEADER_SIZE + i] = (uint8_t)(i & 0xFFu);
        uint32_t tcrc = sp_crc32(tk, SP_TOK_CRC_COVER);
        memcpy(tk + SP_TOK_CRC_COVER, &tcrc, sizeof tcrc);
    }

    /* ── tensor specs ── */
    tspec T[FX_MAXT];
    int n = 0, rc = 0;
    char nm[80];
    rc |= add_q8 (T, &n, "token_embd.weight", FX_V, FX_E, 1u);        /* tied LM head + embed */
    for (uint32_t L = 0; L < FX_NL && !rc; L++) {
        uint32_t s = (L + 1u) * 17u;
        snprintf(nm, sizeof nm, "blk.%u.attn_norm.weight", L);   rc |= add_f32(T, &n, nm, FX_E);
        snprintf(nm, sizeof nm, "blk.%u.attn_q.weight", L);      rc |= add_q8 (T, &n, nm, FX_NH*FX_HD, FX_E, s+1u);
        snprintf(nm, sizeof nm, "blk.%u.attn_k.weight", L);      rc |= add_q8 (T, &n, nm, FX_NKV*FX_HD, FX_E, s+2u);
        snprintf(nm, sizeof nm, "blk.%u.attn_v.weight", L);      rc |= add_q8 (T, &n, nm, FX_NKV*FX_HD, FX_E, s+3u);
        snprintf(nm, sizeof nm, "blk.%u.attn_output.weight", L); rc |= add_q8 (T, &n, nm, FX_E, FX_NH*FX_HD, s+4u);
        snprintf(nm, sizeof nm, "blk.%u.attn_q_norm.weight", L); rc |= add_f32(T, &n, nm, FX_HD);
        snprintf(nm, sizeof nm, "blk.%u.attn_k_norm.weight", L); rc |= add_f32(T, &n, nm, FX_HD);
        snprintf(nm, sizeof nm, "blk.%u.ffn_norm.weight", L);    rc |= add_f32(T, &n, nm, FX_E);
        snprintf(nm, sizeof nm, "blk.%u.ffn_gate.weight", L);    rc |= add_q8 (T, &n, nm, FX_FF, FX_E, s+5u);
        snprintf(nm, sizeof nm, "blk.%u.ffn_up.weight", L);      rc |= add_q8 (T, &n, nm, FX_FF, FX_E, s+6u);
        snprintf(nm, sizeof nm, "blk.%u.ffn_down.weight", L);    rc |= add_q8 (T, &n, nm, FX_E, FX_FF, s+7u);
    }
    rc |= add_f32(T, &n, "output_norm.weight", FX_E);
    if (rc) { for (int i = 0; i < n; i++) free(T[i].data); free(tk); return 1; }

    /* name hashes + sort ascending (xxh64) for the loader's binary search */
    for (int i = 0; i < n; i++) T[i].name_hash = sp_xxh64(T[i].name, strlen(T[i].name), 0);
    qsort(T, (size_t)n, sizeof(tspec), cmp_hash);

    /* ── geometry ── */
    const uint64_t table_off = SP_HEADER_SIZE;
    const uint64_t data_off  = round_up_u64(table_off + (uint64_t)n * SP_TENSOR_ENTRY_SIZE,
                                            SP_DATA_REGION_ALIGN);
    uint64_t cur = 0;
    for (int i = 0; i < n; i++) {
        cur = round_up_u64(cur, SP_TENSOR_ALIGN);
        T[i].offset_in_data = cur;
        cur += T[i].size_bytes;
    }
    const uint64_t data_len  = round_up_u64(cur, SP_TENSOR_ALIGN);
    const uint64_t file_size = data_off + data_len;

    uint8_t *mb = (uint8_t *)calloc(1, (size_t)file_size);
    if (!mb) { for (int i = 0; i < n; i++) free(T[i].data); free(tk); return 1; }

    /* ── tensor table + data ── */
    for (int i = 0; i < n; i++) {
        sp_tensor_entry e; memset(&e, 0, sizeof e);
        snprintf(e.name, sizeof e.name, "%s", T[i].name);
        e.dtype_id = T[i].dtype; e.n_dims = T[i].n_dims;
        for (int d = 0; d < 8; d++) e.dims[d] = T[i].dims[d];
        e.offset_in_data = T[i].offset_in_data; e.size_bytes = T[i].size_bytes;
        e.block_size = T[i].block_size;
        e.block_count = T[i].block_size ? (uint32_t)(T[i].size_bytes / T[i].block_size) : 0u;
        e.name_hash = T[i].name_hash;
        sp_blake3_256(T[i].data, T[i].size_bytes, e.blake3);
        memcpy(mb + table_off + (uint64_t)i * SP_TENSOR_ENTRY_SIZE, &e, SP_TENSOR_ENTRY_SIZE);
        memcpy(mb + data_off + T[i].offset_in_data, T[i].data, (size_t)T[i].size_bytes);
    }

    /* ── header ── */
    {
        sp_model_header h; memset(&h, 0, sizeof h);
        h.magic = SP_MODEL_MAGIC; h.version_major = SP_MODEL_VER_MAJOR;
        h.version_minor = SP_MODEL_VER_MINOR; h.header_size = SP_HEADER_SIZE;
        h.arch_id = a->arch_id; h.arch_struct_size = (uint32_t)sizeof(sp_arch_info);
        h.arch_struct_capacity = 256u;
        memcpy(h.arch_struct, a, sizeof *a);
        sp_sha256(tk, tok_len, h.tokenizer_hash);
        h.vocab_size = FX_V; h.tensor_count = (uint32_t)n;
        h.tensor_table_offset = table_off; h.tensor_data_offset = data_off;
        h.file_size = file_size; h.created_unix_seconds = 1700000000ull;
        memcpy(mb, &h, sizeof h);
        uint32_t hcrc = sp_crc32(mb, SP_HEADER_CRC_COVER);
        memcpy(mb + SP_HEADER_CRC_COVER, &hcrc, sizeof hcrc);
    }

    for (int i = 0; i < n; i++) free(T[i].data);
    info->model_len = (size_t)file_size;
    info->tok_len = tok_len;
    *model_buf = mb;
    *tok_buf = tk;
    return 0;
}

int sp_qwen3_fixture_write(const char *path, const uint8_t *buf, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f) return 1;
    size_t w = fwrite(buf, 1, len, f);
    return (fclose(f) != 0 || w != len) ? 1 : 0;
}
