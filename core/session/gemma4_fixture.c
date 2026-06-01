/* gemma4_fixture.c — see gemma4_fixture.h. Tiny Gemma4-shaped .sp-model with the
 * full weight set sp_model_to_gemma4 reconstructs: AltUp globals + per-layer
 * blocks with per-layer head geometry (SWA vs global) and the per-layer-input
 * (inp_gate/proj/post_norm/layer_output_scale). */
#include "gemma4_fixture.h"
#include "sp/sp_model.h"
#include "sp/sp_hash.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FX_NL     6u
#define FX_E      32u
#define FX_FF     64u
#define FX_V      48u
#define FX_PL     8u      /* n_embd_per_layer */
#define FX_PERIOD 3u      /* global layer when L%period == period-1 */
#define FX_KVFS   3u      /* layers [0,3) own KV; 3..5 reuse */
#define FX_SW     4u      /* SWA sliding window */
/* SWA geometry */
#define FX_HD_S   8u
#define FX_NH_S   4u
#define FX_NKV_S  2u
/* global geometry — REAL Gemma4 shape: n_head / n_head_kv are CONSTANT across
 * layer types, head_dim is per-layer (global 16 / SWA 8), so the Q/K/V projection
 * widths DIFFER per layer (SWA QD=32/KVD=16, global QD=64/KVD=32). This exercises
 * the per-layer-width path in gemma4_forward / kv_step_gemma4. */
#define FX_HD_G   16u
#define FX_NH_G   4u      /* == FX_NH_S  (constant n_head)    */
#define FX_NKV_G  2u      /* == FX_NKV_S (constant n_head_kv) */
#define FX_TOKBLOB 64u
#define FX_MAXT   320

typedef struct {
    char     name[80];
    uint32_t dtype;
    uint32_t n_dims;
    uint64_t dims[8];
    uint32_t block_size;
    uint64_t size_bytes;
    uint8_t *data;
    uint64_t name_hash;
    uint64_t offset_in_data;
} tspec;

static uint64_t round_up_u64(uint64_t n, uint64_t a) { return (n + (a - 1)) & ~(a - 1); }

static int add_q8(tspec *T, int *n, const char *name, uint32_t rows, uint32_t cols, uint32_t salt) {
    if (*n + 1 >= FX_MAXT) return 1;
    tspec *s = &T[*n];
    memset(s, 0, sizeof *s);
    snprintf(s->name, sizeof s->name, "%s", name);
    s->dtype = (uint32_t)SP_DT_OK_Q8; s->n_dims = 2; s->dims[0] = cols; s->dims[1] = rows;
    s->block_size = 1u; s->size_bytes = (uint64_t)rows * cols;
    s->data = (uint8_t *)malloc((size_t)s->size_bytes ? (size_t)s->size_bytes : 1u);
    if (!s->data) return 1;
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < cols; c++) {
            int v = (int)((r * 7u + c * 13u + salt) % 251u) - 125;
            s->data[(size_t)r * cols + c] = (uint8_t)(int8_t)v;
        }
    (*n)++;
    tspec *sc = &T[*n];
    memset(sc, 0, sizeof *sc);
    snprintf(sc->name, sizeof sc->name, "%s.scale", name);
    sc->dtype = (uint32_t)SP_DT_FROBENIUS_SCALE_FP32; sc->n_dims = 1; sc->dims[0] = rows;
    sc->block_size = 4u; sc->size_bytes = (uint64_t)rows * 4u;
    sc->data = (uint8_t *)malloc((size_t)sc->size_bytes);
    if (!sc->data) return 1;
    float *sf = (float *)sc->data;
    for (uint32_t r = 0; r < rows; r++) sf[r] = 0.25f + (float)(r % 7u) * 0.03f;
    (*n)++;
    return 0;
}

static int add_f32(tspec *T, int *n, const char *name, uint32_t len, float base) {
    if (*n >= FX_MAXT) return 1;
    tspec *s = &T[*n];
    memset(s, 0, sizeof *s);
    snprintf(s->name, sizeof s->name, "%s", name);
    s->dtype = (uint32_t)SP_DT_F32; s->n_dims = 1; s->dims[0] = len;
    s->block_size = 4u; s->size_bytes = (uint64_t)len * 4u;
    s->data = (uint8_t *)malloc((size_t)s->size_bytes);
    if (!s->data) return 1;
    float *f = (float *)s->data;
    for (uint32_t i = 0; i < len; i++) f[i] = base + (float)(i % 5u) * 0.02f;
    (*n)++;
    return 0;
}

static int cmp_hash(const void *a, const void *b) {
    uint64_t ha = ((const tspec *)a)->name_hash, hb = ((const tspec *)b)->name_hash;
    return (ha < hb) ? -1 : (ha > hb) ? 1 : 0;
}

int sp_gemma4_fixture_build(uint8_t **model_buf, uint8_t **tok_buf,
                            sp_gemma4_fixture_info *info) {
    if (!model_buf || !tok_buf || !info) return 1;
    memset(info, 0, sizeof *info);
    info->n_layers = FX_NL; info->n_embd = FX_E; info->n_ff = FX_FF; info->n_vocab = FX_V;
    info->n_embd_per_layer = FX_PL;
    info->hd_swa = FX_HD_S; info->nh_swa = FX_NH_S; info->nkv_swa = FX_NKV_S;
    info->hd_global = FX_HD_G; info->nh_global = FX_NH_G; info->nkv_global = FX_NKV_G;
    info->swa_period = FX_PERIOD; info->n_kv_from_start = FX_KVFS; info->sliding_window = FX_SW;
    info->rope_base_global = 1e6f; info->rope_base_swa = 1e4f; info->logit_softcap = 30.0f;

    sp_arch_info *a = &info->arch;
    a->arch_id = (uint32_t)SP_ARCH_ID_GEMMA4; a->vocab_size = FX_V; a->hidden_dim = FX_E;
    a->n_layers = FX_NL; a->n_heads = FX_NH_G; a->n_kv_heads = FX_NKV_G; a->head_dim = FX_HD_G;
    a->max_context = 256u; a->swa_window = FX_SW; a->rope_freq_base = 1e6f;
    a->ffn_variant = 1u; a->norm_variant = 1u; a->tied_embeddings = 1u; a->has_qk_norm = 1u;
    a->n_ff = FX_FF; a->rms_eps = 1e-6f; a->preferred_precision = SP_PRECISION_FP16;
    a->g4_hd_swa = FX_HD_S; a->g4_nh_swa = FX_NH_S; a->g4_nkv_swa = FX_NKV_S;
    a->g4_rope_base_swa = 1e4f; a->g4_n_embd_per_layer = FX_PL;
    a->g4_n_kv_from_start = FX_KVFS; a->g4_logit_softcap = 30.0f; a->g4_swa_period = FX_PERIOD;

    /* tokenizer */
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

    tspec T[FX_MAXT];
    int n = 0, rc = 0;
    char nm[96];
    /* Q/K/V projection widths are PER-LAYER now (qd=nh*hd, kvd=nkv*hd with nh/nkv
     * constant and hd per-layer); computed inside the layer loop below. */

    rc |= add_q8 (T, &n, "token_embd.weight",          FX_V, FX_E, 1u);
    rc |= add_q8 (T, &n, "per_layer_token_embd.weight", FX_V, FX_PL * FX_NL, 2u);
    rc |= add_q8 (T, &n, "per_layer_model_proj.weight", FX_PL * FX_NL, FX_E, 3u);
    rc |= add_f32(T, &n, "per_layer_proj_norm.weight",  FX_PL, 0.9f);
    rc |= add_f32(T, &n, "rope_freqs.weight",           FX_HD_G / 2u, 1.0f);

    for (uint32_t L = 0; L < FX_NL && !rc; L++) {
        const int global = ((L % FX_PERIOD) == FX_PERIOD - 1u);
        const uint32_t hd  = global ? FX_HD_G  : FX_HD_S;
        const uint32_t nh  = global ? FX_NH_G  : FX_NH_S;   /* constant, but keep explicit */
        const uint32_t nkv = global ? FX_NKV_G : FX_NKV_S;
        const uint32_t qd  = nh * hd;     /* per-layer Q width  (SWA 32 / global 64) */
        const uint32_t kvd = nkv * hd;    /* per-layer KV width (SWA 16 / global 32) */
        const uint32_t s = (L + 1u) * 17u;
        snprintf(nm, sizeof nm, "blk.%u.attn_norm.weight",          L); rc |= add_f32(T, &n, nm, FX_E, 0.9f);
        snprintf(nm, sizeof nm, "blk.%u.attn_q.weight",             L); rc |= add_q8 (T, &n, nm, qd,  FX_E, s+1u);
        snprintf(nm, sizeof nm, "blk.%u.attn_k.weight",             L); rc |= add_q8 (T, &n, nm, kvd, FX_E, s+2u);
        snprintf(nm, sizeof nm, "blk.%u.attn_v.weight",             L); rc |= add_q8 (T, &n, nm, kvd, FX_E, s+3u);
        snprintf(nm, sizeof nm, "blk.%u.attn_output.weight",        L); rc |= add_q8 (T, &n, nm, FX_E, qd, s+4u);
        snprintf(nm, sizeof nm, "blk.%u.attn_q_norm.weight",        L); rc |= add_f32(T, &n, nm, hd, 0.95f);
        snprintf(nm, sizeof nm, "blk.%u.attn_k_norm.weight",        L); rc |= add_f32(T, &n, nm, hd, 0.95f);
        snprintf(nm, sizeof nm, "blk.%u.post_attention_norm.weight", L); rc |= add_f32(T, &n, nm, FX_E, 0.9f);
        snprintf(nm, sizeof nm, "blk.%u.ffn_norm.weight",           L); rc |= add_f32(T, &n, nm, FX_E, 0.9f);
        snprintf(nm, sizeof nm, "blk.%u.ffn_gate.weight",           L); rc |= add_q8 (T, &n, nm, FX_FF, FX_E, s+5u);
        snprintf(nm, sizeof nm, "blk.%u.ffn_up.weight",             L); rc |= add_q8 (T, &n, nm, FX_FF, FX_E, s+6u);
        snprintf(nm, sizeof nm, "blk.%u.ffn_down.weight",           L); rc |= add_q8 (T, &n, nm, FX_E, FX_FF, s+7u);
        snprintf(nm, sizeof nm, "blk.%u.post_ffw_norm.weight",      L); rc |= add_f32(T, &n, nm, FX_E, 0.9f);
        snprintf(nm, sizeof nm, "blk.%u.inp_gate.weight",           L); rc |= add_q8 (T, &n, nm, FX_PL, FX_E, s+8u);
        snprintf(nm, sizeof nm, "blk.%u.proj.weight",               L); rc |= add_q8 (T, &n, nm, FX_E, FX_PL, s+9u);
        snprintf(nm, sizeof nm, "blk.%u.post_norm.weight",          L); rc |= add_f32(T, &n, nm, FX_E, 0.9f);
        snprintf(nm, sizeof nm, "blk.%u.layer_output_scale.weight", L); rc |= add_f32(T, &n, nm, 1u, 0.9f);
    }
    rc |= add_f32(T, &n, "output_norm.weight", FX_E, 0.9f);
    if (rc) { for (int i = 0; i < n; i++) free(T[i].data); free(tk); return 1; }

    for (int i = 0; i < n; i++) T[i].name_hash = sp_xxh64(T[i].name, strlen(T[i].name), 0);
    qsort(T, (size_t)n, sizeof(tspec), cmp_hash);

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

    {
        sp_model_header h; memset(&h, 0, sizeof h);
        h.magic = SP_MODEL_MAGIC; h.version_major = SP_MODEL_VER_MAJOR;
        h.version_minor = SP_MODEL_VER_MINOR; h.header_size = SP_HEADER_SIZE;
        h.arch_id = a->arch_id;
        h.arch_struct_size = (uint32_t)sizeof(sp_arch_info);
        h.arch_struct_capacity = 256u;
        memcpy(h.arch_struct, a, sizeof *a);
        sp_sha256(tk, tok_len, h.tokenizer_hash);
        h.vocab_size = FX_V; h.tensor_count = (uint32_t)n;
        h.tensor_table_offset = table_off; h.tensor_data_offset = data_off;
        h.file_size = file_size; h.created_unix_seconds = 1700000002ull;
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

int sp_gemma4_fixture_write(const char *path, const uint8_t *buf, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f) return 1;
    size_t w = fwrite(buf, 1, len, f);
    return (fclose(f) != 0 || w != len) ? 1 : 0;
}
