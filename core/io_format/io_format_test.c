/* io_format_test.c — T_IO_FORMAT: the .sp-model / .sp-tokenizer format layer.
 *
 * Two halves:
 *   1. HEADER_LAYOUT — the #pragma pack(1) structs keep their exact spec sizes
 *      and field offsets (PPT-LAT-SP-MODEL-v0 §3/§4/§7).
 *   2. Runtime loader round-trips — the .sp-model loader (sp_model_load), the
 *      arch query (sp_model_arch), tensor lookup (sp_model_find_tensor), and the
 *      Spinor 0xA5 sentinel sweep (SP_ESPINOR_BADBLOCK). These run against a
 *      synthetic spec-conformant fixture built in-process (sp_model_fixture.c) so
 *      the gate is deterministic and dependency-free on every tier (no multi-GB
 *      real-model artifact; the real-model end-to-end load is presence-gated and
 *      tracked in the offload note). All status codes are the FROZEN L1 ABI codes
 *      (sp/sp_status.h), never ad-hoc per-failure codes. */
#include "sp/sp_test.h"
#include "sp/sp_model.h"
#include "sp/sp_l1.h"
#include "sp/sp_hash.h"
#include "sp_model_fixture.h"
#include <stdlib.h>
#include <string.h>

static void HEADER_LAYOUT(void) {
    SP_CHECK(sizeof(sp_model_header) == 512, "sizeof sp_model_header == 512");
    SP_CHECK(sizeof(sp_tensor_entry) == 256, "sizeof sp_tensor_entry == 256");
    SP_CHECK(sizeof(sp_tok_header) == 128, "sizeof sp_tok_header == 128");
    sp_model_header h; memset(&h, 0, sizeof h);
    SP_CHECK((size_t)((uint8_t *)&h.tokenizer_hash - (uint8_t *)&h) == 280, "tokenizer_hash @280");
    SP_CHECK((size_t)((uint8_t *)&h.header_crc32 - (uint8_t *)&h) == 360, "header_crc32 @360");
    sp_tensor_entry e; memset(&e, 0, sizeof e);
    SP_CHECK((size_t)((uint8_t *)&e.name_hash - (uint8_t *)&e) == 208, "name_hash @208");
    SP_CHECK((size_t)((uint8_t *)&e.blake3 - (uint8_t *)&e) == 176, "blake3 @176");
}

/* Build a fresh fixture and write the (optionally pre-mutated) image to disk.
 * The caller mutates *mb / *tb between build and write for negative cases. */
#define FX_BLOCKS 4u

static void T_IO_FORMAT_LOAD_OK(void) {
    sp_fixture_info info; uint8_t *mb = NULL, *tb = NULL;
    SP_CHECK(sp_fixture_build(&mb, &tb, FX_BLOCKS, &info) == 0, "fixture build");
    SP_CHECK(sp_fixture_write_file("fx_ok.spm", mb, info.model_len) == 0, "write model");
    SP_CHECK(sp_fixture_write_file("fx_ok.spt", tb, info.tok_len) == 0, "write tokenizer");

    /* Deterministic for a given FX_BLOCKS — recorded in the HANDLE offload note. */
    uint8_t dg[32]; sp_sha256(mb, info.model_len, dg);
    fprintf(stderr, "    [fixture] .sp-model SHA-256 (FX_BLOCKS=%u, %zu bytes): ",
            (unsigned)FX_BLOCKS, info.model_len);
    for (unsigned i = 0; i < 32u; i++) fprintf(stderr, "%02x", (unsigned)dg[i]);
    fprintf(stderr, "\n");

    sp_model *m = NULL;
    sp_status st = sp_model_load("fx_ok.spm", "fx_ok.spt", &m);
    SP_CHECK_EQ_I64(st, SP_OK, "load valid -> SP_OK");
    SP_CHECK(m != NULL, "handle non-null");
    if (m) {
        SP_CHECK(sp_model_tensor_count(m) == 2u, "tensor_count == 2");
        sp_model_unload(m);
    }
    free(mb); free(tb);
    remove("fx_ok.spm"); remove("fx_ok.spt");
}

static void T_IO_FORMAT_BAD_MAGIC(void) {
    sp_fixture_info info; uint8_t *mb = NULL, *tb = NULL;
    sp_fixture_build(&mb, &tb, FX_BLOCKS, &info);
    mb[0] ^= 0xFFu;                         /* clobber "SPMD" magic */
    sp_fixture_write_file("fx_bm.spm", mb, info.model_len);
    sp_fixture_write_file("fx_bm.spt", tb, info.tok_len);

    sp_model *m = NULL;
    sp_status st = sp_model_load("fx_bm.spm", "fx_bm.spt", &m);
    SP_CHECK_EQ_I64(st, SP_EBADFORMAT, "bad magic -> SP_EBADFORMAT");
    SP_CHECK(m == NULL, "no handle on error");
    free(mb); free(tb);
    remove("fx_bm.spm"); remove("fx_bm.spt");
}

static void T_IO_FORMAT_BAD_VER(void) {
    sp_fixture_info info; uint8_t *mb = NULL, *tb = NULL;
    sp_fixture_build(&mb, &tb, FX_BLOCKS, &info);
    mb[4] = 1u;                             /* version_major (u16 @4) -> 1, unsupported */
    sp_fixture_write_file("fx_bv.spm", mb, info.model_len);
    sp_fixture_write_file("fx_bv.spt", tb, info.tok_len);

    sp_model *m = NULL;
    sp_status st = sp_model_load("fx_bv.spm", "fx_bv.spt", &m);
    SP_CHECK_EQ_I64(st, SP_EBADFORMAT, "bad version_major -> SP_EBADFORMAT");
    free(mb); free(tb);
    remove("fx_bv.spm"); remove("fx_bv.spt");
}

static void T_IO_FORMAT_BAD_CRC(void) {
    sp_fixture_info info; uint8_t *mb = NULL, *tb = NULL;
    sp_fixture_build(&mb, &tb, FX_BLOCKS, &info);
    /* Corrupt a CRC-covered byte (created_unix_seconds @344, inside [0,360)).
     * NOT the reserved tail @>=364 — that is outside header_crc32 coverage. */
    mb[344] ^= 0xFFu;
    sp_fixture_write_file("fx_bc.spm", mb, info.model_len);
    sp_fixture_write_file("fx_bc.spt", tb, info.tok_len);

    sp_model *m = NULL;
    sp_status st = sp_model_load("fx_bc.spm", "fx_bc.spt", &m);
    SP_CHECK_EQ_I64(st, SP_EBADFORMAT, "header CRC mismatch -> SP_EBADFORMAT");
    free(mb); free(tb);
    remove("fx_bc.spm"); remove("fx_bc.spt");
}

static void T_IO_FORMAT_TOK_HASH(void) {
    sp_fixture_info info; uint8_t *mb = NULL, *tb = NULL;
    sp_fixture_build(&mb, &tb, FX_BLOCKS, &info);
    tb[info.tok_blob_offset] ^= 0xFFu;      /* corrupt tokenizer body -> SHA-256 drifts */
    sp_fixture_write_file("fx_th.spm", mb, info.model_len);
    sp_fixture_write_file("fx_th.spt", tb, info.tok_len);

    sp_model *m = NULL;
    sp_status st = sp_model_load("fx_th.spm", "fx_th.spt", &m);
    SP_CHECK_EQ_I64(st, SP_ETOKENIZER_HASH, "tokenizer SHA mismatch -> SP_ETOKENIZER_HASH");
    free(mb); free(tb);
    remove("fx_th.spm"); remove("fx_th.spt");
}

static void T_IO_FORMAT_FIND(void) {
    sp_fixture_info info; uint8_t *mb = NULL, *tb = NULL;
    sp_fixture_build(&mb, &tb, FX_BLOCKS, &info);
    sp_fixture_write_file("fx_fd.spm", mb, info.model_len);
    sp_fixture_write_file("fx_fd.spt", tb, info.tok_len);

    sp_model *m = NULL;
    sp_status st = sp_model_load("fx_fd.spm", "fx_fd.spt", &m);
    SP_CHECK_EQ_I64(st, SP_OK, "load -> SP_OK");
    if (m) {
        const sp_tensor_entry *hit  = sp_model_find_tensor(m, info.f32_tensor_name);
        const sp_tensor_entry *hit2 = sp_model_find_tensor(m, info.spinor_tensor_name);
        const sp_tensor_entry *miss = sp_model_find_tensor(m, "no.such.tensor");
        SP_CHECK(hit  != NULL, "find hit (f32) -> non-NULL");
        SP_CHECK(hit2 != NULL, "find hit (spinor) -> non-NULL");
        SP_CHECK(miss == NULL, "find miss -> NULL");
        if (hit) SP_CHECK(strcmp(hit->name, info.f32_tensor_name) == 0, "hit name matches");
        sp_model_unload(m);
    }
    free(mb); free(tb);
    remove("fx_fd.spm"); remove("fx_fd.spt");
}

static void T_IO_FORMAT_SPINOR_OK(void) {
    sp_fixture_info info; uint8_t *mb = NULL, *tb = NULL;
    sp_fixture_build(&mb, &tb, FX_BLOCKS, &info);
    sp_fixture_write_file("fx_so.spm", mb, info.model_len);
    sp_fixture_write_file("fx_so.spt", tb, info.tok_len);

    sp_model *m = NULL;
    sp_status st = sp_model_load("fx_so.spm", "fx_so.spt", &m);
    SP_CHECK_EQ_I64(st, SP_OK, "all sentinels 0xA5 -> SP_OK");
    if (m) sp_model_unload(m);
    free(mb); free(tb);
    remove("fx_so.spm"); remove("fx_so.spt");
}

static void T_IO_FORMAT_SPINOR_BAD(void) {
    sp_fixture_info info; uint8_t *mb = NULL, *tb = NULL;
    sp_fixture_build(&mb, &tb, FX_BLOCKS, &info);
    /* Clobber the LAST Spinor block's sentinel -> caught by the default
     * first+last sample inside sp_model_load. */
    size_t off = sp_fixture_spinor_sentinel_at(&info, info.spinor_block_count - 1u);
    mb[off] = 0x00u;
    sp_fixture_write_file("fx_sb.spm", mb, info.model_len);
    sp_fixture_write_file("fx_sb.spt", tb, info.tok_len);

    sp_model *m = NULL;
    sp_status st = sp_model_load("fx_sb.spm", "fx_sb.spt", &m);
    SP_CHECK_EQ_I64(st, SP_ESPINOR_BADBLOCK, "bad sentinel -> SP_ESPINOR_BADBLOCK");
    SP_CHECK(m == NULL, "no handle on error");
    free(mb); free(tb);
    remove("fx_sb.spm"); remove("fx_sb.spt");
}

static void T_IO_FORMAT_ARCH(void) {
    sp_fixture_info info; uint8_t *mb = NULL, *tb = NULL;
    sp_fixture_build(&mb, &tb, FX_BLOCKS, &info);
    sp_fixture_write_file("fx_ar.spm", mb, info.model_len);
    sp_fixture_write_file("fx_ar.spt", tb, info.tok_len);

    sp_model *m = NULL;
    sp_status st = sp_model_load("fx_ar.spm", "fx_ar.spt", &m);
    SP_CHECK_EQ_I64(st, SP_OK, "load -> SP_OK");
    if (m) {
        sp_arch_info got; memset(&got, 0, sizeof got);
        sp_status as = sp_model_arch(m, &got);
        SP_CHECK_EQ_I64(as, SP_OK, "sp_model_arch -> SP_OK");
        SP_CHECK(memcmp(&got, &info.arch, sizeof got) == 0, "arch byte-identity vs embedded payload");
        SP_CHECK_EQ_I64(got.arch_id, SP_ARCH_ID_GEMMA3, "arch_id == GEMMA3");
        SP_CHECK_EQ_I64(got.vocab_size, 256, "vocab_size == 256");
        SP_CHECK_EQ_I64(got.n_layers, 4, "n_layers == 4");
        SP_CHECK_EQ_I64(sp_model_arch(m, NULL), SP_EBADARG, "null out -> SP_EBADARG");
        sp_model_unload(m);
    }
    sp_arch_info tmp;
    SP_CHECK_EQ_I64(sp_model_arch(NULL, &tmp), SP_EBADARG, "null handle -> SP_EBADARG");
    free(mb); free(tb);
    remove("fx_ar.spm"); remove("fx_ar.spt");
}

static void T_IO_FORMAT_SPINOR_FULL(void) {
    sp_fixture_info info; uint8_t *mb = NULL, *tb = NULL;
    sp_fixture_build(&mb, &tb, FX_BLOCKS, &info);   /* blocks 0,1,2,3 */
    /* Clobber a MIDDLE block (index 2): the default first+last sample misses it;
     * the full sweep must catch it. Proves both flag states of the sweep. */
    size_t off = sp_fixture_spinor_sentinel_at(&info, 2u);
    mb[off] = 0x00u;
    sp_fixture_write_file("fx_sf.spm", mb, info.model_len);
    sp_fixture_write_file("fx_sf.spt", tb, info.tok_len);

    sp_model *m = NULL;
    sp_status st = sp_model_load("fx_sf.spm", "fx_sf.spt", &m);
    SP_CHECK_EQ_I64(st, SP_OK, "mid-block clobber: default load passes (samples first+last)");
    if (m) {
        SP_CHECK_EQ_I64(sp_model_verify_spinors(m, 0), SP_OK, "verify(full=0) misses mid block");
        SP_CHECK_EQ_I64(sp_model_verify_spinors(m, 1), SP_ESPINOR_BADBLOCK, "verify(full=1) catches mid block");
        sp_model_unload(m);
    }
    free(mb); free(tb);
    remove("fx_sf.spm"); remove("fx_sf.spt");
}

int main(void) {
    SP_RUN(HEADER_LAYOUT);
    SP_RUN(T_IO_FORMAT_LOAD_OK);
    SP_RUN(T_IO_FORMAT_BAD_MAGIC);
    SP_RUN(T_IO_FORMAT_BAD_VER);
    SP_RUN(T_IO_FORMAT_BAD_CRC);
    SP_RUN(T_IO_FORMAT_TOK_HASH);
    SP_RUN(T_IO_FORMAT_FIND);
    SP_RUN(T_IO_FORMAT_SPINOR_OK);
    SP_RUN(T_IO_FORMAT_SPINOR_BAD);
    SP_RUN(T_IO_FORMAT_ARCH);
    SP_RUN(T_IO_FORMAT_SPINOR_FULL);
    return SP_DONE();
}
