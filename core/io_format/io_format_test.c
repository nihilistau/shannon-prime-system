/* io_format_test.c — T_IO_FORMAT: the frozen .sp-model byte layout. The structural
 * half of the FMT format gate (the part the hash test shed): the #pragma pack(1)
 * structs must keep their exact spec sizes and field offsets after the move into the
 * math core. Pure layout assertions; the loader's runtime round-trip stays with the
 * model layer above (the FMT round-trip is engine-side). Linking the io_format
 * library here also proves the relocated loader + error compile and link cleanly
 * against the math-core hash dependency. */
#include "sp/sp_test.h"
#include "sp/sp_model.h"
#include <string.h>

static void HEADER_LAYOUT(void) {
    /* The pragma-packed structs must match the spec offsets exactly (PPT-LAT-SP-MODEL-v0 §3/§4/§7). */
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

int main(void) { SP_RUN(HEADER_LAYOUT); return SP_DONE(); }
