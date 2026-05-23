/* gguf_test.c — T_GGUF: link + trivial-path smoke for the relocated GGUF v3 parser.
 * Deep parse correctness is exercised by the engine's GGUF-loading regression and the
 * eventual forward round-trip; here we only confirm the parser builds and links in the
 * math core and its documented error-by-NULL and type-name paths behave. */
#include "sp/sp_test.h"
#include "sp/gguf.h"

static void GGUF_SMOKE(void) {
    /* Opening a nonexistent file fails gracefully (the documented error-by-NULL path). */
    SP_CHECK(gguf_open("definitely-no-such-file-12345.gguf") == NULL, "gguf_open(bad path) -> NULL");
    /* ggml_type_name maps an on-disk dtype id to a non-empty name. */
    const char *fn = ggml_type_name((uint32_t)GGML_T_F32);
    SP_CHECK(fn != NULL && fn[0] != '\0', "ggml_type_name(F32) is a non-empty name");
}

int main(void) { SP_RUN(GGUF_SMOKE); return SP_DONE(); }
