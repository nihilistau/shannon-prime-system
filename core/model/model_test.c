/* model_test.c — T_MODEL: link + trivial-path smoke for the relocated model-load
 * lifecycle (sp/model.h: qwen3_load / qwen3_free / qwen3_release_source). Like the
 * GGUF module's smoke, deep correctness — that a real Qwen3/Gemma3 GGUF binds to the
 * right config and tensor table — cannot be a committed unit test (it needs a multi-
 * hundred-MB model fixture absent from CI). That proof lives in (1) an off-CI real
 * load-smoke run by hand against a local Qwen3-0.6B GGUF and (2) the eventual forward
 * round-trip against the engine oracle. Here we only confirm the binding builds and
 * links in the math core and its documented error paths behave. */
#include "sp/sp_test.h"
#include "sp/model.h"

static void MODEL_SMOKE(void) {
    /* A non-existent / non-GGUF path fails gracefully (gguf_open -> NULL -> qwen3_load NULL). */
    SP_CHECK(qwen3_load("definitely-no-such-model-12345.gguf") == NULL,
             "qwen3_load(bad path) -> NULL");
    /* Freeing NULL and freeing a never-allocated model are documented no-ops (no crash). */
    qwen3_free(NULL);
    /* release_source on a NULL model is a defined error return, not a crash. */
    SP_CHECK(qwen3_release_source(NULL) == 1, "qwen3_release_source(NULL) -> error (1)");
}

int main(void) { SP_RUN(MODEL_SMOKE); return SP_DONE(); }
