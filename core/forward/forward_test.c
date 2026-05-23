/* forward_test.c — T_FORWARD: link + bad-arg smoke for the relocated qwen3 forward
 * orchestration (sp/model.h: qwen3_forward / qwen3_forward_ex / qwen3_generate_kv).
 * The forward composes the whole transformer over a loaded model, so it cannot be a
 * committed unit test — its real correctness is a logits round-trip against the engine's
 * forward (the §8.6.1 distributional floor / bit-exact in scalar mode), which the engine's
 * existing E_CPU_2 / GEN_KV regression re-runs against this implementation when the engine
 * consumes the migrated library. It was also exercised off-CI end-to-end on the real
 * Qwen3-0.6B (finite logits, sane single argmax through the full composed prefill). Here we
 * only confirm the module builds and links and the one documented bad-arg guard behaves.
 * Calling qwen3_generate_kv pulls this translation unit's object, link-exercising the whole
 * forward (its references to every migrated kernel and overlay codec must resolve). */
#include "sp/sp_test.h"
#include "sp/model.h"   /* qwen3_generate_kv and the forward prototypes */

static void FORWARD_SMOKE(void) {
    /* The only NULL-safe documented path: a NULL model is rejected with -1, not a crash. */
    int32_t dummy = 0;
    SP_CHECK_EQ_I64(qwen3_generate_kv(NULL, &dummy, 1, 0, -1), -1,
                    "qwen3_generate_kv(NULL model) -> -1 (and links the whole forward module)");
}

int main(void) { SP_RUN(FORWARD_SMOKE); return SP_DONE(); }
