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
#include "sp/poly_ring.h"          /* NTT.5c Stage 2: direct sp_pr admission */
#include "sp/poly_ring_bluestein.h"/* NTT.5c Stage 2: Bluestein admission */

#include <stdio.h>     /* snprintf */

static void FORWARD_SMOKE(void) {
    /* The NULL-safe documented paths: a NULL model is rejected with -1, not a crash. Calling
     * both generation entry points pulls their objects (and transitively the prefill/forward
     * they drive), link-exercising the whole forward module against the migrated closure. */
    int32_t dummy = 0;
    SP_CHECK_EQ_I64(qwen3_generate_kv(NULL, &dummy, 1, 0, -1), -1,
                    "qwen3_generate_kv(NULL model) -> -1 (persistent-KV decode guard)");
    SP_CHECK_EQ_I64(qwen3_generate(NULL, &dummy, 1, 0, -1), -1,
                    "qwen3_generate(NULL model) -> -1 (greedy O(n^2) generate guard)");
}

/* T_NTT5C_HD_256_NO_REGRESSION (NTT.5c Stage 2 gate).
 *
 * Validates that the NTT.5c HD-dispatch logic (forward.c + qwen25.c overlay
 * init block) correctly picks DIRECT sp_pr_init for HD ∈ {128, 256, 512}
 * and Bluestein sp_pr_bluestein_init for HD ∈ {2..256, ≠ 512}. This is the
 * "no regression at the direct path" gate: HD=128/256/512 still hit the
 * pre-NTT.5c code path (direct sp_pr_init returns valid ctx; pr_b stays
 * NULL; the inner loop dispatches to sp_pr_inner unchanged).
 *
 * Methodology: exercise the admission decisions directly. The dispatch
 * branch in forward.c/qwen25.c is:
 *   if (HD == 128 || HD == 256 || HD == 512) pr = sp_pr_init(HD);
 *   else if (HD >= 2 && HD <= 256 && PoT(HD))  pr_b = sp_pr_bluestein_init(HD);
 *
 * We confirm that the underlying init functions admit/reject as the
 * dispatch logic assumes. This is a structural/regression gate; the actual
 * end-to-end HD=128/256 forward via real model is covered by the engine's
 * E_CPU_2 / GEN_KV regression suite (per closure-style language above).
 *
 * Per feedback-no-silent-gate-revisions: if any admission disagrees, the
 * NTT.5c HD-dispatch logic is wrong (would route a usable HD to the
 * disabled overlay branch). Surface UPSTREAM. */
static void T_NTT5C_HD_256_NO_REGRESSION(void) {
    /* Direct sp_pr_init admits HD ∈ {128, 256, 512} (unchanged from NTT.4). */
    sp_pr_ctx *pr128 = sp_pr_init(128); SP_CHECK(pr128 != NULL, "sp_pr_init(128) admits (direct path)");
    sp_pr_ctx *pr256 = sp_pr_init(256); SP_CHECK(pr256 != NULL, "sp_pr_init(256) admits (direct path)");
    sp_pr_ctx *pr512 = sp_pr_init(512); SP_CHECK(pr512 != NULL, "sp_pr_init(512) admits (direct path)");
    sp_pr_free(pr128); sp_pr_free(pr256); sp_pr_free(pr512);

    /* Bluestein admits the {2..256} ∖ {512} set (NTT.5a closure §"Admissible-N analysis"). */
    static const uint32_t kPoT[8] = {2, 4, 8, 16, 32, 64, 128, 256};
    for (int i = 0; i < 8; i++) {
        sp_pr_bluestein_ctx *b = sp_pr_bluestein_init(kPoT[i]);
        char msg[80]; snprintf(msg, sizeof msg, "sp_pr_bluestein_init(%u) admits (Bluestein path)", kPoT[i]);
        SP_CHECK(b != NULL, msg);
        sp_pr_bluestein_free(b);
    }
    /* Bluestein rejects 512 (would need M=1024; frozen primes can't), as expected. */
    sp_pr_bluestein_ctx *b512 = sp_pr_bluestein_init(512);
    SP_CHECK(b512 == NULL, "sp_pr_bluestein_init(512) returns NULL (frozen-prime cap; direct path handles 512)");
    sp_pr_bluestein_free(b512);

    /* Non-power-of-2 rejected by both paths — HD with odd factors silently
     * disables overlay (overlay_active stays 0 → fp32 fallback per NTT.5c
     * forward.c logic). */
    static const uint32_t kOdd[4] = {96, 192, 288, 384};
    for (int i = 0; i < 4; i++) {
        SP_CHECK(sp_pr_init(kOdd[i]) == NULL, "sp_pr_init rejects non-{128,256,512}");
        SP_CHECK(sp_pr_bluestein_init(kOdd[i]) == NULL, "sp_pr_bluestein_init rejects non-power-of-2");
    }
}

int main(void) {
    SP_RUN(FORWARD_SMOKE);
    SP_RUN(T_NTT5C_HD_256_NO_REGRESSION);
    return SP_DONE();
}
