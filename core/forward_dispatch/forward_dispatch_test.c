/* forward_dispatch_test.c — T_FWD_DISPATCH: link + gate-knob smoke for the relocated
 * weight-lift kernels (sp/forward_dispatch.h). sp_matmul / sp_embed_row / sp_as_f32 all
 * take a whole qwen3_model and read a GGUF mapping or the packed arena, so their real
 * (bit-exact) gate needs a model and is run off-CI — an inline scalar-reference parity
 * check of the pure-f32 sp_matmul path and sp_embed_row against a loaded Qwen3-0.6B (the
 * inline reference equals the engine's scalar matmul by construction), with the arena /
 * Frob / F16-activation paths sealed by the forward round-trip. The committed test stays
 * fixture-free: it confirms the module builds and links and the gate-knob state behaves. */
#include "sp/sp_test.h"
#include "sp/forward_dispatch.h"   /* sp_kernels_read_env (+ sp/model.h for qwen3_q4_stats) */

static void FWD_DISPATCH_SMOKE(void) {
    /* Reading the env-knob state is side-effect-only and resets the Q4 calibration
     * counters; afterwards the stats read back zeroed. This also link-exercises the
     * module (sp_matmul / sp_embed_row / sp_as_f32 symbols must resolve to build it). */
    sp_kernels_read_env();
    long promoted = -1, rows = -1;
    qwen3_q4_stats(&promoted, &rows);
    SP_CHECK_EQ_I64(promoted, 0, "Q4 promotion count is zero after a fresh env read");
    SP_CHECK_EQ_I64(rows, 0,     "Q4 rows-seen count is zero after a fresh env read");
}

int main(void) { SP_RUN(FWD_DISPATCH_SMOKE); return SP_DONE(); }
