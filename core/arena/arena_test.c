/* arena_test.c — T_ARENA: container-path invariants for the relocated packed-weight
 * arena (sp/arena.h). Exercises the .sp-model "adopt already-packed" path entirely
 * with a hand-built synthetic packed tensor — no model, no GGUF, no fixtures: pack a
 * tiny known matrix with the real Frobenius codec, adopt two named copies into an
 * arena, then check name lookup, the precision/row/byte accounting, the inline
 * dequant lift round-trip, and the ownership-transfer free path. The builder half
 * (sp_arena_build over a loaded qwen3_model) is compiled here but its correctness
 * gate is the forward round-trip, which needs a real model. */
#include "sp/sp_test.h"
#include "sp/arena.h"
#include "sp/frobenius_lift.h"
#include <string.h>

/* Synthetic weight: 2 rows x 4 cols, known per-row max-abs (the Frobenius scale):
 * row 0 max-abs 2.0, row 1 max-abs 3.0. The Q8 round-trip error of any element is
 * bounded by half a code step = scale/254 < scale/127, so scale/127 is a safe tol. */
#define ROWS 2
#define COLS 4
static const float WEIGHT[ROWS][COLS] = {
    { 1.0f, -2.0f, 0.5f,  0.0f },
    { 3.0f,  0.25f, -1.0f, 2.0f },
};
static const float ROW_SCALE[ROWS] = { 2.0f, 3.0f };

static int weight_get_row(void *ctx, int j, float *dst) {
    (void)ctx;
    memcpy(dst, WEIGHT[j], COLS * sizeof(float));
    return 0;
}

static int nearf(float a, float b, float tol) { float d = a - b; return (d < 0 ? -d : d) <= tol; }

static void ARENA_CONTAINER(void) {
    /* Pack the synthetic matrix Q8 twice into two independent packed tensors so each
     * owns its own buffers (the arena adopts both by value; no aliasing / double-free). */
    sp_frob_packed_tensor pt0 = {0}, pt1 = {0};
    SP_CHECK(sp_frob_pack_tensor(ROWS, COLS, 8, 0.0f, weight_get_row, NULL, &pt0, NULL) == 0,
             "frobenius pack of synthetic weight (slot 0) succeeds");
    SP_CHECK(sp_frob_pack_tensor(ROWS, COLS, 8, 0.0f, weight_get_row, NULL, &pt1, NULL) == 0,
             "frobenius pack of synthetic weight (slot 1) succeeds");
    size_t want_bytes = sp_frob_packed_tensor_bytes(&pt0) + sp_frob_packed_tensor_bytes(&pt1);

    sp_arena_tensor ts[2];
    memset(ts, 0, sizeof ts);
    snprintf(ts[0].name, sizeof ts[0].name, "%s", "blk.0.ffn_gate.weight"); ts[0].pt = pt0;
    snprintf(ts[1].name, sizeof ts[1].name, "%s", "blk.0.ffn_up.weight");   ts[1].pt = pt1;

    /* Adopt into an arena — the arena now owns the pt buffers (we must not free them). */
    sp_arena *a = sp_arena_from_packed(ts, 2, 8);
    SP_CHECK(a != NULL, "sp_arena_from_packed adopts two named packed tensors");

    /* Accounting: precision passthrough, total rows summed, bytes summed, no Q4 promotions. */
    SP_CHECK_EQ_I64(sp_arena_precision(a), 8,            "arena precision recorded as 8");
    SP_CHECK_EQ_I64(sp_arena_total_rows(a), (long)(2 * ROWS), "arena total_rows = sum of the two tensors' rows");
    SP_CHECK_EQ_I64(sp_arena_promoted(a), 0,             "Q8 arena has no promoted rows");
    SP_CHECK(sp_arena_bytes(a) == want_bytes,            "arena bytes = sum of the adopted tensors' packed bytes");

    /* Name lookup: hits the right slot, misses cleanly. */
    const sp_arena_tensor *up = sp_arena_find(a, "blk.0.ffn_up.weight");
    SP_CHECK(up != NULL && up->pt.rows == ROWS && up->pt.cols == COLS, "find returns the named slot's packed tensor");
    SP_CHECK(sp_arena_find(a, "no.such.tensor") == NULL, "find of an unknown name -> NULL");

    /* Inline dequant lift round-trips row 0 within the Q8 code-step tolerance. */
    float out[COLS];
    SP_CHECK(sp_arena_dequant_row(up, 0, out) == 0, "dequant of an adopted row succeeds");
    float tol0 = ROW_SCALE[0] / (float)SP_FROB_QMAX;   /* one Q8 code step on row 0's scale */
    int rt = 1; for (int c = 0; c < COLS; c++) if (!nearf(out[c], WEIGHT[0][c], tol0)) rt = 0;
    SP_CHECK(rt, "dequant lift reconstructs row 0 within one Q8 code step of the source");

    /* Frees the adopted buffers exactly once — the originals' pointers now live only
     * inside the arena, so we deliberately do NOT sp_frob_packed_free pt0/pt1. */
    sp_arena_free(a);

    /* Documented bad-arg paths return NULL / 0 without crashing. */
    SP_CHECK(sp_arena_from_packed(NULL, 0, 8) == NULL, "from_packed(NULL) -> NULL");
    SP_CHECK(sp_arena_from_packed(ts, 2, 5) == NULL,   "from_packed with a bad precision -> NULL");
    SP_CHECK(sp_arena_precision(NULL) == 0,            "accessors on NULL arena are zero-valued");
}

int main(void) { SP_RUN(ARENA_CONTAINER); return SP_DONE(); }
