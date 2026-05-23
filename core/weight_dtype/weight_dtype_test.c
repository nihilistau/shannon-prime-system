/* weight_dtype_test.c — T_WDTYPE: F16<->F32 conversion and per-row dequant invariants
 * for the relocated weight-dtype leaf (sp/weight_dtype.h). Hand-verifiable cases, no
 * model or fixtures. */
#include "sp/sp_test.h"
#include "sp/weight_dtype.h"
#include <string.h>

static int nearf(float a, float b, float tol) { float d = a - b; return (d < 0 ? -d : d) <= tol; }

static void WDTYPE(void) {
    /* Known half bit-patterns -> f32 (exact for these values). */
    SP_CHECK(sp_f16_to_f32(0x3C00u) == 1.0f,  "half 0x3C00 -> 1.0");
    SP_CHECK(sp_f16_to_f32(0x0000u) == 0.0f,  "half 0x0000 -> 0.0");
    SP_CHECK(sp_f16_to_f32(0xC000u) == -2.0f, "half 0xC000 -> -2.0");

    /* f16-representable values round-trip exactly f16 -> f32 -> f16. */
    static const uint16_t hs[4] = { 0x3C00u, 0xC000u, 0x3800u, 0x4900u };  /* 1, -2, 0.5, 10 */
    int rt = 1; for (int i = 0; i < 4; i++) if (sp_f32_to_f16(sp_f16_to_f32(hs[i])) != hs[i]) rt = 0;
    SP_CHECK(rt, "f16-representable values round-trip f16->f32->f16 exactly");

    /* Dequant of an F32 row is a byte passthrough. */
    {
        float in[4] = {1.5f, -2.0f, 3.25f, 0.0f}, out[4];
        SP_CHECK(sp_dequant_row(in, SP_WDT_F32, 4, out) == 0 && memcmp(in, out, sizeof in) == 0,
                 "dequant F32 row is a passthrough");
    }

    /* Dequant of an F16 row equals the elementwise half->single conversion. */
    {
        uint16_t in[3] = { 0x3C00u, 0x4000u, 0xC000u }; float out[3];   /* 1, 2, -2 */
        sp_dequant_row(in, SP_WDT_F16, 3, out);
        SP_CHECK(out[0] == 1.0f && out[1] == 2.0f && out[2] == -2.0f, "dequant F16 row -> {1,2,-2}");
    }

    /* Dequant a hand-built Q8_0 block (f16 scale 1.0, int8 codes 1..32) -> code*scale. */
    {
        uint8_t blk[34]; uint16_t d = sp_f32_to_f16(1.0f); memcpy(blk, &d, 2);
        for (int i = 0; i < 32; i++) ((int8_t *)blk)[2 + i] = (int8_t)(i + 1);
        float out[32]; sp_dequant_row(blk, SP_WDT_Q8_0, 32, out);
        int ok = 1; for (int i = 0; i < 32; i++) if (!nearf(out[i], (float)(i + 1), 1e-4f)) ok = 0;
        SP_CHECK(ok, "dequant Q8_0 block (scale 1.0) -> codes 1..32");
    }
}

int main(void) { SP_RUN(WDTYPE); return SP_DONE(); }
