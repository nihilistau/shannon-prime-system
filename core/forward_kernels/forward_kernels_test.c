/* forward_kernels_test.c — T_FWDK: hand-verifiable algebraic invariants of the
 * portable scalar reference kernels (sp/forward_kernels.h). Cheap properties, no
 * model or fixtures. */
#include "sp/sp_test.h"
#include "sp/forward_kernels.h"
#include <math.h>

static int nearf(float a, float b, float tol) { float d = a - b; return (d < 0 ? -d : d) <= tol; }

static void T_FWDK(void) {
    /* Dot of small integers is exact in float. */
    {
        const float a[3] = {1, 2, 3}, b[3] = {4, 5, 6};
        SP_CHECK(sp_dot_f32(a, b, 3) == 32.0f, "sp_dot_f32 exact on small integers (1*4+2*5+3*6=32)");
    }

    /* RMSNorm with unit weights yields an output whose RMS is ~1. */
    {
        float x[4] = {1.0f, -2.0f, 3.0f, -4.0f}, w[4] = {1, 1, 1, 1}, out[4];
        sp_rmsnorm(x, w, 4, 1e-6f, out);
        double ss = 0; for (int i = 0; i < 4; i++) ss += (double)out[i] * out[i];
        SP_CHECK(nearf((float)sqrt(ss / 4.0), 1.0f, 1e-3f), "sp_rmsnorm: unit-weight output RMS ~ 1");
    }

    /* NEOX RoPE preserves the magnitude of each rotated coordinate pair. */
    {
        const int d = 8; float v[8] = {0.5f, -1.0f, 2.0f, 0.25f, 1.5f, -0.5f, 0.75f, -2.0f}, v0[8];
        for (int i = 0; i < d; i++) v0[i] = v[i];
        sp_rope_neox(v, d, 7, 10000.0f);
        int half = d / 2, ok = 1;
        for (int i = 0; i < half; i++) {
            float m0 = sqrtf(v0[i] * v0[i] + v0[i + half] * v0[i + half]);
            float m1 = sqrtf(v[i] * v[i] + v[i + half] * v[i + half]);
            if (!nearf(m0, m1, 1e-4f)) ok = 0;
        }
        SP_CHECK(ok, "sp_rope_neox is norm-preserving per coordinate pair");
    }

    /* Single-position attention: softmax over one score is 1, so out == that V head. */
    {
        const int HD = 4, KVD = 4; float qh[4] = {1, 1, 1, 1};
        float KC[4] = {0.1f, 0.2f, 0.3f, 0.4f}, VC[4] = {9.0f, -8.0f, 7.0f, -6.0f};
        float sc[1], out[4];
        sp_attn_head(qh, KC, VC, /*pos*/0, KVD, /*kvh*/0, HD, /*ascale*/0.5f, /*win*/-1, sc, out);
        int ok = 1; for (int i = 0; i < HD; i++) if (!nearf(out[i], VC[i], 1e-4f)) ok = 0;
        SP_CHECK(ok, "sp_attn_head at pos 0 returns the sole position's V (softmax of one = 1)");
    }
}

int main(void) { SP_RUN(T_FWDK); return SP_DONE(); }
