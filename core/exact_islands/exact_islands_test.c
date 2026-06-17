/* exact_islands_test.c -- T_EXACT_ISLANDS: gate the three exact-integer fp32-island
 * kernels (sp/exact_islands.h) on the two properties the BYTE-EXACT contract names:
 *   (1) FIDELITY  -- each integer kernel matches its float reference to ~1e-5..1e-6
 *       (lossless for inference; the §3 prototype table).
 *   (2) BYTE-EXACTNESS -- the reductions are reduction-order-immune, so feeding the
 *       same data in a permuted order yields a BIT-IDENTICAL scale/denominator
 *       (proved by exact float == on the correspondingly-permuted outputs). This is
 *       the property the float bridges (1/sqrt, exp in float) lack.
 * No model, no fixtures -- a deterministic LCG vector, hand-checkable. */
#include "sp/sp_test.h"
#include "sp/exact_islands.h"
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* deterministic uniform in [-rng, rng] (no libc rand, fully reproducible). */
static uint64_t g_s = 0x9e3779b97f4a7c15ULL;
static double udev(double rng) {
    g_s = g_s * 6364136223846793005ULL + 1442695040888963407ULL;
    double u = (double)(g_s >> 11) / (double)(1ULL << 53);   /* [0,1) */
    return (2.0 * u - 1.0) * rng;
}
static double relerr(const double *a, const double *b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) { double d = a[i] - b[i]; num += d * d; den += a[i] * a[i]; }
    return sqrt(num) / (sqrt(den) + 1e-30);
}

static void T_EXACT_ISLANDS(void) {
    enum { E = 3840, M = 256 };
    static float x[E], w[E], outi[E], outr[E];
    static double yf[E], yi[E];

    /* ---------- Island 1: RMSNorm ---------- */
    g_s = 1;
    for (int i = 0; i < E; i++) { x[i] = (float)udev(2.0); w[i] = (float)(1.0 + udev(0.1)); }
    /* float reference: out = x * (1/sqrt(mean x^2)) * w (eps omitted, == exact route) */
    double ss = 0; for (int i = 0; i < E; i++) ss += (double)x[i] * x[i];
    double scale = 1.0 / sqrt(ss / (double)E);
    for (int i = 0; i < E; i++) yf[i] = (double)x[i] * scale * (double)w[i];
    sp_rmsnorm_exact(x, w, E, outi);
    for (int i = 0; i < E; i++) yi[i] = outi[i];
    SP_CHECK(relerr(yi, yf, E) < 1e-4, "RMSNorm exact-integer matches float (relerr < 1e-4)");

    /* reduction-order immunity: reversed input, unit weight -> outputs must reverse
     * BIT-IDENTICALLY (only true if sum x^2, hence the scale, was bit-identical). */
    {
        static float xr[E], wone[E], oa[E];
        for (int i = 0; i < E; i++) { wone[i] = 1.0f; xr[i] = x[E - 1 - i]; }
        sp_rmsnorm_exact(x,  wone, E, oa);
        sp_rmsnorm_exact(xr, wone, E, outr);
        int ok = 1; for (int i = 0; i < E; i++) if (outr[i] != oa[E - 1 - i]) ok = 0;
        SP_CHECK(ok, "RMSNorm scale is reduction-order-immune (byte-identical under permutation)");
    }

    /* ---------- Island 2: softmax ---------- */
    {
        static float z[M], zr[M];
        static double pf[M], pi[M], pr[M];
        g_s = 7;
        for (int i = 0; i < M; i++) z[i] = (float)udev(8.0);
        double mx = -1e30; for (int i = 0; i < M; i++) if (z[i] > mx) mx = z[i];
        double se = 0; for (int i = 0; i < M; i++) { pf[i] = exp((double)z[i] - mx); se += pf[i]; }
        for (int i = 0; i < M; i++) pf[i] /= se;
        sp_softmax_exact(z, M, pi);
        double mad = 0; for (int i = 0; i < M; i++) { double d = fabs(pf[i] - pi[i]); if (d > mad) mad = d; }
        SP_CHECK(mad < 1e-5, "softmax exact-integer matches float (max|dp| < 1e-5)");

        for (int i = 0; i < M; i++) zr[i] = z[M - 1 - i];
        sp_softmax_exact(zr, M, pr);
        int ok = 1; for (int i = 0; i < M; i++) if (pr[i] != pi[M - 1 - i]) ok = 0;
        SP_CHECK(ok, "softmax denominator is reduction-order-immune (byte-identical under permutation)");
    }

    /* ---------- Island 3: GELU-tanh ---------- */
    {
        enum { G = 512 };
        static float gx[G], gi[G], gi2[G];
        static double gf[G], gd[G];
        g_s = 11;
        const double k = sqrt(2.0 / M_PI);
        for (int i = 0; i < G; i++) {
            double xv = udev(3.0); gx[i] = (float)xv;
            gf[i] = 0.5 * xv * (1.0 + tanh(k * (xv + 0.044715 * xv * xv * xv)));
        }
        sp_gelu_exact(gx, G, gi);
        for (int i = 0; i < G; i++) gd[i] = gi[i];
        SP_CHECK(relerr(gd, gf, G) < 1e-4, "GELU-tanh exact-integer matches float (relerr < 1e-4)");

        sp_gelu_exact(gx, G, gi2);   /* determinism: same input -> byte-identical output */
        int ok = 1; for (int i = 0; i < G; i++) if (gi2[i] != gi[i]) ok = 0;
        SP_CHECK(ok, "GELU-tanh is a deterministic integer function (repeat call byte-identical)");
    }
}

int main(void) { SP_RUN(T_EXACT_ISLANDS); return SP_DONE(); }
