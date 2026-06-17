/* exact_islands.c -- exact-integer fp32-island kernels; see sp/exact_islands.h.
 * Direct C ports of the proven offline prototypes (engine tools/ring3/
 * g_norm_integer.py + g_islands_integer.py). Every integer constant below was
 * lifted from those prototypes so the C path is bit-identical to the Python that
 * gated GREEN. Portable C11 + __int128 (the canonical CPU build is MinGW gcc).
 *
 * Two C-vs-Python correspondences that keep the ports exact:
 *   - Python  x >> k  on negatives floors (arithmetic shift); on two's-complement
 *     gcc/clang the C  >>  on a signed value does the same. We rely on that.
 *   - Python  a // (2^k)  is FLOOR division; the prototypes only ever floor-divide
 *     by a power of two (Z = 2^14 / 2^16), so we use the arithmetic right shift
 *     `>> k` (NOT C `/`, which truncates toward zero and would differ for a<0).
 */
#include "sp/exact_islands.h"

#include <math.h>    /* llround */
#include <stdint.h>
#include <stddef.h>

/* ---- shared fixed-point exp via 2^x (FB=30 integer poly, coeffs (ln2)^k/k!) ---- */
#define FB   30
#define ONE  (1LL << FB)
static const int64_t LOG2E = 1549082005LL;      /* round(log2(e) * 2^30) */
static const int64_t EXPC[7] = {                /* round((ln2)^k / k! * 2^30) */
    1073741824LL, 744261118LL, 257941248LL, 59597083LL,
    10327387LL,   1431680LL,   165394LL
};

/* exp2_frac(r) ~= 2^(r/2^FB) for r in [0, ONE], FB-fixed result. */
static int64_t exp2_frac(int64_t r) {
    int64_t acc = EXPC[6];
    for (int k = 5; k >= 0; k--) acc = ((acc * r) >> FB) + EXPC[k];
    return acc;
}

/* exp_fixed(d) = e^d for d <= 0, FB-fixed in and out. d = -2*a in tanh, or the
 * (z-max) logit gap in softmax -- always <= 0 so the result is in [0, ONE]. */
static int64_t exp_fixed(int64_t d) {
    int64_t g = (int64_t)(-(((__int128)d * LOG2E) >> FB));   /* g = -d*log2(e) >= 0 */
    int64_t n = g >> FB;
    int64_t r = g - (n << FB);
    if (n >= 32) return 0;                       /* underflow -> 0 (matches Python) */
    if (r) return exp2_frac(ONE - r) >> (n + 1);
    return ONE >> n;
}

/* ---- Island 1: exact-integer RMSNorm (Q=16, IB=20, Qw=16) ---- */
/* floor(sqrt(v)) over u64 -- exact integer isqrt (== Python math.isqrt). */
static uint64_t isqrt_u64(uint64_t v) {
    uint64_t x = 0, b = 1ULL << 62;
    while (b > v) b >>= 2;
    while (b) {
        if (v >= x + b) { v -= x + b; x = (x >> 1) + b; }
        else x >>= 1;
        b >>= 2;
    }
    return x;
}

void sp_rmsnorm_exact(const float *x, const float *w, int n, float *out) {
    enum { Q = 16, IB = 20, Qw = 16 };
    int64_t sumsq = 0;                           /* EXACT: sum (round(x*2^Q))^2 */
    for (int i = 0; i < n; i++) {
        int64_t xi = (int64_t)llround((double)x[i] * (double)(1 << Q));
        sumsq += xi * xi;
    }
    if (sumsq == 0) { for (int i = 0; i < n; i++) out[i] = 0.0f; return; }
    /* inv = round(2^IB * sqrt(n / sum x^2)) = isqrt( (n << 2*(Q+IB)) / sum x^2 ). */
    __int128 num = (__int128)n << (2 * (Q + IB));
    uint64_t q = (uint64_t)(num / (__int128)sumsq);
    int64_t inv = (int64_t)isqrt_u64(q);
    const double denom = (double)(1ULL << (Q + IB + Qw));
    for (int i = 0; i < n; i++) {
        int64_t xi = (int64_t)llround((double)x[i] * (double)(1 << Q));
        int64_t wi = (int64_t)llround((double)w[i] * (double)(1 << Qw));
        __int128 yi = (__int128)xi * inv * wi;   /* 2^(Q+IB+Qw) fixed-point */
        out[i] = (float)((double)(int64_t)yi / denom);
    }
}

/* ---- Island 2: exact-integer softmax (Z=2^14) ---- */
void sp_softmax_exact(const float *z, int m, double *p) {
    enum { ZB = 14 };
    int64_t mx = INT64_MIN;
    for (int i = 0; i < m; i++) {
        int64_t zi = (int64_t)llround((double)z[i] * (double)(1 << ZB));
        if (zi > mx) mx = zi;
    }
    int64_t S = 0;                               /* EXACT integer denominator */
    for (int i = 0; i < m; i++) {
        int64_t zi = (int64_t)llround((double)z[i] * (double)(1 << ZB));
        int64_t d  = ((zi - mx) * ONE) >> ZB;    /* floor-div by 2^ZB (z<=max => d<=0) */
        p[i] = (double)exp_fixed(d);             /* stash e_i in p, normalise below */
        S += (int64_t)p[i];
    }
    for (int i = 0; i < m; i++) p[i] /= (double)S;
}

/* ---- Island 3: exact-integer GELU-tanh (Z=2^16) ---- */
static const int64_t GK = 856722024LL;           /* round(sqrt(2/pi) * 2^30) */
static const int64_t GA = 48012366LL;            /* round(0.044715  * 2^30) */

/* tanh(t) in FB-fixed via the shared exp primitive: sign * (1 - 2 e^{-2|t|}/(1+e^{-2|t|})). */
static int64_t tanh_fixed(int64_t t) {
    int s = (t >= 0) ? 1 : -1;
    int64_t a = (t >= 0) ? t : -t;
    int64_t e2 = exp_fixed(-(2 * a));
    int64_t num = (2 * e2) << FB;                /* 2*e2 <= 2*ONE; <<FB fits int64 */
    return (int64_t)s * (ONE - num / (ONE + e2));
}

void sp_gelu_exact(const float *x, int n, float *out) {
    enum { ZB = 16 };
    for (int i = 0; i < n; i++) {
        int64_t xq = (int64_t)llround((double)x[i] * (double)(1 << ZB));
        int64_t X  = (xq * ONE) >> ZB;           /* floor-div by 2^ZB */
        __int128 X2 = ((__int128)X * X) >> FB;
        int64_t x3  = (int64_t)((X2 * X) >> FB);
        __int128 inner128 = ((__int128)GK * (X + (int64_t)(((__int128)GA * x3) >> FB))) >> FB;
        int64_t  t  = tanh_fixed((int64_t)inner128);
        int64_t  g  = (int64_t)(((__int128)(X >> 1) * (ONE + t)) >> FB);
        out[i] = (float)((double)g / (double)ONE);
    }
}
