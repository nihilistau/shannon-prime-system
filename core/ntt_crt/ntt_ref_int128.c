/* ntt_ref_int128.c — TEST-ONLY parity oracle for the CRT negacyclic NTT.
 *
 * This file is compiled ONLY into the test executable (via TEST_SOURCES in
 * core/ntt_crt/CMakeLists.txt), never into the production library sp_ntt_crt.
 * It is the regression anchor the kernel is validated against, so it is kept
 * deliberately dumb: a schoolbook O(N^2) negacyclic convolution accumulated in
 * __int128, then reduced to the signed centered residue mod M. Because it lives
 * only in the test, __int128 is fine here (and unsupported by MSVC, which is
 * exactly why T_NTT_3 is deferred to the MSVC wave).
 *
 * Negacyclic ring: Z[x]/(x^N + 1). Coefficient k of a*b is
 *     sum_{i+j=k}   a_i b_j   -   sum_{i+j=k+N} a_i b_j
 * i.e. products that wrap past degree N flip sign (x^N = -1).
 */
#include "sp/ntt_crt.h"

#include <stdint.h>

void ntt_ref_negacyclic_mul(uint32_t N, const int32_t *a, const int32_t *b,
                            int64_t *out) {
    const __int128 M = (__int128)SP_NTT_M;

    for (uint32_t k = 0; k < N; k++) {
        __int128 acc = 0;
        for (uint32_t i = 0; i < N; i++) {
            uint32_t j;
            __int128 term;
            if (i <= k) {
                j = k - i;                       /* i + j = k        -> +a_i b_j */
                term = (__int128)a[i] * (__int128)b[j];
                acc += term;
            }
            /* the wrapped contribution: i + j = k + N -> -a_i b_j */
            if (k + N >= i && (k + N - i) < N) {
                j = k + N - i;
                term = (__int128)a[i] * (__int128)b[j];
                acc -= term;
            }
        }
        /* reduce to centered residue in (-M/2, M/2] */
        __int128 v = acc % M;
        if (v < 0) v += M;
        if (v > M / 2) v -= M;
        out[k] = (int64_t)v;
    }
}
