/* resdot.c — the keystore hot loop: modular residue dot (portable reference).
 *
 * ISOLATED in its own translation unit ON PURPOSE: the engine overrides
 * sp_pr_resdot with an AVX2 kernel in an always-pulled object (cpu_overlay.c),
 * so in engine links this member is never pulled and the canonical scoring
 * runs at engine speed — the same linker-resolution seam as the decode split.
 *
 * Deferred reduction: a[i], b[i] < q < 2^30 => each product < 2^60; FIFTEEN
 * products sum exactly below 2^64 (15*2^60 + q < 2^64), so we reduce once per
 * 15-element chunk instead of per element. Exact: integer adds never wrap,
 * and (x + y) mod q folding per chunk preserves the total mod q.
 */
#include "sp/poly_ring.h"

uint32_t sp_pr_resdot(const uint32_t *a, const uint32_t *b, uint32_t n, uint32_t q) {
    uint64_t acc = 0;
    uint32_t i = 0;
    while (i < n) {
        uint32_t lim = i + 15u; if (lim > n) lim = n;
        for (; i < lim; i++) acc += (uint64_t)a[i] * b[i];
        acc %= q;                              /* one reduction per 15 elements */
    }
    return (uint32_t)acc;
}
