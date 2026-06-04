/* arm_scan.c — portable reference for the popcount candidate scan.
 *
 * THIS SYMBOL'S OWN ARCHIVE MEMBER, deliberately (the resdot/ntt_batch
 * linker seam): the engine overrides sp_arm_scan_sig with an
 * AVX512-VPOPCNTDQ + OMP kernel by defining the symbol in an always-pulled
 * object; binaries that don't get this canonical loop.
 *
 * THE 32k WALL: streaming ingest scans O(pos) candidates per head per token
 * — the quadratic term of long-context prefill (~2.4e14 popcounts at 32k on
 * Qwen3 geometry). The head-major sigk layout (arm.h) makes `sigs` a
 * contiguous stride-1 u64 slice, so the override is pure streaming loads.
 *
 * EXACTNESS CONTRACT: any override must produce IDENTICAL cand entries
 * (same float score, same index, same order) — the host-side quickselect
 * consumes these values, so score-array equality implies selection identity
 * (gate T_ARM_SIG scan check + N=512 NIAH parity).
 */
#include "sp/arm.h"

static int scan_popcount64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    /* portable SWAR fallback (toolchain-independent reference) */
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (int)((x * 0x0101010101010101ULL) >> 56);
#endif
}

void sp_arm_scan_sig(uint64_t qsig, const uint64_t *sigs, int n, int s0,
                     sp_arm_sidx *cand) {
    for (int i = 0; i < n; i++) {
        cand[i].s = -(float)scan_popcount64(qsig ^ sigs[i]);
        cand[i].i = s0 + i;
    }
}
