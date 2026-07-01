/* sp/kste_md.h — KSTE "magnitude-as-depth" encoder + Dickson dominance (v2, experimental).
 *
 * This is a FRESH re-derivation of the fuller Paper III/IV encoder (the 60-node
 * magnitude-as-depth tree with count-based Tier-0/Tier-1 signatures and the
 * Dickson N^14 product order), written from the published spec — NOT copied from
 * the anti-contaminate-gated old shannon-prime repos. It exists ALONGSIDE the
 * frozen v1 `sp/kste.h` (order-statistic componentwise order), which is left
 * untouched (its wire format is frozen; the PoUW sieve depends on it).
 *
 * Why v2: the v1 in-tree encoder is magnitude-dominated (order-stats all move
 * together with scale), so nearly all pairs are comparable -> weak cluster
 * discrimination (measured intra/inter ~1.2x, G-SIEVE-MEASURE-2026-07-01). The
 * count-based signature below adds sign-split and chain-shape components that
 * move INDEPENDENTLY, which is what buys discrimination under Dickson's order.
 *
 * Encoder (integer-only, deterministic):
 *   1. rank the k components by |value| descending.
 *   2. ANCHORS  = the top SP_KMD_ANCHORS by magnitude -> label A, level-1 children of root.
 *   3. RESIDUALS= the next SP_KMD_RESID by magnitude. Each residual becomes a CHAIN
 *      of label B (value>=0) or C (value<0), of length L = magnitude-as-depth:
 *        L = 1 + (LMAX-1) * |v| / amax           (1..LMAX, integer)
 *      hung under an anchor. Total nodes capped at SP_KMD_NODEBUDGET.
 *   4. Signature = sigma0 (5 node counts) (+) sigma1 (3x3 ancestor-pair counts) in N^14.
 *
 * Dickson dominance = elementwise product order on the N^14 signature
 * (Dickson 1913 wqo; PRA-strength). tri-state via sp_dom_t (reused from sp/kste.h).
 */
#ifndef SP_KSTE_MD_H
#define SP_KSTE_MD_H

#include <stdint.h>
#include "sp/kste.h"   /* sp_dom_t {SP_INCOMPARABLE,SP_DOMINATES,SP_DOMINATED,SP_EQUIVALENT} */

#ifdef __cplusplus
extern "C" {
#endif

#define SP_KMD_ANCHORS     14   /* top-magnitude components -> label A            */
#define SP_KMD_RESID       60   /* next-magnitude components -> B/C chains        */
#define SP_KMD_NODEBUDGET  60   /* max non-anchor (chain) nodes                   */
#define SP_KMD_LMAX         8   /* max chain length (magnitude-as-depth ceiling)  */
#define SP_KMD_SIGDIM      14   /* sigma0(5) + sigma1(9)                          */

/* N^14 count signature. Layout:
 *   [0]=nA  [1]=nB  [2]=nC  [3]=dmax  [4]=ntotal
 *   [5..13] = 3x3 ancestor-pair counts, row-major (anc,desc) over {A,B,C}:
 *     [5]=AA [6]=AB [7]=AC  [8]=BA [9]=BB [10]=BC  [11]=CA [12]=CB [13]=CC
 * All components are non-negative counts; the product order on them is a wqo. */
typedef struct { int32_t v[SP_KMD_SIGDIM]; } sp_kste_md_sig_t;

/* Encode a K-vector (int32, length k) into its count signature. Deterministic,
 * integer-only, pure. */
void sp_kste_md_encode(const int32_t *vec, int k, sp_kste_md_sig_t *out);

/* Dickson dominance of a relative to b (elementwise product order on N^14). */
sp_dom_t sp_kste_md_dom(const sp_kste_md_sig_t *a, const sp_kste_md_sig_t *b);

#ifdef __cplusplus
}
#endif
#endif /* SP_KSTE_MD_H */
