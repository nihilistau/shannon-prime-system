/* kste_md.c — KSTE magnitude-as-depth encoder + Dickson dominance (v2).
 * Fresh from the Paper III/IV spec (anti-contamination: no old-repo code).
 * See sp/kste_md.h for the algorithm and signature layout.
 *
 * Integer-only and deterministic: the same (vec,k) yields the same signature on
 * every platform (no float, no memcpy of wider types, stable sort tie-break by
 * original index).
 */
#include "sp/kste_md.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* |v| into int64 (handles INT32_MIN without UB). */
static int64_t abs64(int32_t x) { return x < 0 ? -(int64_t)x : (int64_t)x; }

/* rank record: magnitude + original index (index breaks ties deterministically). */
typedef struct { int64_t a; int32_t idx; } av_t;

static int av_desc(const void *p, const void *q) {
    const av_t *x = (const av_t *)p, *y = (const av_t *)q;
    if (x->a > y->a) return -1;
    if (x->a < y->a) return  1;
    return (x->idx < y->idx) ? -1 : (x->idx > y->idx) ? 1 : 0;  /* stable */
}

void sp_kste_md_encode(const int32_t *vec, int k, sp_kste_md_sig_t *out) {
    memset(out, 0, sizeof(*out));
    if (k <= 0 || vec == NULL) return;

    av_t *arr = (av_t *)malloc((size_t)k * sizeof(av_t));
    if (!arr) return;
    for (int i = 0; i < k; i++) { arr[i].a = abs64(vec[i]); arr[i].idx = i; }
    qsort(arr, (size_t)k, sizeof(av_t), av_desc);

    int64_t amax = arr[0].a;
    if (amax <= 0) amax = 1;                    /* all-zero vector -> neutral */

    /* --- anchors: top SP_KMD_ANCHORS with nonzero magnitude (label A) --- */
    int nA = 0;
    for (int i = 0; i < SP_KMD_ANCHORS && i < k; i++)
        if (arr[i].a > 0) nA++;

    /* --- residuals: next SP_KMD_RESID, each a B/C chain of magnitude-as-depth --- */
    int budget = SP_KMD_NODEBUDGET;             /* max chain nodes */
    int used = 0, dmax = 0;
    int64_t nB = 0, nC = 0, MBB = 0, MCC = 0;   /* counts + internal ancestor pairs */

    for (int r = SP_KMD_ANCHORS; r < k && r < SP_KMD_ANCHORS + SP_KMD_RESID; r++) {
        int64_t a = arr[r].a;
        if (a == 0) break;
        /* magnitude-as-depth: L in 1..LMAX, from |v|/amax (integer math) */
        int L = 1 + (int)(((int64_t)(SP_KMD_LMAX - 1) * a) / amax);
        if (L < 1) L = 1;
        if (L > SP_KMD_LMAX) L = SP_KMD_LMAX;
        if (used + L > budget) { L = budget - used; }
        if (L <= 0) break;
        used += L;

        int positive = (vec[arr[r].idx] >= 0);
        int64_t internal = (int64_t)L * (L - 1) / 2;   /* (X,X) ancestor pairs in the chain */
        if (positive) { nB += L; MBB += internal; }
        else          { nC += L; MCC += internal; }
        if (L + 1 > dmax) dmax = L + 1;                /* anchor(depth1) + chain */
    }

    int64_t ntot = (int64_t)nA + nB + nC;
    /* sigma0 */
    out->v[0] = nA;
    out->v[1] = (int32_t)nB;
    out->v[2] = (int32_t)nC;
    out->v[3] = dmax;
    out->v[4] = (int32_t)ntot;
    /* sigma1 3x3 ancestor-pair counts (row-major anc,desc over {A,B,C}) */
    out->v[5]  = 0;              /* AA: anchors are siblings, never anc/desc     */
    out->v[6]  = (int32_t)nB;    /* AB: each B node has an A ancestor            */
    out->v[7]  = (int32_t)nC;    /* AC: each C node has an A ancestor            */
    out->v[8]  = 0;              /* BA */
    out->v[9]  = (int32_t)MBB;   /* BB: within-chain B ancestor pairs           */
    out->v[10] = 0;              /* BC: chains are single-label                  */
    out->v[11] = 0;              /* CA */
    out->v[12] = 0;              /* CB */
    out->v[13] = (int32_t)MCC;   /* CC: within-chain C ancestor pairs           */

    free(arr);
}

sp_dom_t sp_kste_md_dom(const sp_kste_md_sig_t *a, const sp_kste_md_sig_t *b) {
    int ge = 1, le = 1, eq = 1;   /* a>=b elementwise / a<=b / a==b */
    for (int i = 0; i < SP_KMD_SIGDIM; i++) {
        int32_t x = a->v[i], y = b->v[i];
        if (x < y) ge = 0;
        if (x > y) le = 0;
        if (x != y) eq = 0;
    }
    if (eq) return SP_EQUIVALENT;
    if (ge) return SP_DOMINATES;
    if (le) return SP_DOMINATED;
    return SP_INCOMPARABLE;
}
