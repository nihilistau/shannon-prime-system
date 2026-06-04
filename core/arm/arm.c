/* arm.c — ARM (Algebraic Resonance Memory) two-ring core. See sp/arm.h.
 *
 * Faithful port of the engine's proven C2.1 production code (cpu_forward.c:
 * recall_build_R / recall_project / recall_select + quickselect + r1slot) into
 * the math core, plus the portable stdio Ring-2 reference backend. Behavior is
 * IDENTICAL to the engine implementation (same frozen seed, same arithmetic
 * order, same selection semantics) so sidecars and gates carry over unchanged.
 */
#include "sp/arm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── frozen ±1 projection ─────────────────────────────────────────────────── */

static uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void sp_arm_build_R(signed char *R, int r, int hd) {
    uint64_t s = SP_ARM_PROJ_SEED;
    for (int i = 0; i < r * hd; i++) R[i] = (splitmix64(&s) & 1) ? 1 : -1;
}

void sp_arm_project(const signed char *R, int r, int hd,
                    const float *vec, float *proj) {
    for (int p = 0; p < r; p++) {
        const signed char *Rp = R + (size_t)p * hd;
        float a = 0.0f;
        for (int d = 0; d < hd; d++) a += (float)Rp[d] * vec[d];
        proj[p] = a;
    }
}

/* ── quickselect (Hoare, median-of-three Lomuto partition, DESCENDING) ────── */

static void sidx_swap(sp_arm_sidx *a, int i, int j) {
    sp_arm_sidx t = a[i]; a[i] = a[j]; a[j] = t;
}

static int qsel_partition(sp_arm_sidx *a, int lo, int hi) {
    int mid = lo + (hi - lo) / 2;
    if (a[mid].s > a[lo].s)  sidx_swap(a, lo, mid);
    if (a[hi].s  > a[lo].s)  sidx_swap(a, lo, hi);
    if (a[hi].s  > a[mid].s) sidx_swap(a, mid, hi);
    float pivot = a[mid].s;
    sidx_swap(a, mid, hi);                                  /* park pivot at hi */
    int store = lo;
    for (int j = lo; j < hi; j++)
        if (a[j].s > pivot) { sidx_swap(a, j, store); store++; }
    sidx_swap(a, store, hi);                                /* pivot to final slot */
    return store;
}

static void qsel_topk(sp_arm_sidx *a, int n, int k) {
    if (k <= 0 || k >= n) return;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int p = qsel_partition(a, lo, hi);
        if (p == k) break;
        else if (p < k) lo = p + 1;
        else hi = p - 1;
    }
}

/* ── recall selection ─────────────────────────────────────────────────────── */

int sp_arm_select(const signed char *R, int r, int hd, const float *qh,
                  const float *projk, size_t L, int P, int NKV, int kvh,
                  int B, int W0, int sink0, int pos, sp_arm_sidx *cand, int *ri) {
    if (B <= 0 || pos + 1 <= B) { for (int s = 0; s <= pos; s++) ri[s] = s; return pos + 1; }
    float pq[SP_ARM_R_MAX];
    sp_arm_project(R, r, hd, qh, pq);
    int W = W0;       if (W > pos + 1) W = pos + 1;
    int sink = sink0; if (sink > pos + 1) sink = pos + 1;
    int cand_hi = pos + 1 - W;
    if (cand_hi < sink) cand_hi = sink;
    if (cand_hi > pos + 1) cand_hi = pos + 1;
    int topk = B - W - sink; if (topk < 0) topk = 0;
    if (topk > cand_hi - sink) topk = cand_hi - sink;
    int nc = 0;                                             /* score candidates */
    for (int s = sink; s < cand_hi; s++) {
        const float *pk = projk + (((size_t)L * P + s) * NKV + kvh) * (size_t)r;
        float a = 0.0f;
        for (int p = 0; p < r; p++) a += pq[p] * pk[p];
        cand[nc].s = a; cand[nc].i = s; nc++;
    }
    qsel_topk(cand, nc, topk);                              /* expected O(N) */
    int m = 0;
    for (int s = 0; s < sink; s++) ri[m++] = s;             /* pinned sink anchors */
    for (int t = 0; t < topk; t++) ri[m++] = cand[t].i;     /* top-k (order-free) */
    for (int s = cand_hi; s <= pos; s++) ri[m++] = s;       /* recent window */
    return m;
}

/* ── bit-packed popcount router (SimHash overlay; see arm.h contract) ───────
 * The candidate scoring scan lives in arm_scan.c (its own archive member —
 * the engine-overridable seam); this TU keeps projection + selection. */

uint64_t sp_arm_project_sig(const signed char *R, int r, int hd, const float *vec) {
    uint64_t sig = 0;
    for (int p = 0; p < r; p++) {
        const signed char *Rp = R + (size_t)p * hd;
        float a = 0.0f;
        for (int d = 0; d < hd; d++) a += (float)Rp[d] * vec[d];
        if (a >= 0.0f) sig |= (1ULL << p);     /* sign bit of the SAME float dot */
    }
    return sig;
}

int sp_arm_select_sig(const signed char *R, int r, int hd, const float *qh,
                      const uint64_t *sigk, size_t L, int P, int NKV, int kvh,
                      int B, int W0, int sink0, int pos, sp_arm_sidx *cand, int *ri) {
    if (B <= 0 || pos + 1 <= B) { for (int s = 0; s <= pos; s++) ri[s] = s; return pos + 1; }
    const uint64_t qsig = sp_arm_project_sig(R, r, hd, qh);
    int W = W0;       if (W > pos + 1) W = pos + 1;
    int sink = sink0; if (sink > pos + 1) sink = pos + 1;
    int cand_hi = pos + 1 - W;
    if (cand_hi < sink) cand_hi = sink;
    if (cand_hi > pos + 1) cand_hi = pos + 1;
    int topk = B - W - sink; if (topk < 0) topk = 0;
    if (topk > cand_hi - sink) topk = cand_hi - sink;
    /* head-major sidecar: this head's signatures are stride-1 — hand the
     * contiguous slice to the scan seam (engine: VPOPCNTDQ + OMP). */
    const uint64_t *hs = sigk + ((size_t)L * NKV + kvh) * (size_t)P;
    int nc = cand_hi - sink;
    sp_arm_scan_sig(qsig, hs + sink, nc, sink, cand);
    qsel_topk(cand, nc, topk);
    int m = 0;
    for (int s = 0; s < sink; s++) ri[m++] = s;             /* pinned sink anchors */
    for (int t = 0; t < topk; t++) ri[m++] = cand[t].i;     /* top-k (order-free) */
    for (int s = cand_hi; s <= pos; s++) ri[m++] = s;       /* recent window */
    return m;
}

/* ── Ring-1 slot map ──────────────────────────────────────────────────────── */

int sp_arm_r1slot(int s, int offloading, int sink, int w) {
    return offloading ? (s < sink ? s : sink + (s - sink) % w) : s;
}

/* ── portable stdio Ring-2 reference backend ──────────────────────────────── */

/* 64-bit-safe seek across MSVC/MinGW/POSIX. */
#if defined(_WIN32)
#  define sp_arm_fseek64 _fseeki64
#else
#  define sp_arm_fseek64(f, off, wh) fseeko((f), (off_t)(off), (wh))
#endif

typedef struct { FILE *f[2]; } sp_arm_ring2_stdio;

static int stdio_write_block(void *handle, int which, uint64_t off,
                             const void *src, size_t len) {
    sp_arm_ring2_stdio *st = (sp_arm_ring2_stdio *)handle;
    if (!st || which < 0 || which > 1 || !st->f[which]) return 1;
    if (sp_arm_fseek64(st->f[which], (int64_t)off, SEEK_SET)) return 1;
    return fwrite(src, 1, len, st->f[which]) == len ? 0 : 1;
}

static int stdio_read_block(void *handle, int which, uint64_t off,
                            void *dst, size_t len) {
    sp_arm_ring2_stdio *st = (sp_arm_ring2_stdio *)handle;
    if (!st || which < 0 || which > 1 || !st->f[which]) return 1;
    if (sp_arm_fseek64(st->f[which], (int64_t)off, SEEK_SET)) return 1;
    return fread(dst, 1, len, st->f[which]) == len ? 0 : 1;
}

static void stdio_close(void *handle) {
    sp_arm_ring2_stdio *st = (sp_arm_ring2_stdio *)handle;
    if (!st) return;
    if (st->f[0]) fclose(st->f[0]);
    if (st->f[1]) fclose(st->f[1]);
    free(st);
}

int sp_arm_ring2_stdio_open(const char *dir, sp_arm_ring2_backend *out) {
    if (!dir || !out) return 1;
    sp_arm_ring2_stdio *st = (sp_arm_ring2_stdio *)calloc(1, sizeof(*st));
    if (!st) return 1;
    char path[1024];
    static const char *names[2] = { "sp_arm_ring2_k.bin", "sp_arm_ring2_v.bin" };
    size_t dl = strlen(dir);
    int sep = (dl > 0 && dir[dl - 1] != '/' && dir[dl - 1] != '\\');
    for (int w = 0; w < 2; w++) {
        snprintf(path, sizeof(path), "%s%s%s", dir, sep ? "/" : "", names[w]);
        st->f[w] = fopen(path, "w+b");                     /* fresh store per open */
        if (!st->f[w]) { stdio_close(st); return 1; }
    }
    out->handle        = st;
    out->write_block   = stdio_write_block;
    out->read_block    = stdio_read_block;
    out->read_batch    = NULL;                             /* serial reference */
    out->alloc_aligned = NULL;                             /* malloc is fine here */
    out->free_aligned  = NULL;
    out->close         = stdio_close;
    return 0;
}

/* ── platform-backend registration (the L1 hook) ──────────────────────────── */

static sp_arm_ring2_backend g_registered;
static int g_registered_on = 0;

void sp_arm_ring2_register(const sp_arm_ring2_backend *be) {
    if (be) { g_registered = *be; g_registered_on = 1; }
    else    { g_registered_on = 0; }
}

int sp_arm_ring2_registered(sp_arm_ring2_backend *out) {
    if (!g_registered_on || !out) return 0;
    *out = g_registered;
    return 1;
}
