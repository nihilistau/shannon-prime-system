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

/* General projection (G-P3-GEOM): R row stride hd_max, vector length hd.
 * The legacy uniform projection is the hd == hd_max special case below —
 * identical loop, identical arithmetic order, so delegation is bit-exact. */
void sp_arm_project_geom(const signed char *R, int r, int hd_max, int hd,
                         const float *vec, float *proj) {
    for (int p = 0; p < r; p++) {
        const signed char *Rp = R + (size_t)p * (size_t)hd_max;
        float a = 0.0f;
        for (int d = 0; d < hd; d++) a += (float)Rp[d] * vec[d];
        proj[p] = a;
    }
}

void sp_arm_project(const signed char *R, int r, int hd,
                    const float *vec, float *proj) {
    sp_arm_project_geom(R, r, hd, hd, vec, proj);
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

/* ── recall-hit telemetry (C1L.2: the LRU / association signal) ──────────────
 * Opt-in per-position counter. When a buffer is attached, sp_arm_select{,_sig}
 * increment g_arm_hits[s] for each CONTENT-selected top-k position s (the sinks
 * and the recent window are structural — always chosen — so they carry no
 * coldness signal and are NOT counted). Cold positions = low hit count = the
 * curator's eviction candidates. Detached (NULL) => zero work, selection and
 * the decoded sequence are bit-identical (the telemetry-null guarantee). */
static int *g_arm_hits = NULL;
static int  g_arm_hits_n = 0;

void sp_arm_hits_attach(int *buf, int n) { g_arm_hits = buf; g_arm_hits_n = (buf ? n : 0); }
void sp_arm_hits_detach(void)            { g_arm_hits = NULL; g_arm_hits_n = 0; }

/* ── cold-evict mask (C1L.2 Step 2: the curator's consolidation knob) ────────
 * mask[s]=1 => position s is EVICTED from the store and skipped as a recall
 * candidate. Evicting the COLD set (positions with zero content-hits) is
 * LOSSLESS — a position that never won top-k cannot be removed from any recall
 * set, so the decode is bit-identical while the episode shrinks. Evicting a HOT
 * position changes the recall set => the decode diverges (the gate that the
 * curator rewinds). Attached on the f32 router (sp_arm_select); NULL = no
 * eviction, bit-exact. */
static const unsigned char *g_arm_evict = NULL;
static int g_arm_evict_n = 0;
void sp_arm_evict_attach(const unsigned char *mask, int n) { g_arm_evict = mask; g_arm_evict_n = (mask ? n : 0); }
void sp_arm_evict_detach(void)                             { g_arm_evict = NULL; g_arm_evict_n = 0; }

/* ── recall selection ─────────────────────────────────────────────────────── */

/* General per-layer-class selection (G-P3-GEOM): sidecar block at g->off,
 * row addressing by g->nkv, projection dims {hd_max, g->hd}. The legacy
 * uniform sp_arm_select below delegates here with off = L*P*NKV*r and
 * hd_max = hd, reproducing the legacy addressing exactly (bit-identical). */
int sp_arm_select_geom(const signed char *R, int r, int hd_max,
                       const sp_arm_geom *g, const float *qh,
                       const float *projk, int P, int kvh,
                       int B, int W0, int sink0, int pos,
                       sp_arm_sidx *cand, int *ri) {
    (void)P;   /* f32 sidecar rows are position-major within the block */
    if (B <= 0 || pos + 1 <= B) { for (int s = 0; s <= pos; s++) ri[s] = s; return pos + 1; }
    float pq[SP_ARM_R_MAX];
    sp_arm_project_geom(R, r, hd_max, g->hd, qh, pq);
    int W = W0;       if (W > pos + 1) W = pos + 1;
    int sink = sink0; if (sink > pos + 1) sink = pos + 1;
    int cand_hi = pos + 1 - W;
    if (cand_hi < sink) cand_hi = sink;
    if (cand_hi > pos + 1) cand_hi = pos + 1;
    int topk = B - W - sink; if (topk < 0) topk = 0;
    if (topk > cand_hi - sink) topk = cand_hi - sink;
    int nc = 0;                                             /* score candidates */
    for (int s = sink; s < cand_hi; s++) {
        if (g_arm_evict && s < g_arm_evict_n && g_arm_evict[s]) continue;  /* C1L.2: evicted from the store */
        const float *pk = projk + g->off + ((size_t)s * (size_t)g->nkv + (size_t)kvh) * (size_t)r;
        float a = 0.0f;
        for (int p = 0; p < r; p++) a += pq[p] * pk[p];
        cand[nc].s = a; cand[nc].i = s; nc++;
    }
    if (topk > nc) topk = nc;                               /* fewer survivors after eviction */
    qsel_topk(cand, nc, topk);                              /* expected O(N) */
    int m = 0;
    for (int s = 0; s < sink; s++) ri[m++] = s;             /* pinned sink anchors */
    for (int t = 0; t < topk; t++) {                        /* top-k (order-free) */
        const int hs = cand[t].i; ri[m++] = hs;
        if (g_arm_hits && hs < g_arm_hits_n) g_arm_hits[hs]++;  /* C1L.2 coldness signal */
    }
    for (int s = cand_hi; s <= pos; s++) ri[m++] = s;       /* recent window */
    return m;
}

int sp_arm_select(const signed char *R, int r, int hd, const float *qh,
                  const float *projk, size_t L, int P, int NKV, int kvh,
                  int B, int W0, int sink0, int pos, sp_arm_sidx *cand, int *ri) {
    sp_arm_geom g;
    g.nkv = NKV; g.hd = hd;
    g.off = (size_t)L * (size_t)P * (size_t)NKV * (size_t)r;
    return sp_arm_select_geom(R, r, hd, &g, qh, projk, P, kvh,
                              B, W0, sink0, pos, cand, ri);
}

/* ── bit-packed popcount router (SimHash overlay; see arm.h contract) ───────
 * The candidate scoring scan lives in arm_scan.c (its own archive member —
 * the engine-overridable seam); this TU keeps projection + selection. */

/* General sign-bit projection (G-P3-GEOM): same float dot as the f32 geom
 * projection (R row stride hd_max, vector length hd), sign bit per row. */
uint64_t sp_arm_project_sig_geom(const signed char *R, int r, int hd_max,
                                 int hd, const float *vec) {
    uint64_t sig = 0;
    for (int p = 0; p < r; p++) {
        const signed char *Rp = R + (size_t)p * (size_t)hd_max;
        float a = 0.0f;
        for (int d = 0; d < hd; d++) a += (float)Rp[d] * vec[d];
        if (a >= 0.0f) sig |= (1ULL << p);     /* sign bit of the SAME float dot */
    }
    return sig;
}

uint64_t sp_arm_project_sig(const signed char *R, int r, int hd, const float *vec) {
    return sp_arm_project_sig_geom(R, r, hd, hd, vec);
}

/* General per-layer-class sig selection (G-P3-GEOM): head-major block at
 * g->off, head stride P. Legacy sp_arm_select_sig delegates with
 * off = L*NKV*P and hd_max = hd (bit-identical addressing). */
int sp_arm_select_sig_geom(const signed char *R, int r, int hd_max,
                           const sp_arm_geom *g, const float *qh,
                           const uint64_t *sigk, int P, int kvh,
                           int B, int W0, int sink0, int pos,
                           sp_arm_sidx *cand, int *ri) {
    if (B <= 0 || pos + 1 <= B) { for (int s = 0; s <= pos; s++) ri[s] = s; return pos + 1; }
    const uint64_t qsig = sp_arm_project_sig_geom(R, r, hd_max, g->hd, qh);
    int W = W0;       if (W > pos + 1) W = pos + 1;
    int sink = sink0; if (sink > pos + 1) sink = pos + 1;
    int cand_hi = pos + 1 - W;
    if (cand_hi < sink) cand_hi = sink;
    if (cand_hi > pos + 1) cand_hi = pos + 1;
    int topk = B - W - sink; if (topk < 0) topk = 0;
    if (topk > cand_hi - sink) topk = cand_hi - sink;
    /* head-major sidecar: this head's signatures are stride-1 — hand the
     * contiguous slice to the scan seam (engine: VPOPCNTDQ + OMP). */
    const uint64_t *hs = sigk + g->off + (size_t)kvh * (size_t)P;
    int nc = cand_hi - sink;
    sp_arm_scan_sig(qsig, hs + sink, nc, sink, cand);
    qsel_topk(cand, nc, topk);
    int m = 0;
    for (int s = 0; s < sink; s++) ri[m++] = s;             /* pinned sink anchors */
    for (int t = 0; t < topk; t++) {                        /* top-k (order-free) */
        const int hs2 = cand[t].i; ri[m++] = hs2;
        if (g_arm_hits && hs2 < g_arm_hits_n) g_arm_hits[hs2]++;  /* C1L.2 coldness signal */
    }
    for (int s = cand_hi; s <= pos; s++) ri[m++] = s;       /* recent window */
    return m;
}

int sp_arm_select_sig(const signed char *R, int r, int hd, const float *qh,
                      const uint64_t *sigk, size_t L, int P, int NKV, int kvh,
                      int B, int W0, int sink0, int pos, sp_arm_sidx *cand, int *ri) {
    sp_arm_geom g;
    g.nkv = NKV; g.hd = hd;
    g.off = (size_t)L * (size_t)NKV * (size_t)P;
    return sp_arm_select_sig_geom(R, r, hd, &g, qh, sigk, P, kvh,
                                  B, W0, sink0, pos, cand, ri);
}

/* Cumulative per-layer sidecar block offsets for heterogeneous geometry.
 * kind 0 = f32 projk (block = P*nkv*r floats); kind 1 = sig (block = nkv*P
 * u64). Returns the total element count (the sidecar allocation size). */
size_t sp_arm_geom_layout(sp_arm_geom *g, int n_layers, int P, int r, int kind) {
    size_t off = 0;
    for (int L = 0; L < n_layers; L++) {
        g[L].off = off;
        off += (kind == 0)
             ? (size_t)P * (size_t)g[L].nkv * (size_t)r
             : (size_t)g[L].nkv * (size_t)P;
    }
    return off;
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

/* fmode: "w+b" = fresh store (spill path); "rb" = read-only LOAD (replay path,
 * non-truncating — opens an existing persisted episode for recall). */
static int stdio_open_mode(const char *dir, sp_arm_ring2_backend *out, const char *fmode) {
    if (!dir || !out) return 1;
    sp_arm_ring2_stdio *st = (sp_arm_ring2_stdio *)calloc(1, sizeof(*st));
    if (!st) return 1;
    char path[1024];
    static const char *names[2] = { "sp_arm_ring2_k.bin", "sp_arm_ring2_v.bin" };
    size_t dl = strlen(dir);
    int sep = (dl > 0 && dir[dl - 1] != '/' && dir[dl - 1] != '\\');
    for (int w = 0; w < 2; w++) {
        snprintf(path, sizeof(path), "%s%s%s", dir, sep ? "/" : "", names[w]);
        st->f[w] = fopen(path, fmode);
        if (!st->f[w]) { stdio_close(st); return 1; }
    }
    out->handle        = st;
    out->write_block   = stdio_write_block;
    out->read_block    = stdio_read_block;
    out->read_batch    = NULL;                             /* serial reference */
    out->alloc_aligned = NULL;                             /* malloc is fine here */
    out->free_aligned  = NULL;
    out->close         = stdio_close;
    out->read_batch2   = NULL;                             /* serial reference */
    return 0;
}

int sp_arm_ring2_stdio_open(const char *dir, sp_arm_ring2_backend *out) {
    return stdio_open_mode(dir, out, "w+b");               /* fresh store per open (spill) */
}

/* C1L.0b: non-truncating read-only LOAD of a persisted episode (replay-decode).
 * Replay is read-path only; a curated episode is written via the spill open. */
int sp_arm_ring2_stdio_open_ro(const char *dir, sp_arm_ring2_backend *out) {
    return stdio_open_mode(dir, out, "rb");                /* load existing, no truncate */
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
