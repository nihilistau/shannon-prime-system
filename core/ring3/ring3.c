/* ring3.c — Ring-3 VSA layer + NIGHTSHIFT consolidation, native C.
 *
 * Port of tools/ring3/ok_bind.py + tools/ring3/g_r3_nightshift.py onto the
 * exact-integer negacyclic CRT-NTT (core/poly_ring, sp_pr_mul). The NTT is NOT
 * reimplemented — every bind/unbind routes through sp_pr_mul, so the M-coeff
 * vectors are bit-identical to the Python reference once both share the
 * canonical splitmix64 ±1 generator (sp_r3_carrier/sp_r3_idvec here; ok_bind.py
 * updated to the same). See sp/ring3.h for the contract.
 *
 * Anti-contamination: derived from the two named Python references + the
 * sp/poly_ring.h API. No other legacy source read.
 */
#include "sp/ring3.h"
#include "sp/poly_ring.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── splitmix64 ±1 generator (matches g_r3_nightshift.smix) ────────────────── */
static void smix_pm1(uint64_t seed, int n, int8_t *out) {
    uint64_t s = seed;
    for (int i = 0; i < n; i++) {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z =  z ^ (z >> 31);
        out[i] = (z & 1ull) ? (int8_t)1 : (int8_t)-1;
    }
}

void sp_r3_carrier(uint64_t seed, int D, int8_t *out) {
    smix_pm1(seed, D, out);
}
void sp_r3_idvec(uint64_t seed, int D, int8_t *out) {
    smix_pm1(seed ^ SP_R3_ID_XOR, D, out);
}

/* ── block tiling (matches ok_bind._blocks) ─────────────────────────────────
 * Returns block count, fills blk[] with the {128,256,512} block sizes. Returns
 * -1 if D is not tileable. */
static int r3_blocks(int D, int *blk) {
    if (D == 128 || D == 256 || D == 512) { blk[0] = D; return 1; }
    if (D == 1024) { blk[0] = 512; blk[1] = 512; return 2; }
    int nb = 0, r = D;
    while (r >= 512) { if (nb >= 64) return -1; blk[nb++] = 512; r -= 512; }
    if (r == 128 || r == 256 || r == 512) { blk[nb++] = r; }
    else if (r > 0) return -1;
    return nb;
}

int sp_r3_dim_ok(int D) {
    int blk[64];
    return (D > 0 && r3_blocks(D, blk) > 0) ? 1 : 0;
}

/* involution out[0]=a[0]; out[1:]=-a[N-1:0:-1] (matches ok_bind._involute). */
static void r3_involute_i32(const int8_t *a, int N, int32_t *out) {
    out[0] = a[0];
    for (int j = 1; j < N; j++) out[j] = -(int32_t)a[N - j];
}

/* ── bind / unbind ─────────────────────────────────────────────────────────
 * Both tile D into negacyclic blocks and call sp_pr_mul per block. addr/idv are
 * ±1 int8; sp_pr_mul wants int32, so each block is widened into a scratch. */
int sp_r3_bind(const int8_t *addr, const int8_t *idv, int D, int64_t *M_out) {
    int blk[64];
    int nb = r3_blocks(D, blk);
    if (nb <= 0) return 1;
    int32_t a32[512], b32[512];
    int o = 0, rc = 1;
    for (int bi = 0; bi < nb; bi++) {
        int N = blk[bi];
        sp_pr_ctx *ctx = sp_pr_init((uint32_t)N);
        if (!ctx) return 1;
        for (int j = 0; j < N; j++) { a32[j] = addr[o + j]; b32[j] = idv[o + j]; }
        sp_pr_mul(ctx, a32, b32, M_out + o);
        sp_pr_free(ctx);
        o += N;
    }
    rc = 0;
    return rc;
}

int sp_r3_unbind(const int64_t *M, const int8_t *addr, int D, int64_t *out) {
    int blk[64];
    int nb = r3_blocks(D, blk);
    if (nb <= 0) return 1;
    int32_t m32[512], inv[512];
    int o = 0;
    for (int bi = 0; bi < nb; bi++) {
        int N = blk[bi];
        sp_pr_ctx *ctx = sp_pr_init((uint32_t)N);
        if (!ctx) return 1;
        /* M coefficients of a ±1 (x) ±1 negacyclic block are bounded by N<=512,
         * so they fit int32 with vast room — widen for sp_pr_mul. */
        for (int j = 0; j < N; j++) m32[j] = (int32_t)M[o + j];
        r3_involute_i32(addr + o, N, inv);
        sp_pr_mul(ctx, m32, inv, out + o);
        sp_pr_free(ctx);
        o += N;
    }
    return 0;
}

int sp_r3_superpose(int64_t *M, const int8_t *addr, const int8_t *idv, int D) {
    int64_t contrib[1024];
    if (D > 1024) return 1;            /* the supported tiling tops out at 1024 here */
    if (sp_r3_bind(addr, idv, D, contrib)) return 1;
    for (int i = 0; i < D; i++) M[i] += contrib[i];
    return 0;
}

/* ── cosine in double (matches ok_bind.cos) ─────────────────────────────────
 * a.b / (||a|| ||b|| + 1e-12), all in float64. */
double sp_r3_cos_i64(const int64_t *a, const int64_t *b, int D) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < D; i++) {
        double av = (double)a[i], bv = (double)b[i];
        dot += av * bv; na += av * av; nb += bv * bv;
    }
    return dot / (sqrt(na) * sqrt(nb) + 1e-12);
}
double sp_r3_cos_i64_i8(const int64_t *a, const int8_t *b, int D) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < D; i++) {
        double av = (double)a[i], bv = (double)b[i];
        dot += av * bv; na += av * av; nb += bv * bv;
    }
    return dot / (sqrt(na) * sqrt(nb) + 1e-12);
}

/* ── ActiveVector ──────────────────────────────────────────────────────────── */
int sp_r3_vector_init(sp_r3_vector *v, int D) {
    if (!v || !sp_r3_dim_ok(D)) return 1;
    memset(v, 0, sizeof(*v));
    v->D = D;
    v->M = calloc((size_t)D, sizeof(int64_t));
    if (!v->M) return 1;
    return 0;
}

void sp_r3_vector_free(sp_r3_vector *v) {
    if (!v) return;
    free(v->M); v->M = NULL;
    for (int i = 0; i < v->n; i++) { free(v->addr[i]); free(v->idv[i]); }
    v->n = 0;
}

void sp_r3_vector_seal(sp_r3_vector *v) { if (v) v->sealed = 1; }

/* whole-set recall@1: for every bound episode q, unbind(M, addr_q) must have
 * id_q as argmax cosine over all bound ids, with margin > 0 (matches the
 * Python _recall_ok). */
int sp_r3_recall_ok(const sp_r3_vector *v) {
    if (!v || v->n == 0) return 1;
    int D = v->D;
    int64_t *est = malloc((size_t)D * sizeof(int64_t));
    if (!est) return 0;
    int ok = 1;
    for (int q = 0; q < v->n && ok; q++) {
        if (sp_r3_unbind(v->M, v->addr[q], D, est)) { ok = 0; break; }
        double sim_q = sp_r3_cos_i64_i8(est, v->idv[q], D);
        double best_other = 0.0; int have_other = 0; int best_is_q = 1;
        for (int e = 0; e < v->n; e++) {
            double s = sp_r3_cos_i64_i8(est, v->idv[e], D);
            if (e != q) {
                if (!have_other || s > best_other) { best_other = s; have_other = 1; }
            }
            if (e != q && s > sim_q) best_is_q = 0;      /* someone beats q -> argmax!=q */
        }
        double margin = have_other ? (sim_q - best_other) : sim_q;
        if (!best_is_q || margin <= 0.0) ok = 0;
    }
    free(est);
    return ok;
}

int sp_r3_try_bind(sp_r3_vector *v, uint64_t seed) {
    if (!v || v->sealed || v->n >= SP_R3_CAP) return 0;
    int D = v->D;
    int8_t *addr = malloc((size_t)D);
    int8_t *idv  = malloc((size_t)D);
    if (!addr || !idv) { free(addr); free(idv); return 0; }
    sp_r3_carrier(seed, D, addr);
    sp_r3_idvec  (seed, D, idv);

    /* SHADOW: M' = M + bind(addr,id); accept only if whole-set recall holds. */
    int slot = v->n;
    v->addr[slot] = addr; v->idv[slot] = idv; v->seeds[slot] = seed;
    int64_t *saved = malloc((size_t)D * sizeof(int64_t));
    if (!saved) { v->addr[slot] = NULL; v->idv[slot] = NULL; free(addr); free(idv); return 0; }
    memcpy(saved, v->M, (size_t)D * sizeof(int64_t));
    if (sp_r3_superpose(v->M, addr, idv, D)) {       /* commit candidate to M */
        memcpy(v->M, saved, (size_t)D * sizeof(int64_t));
        free(saved); v->addr[slot] = NULL; v->idv[slot] = NULL; free(addr); free(idv);
        return 0;
    }
    v->n = slot + 1;
    if (sp_r3_recall_ok(v)) { free(saved); return 1; }     /* PROMOTE */
    /* gate fail: roll back M + drop the candidate episode */
    memcpy(v->M, saved, (size_t)D * sizeof(int64_t));
    free(saved);
    v->n = slot;
    v->addr[slot] = NULL; v->idv[slot] = NULL;
    free(addr); free(idv);
    return 0;
}

/* ── nightshift driver (mirrors g_r3_nightshift.nightshift) ─────────────────── */
int sp_r3_nightshift(int D, const uint64_t *seeds, int n,
                     sp_r3_nightshift_result *out) {
    if (!sp_r3_dim_ok(D) || !seeds || !out) return 1;
    memset(out, 0, sizeof(*out));

    sp_r3_vector vecs[64];
    int nv = 0;
    if (sp_r3_vector_init(&vecs[nv++], D)) return 1;

    for (int i = 0; i < n; i++) {
        if (!sp_r3_try_bind(&vecs[nv - 1], seeds[i])) {
            sp_r3_vector_seal(&vecs[nv - 1]);
            if (nv >= 64) { for (int k = 0; k < nv; k++) sp_r3_vector_free(&vecs[k]); return 1; }
            if (sp_r3_vector_init(&vecs[nv++], D)) {
                for (int k = 0; k < nv - 1; k++) sp_r3_vector_free(&vecs[k]);
                return 1;
            }
            if (!sp_r3_try_bind(&vecs[nv - 1], seeds[i])) {
                /* fresh vector must accept (the Python asserts this) */
                for (int k = 0; k < nv; k++) sp_r3_vector_free(&vecs[k]);
                return 1;
            }
        }
        out->consolidated++;
    }

    int all_ok = 1, sealed = 0;
    for (int k = 0; k < nv; k++) {
        if (!sp_r3_recall_ok(&vecs[k])) all_ok = 0;
        if (vecs[k].sealed) sealed++;
        out->sizes[k] = vecs[k].n;
    }
    out->n_vectors = nv;
    out->sealed = sealed;
    out->all_recall_ok = all_ok;

    /* production (D>=1024): non-last vectors sealed at CAP. small D: gate fires
     * before CAP (max vector size < CAP). */
    if (D >= 1024) {
        int cap = 1;
        for (int k = 0; k < nv - 1; k++) if (vecs[k].n != SP_R3_CAP) cap = 0;
        out->cap_seal = cap;
        out->gate_seal = 1;
    } else {
        int mx = 0;
        for (int k = 0; k < nv; k++) if (vecs[k].n > mx) mx = vecs[k].n;
        out->gate_seal = (mx < SP_R3_CAP) ? 1 : 0;
        out->cap_seal = 1;
    }

    for (int k = 0; k < nv; k++) sp_r3_vector_free(&vecs[k]);
    return 0;
}
