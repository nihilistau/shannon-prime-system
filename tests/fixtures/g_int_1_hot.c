/* g_int_1_hot.c — G-INT-1 HOT PERSISTENCE (revised, two-tier f32 payload).
 *
 * CONTRACT-PPT-ARM-LAT-INTEGRATION (lattice ab8884e), amended two-tier Ring-2:
 *   HOT tier  : the recall PAYLOAD is LOSSLESS f32 K/V (NOT spinor — 0.7% int8
 *               noise contaminates the causal oracle's dLL). This gate.
 *   COLD tier : spinor byte-exact storage, sig flips 1/256 on lossy recompute
 *               — DEFERRED to G-INT-3 (see prior g_int_1.c honest negative).
 *   INDEX     : the EXACT 256-bit C2 sig captured at ingest (here persisted
 *               verbatim alongside the f32 payload).
 *
 * HOT round-trip — NO spinor on this path:
 *  1. Load f32 K -> gk [ng=8][npos][512] (recall.rs global-layer extract L%6==5),
 *     lossless exactly as captured. Compute the EXACT C2 sig once at ingest.
 *  2. Write the raw f32 payload bytes of gk to Ring-2 (K stream, which=0).
 *     Persist the 256-bit sig as a small index write (V stream, which=1).
 *  3. Read both back.
 *  4. GATE PASS iff (a) memcmp(written,read, ng*npos*512*4)==0, and
 *     (b) sig Hamming==0 (stored round-trip AND recompute-from-readback).
 */
#include "sp/arm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define NL      48
#define HD      512
#define PERIOD  6
#define R_BITS  256
#define SEED    0x5350524F4A2BULL

static void build_R(signed char *R) {
    uint64_t s = SEED;
    for (size_t i = 0; i < (size_t)R_BITS * HD; i++) {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^=  z >> 31;
        R[i] = (z & 1ULL) ? 1 : -1;
    }
}

static void signature(const signed char *R, const float *gk, size_t n_vec,
                      uint64_t words[4]) {
    double *acc = (double*)calloc(R_BITS, sizeof(double));
    for (size_t v = 0; v < n_vec; v++) {
        const float *kb = gk + v * HD;
        for (int b = 0; b < R_BITS; b++) {
            const signed char *rb = R + (size_t)b * HD;
            double dot = 0.0;
            for (int d = 0; d < HD; d++) dot += (double)rb[d] * (double)kb[d];
            acc[b] += dot;
        }
    }
    words[0]=words[1]=words[2]=words[3]=0;
    double nv = (double)n_vec;
    for (int b = 0; b < R_BITS; b++) {
        double mean = acc[b] / nv;
        if (mean > 0.0) words[b/64] |= (1ULL << (b%64));
    }
    free(acc);
}

static int hamming256(const uint64_t a[4], const uint64_t b[4]) {
    int h = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t x = a[i] ^ b[i];
        while (x) { x &= (x-1); h++; }
    }
    return h;
}

int main(int argc, char **argv) {
    const char *ep_dir   = (argc > 1) ? argv[1] : ".";
    const char *ring_dir = (argc > 2) ? argv[2] : ".";
    int npos             = (argc > 3) ? atoi(argv[3]) : 22;
    const char *log_path = (argc > 4) ? argv[4] : "G-INT-1.log";

    char kpath[1024];
    snprintf(kpath, sizeof kpath, "%s/ep.k", ep_dir);
    FILE *f = fopen(kpath, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", kpath); return 2; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    size_t n_f32 = (size_t)fsz / 4;
    int p_total = (int)(n_f32 / ((size_t)NL * HD));
    if (p_total == 0) { fprintf(stderr, "ep.k too small\n"); return 2; }
    if (npos > p_total) npos = p_total;
    float *raw = (float*)malloc((size_t)n_f32 * sizeof(float));
    if (fread(raw, 4, n_f32, f) != n_f32) { fprintf(stderr, "short read\n"); return 2; }
    fclose(f);

    int globals[NL]; int ng = 0;
    for (int l = 0; l < NL; l++) if (l % PERIOD == PERIOD-1) globals[ng++] = l;
    size_t n_vec = (size_t)ng * npos;
    size_t payload_floats = n_vec * HD;
    size_t payload_bytes  = payload_floats * sizeof(float);
    float *gk = (float*)malloc(payload_bytes);
    for (int gi = 0; gi < ng; gi++) {
        int l = globals[gi];
        for (int p = 0; p < npos; p++) {
            size_t src0 = ((size_t)l * p_total + p) * HD;
            size_t dst0 = ((size_t)gi * npos + p) * HD;
            for (int d = 0; d < HD; d++) gk[dst0 + d] = raw[src0 + d];
        }
    }
    free(raw);

    signed char *R = (signed char*)malloc((size_t)R_BITS * HD);
    build_R(R);
    uint64_t sig_ingest[4];
    signature(R, gk, n_vec, sig_ingest);

    sp_arm_ring2_backend be;
    if (sp_arm_ring2_stdio_open(ring_dir, &be) != 0) {
        fprintf(stderr, "ring2 stdio open failed in %s\n", ring_dir); return 2;
    }

    const size_t CHUNK = HD * sizeof(float);
    int w_err = 0;
    for (size_t v = 0; v < n_vec; v++) {
        uint64_t off = (uint64_t)v * CHUNK;
        if (be.write_block(be.handle, 0, off,
                           (const uint8_t*)gk + off, CHUNK) != 0) { w_err = 1; break; }
    }
    if (!w_err) {
        if (be.write_block(be.handle, 1, 0, sig_ingest, sizeof sig_ingest) != 0) w_err = 1;
    }
    if (w_err) { fprintf(stderr, "write_block failed\n"); be.close(be.handle); return 2; }

    float *gk_read = (float*)malloc(payload_bytes);
    int r_err = 0;
    for (size_t v = 0; v < n_vec; v++) {
        uint64_t off = (uint64_t)v * CHUNK;
        if (be.read_block(be.handle, 0, off,
                         (uint8_t*)gk_read + off, CHUNK) != 0) { r_err = 1; break; }
    }
    uint64_t sig_stored[4] = {0,0,0,0};
    if (!r_err) {
        if (be.read_block(be.handle, 1, 0, sig_stored, sizeof sig_stored) != 0) r_err = 1;
    }
    be.close(be.handle);
    if (r_err) { fprintf(stderr, "read_block failed\n"); return 2; }

    int payload_cmp = memcmp(gk, gk_read, payload_bytes);
    long byte_diffs = 0; long first_diff = -1;
    const uint8_t *pa = (const uint8_t*)gk, *pb = (const uint8_t*)gk_read;
    for (size_t i = 0; i < payload_bytes; i++) {
        if (pa[i] != pb[i]) { byte_diffs++; if (first_diff < 0) first_diff = (long)i; }
    }
    int ham_stored = hamming256(sig_ingest, sig_stored);
    uint64_t sig_recompute[4];
    signature(R, gk_read, n_vec, sig_recompute);
    int ham_recompute = hamming256(sig_ingest, sig_recompute);

    int green = (payload_cmp == 0 && byte_diffs == 0 &&
                 ham_stored == 0 && ham_recompute == 0);

    FILE *lg = fopen(log_path, "a");
    FILE *outs[2] = { stdout, lg };
    for (int o = 0; o < 2; o++) {
        FILE *out = outs[o]; if (!out) continue;
        fprintf(out, "\n=== G-INT-1 HOT (revised, two-tier f32 payload) ===\n");
        fprintf(out, "contract         : CONTRACT-PPT-ARM-LAT-INTEGRATION (amended, lattice ab8884e)\n");
        fprintf(out, "tier             : HOT recall payload = LOSSLESS f32 K (no spinor); INDEX = exact 256-bit C2 sig\n");
        fprintf(out, "episode dir      : %s\n", ep_dir);
        fprintf(out, "ring2 dir        : %s\n", ring_dir);
        fprintf(out, "ep.k bytes       : %ld  (f32=%zu, NL=%d HD=%d)\n", fsz, n_f32, NL, HD);
        fprintf(out, "p_total (floor)  : %d\n", p_total);
        fprintf(out, "npos             : %d\n", npos);
        fprintf(out, "n_global (L%%6==5): %d\n", ng);
        fprintf(out, "gk vectors       : %zu  ([ng=%d][npos=%d][HD=%d])\n", n_vec, ng, npos, HD);
        fprintf(out, "--- (a) PAYLOAD byte-identical (HOT f32 round-trip) ---\n");
        fprintf(out, "payload floats   : %zu\n", payload_floats);
        fprintf(out, "payload bytes    : %zu   (= ng*npos*512*4)\n", payload_bytes);
        fprintf(out, "memcmp result    : %d   (MUST be 0)\n", payload_cmp);
        fprintf(out, "byte diffs       : %ld   (MUST be 0)\n", byte_diffs);
        fprintf(out, "first diff off   : %ld   (-1 = none)\n", first_diff);
        fprintf(out, "--- (b) C2 sig Hamming (exact index) ---\n");
        fprintf(out, "sig stored r/trip: %d / 256   (MUST be 0)\n", ham_stored);
        fprintf(out, "sig recompute    : %d / 256   (recomputed from read-back f32; MUST be 0)\n", ham_recompute);
        fprintf(out, "sig_ingest       : %016llx%016llx%016llx%016llx\n",
                (unsigned long long)sig_ingest[3],(unsigned long long)sig_ingest[2],
                (unsigned long long)sig_ingest[1],(unsigned long long)sig_ingest[0]);
        fprintf(out, "sig_stored       : %016llx%016llx%016llx%016llx\n",
                (unsigned long long)sig_stored[3],(unsigned long long)sig_stored[2],
                (unsigned long long)sig_stored[1],(unsigned long long)sig_stored[0]);
        fprintf(out, "sig_recompute    : %016llx%016llx%016llx%016llx\n",
                (unsigned long long)sig_recompute[3],(unsigned long long)sig_recompute[2],
                (unsigned long long)sig_recompute[1],(unsigned long long)sig_recompute[0]);
        fprintf(out, "=== VERDICT : %s ===\n",
                green ? "G-INT-1 HOT GREEN" : "RING-2 BYTE-FIDELITY BUG");
    }
    if (lg) fclose(lg);

    free(R); free(gk); free(gk_read);
    return green ? 0 : 1;
}
