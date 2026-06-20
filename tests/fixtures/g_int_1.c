/* g_int_1.c — G-INT-1: NIGHTSHIFT episode K survives VHT2 spinor -> Ring-2 -> reload.
 *
 * CONTRACT-PPT-ARM-LAT-INTEGRATION first wire (lattice 7b58841).
 * Corrected gate: storage byte-EXACT + compression C2-index-PRESERVING.
 *
 * Input : a REAL captured episode's ep.k (raw LE f32 [NL=48][P][HD=512]).
 *         Extract GLOBAL layers (L%6==5) -> gk [ng=8][npos][512] (recall.rs layout).
 * Per gk vector v[512]:
 *   sp_spinor_encode_vec(v,512,blocks) (10 blocks) -> pack each (63B)
 *   -> WRITE via Ring-2 stdio backend -> READ back -> unpack -> decode -> v'[512]
 * Gate (a) STORAGE byte-exact : packed bytes read back bit-identical (memcmp==0)
 *                               AND unpack(pack(b))==b for all blocks.
 * Gate (b) COMPRESSION recall-preserving : 256-bit C2 sign-projection signature
 *          (recall.rs Projection: SEED, R_BITS=256, HD=512, sig[b]=sign(mean over
 *          (layer,pos) of R[b].K)) over ORIGINAL gk and RECON gk -> Hamming==0.
 * Reports : storage diffs, sig Hamming, mean per-vector recon relL2, byte-shrink,
 *           blocks/vector, npos.
 */
#include "sp/spinor_block.h"
#include "sp/arm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define NL      48
#define HD      512
#define PERIOD  6
#define R_BITS  256
#define SEED    0x5350524F4A2BULL

/* ---- recall.rs::Projection mirror -------------------------------------------
 * smix(SEED, R_BITS*HD) -> +-1 stream (z&1 ? +1 : -1), row-major R[R_BITS][HD]. */
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

/* sig[b] = sign( mean over (global layer, pos) of R[b].K ), packed 4x u64 LE-bit.
 * gk is [n_vec][HD] row-major (n_vec = n_global*npos). f64 accumulation (mirror). */
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

    /* ---- load ep.k -------------------------------------------------------- */
    char kpath[1024];
    snprintf(kpath, sizeof kpath, "%s/ep.k", ep_dir);
    FILE *f = fopen(kpath, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", kpath); return 2; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    size_t n_f32 = (size_t)fsz / 4;
    int p_total = (int)(n_f32 / ((size_t)NL * HD));        /* floor (recall.rs) */
    if (p_total == 0) { fprintf(stderr, "ep.k too small\n"); return 2; }
    if (npos > p_total) npos = p_total;
    float *raw = (float*)malloc((size_t)n_f32 * sizeof(float));
    if (fread(raw, 4, n_f32, f) != n_f32) { fprintf(stderr, "short read\n"); return 2; }
    fclose(f);

    /* global layers L%PERIOD==PERIOD-1 -> gk [ng][npos][HD] (recall.rs layout). */
    int globals[NL]; int ng = 0;
    for (int l = 0; l < NL; l++) if (l % PERIOD == PERIOD-1) globals[ng++] = l;
    size_t n_vec = (size_t)ng * npos;
    float *gk = (float*)malloc(n_vec * HD * sizeof(float));
    for (int gi = 0; gi < ng; gi++) {
        int l = globals[gi];
        for (int p = 0; p < npos; p++) {
            size_t src0 = ((size_t)l * p_total + p) * HD;
            size_t dst0 = ((size_t)gi * npos + p) * HD;
            for (int d = 0; d < HD; d++) gk[dst0 + d] = raw[src0 + d];
        }
    }
    free(raw);

    /* ---- open Ring-2 stdio backend --------------------------------------- */
    sp_arm_ring2_backend be;
    if (sp_arm_ring2_stdio_open(ring_dir, &be) != 0) {
        fprintf(stderr, "ring2 stdio open failed in %s\n", ring_dir); return 2;
    }

    int blocks_per_vec = sp_spinor_blocks_for(HD);        /* 512 -> 10 */
    const size_t BLK = 63;                                /* packed block bytes */

    /* recon buffer for the whole gk */
    float *gk_recon = (float*)malloc(n_vec * HD * sizeof(float));

    long storage_diffs = 0;       /* (a) memcmp mismatches read-vs-written */
    long unpack_diffs  = 0;       /* (a) unpack(pack(b)) != b */
    double rel_sum = 0.0; size_t rel_n = 0;   /* (b) per-vector relL2 */

    sp_spinor_block_t *blks = (sp_spinor_block_t*)malloc((size_t)blocks_per_vec * sizeof(sp_spinor_block_t));

    for (size_t v = 0; v < n_vec; v++) {
        const float *vec = gk + v * HD;
        /* encode -> pack -> write each block to Ring-2 (K stream, which=0) */
        sp_spinor_encode_vec(vec, HD, blks);
        for (int b = 0; b < blocks_per_vec; b++) {
            uint8_t packed[63];
            sp_spinor_pack(&blks[b], packed);
            uint64_t off = ((uint64_t)v * blocks_per_vec + b) * BLK;
            if (be.write_block(be.handle, 0, off, packed, BLK) != 0) {
                fprintf(stderr, "write_block failed v=%zu b=%d\n", v, b); return 2;
            }
        }
    }
    /* read back, verify storage byte-exactness + unpack bijection, decode -> recon */
    for (size_t v = 0; v < n_vec; v++) {
        const float *vec = gk + v * HD;
        sp_spinor_encode_vec(vec, HD, blks);          /* re-derive expected packed */
        for (int b = 0; b < blocks_per_vec; b++) {
            uint8_t expected[63], readback[63];
            sp_spinor_pack(&blks[b], expected);
            uint64_t off = ((uint64_t)v * blocks_per_vec + b) * BLK;
            if (be.read_block(be.handle, 0, off, readback, BLK) != 0) {
                fprintf(stderr, "read_block failed v=%zu b=%d\n", v, b); return 2;
            }
            if (memcmp(expected, readback, BLK) != 0) storage_diffs++;
            /* bijection: unpack(pack(b)) == b (compare full 63-byte image) */
            sp_spinor_block_t rb; sp_spinor_unpack(readback, &rb);
            uint8_t repack[63]; sp_spinor_pack(&rb, repack);
            if (memcmp(readback, repack, BLK) != 0) unpack_diffs++;
            /* feed the read-back block into the recon stream */
            sp_spinor_unpack(readback, &blks[b]);
        }
        /* decode the read-back blocks -> v'[512] */
        float *vrec = gk_recon + v * HD;
        sp_spinor_decode_vec(blks, HD, vrec);
        /* per-vector relL2 = ||v - v'|| / ||v|| */
        double num = 0.0, den = 0.0;
        for (int d = 0; d < HD; d++) {
            double dv = (double)vec[d] - (double)vrec[d];
            num += dv * dv; den += (double)vec[d] * (double)vec[d];
        }
        if (den > 0.0) { rel_sum += sqrt(num) / sqrt(den); rel_n++; }
    }
    be.close(be.handle);

    /* ---- (b) C2 signature over original vs recon -------------------------- */
    signed char *R = (signed char*)malloc((size_t)R_BITS * HD);
    build_R(R);
    uint64_t sig_orig[4], sig_recon[4];
    signature(R, gk, n_vec, sig_orig);
    signature(R, gk_recon, n_vec, sig_recon);
    int ham = hamming256(sig_orig, sig_recon);

    double mean_rel = rel_n ? rel_sum / (double)rel_n : 0.0;
    /* shrink: f32 gk bytes vs spinor store bytes */
    double f32_bytes    = (double)n_vec * HD * 4.0;
    double spinor_bytes = (double)n_vec * blocks_per_vec * (double)BLK;
    double shrink       = f32_bytes / spinor_bytes;

    int green = (storage_diffs == 0 && unpack_diffs == 0 && ham == 0);

    /* ---- emit log --------------------------------------------------------- */
    FILE *lg = fopen(log_path, "w");
    FILE *outs[2] = { stdout, lg };
    for (int o = 0; o < 2; o++) {
        FILE *out = outs[o]; if (!out) continue;
        fprintf(out, "=== G-INT-1 : NIGHTSHIFT episode K  spinor -> Ring-2 -> reload ===\n");
        fprintf(out, "episode dir      : %s\n", ep_dir);
        fprintf(out, "ring2 dir        : %s\n", ring_dir);
        fprintf(out, "ep.k bytes       : %ld  (f32=%zu, NL=%d HD=%d)\n", fsz, n_f32, NL, HD);
        fprintf(out, "p_total (floor)  : %d\n", p_total);
        fprintf(out, "npos             : %d\n", npos);
        fprintf(out, "n_global (L%%6==5): %d\n", ng);
        fprintf(out, "gk vectors       : %zu  ([ng=%d][npos=%d][HD=%d])\n", n_vec, ng, npos, HD);
        fprintf(out, "blocks / vector  : %d  (sp_spinor_blocks_for(512))\n", blocks_per_vec);
        fprintf(out, "packed block     : %zu bytes\n", BLK);
        fprintf(out, "--- (a) STORAGE byte-exact ---\n");
        fprintf(out, "storage diffs    : %ld   (read-vs-written memcmp; MUST be 0)\n", storage_diffs);
        fprintf(out, "unpack diffs     : %ld   (unpack(pack(b))!=b; MUST be 0)\n", unpack_diffs);
        fprintf(out, "--- (b) COMPRESSION C2-index survival ---\n");
        fprintf(out, "sig Hamming      : %d / 256   (MUST be 0)\n", ham);
        fprintf(out, "sig_orig         : %016llx%016llx%016llx%016llx\n",
                (unsigned long long)sig_orig[3],(unsigned long long)sig_orig[2],
                (unsigned long long)sig_orig[1],(unsigned long long)sig_orig[0]);
        fprintf(out, "sig_recon        : %016llx%016llx%016llx%016llx\n",
                (unsigned long long)sig_recon[3],(unsigned long long)sig_recon[2],
                (unsigned long long)sig_recon[1],(unsigned long long)sig_recon[0]);
        fprintf(out, "--- honest compression cost ---\n");
        fprintf(out, "mean recon relL2 : %.6e   (per-vector lossy cost)\n", mean_rel);
        fprintf(out, "f32 store bytes  : %.0f\n", f32_bytes);
        fprintf(out, "spinor store     : %.0f bytes\n", spinor_bytes);
        fprintf(out, "shrink (f32/spin): %.4fx\n", shrink);
        fprintf(out, "=== VERDICT : %s ===\n", green ? "G-INT-1 GREEN" : "HONEST NEGATIVE");
    }
    if (lg) fclose(lg);

    free(R); free(gk); free(gk_recon); free(blks);
    return green ? 0 : 1;
}
