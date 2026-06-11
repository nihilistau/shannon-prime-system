/* xbar_manifest_gate.c — XBAR P3.0 standalone gate G-P3-0 (CONTRACT-XBAR-P3 §2).
 *
 * No model, no decode (the curator_replay.c discipline). Grades the episode/
 * owner-map manifest in isolation against three pre-stated checks:
 *
 *   (1) ROUND-TRIP byte-exact — build → serialize → deserialize → re-serialize
 *       reproduces the same bytes, and the deserialized record array is
 *       field-identical.
 *   (2) UNIFORM-NULL — on uniform geometry (qwen3 dims) AND on the KVD-constant
 *       12B (global 1x512 == SWA 2x256), the manifest-driven block offset
 *       reproduces the legacy ((L*P)+pos)*KVD*4 layout EXACTLY.
 *   (3) JAGGED ORACLE — on the genuinely-jagged E2B (global 512 / SWA 256 owner
 *       rows + 15 owners / 20 sharers), every block offset matches an
 *       INDEPENDENT brute-force prefix-sum + owner-indirection oracle.
 *
 * Falsification (pre-stated, CONTRACT §2): any manifest offset differing from the
 * legacy layout on uniform/const geometry, or from the oracle on jagged.
 *
 * Build/run (WSL/MinGW, libc only):
 *   gcc -O2 -std=c11 -Iinclude tools/curator/xbar_manifest_gate.c \
 *       core/xbar/xbar_episode.c -o /tmp/xbar_gate && /tmp/xbar_gate
 */
#include "sp/xbar_episode.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* fill geom[NL] from a class rule global=(L%period==period-1) + per-class dims */
static void mk_geom(sp_xbar_layer_geom *g, int NL, int period,
                    int g_nkv, int g_hd, int s_nkv, int s_hd)
{
    for (int L = 0; L < NL; L++) {
        int global = ((L % period) == period - 1);
        g[L].cls = global ? SP_XBAR_CLASS_GLOBAL : SP_XBAR_CLASS_SWA;
        g[L].nkv = global ? g_nkv : s_nkv;
        g[L].hd  = global ? g_hd  : s_hd;
        g[L].nh  = global ? 4 : 8;
        g[L].window = global ? -1 : 1024;
        g[L].rope_base = global ? 1000000.0f : 10000.0f;
        g[L].has_freq_factors = global ? 1 : 0;
        g[L].vless = global ? 1 : 0;   /* 12B dense globals are V-less */
    }
}

static int kvd_of(const sp_xbar_layer_geom *g, int L) { return g[L].nkv * g[L].hd; }

/* ── (1) round-trip ─────────────────────────────────────────────────────────── */
static int check_roundtrip(const char *name, const sp_xbar_manifest *mf)
{
    size_t n = sp_xbar_manifest_serial_size(mf);
    uint8_t *b1 = malloc(n), *b2 = malloc(n);
    size_t w1 = sp_xbar_manifest_serialize(mf, b1, n);
    sp_xbar_manifest mf2; memset(&mf2, 0, sizeof mf2);
    int de = sp_xbar_manifest_deserialize(&mf2, b1, w1);
    size_t w2 = sp_xbar_manifest_serialize(&mf2, b2, n);
    int bytes_eq = (w1 == n && w2 == n && de == 0 && memcmp(b1, b2, n) == 0);
    /* deep field equality of the record arrays + header scalars */
    int fields_eq = (mf2.NL == mf->NL && mf2.P == mf->P && mf2.period == mf->period &&
                     mf2.kvfs == mf->kvfs && mf2.r == mf->r && mf2.version == mf->version &&
                     mf2.proj_seed == mf->proj_seed && mf2.store_bytes == mf->store_bytes &&
                     memcmp(mf2.artifact_sha, mf->artifact_sha, SP_XBAR_SHA_LEN) == 0);
    if (fields_eq)
        for (int L = 0; L < mf->NL; L++)
            if (memcmp(&mf2.layers[L], &mf->layers[L], sizeof(sp_xbar_layer)) != 0) { fields_eq = 0; break; }
    int ok = bytes_eq && fields_eq;
    printf("    [%-5s] serialize->deserialize->serialize byte-exact + fields : %s\n",
           name, ok ? "PASS" : "FAIL");
    free(b1); free(b2); sp_xbar_manifest_free(&mf2);
    return ok;
}

/* ── (2) uniform / KVD-const null: legacy ((L*P)+pos)*KVD*4 ──────────────────── */
static int check_uniform(const char *name, const sp_xbar_manifest *mf, int KVD)
{
    int ok = 1, P = mf->P;
    const int poses[] = { 0, 1, 2, 7, 63, 100, P - 1 };
    for (int L = 0; L < mf->NL && ok; L++)
        for (size_t pi = 0; pi < sizeof poses / sizeof poses[0]; pi++) {
            int pos = poses[pi];
            uint64_t got = sp_xbar_block_off(mf, L, pos);
            uint64_t exp = ((uint64_t)((size_t)L * P + pos)) * (uint64_t)KVD * 4u;
            if (got != exp) { ok = 0;
                printf("      MISMATCH L=%d pos=%d got=%llu exp=%llu\n",
                       L, pos, (unsigned long long)got, (unsigned long long)exp); break; }
        }
    printf("    [%-5s] block(L,pos) == ((L*P)+pos)*%d*4  (KVD const)          : %s\n",
           name, KVD, ok ? "PASS" : "FAIL");
    return ok;
}

/* ── (3) jagged oracle: independent prefix-sum + owner indirection ───────────── */
static int check_jagged(const char *name, const sp_xbar_manifest *mf,
                        const sp_xbar_layer_geom *g, int kvfs)
{
    int NL = mf->NL, P = mf->P, ok = 1;
    uint64_t *off = malloc((size_t)NL * sizeof(uint64_t));
    int *own = malloc((size_t)NL * sizeof(int));
    /* owners: prefix-sum of P*kvd*4 */
    uint64_t cur = 0;
    for (int L = 0; L < NL; L++)
        if (L < kvfs) { own[L] = L; off[L] = cur; cur += (uint64_t)P * (uint64_t)kvd_of(g, L) * 4u; }
    /* sharers: own = kvfs-1 (global) / kvfs-2 (SWA); off = owner's off */
    for (int L = 0; L < NL; L++)
        if (L >= kvfs) {
            int global = (g[L].cls == SP_XBAR_CLASS_GLOBAL);
            int src = kvfs - (global ? 1 : 2);
            own[L] = src; off[L] = off[src];
        }
    /* compare own[L], owns_kv, off, store_bytes, and block(L,pos) */
    if (cur != mf->store_bytes) { ok = 0; printf("      store_bytes oracle=%llu mf=%llu\n",
        (unsigned long long)cur, (unsigned long long)mf->store_bytes); }
    const int poses[] = { 0, 1, 5, 42, 511, P - 1 };
    for (int L = 0; L < NL && ok; L++) {
        if (mf->layers[L].own != own[L]) { ok = 0; printf("      own[%d] oracle=%d mf=%d\n", L, own[L], mf->layers[L].own); break; }
        if (mf->layers[L].owns_kv != (uint8_t)(L < kvfs)) { ok = 0; printf("      owns_kv[%d] wrong\n", L); break; }
        int kvd_own = kvd_of(g, own[L]);   /* owner's kvd (== L's class kvd) */
        for (size_t pi = 0; pi < sizeof poses / sizeof poses[0]; pi++) {
            int pos = poses[pi];
            uint64_t got = sp_xbar_block_off(mf, L, pos);
            uint64_t exp = off[L] + (uint64_t)pos * (uint64_t)kvd_own * 4u;
            if (got != exp) { ok = 0; printf("      MISMATCH L=%d pos=%d got=%llu exp=%llu\n",
                L, pos, (unsigned long long)got, (unsigned long long)exp); break; }
        }
    }
    printf("    [%-5s] block(L,pos) == brute-force prefix-sum+owner-indirect    : %s\n",
           name, ok ? "PASS" : "FAIL");
    free(off); free(own);
    return ok;
}

int main(void)
{
    uint8_t sha[SP_XBAR_SHA_LEN];
    for (int i = 0; i < SP_XBAR_SHA_LEN; i++) sha[i] = (uint8_t)(0xA0 + i);
    const uint64_t SEED = SP_ARM_PROJ_SEED;
    const int P = 2048;
    int all = 1;

    printf("G-P3-0  XBAR P3.0 episode/owner-map manifest (CONTRACT-XBAR-P3 §2)\n");
    printf("  P=%d  proj_seed=0x%llX\n\n", P, (unsigned long long)SEED);

    /* ── qwen3 uniform: NL=28, all owners, nkv8/hd128 -> KVD=1024 ── */
    {
        int NL = 28, period = 7, kvfs = NL;
        sp_xbar_layer_geom *g = malloc((size_t)NL * sizeof *g);
        mk_geom(g, NL, period, 8, 128, 8, 128);   /* uniform: both classes 8x128 */
        sp_xbar_manifest mf; memset(&mf, 0, sizeof mf);
        int rc = sp_xbar_manifest_build(&mf, NL, P, period, kvfs, 32, SEED, sha, g);
        printf("  qwen3-uniform  NL=%d kvfs=%d period=%d  KVD=1024 (all owners)\n", NL, kvfs, period);
        if (rc) { printf("    BUILD FAILED rc=%d\n", rc); all = 0; }
        else {
            all &= check_roundtrip("qwen3", &mf);
            all &= check_uniform ("qwen3", &mf, 1024);
        }
        sp_xbar_manifest_free(&mf); free(g);
    }
    printf("\n");

    /* ── 12B: NL=48, period6, all owners, global 1x512 / SWA 2x256 -> KVD const 512 ── */
    {
        int NL = 48, period = 6, kvfs = NL;
        sp_xbar_layer_geom *g = malloc((size_t)NL * sizeof *g);
        mk_geom(g, NL, period, 1, 512, 2, 256);
        sp_xbar_manifest mf; memset(&mf, 0, sizeof mf);
        int rc = sp_xbar_manifest_build(&mf, NL, P, period, kvfs, 32, SEED, sha, g);
        printf("  gemma4-12B     NL=%d kvfs=%d period=%d  KVD=512 const (1x512==2x256)\n", NL, kvfs, period);
        if (rc) { printf("    BUILD FAILED rc=%d\n", rc); all = 0; }
        else {
            all &= check_roundtrip("12B", &mf);
            all &= check_uniform ("12B", &mf, 512);
        }
        sp_xbar_manifest_free(&mf); free(g);
    }
    printf("\n");

    /* ── E2B: NL=35, period5, kvfs=15, global 1x512 / SWA 2x256 -> JAGGED 512/256 ── */
    {
        int NL = 35, period = 5, kvfs = 15;
        sp_xbar_layer_geom *g = malloc((size_t)NL * sizeof *g);
        mk_geom(g, NL, period, 1, 512, 2, 256);
        sp_xbar_manifest mf; memset(&mf, 0, sizeof mf);
        int rc = sp_xbar_manifest_build(&mf, NL, P, period, kvfs, 32, SEED, sha, g);
        printf("  gemma4-E2B     NL=%d kvfs=%d period=%d  JAGGED global512/swa256, %d owners/%d sharers\n",
               NL, kvfs, period, kvfs, NL - kvfs);
        if (rc) { printf("    BUILD FAILED rc=%d\n", rc); all = 0; }
        else {
            all &= check_roundtrip("E2B", &mf);
            all &= check_jagged   ("E2B", &mf, g, kvfs);
        }
        sp_xbar_manifest_free(&mf); free(g);
    }

    printf("\n  G-P3-0: %s\n", all ? "PASS" : "FAIL");
    return all ? 0 : 1;
}
