/* curator_replay.c — XBAR C1-lite C1L.0a: episode persistence + router-index
 * re-projection determinism (CONTRACT-XBAR-C1-lite §2).
 *
 * The seam survey (decode.c) established: the recall router index `projk` is the
 * frozen ±1 Rademacher projection of each post-RoPE K (sp_arm_project), and the
 * SAME K is what spills to the Ring-2 store. Therefore a persisted episode does
 * NOT need to serialize projk — it is RECOVERABLE by re-projecting the stored K
 * on reload. This is the foundation of the replay-decode mode (C1L.0b): load a
 * persisted {K, V} episode, rebuild projk, decode against it.
 *
 * This harness proves that foundation against the REAL arm.c functions, with no
 * decode surgery and no model: synth an episode, persist K, reload, re-project,
 * and gate that the rebuilt router index is BIT-IDENTICAL to the original.
 *
 * It also writes the episode-persistence layout NIGHTSHIFT will reuse:
 *   <dir>/k.bin       NPOS*NKV*HD f32  (the verbatim K stream — the store)
 *   <dir>/manifest.txt  npos nkv hd r seed  (geometry to rebuild on reload)
 *
 * Build/run (WSL, links the real ARM module, no model):
 *   gcc -O2 -std=c11 -Iinclude tools/curator/curator_replay.c core/arm/arm.c -lm -o /tmp/replay && /tmp/replay
 */
#define _POSIX_C_SOURCE 200809L
#include "sp/arm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static float frand(unsigned *s) { *s = *s * 1103515245u + 12345u; return (float)((*s >> 9) & 0xFFFF) / 32768.0f - 1.0f; }

int main(void) {
    const int NPOS = 64, NKV = 8, HD = 128, R = 32;   /* qwen3-class geometry */
    const char *dir = "/tmp/xbar_episode";
    mkdir(dir, 0755);
    setenv("SP_ARM_PROJ_SEED", "424242", 0);          /* frozen projection seed */

    /* frozen ±1 matrix (the router's projection) */
    signed char *Rm = malloc((size_t)R * HD);
    sp_arm_build_R(Rm, R, HD);

    /* synth episode K: [NPOS][NKV][HD] (stands in for post-RoPE stored K) */
    const size_t NK = (size_t)NPOS * NKV * HD;
    float *K = malloc(NK * sizeof(float));
    unsigned s = 99991;
    for (size_t i = 0; i < NK; i++) K[i] = frand(&s) * 2.0f;

    /* (1) ORIGINAL router index: project each (pos,kvh) head-vector in RAM */
    const size_t NP = (size_t)NPOS * NKV * R;
    float *projk_orig = malloc(NP * sizeof(float));
    for (int p = 0; p < NPOS; p++)
        for (int h = 0; h < NKV; h++)
            sp_arm_project(Rm, R, HD, K + ((size_t)p * NKV + h) * HD,
                           projk_orig + ((size_t)p * NKV + h) * R);

    /* (2) PERSIST the episode: write K store + manifest */
    char path[1024];
    snprintf(path, sizeof path, "%s/k.bin", dir);
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(K, sizeof(float), NK, f) != NK) { fprintf(stderr, "k.bin write fail\n"); return 1; }
    fclose(f);
    snprintf(path, sizeof path, "%s/manifest.txt", dir);
    f = fopen(path, "w"); fprintf(f, "npos=%d nkv=%d hd=%d r=%d seed=424242\n", NPOS, NKV, HD, R); fclose(f);

    /* (3) RELOAD the episode (read-only — NOT the decode's w+b truncate path) and
     *     REBUILD the router index by re-projecting the stored K. */
    float *Kload = malloc(NK * sizeof(float));
    snprintf(path, sizeof path, "%s/k.bin", dir);
    f = fopen(path, "rb");
    if (!f || fread(Kload, sizeof(float), NK, f) != NK) { fprintf(stderr, "k.bin read fail\n"); return 1; }
    fclose(f);
    int store_roundtrip = (memcmp(K, Kload, NK * sizeof(float)) == 0);

    signed char *Rm2 = malloc((size_t)R * HD);        /* rebuilt from the SAME seed */
    sp_arm_build_R(Rm2, R, HD);
    float *projk_rebuilt = malloc(NP * sizeof(float));
    for (int p = 0; p < NPOS; p++)
        for (int h = 0; h < NKV; h++)
            sp_arm_project(Rm2, R, HD, Kload + ((size_t)p * NKV + h) * HD,
                           projk_rebuilt + ((size_t)p * NKV + h) * R);

    /* (4) GATE G-C1L-0a: rebuilt router index == original, BIT-IDENTICAL */
    int proj_bitexact = (memcmp(projk_orig, projk_rebuilt, NP * sizeof(float)) == 0);
    int R_bitexact    = (memcmp(Rm, Rm2, (size_t)R * HD) == 0);

    printf("G-C1L-0a episode persistence + re-projection determinism\n");
    printf("  store K round-trip (write->read) bit-identical : %s\n", store_roundtrip ? "PASS" : "FAIL");
    printf("  frozen R rebuilt from seed bit-identical        : %s\n", R_bitexact ? "PASS" : "FAIL");
    printf("  router index projk re-projected bit-identical   : %s\n", proj_bitexact ? "PASS" : "FAIL");
    int ok = store_roundtrip && R_bitexact && proj_bitexact;
    printf("\n  C1L.0a: %s  (projk is recoverable from the persisted K store — replay-decode foundation)\n",
           ok ? "PASS" : "FAIL");
    printf("  episode persisted at %s/{k.bin,manifest.txt}\n", dir);
    free(Rm); free(Rm2); free(K); free(Kload); free(projk_orig); free(projk_rebuilt);
    return ok ? 0 : 1;
}
