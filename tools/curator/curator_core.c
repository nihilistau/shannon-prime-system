/* curator_core.c — XBAR C1-lite: the transactional curator core (Memo v0).
 *
 * CONTRACT-XBAR-C1-lite §2, stage C1L.1: the propose -> gate -> promote/rewind
 * "immune system" (RFC-XBAR §3 rule 3 + §3.1 Ring 2'). This file proves the
 * TRANSACTIONAL MACHINERY in isolation on synthetic episodes — clone-isolation,
 * a pluggable gate, ATOMIC promote, SAFE rewind, append-only receipts — the
 * G-C1L-1 control-flow null. It deliberately does NOT need the replay-decode
 * mode (C1L.0) or a model: the identity curator yields a byte-identical shadow,
 * so "gate PASS" is true by construction, isolating the plumbing from the
 * quality gate that plugs in later.
 *
 * Episode = a directory holding {k.bin, v.bin, manifest.txt}. The transactional
 * ops mirror the real Ring-2 store (sp_arm_ring2_{k,v}.bin) so this core wires
 * onto the live two-ring once C1L.0 lands the persisted-episode replay.
 *
 * Build/run (WSL, no engine link):
 *   gcc -O2 -std=c11 -Wall -Wextra tools/curator/curator_core.c -o /tmp/curator && /tmp/curator
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>     /* rmdir */
#include <errno.h>

/* ── tiny fs helpers ─────────────────────────────────────────────────────── */
static int copy_file(const char *src, const char *dst) {
    FILE *a = fopen(src, "rb"), *b = NULL;
    if (!a) return -1;
    b = fopen(dst, "wb");
    if (!b) { fclose(a); return -1; }
    char buf[1 << 16]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, a)) > 0)
        if (fwrite(buf, 1, n, b) != n) { fclose(a); fclose(b); return -1; }
    fclose(a); fclose(b);
    return 0;
}
static long file_size(const char *p) { struct stat st; return stat(p, &st) ? -1 : (long)st.st_size; }
static int files_identical(const char *p, const char *q) {
    FILE *a = fopen(p, "rb"), *b = fopen(q, "rb");
    if (!a || !b) { if (a) fclose(a); if (b) fclose(b); return 0; }
    int eq = 1; int ca, cb;
    do { ca = fgetc(a); cb = fgetc(b); if (ca != cb) { eq = 0; break; } } while (ca != EOF);
    fclose(a); fclose(b);
    return eq;
}
static void joinp(char *out, size_t n, const char *dir, const char *name) {
    snprintf(out, n, "%s/%s", dir, name);
}
static const char *EP_FILES[3] = { "k.bin", "v.bin", "manifest.txt" };

/* ── episode transactional ops ───────────────────────────────────────────── */
/* clone canonical -> shadow (isolation: the curator never touches canonical). */
static int episode_clone(const char *canon, const char *shadow) {
    mkdir(shadow, 0755);
    char s[1024], d[1024];
    for (int i = 0; i < 3; i++) {
        joinp(s, sizeof s, canon, EP_FILES[i]);
        joinp(d, sizeof d, shadow, EP_FILES[i]);
        if (copy_file(s, d)) return -1;
    }
    return 0;
}
/* gate: byte-identity delta for the null (0 = identical = PASS). The real gate
 * (C1L.0) swaps this for a PPL/recall delta on the replayed episode; the
 * signature (canon, shadow) -> delta is the seam the quality gate plugs into. */
static long gate_byte_delta(const char *canon, const char *shadow) {
    char c[1024], s[1024]; long diff = 0;
    for (int i = 0; i < 3; i++) {
        joinp(c, sizeof c, canon, EP_FILES[i]);
        joinp(s, sizeof s, shadow, EP_FILES[i]);
        if (!files_identical(c, s)) diff++;
    }
    return diff;     /* count of differing files; 0 = byte-identical episode */
}
/* receipt: append-only audit log of every transaction (the auditable dream). */
static void receipt(const char *canon, const char *action, const char *curator,
                    long delta, long size_before, long size_after) {
    char rp[1024]; joinp(rp, sizeof rp, canon, "../receipts.log");
    FILE *f = fopen(rp, "ab");
    if (!f) return;
    fprintf(f, "%ld\t%s\tcurator=%s\tdelta=%ld\tsize %ld->%ld\n",
            (long)time(NULL), action, curator, delta, size_before, size_after);
    fclose(f);
}
static long episode_bytes(const char *dir) {
    char p[1024]; long t = 0;
    for (int i = 0; i < 3; i++) { joinp(p, sizeof p, dir, EP_FILES[i]); long s = file_size(p); if (s > 0) t += s; }
    return t;
}
/* PROMOTE: atomically replace canonical with shadow (rename = atomic per-file
 * on the same fs), then remove the shadow dir. */
static int episode_promote(const char *canon, const char *shadow) {
    char s[1024], c[1024];
    for (int i = 0; i < 3; i++) {
        joinp(s, sizeof s, shadow, EP_FILES[i]);
        joinp(c, sizeof c, canon, EP_FILES[i]);
        if (rename(s, c)) return -1;        /* atomic swap-in */
    }
    rmdir(shadow);
    return 0;
}
/* REWIND: discard the shadow entirely; canonical is byte-untouched (it was
 * never written — the whole point of clone-isolation). */
static int episode_rewind(const char *shadow) {
    char s[1024];
    for (int i = 0; i < 3; i++) { joinp(s, sizeof s, shadow, EP_FILES[i]); remove(s); }
    return rmdir(shadow);
}

/* ── curator functions (operate ONLY on the shadow) ──────────────────────── */
/* v0 identity — proposes the episode unchanged (the null curator). */
static void curator_identity(const char *shadow) { (void)shadow; }

/* C1L.2 stub — cold-evict: drop the coldest `drop` positions of K/V by an
 * access-count array (the LRU/association signal from sp_arm_select). Shown
 * here on synthetic fixed-size positions to demonstrate the size delta; the
 * REAL eviction + projk-prune + PPL gate is C1L.2 proper. */
static void curator_cold_evict(const char *shadow, const int *access, int npos,
                               int pos_bytes, int drop) {
    /* rank positions by access asc, keep the top (npos-drop) by rewriting k/v
     * with only the survivors. Synthetic: each position is pos_bytes in each
     * stream. */
    int *order = malloc((size_t)npos * sizeof(int));
    for (int i = 0; i < npos; i++) order[i] = i;
    for (int i = 0; i < npos; i++)            /* simple selection sort by access asc */
        for (int j = i + 1; j < npos; j++)
            if (access[order[j]] < access[order[i]]) { int t = order[i]; order[i] = order[j]; order[j] = t; }
    /* survivors = order[drop..npos) ; keep original position order among them */
    char keep[4096]; memset(keep, 0, sizeof keep);
    for (int i = drop; i < npos; i++) keep[order[i]] = 1;
    for (int s = 0; s < 2; s++) {
        char path[1024]; joinp(path, sizeof path, shadow, EP_FILES[s]);
        long sz = file_size(path); if (sz < 0) continue;
        char *buf = malloc((size_t)sz); FILE *f = fopen(path, "rb");
        if (!f || fread(buf, 1, (size_t)sz, f) != (size_t)sz) { if (f) fclose(f); free(buf); continue; }
        fclose(f);
        f = fopen(path, "wb");
        for (int p = 0; p < npos; p++)
            if (keep[p]) fwrite(buf + (long)p * pos_bytes, 1, (size_t)pos_bytes, f);
        fclose(f); free(buf);
    }
    free(order);
}

/* ── synthetic episode + the C1L.1 null gates ────────────────────────────── */
static void write_synth_episode(const char *dir, int npos, int pos_bytes) {
    mkdir(dir, 0755);
    char p[1024];
    for (int s = 0; s < 2; s++) {
        joinp(p, sizeof p, dir, EP_FILES[s]);
        FILE *f = fopen(p, "wb");
        for (int i = 0; i < npos * pos_bytes; i++) fputc((s * 131 + i * 7) & 0xFF, f);
        fclose(f);
    }
    joinp(p, sizeof p, dir, EP_FILES[2]);
    FILE *m = fopen(p, "w"); fprintf(m, "npos=%d pos_bytes=%d\n", npos, pos_bytes); fclose(m);
}

int main(void) {
    const char *root = "/tmp/xbar_c1lite";
    char canon[1024], shadow[1024];
    joinp(canon, sizeof canon, root, "episode");
    joinp(shadow, sizeof shadow, root, "episode.shadow");
    mkdir(root, 0755);
    const int NPOS = 64, POSB = 256;

    int fails = 0;

    /* ---- G-C1L-1a: identity propose -> gate PASS -> promote -> canonical UNCHANGED ---- */
    write_synth_episode(canon, NPOS, POSB);
    /* snapshot canonical bytes for the unchanged-assertion */
    char kc[1024]; joinp(kc, sizeof kc, canon, "k.bin");
    char snap[1024]; joinp(snap, sizeof snap, root, "k_snapshot.bin"); copy_file(kc, snap);
    long sz0 = episode_bytes(canon);

    episode_clone(canon, shadow);
    curator_identity(shadow);
    long d1 = gate_byte_delta(canon, shadow);
    int g1_gate = (d1 == 0);                          /* identity must be byte-identical */
    if (g1_gate) episode_promote(canon, shadow);
    int g1_unchanged = files_identical(kc, snap);     /* canonical k.bin == pre-promote */
    long sz1 = episode_bytes(canon);
    receipt(canon, g1_gate ? "PROMOTE" : "REJECT", "identity", d1, sz0, sz1);
    int g1 = g1_gate && g1_unchanged && (sz1 == sz0);
    printf("G-C1L-1a identity-promote-null : gate_delta=%ld promote=%d canon_unchanged=%d  -> %s\n",
           d1, g1_gate, g1_unchanged, g1 ? "PASS" : "FAIL");
    fails += !g1;

    /* ---- G-C1L-1b: forced-reject -> rewind -> canonical UNTOUCHED + shadow gone ---- */
    copy_file(kc, snap);                               /* refresh snapshot */
    episode_clone(canon, shadow);
    /* mutate the shadow to force a non-zero gate delta (a "bad" consolidation) */
    { char sk[1024]; joinp(sk, sizeof sk, shadow, "k.bin");
      FILE *f = fopen(sk, "r+b"); if (f) { fputc(0xFF, f); fclose(f); } }
    long d2 = gate_byte_delta(canon, shadow);
    int g2_reject = (d2 > 0);                          /* gate sees the change -> reject */
    if (g2_reject) episode_rewind(shadow);
    int g2_unchanged = files_identical(kc, snap);      /* canonical never written */
    struct stat stx; int g2_shadow_gone = (stat(shadow, &stx) != 0);
    receipt(canon, "REWIND", "forced-bad", d2, sz0, sz0);
    int g2 = g2_reject && g2_unchanged && g2_shadow_gone;
    printf("G-C1L-1b forced-reject-rewind  : gate_delta=%ld reject=%d canon_untouched=%d shadow_gone=%d -> %s\n",
           d2, g2_reject, g2_unchanged, g2_shadow_gone, g2 ? "PASS" : "FAIL");
    fails += !g2;

    /* ---- C1L.2 PREVIEW (structural only): cold-evict shrinks the episode ----
     * The quality gate here is STUBBED (no model) — this proves the curator FN
     * + size delta + transactional commit; the real PPL/recall gate is C1L.0. */
    int access[64]; for (int i = 0; i < NPOS; i++) access[i] = (i * 2654435761u) % 100;  /* synthetic hit counts */
    episode_clone(canon, shadow);
    long pre = episode_bytes(shadow);
    curator_cold_evict(shadow, access, NPOS, POSB, /*drop=*/16);   /* evict 16 coldest */
    long post = episode_bytes(shadow);
    int g3_shrunk = (post < pre);
    /* structural-only gate: a real C1L.2 gates on PPL delta BEFORE promote */
    if (g3_shrunk) episode_promote(canon, shadow);
    long szc = episode_bytes(canon);
    receipt(canon, "PROMOTE", "cold-evict-16(STRUCTURAL-ONLY)", -1, sz0, szc);
    printf("C1L.2-preview cold-evict       : bytes %ld->%ld (shrunk=%d)  [PPL gate = C1L.0, NOT yet applied]\n",
           pre, post, g3_shrunk);

    printf("\nreceipts -> %s/receipts.log\n", root);
    printf("C1L.1 transactional null: %s (%d gate fail%s)\n",
           fails ? "FAIL" : "PASS", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
