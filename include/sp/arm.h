/* arm.h — ARM (Algebraic Resonance Memory): the two-ring KV memory core.
 *
 * Promoted from the engine's proven C2.1 production implementation
 * (shannon-prime-system-engine src/backends/cpu/cpu_forward.c, commits
 * 67f4997..7896bc4: recall router wired live, Ring-1 sink+window shrink 910x
 * @32k, Optane Ring-2 7.57us/read, compact-and-spill fusion) into a
 * first-class math-core module. The forward (core/forward) consumes this; the
 * platform Ring-2 store (Optane ReadFile+IOCP) stays engine-side and registers
 * through the abstract backend below.
 *
 * Three pieces:
 *
 *  1. RECALL ROUTER — frozen ±1 Rademacher projection (r x head_dim, seed
 *     SP_ARM_PROJ_SEED via integer SplitMix64 => identical across steps/
 *     backends/runs, so stored projections recall correctly). Selection =
 *     pinned sinks ∪ top-(B-W-sink) by projected score ∪ recent-W window,
 *     expected-O(N) quickselect. B=0 or context within budget => identity
 *     (full [0,pos]) — the bit-exact-when-off parity guarantee.
 *
 *  2. RING-1 SLOT MAP — physical slot for logical token s when offloading:
 *     sinks pinned at [0,sink), the W-window at [sink,sink+W) by modulo
 *     (structural eviction). Identity when not offloading.
 *
 *  3. RING-2 ABSTRACT BACKEND — the spilled-history block store as an
 *     L1-registrable interface (same pattern as the compute-backend
 *     registration). math-core ships the portable stdio reference
 *     implementation; the engine registers the Optane NO_BUFFERING+IOCP
 *     backend; a QUIC peer is the same interface over the mesh (the stored
 *     unit and the wire unit are one object — Trick #8/#9).
 *
 * Everything here is a PERF/CAPACITY overlay, never a correctness dependency:
 * all knobs OFF => the forward's exact full-context baseline.
 */
#ifndef SP_ARM_H
#define SP_ARM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── recall router ────────────────────────────────────────────────────────── */

#define SP_ARM_R_MAX 64
/* "SPROJ+" — frozen projection seed. Bumping it is a projection-version change
 * (stored sidecars become unreadable); do not change without a contract. */
#define SP_ARM_PROJ_SEED 0x5350524F4A2BULL

/* [score,index] pair for the quickselect router (per-thread scratch). */
typedef struct { float s; int i; } sp_arm_sidx;

/* Fill R[r*hd] with ±1 deterministically from SP_ARM_PROJ_SEED. */
void sp_arm_build_R(signed char *R, int r, int hd);

/* proj[p] = R[p,:]·vec for p in [0,r)  (±1 matrix => add/sub only). */
void sp_arm_project(const signed char *R, int r, int hd,
                    const float *vec, float *proj);

/* One query head's recall set into ri[0,m): full [0,pos] (exact baseline)
 * unless B>0 and pos+1>B, then sinks ∪ top-(B-W-sink) candidates by projected
 * score ∪ recent-W. projk is the stored K-projection sidecar laid out as
 * projk[((L*P + s)*NKV + kvh)*r + p]. cand is (pos+1)-entry scratch. Returns m.
 * Selection is expected O(N) quickselect; the top-k SET matches max-extract
 * except at exact float ties (projections ~never tie); softmax is order-free. */
int sp_arm_select(const signed char *R, int r, int hd, const float *qh,
                  const float *projk, size_t L, int P, int NKV, int kvh,
                  int B, int W, int sink, int pos, sp_arm_sidx *cand, int *ri);

/* ── bit-packed popcount router (SimHash overlay — gated, strictly lossier) ──
 *
 * The r-float projk sidecar is the last full-P RAM resident (~940 MB @32k).
 * Packing the SIGN of each Rademacher projection into one u64 per (pos,kvh)
 * shrinks it 32x (r=32: 128 B -> 8 B) and turns scoring into hardware
 * popcount(qsig ^ ksig) — Hamming distance, the SimHash angle estimator.
 *
 * HONEST CONTRACT: this is a LOSSIER estimator than the f32 projection dot
 * (1 bit/row vs 32; scores take only r+1 distinct values, so top-k boundary
 * ties are common and resolved deterministically-but-arbitrarily). It is an
 * overlay knob (SP_RECALL_BITS), NEVER a default: any new (N, B, r) regime
 * must re-pass the NIAH retrieval + PPL deflection gates before being
 * trusted. Bit-exact-when-off parity is unchanged. */

/* Pack the r sign bits of the projection of vec (bit p = proj[p] >= 0). */
uint64_t sp_arm_project_sig(const signed char *R, int r, int hd, const float *vec);

/* sp_arm_select with the bit-packed sidecar: identical selection structure
 * (sinks ∪ top-(B-W-sink) ∪ recent-W, identity when B=0 or within budget),
 * but candidates are ranked by ASCENDING popcount(qsig ^ ksig).
 *
 * sigk LAYOUT v2 — HEAD-MAJOR: sigk[(L*NKV + kvh)*P + s]. Each head's
 * signature stream is a CONTIGUOUS stride-1 u64 array, so the candidate scan
 * is pure streaming (vector loads, hardware-prefetcher-perfect) instead of
 * a strided gather. (v1 interleaved heads position-major; nothing persisted
 * v1 — the sidecar is rebuilt per run, so this is not a format break.) */
int sp_arm_select_sig(const signed char *R, int r, int hd, const float *qh,
                      const uint64_t *sigk, size_t L, int P, int NKV, int kvh,
                      int B, int W, int sink, int pos, sp_arm_sidx *cand, int *ri);

/* THE 32k WALL — the candidate scoring scan, hoisted into its OWN archive
 * member (arm_scan.c — the resdot/ntt_batch seam): score n candidates
 * s = s0..s0+n-1 of one head's contiguous signature slice,
 *     cand[i] = { -(float)popcount(qsig ^ sigs[i]), s0 + i }.
 * This is the quadratic term of streaming ingest (O(pos) per head per token);
 * the engine overrides it with AVX512-VPOPCNTDQ (8 u64/instr) + OMP chunking.
 * EXACTNESS CONTRACT: any override must produce IDENTICAL cand entries
 * (same float score, same index, same order) — quickselect runs host-side on
 * these values, so score-array equality ⇒ selection identity. */
void sp_arm_scan_sig(uint64_t qsig, const uint64_t *sigs, int n, int s0,
                     sp_arm_sidx *cand);

/* ── Ring-1 slot map ──────────────────────────────────────────────────────── */

/* Physical Ring-1 slot for logical token position s: sinks pinned at [0,sink),
 * the W-window at [sink,sink+W) by modulo. Identity when offloading==0. */
int sp_arm_r1slot(int s, int offloading, int sink, int w);

/* ── Ring-2 abstract backend ──────────────────────────────────────────────── */

/* Block store for the spilled K/V history. which: 0 = K stream, 1 = V stream.
 * Offsets are byte offsets within the stream; len is the block size the store
 * was opened with. All fns return 0 on success, non-zero on error.
 * OPTIONAL members (NULL => caller falls back):
 *   read_batch     — batched reads (Optane IOCP queue-depth amortization);
 *                    NULL => callers loop read_block.
 *   alloc_aligned / free_aligned — landing-buffer allocator for backends with
 *                    direct-I/O alignment requirements (NO_BUFFERING sector
 *                    alignment); NULL => callers use malloc/free. */
typedef struct {
    void *handle;
    int   (*write_block)(void *handle, int which, uint64_t off,
                         const void *src, size_t len);
    int   (*read_block)(void *handle, int which, uint64_t off,
                        void *dst, size_t len);
    int   (*read_batch)(void *handle, const int *which, const uint64_t *off,
                        void *const *dst, size_t len, int n);   /* optional */
    void *(*alloc_aligned)(void *handle, size_t bytes);          /* optional */
    void  (*free_aligned)(void *handle, void *p);                /* optional */
    void  (*close)(void *handle);
} sp_arm_ring2_backend;

/* Portable stdio reference backend (fopen/seek/read/write; two files
 * "sp_arm_ring2_k.bin" / "sp_arm_ring2_v.bin" under `dir`). The L1 reference
 * implementation — correctness twin of the engine's Optane store, not its
 * performance twin. Returns 0 and fills *out on success. */
int sp_arm_ring2_stdio_open(const char *dir, sp_arm_ring2_backend *out);

/* ── platform-backend registration (the L1 hook) ─────────────────────────────
 * A platform store (engine Optane NO_BUFFERING+IOCP, a QUIC peer, ...) wraps
 * itself in sp_arm_ring2_backend and registers ONCE at startup; the canonical
 * decode then uses the registered backend instead of opening the stdio
 * reference. Ownership stays with the registrant: the decode treats a
 * registered backend as BORROWED (never calls close on it); the registrant
 * unregisters (be=NULL) before tearing the store down. Registration is
 * startup-time, not thread-safe against concurrent decodes. */
void sp_arm_ring2_register(const sp_arm_ring2_backend *be);

/* Copy the registered backend into *out and return 1; return 0 if none. */
int  sp_arm_ring2_registered(sp_arm_ring2_backend *out);

#ifdef __cplusplus
}
#endif
#endif /* SP_ARM_H */
