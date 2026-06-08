/* arm_genkv_gate.c — T_ARM_GENKV: Stage-D RUN-gates for the two-ring + NTT decode
 * through the canonical qwen3_generate_kv, on the synthetic qwen3 fixture
 * (qwen3_fixture.c: tiny but spec-conformant .sp-model, HD=8 => the NTT decode
 * overlay engages via Bluestein). This is the run-validation forward_test.c
 * cannot give (it is link-only): every gate below actually decodes.
 *
 * The load-bearing trick: with a budget B >= P the recall selection is the
 * IDENTITY set, so the ARM machinery (Ring-1 sink+window ring buffer,
 * projection sidecar, Ring-2 spill + fetch — mock RAM, stdio backend, and the
 * compact-and-spill fusion) is fully exercised while the attention output must
 * be EXACTLY the baseline: every old token's K/V is structurally evicted from
 * Ring-1 and must come back byte-identical from Ring-2, or the sequence
 * diverges. That is the same structural-parity argument the engine's C2.1
 * gates used (poison-free variant: eviction is the poison).
 *
 *   T_GENKV_DETERMINISM        baseline decode is deterministic (run twice).
 *   T_GENKV_ARM_MOCK_PARITY    ring buffer + mock-RAM Ring-2, B>=P == baseline.
 *   T_GENKV_ARM_BACKEND_PARITY same through the stdio Ring-2 backend.
 *   T_GENKV_ARM_FUSE_PARITY    compact-and-spill boundary path == baseline.
 *   T_GENKV_ARM_SPARSE_RUNS    real sparse budget (B<P) completes + deterministic.
 *   T_GENKV_NTT_TOP1           SP_ENGINE_NTT_ATTN=1 decode == baseline sequence
 *                              (the decode-side gate for the NTT fusion).
 */
#include "sp/sp_test.h"
#include "sp/sp_l1.h"
#include "sp/sp_model.h"
#include "sp/model.h"
#include "sp/arm.h"
#include "qwen3_fixture.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifdef _WIN32
#  include <direct.h>
#  define sp_mkdir(d) _mkdir(d)
#  define sp_rmdir(d) _rmdir(d)
#else
#  include <sys/stat.h>
#  define sp_mkdir(d) mkdir((d), 0777)
#  define sp_rmdir(d) rmdir(d)
#endif

#define NPROMPT 4
#define NGEN    24
#define PTOT    (NPROMPT + NGEN)

static void knob(const char *k, const char *v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

/* All ARM/NTT knobs to the OFF baseline. */
static void knobs_off(void) {
    knob("SP_ENGINE_NTT_ATTN", "0");
    knob("SP_RECALL_B", "0");  knob("SP_RECALL_R", "16");
    knob("SP_RECALL_W", "64"); knob("SP_RECALL_SINK", "4");
    knob("SP_RING2", "0");     knob("SP_RING2_DISK", "0");
    knob("SP_RING2_DIR", "."); knob("SP_RECALL_DECODE_ONLY", "0");
    knob("SP_RECALL_FUSE", "0");
    knob("SP_REPLAY", "0"); knob("SP_REPLAY_DIR", "."); knob("SP_REPLAY_NPOS", "0");
}

/* Small ring config so the P=28 decode genuinely evicts/offloads:
 * sink=2, W=4 -> Ring-1 cap 6 slots; everything older is Ring-2-served. */
static void knobs_ring_common(void) {
    knob("SP_RECALL_W", "4"); knob("SP_RECALL_SINK", "2"); knob("SP_RECALL_R", "16");
}

static qwen3_model *g_qm = NULL;
static sp_model    *g_sm = NULL;

static int load_model(void) {
    sp_qwen3_fixture_info info;
    uint8_t *mb = NULL, *tb = NULL;
    if (sp_qwen3_fixture_build(&mb, &tb, &info)) return 1;
    int rc = sp_qwen3_fixture_write("fx_arm.spm", mb, info.model_len)
           | sp_qwen3_fixture_write("fx_arm.spt", tb, info.tok_len);
    free(mb); free(tb);
    if (rc) return 1;
    if (sp_model_load("fx_arm.spm", "fx_arm.spt", &g_sm) != SP_OK) return 1;
    g_qm = sp_model_to_qwen3(g_sm);
    return g_qm ? 0 : 1;
}

/* Run one greedy decode; returns total length (or <0). seq must hold PTOT. */
static int run_decode(int32_t *seq) {
    seq[0] = 1; seq[1] = 7; seq[2] = 3; seq[3] = 42;   /* < n_vocab=48 */
    return qwen3_generate_kv(g_qm, seq, NPROMPT, NGEN, -1);
}

static int seq_equal(const int32_t *a, const int32_t *b) {
    return memcmp(a, b, (size_t)PTOT * sizeof(int32_t)) == 0;
}

static void dump_seq(const char *tag, const int32_t *s) {
    fprintf(stderr, "    [%s]", tag);
    for (int i = 0; i < PTOT; i++) fprintf(stderr, " %d", s[i]);
    fprintf(stderr, "\n");
}

static int32_t g_base[PTOT];   /* the knobs-off baseline sequence */

static void T_GENKV_DETERMINISM(void) {
    knobs_off();
    int32_t again[PTOT];
    SP_CHECK_EQ_I64(run_decode(g_base), PTOT, "baseline decode completes (full length)");
    SP_CHECK_EQ_I64(run_decode(again),  PTOT, "baseline re-run completes");
    SP_CHECK(seq_equal(g_base, again), "baseline decode is deterministic");
    dump_seq("baseline", g_base);
}

static void T_GENKV_ARM_MOCK_PARITY(void) {
    knobs_off(); knobs_ring_common();
    knob("SP_RECALL_B", "64");          /* B >= P: identity selection */
    knob("SP_RING2", "1");              /* mock RAM Ring-2, Ring-1 = 6-slot ring */
    int32_t got[PTOT];
    SP_CHECK_EQ_I64(run_decode(got), PTOT, "mock two-ring decode completes");
    if (!seq_equal(g_base, got)) dump_seq("mock-ring", got);
    SP_CHECK(seq_equal(g_base, got),
             "two-ring (ring buffer + mock Ring-2 fetch), identity budget == baseline");
    knobs_off();
}

static void T_GENKV_ARM_BACKEND_PARITY(void) {
    knobs_off(); knobs_ring_common();
    knob("SP_RECALL_B", "64");
    knob("SP_RING2", "1"); knob("SP_RING2_DISK", "1"); knob("SP_RING2_DIR", ".");
    int32_t got[PTOT];
    SP_CHECK_EQ_I64(run_decode(got), PTOT, "backend two-ring decode completes");
    if (!seq_equal(g_base, got)) dump_seq("backend-ring", got);
    SP_CHECK(seq_equal(g_base, got),
             "two-ring through the stdio Ring-2 backend, identity budget == baseline");
    knobs_off();
    remove("sp_arm_ring2_k.bin"); remove("sp_arm_ring2_v.bin");
}

static void T_GENKV_ARM_FUSE_PARITY(void) {
    knobs_off(); knobs_ring_common();
    knob("SP_RECALL_B", "64");
    knob("SP_RING2", "1"); knob("SP_RING2_DISK", "1"); knob("SP_RING2_DIR", ".");
    knob("SP_RECALL_FUSE", "1");        /* dense full-P prefill, bulk-spill at boundary */
    int32_t got[PTOT];
    SP_CHECK_EQ_I64(run_decode(got), PTOT, "fuse two-ring decode completes");
    if (!seq_equal(g_base, got)) dump_seq("fuse-ring", got);
    SP_CHECK(seq_equal(g_base, got),
             "compact-and-spill fusion, identity budget == baseline");
    knobs_off();
    remove("sp_arm_ring2_k.bin"); remove("sp_arm_ring2_v.bin");
}

static void T_GENKV_ARM_SPARSE_RUNS(void) {
    knobs_off(); knobs_ring_common();
    knob("SP_RECALL_B", "8");           /* real sparse budget: 2 sinks + 2 top-k + 4 window */
    knob("SP_RING2", "1");
    int32_t a[PTOT], b[PTOT];
    SP_CHECK_EQ_I64(run_decode(a), PTOT, "sparse two-ring decode completes");
    SP_CHECK_EQ_I64(run_decode(b), PTOT, "sparse re-run completes");
    SP_CHECK(seq_equal(a, b), "sparse two-ring decode is deterministic");
    dump_seq("sparse-B8", a);
    knobs_off();
}

/* ── Stage C: registered platform backend (counting in-memory store) ────────
 * Registers a custom backend with read_batch + the aligned-alloc hooks and
 * proves the canonical decode (a) routes through it instead of the stdio
 * reference, (b) stays byte-identical to baseline at identity budget, and
 * (c) actually exercised every hook (counters > 0). This is the same seam the
 * engine's Optane NO_BUFFERING+IOCP store registers through. */
#define CBE_CAP (1u << 20)
typedef struct {
    unsigned char *mem[2];
    long writes, reads, batches, allocs, frees;
} counting_be;
static counting_be g_cbe;

static int cbe_write(void *h, int which, uint64_t off, const void *src, size_t len) {
    counting_be *b = (counting_be *)h;
    if (off + len > CBE_CAP) return 1;
    memcpy(b->mem[which] + off, src, len); b->writes++; return 0;
}
static int cbe_read(void *h, int which, uint64_t off, void *dst, size_t len) {
    counting_be *b = (counting_be *)h;
    if (off + len > CBE_CAP) return 1;
    memcpy(dst, b->mem[which] + off, len); b->reads++; return 0;
}
static int cbe_read_batch(void *h, const int *which, const uint64_t *off,
                          void *const *dst, size_t len, int n) {
    counting_be *b = (counting_be *)h;
    for (int i = 0; i < n; i++)
        if (cbe_read(h, which[i], off[i], dst[i], len)) return 1;
    b->batches++; return 0;
}
static void *cbe_alloc(void *h, size_t bytes) { ((counting_be *)h)->allocs++; return malloc(bytes); }
static void  cbe_free(void *h, void *p)       { ((counting_be *)h)->frees++;  free(p); }

static void T_GENKV_REGISTERED_BACKEND(void) {
    memset(&g_cbe, 0, sizeof(g_cbe));
    g_cbe.mem[0] = (unsigned char *)malloc(CBE_CAP);
    g_cbe.mem[1] = (unsigned char *)malloc(CBE_CAP);
    SP_CHECK(g_cbe.mem[0] && g_cbe.mem[1], "counting backend buffers");

    sp_arm_ring2_backend be;
    memset(&be, 0, sizeof(be));           /* optional members (read_batch2, ...) NULL */
    be.handle = &g_cbe;
    be.write_block = cbe_write; be.read_block = cbe_read; be.read_batch = cbe_read_batch;
    be.alloc_aligned = cbe_alloc; be.free_aligned = cbe_free;
    be.close = NULL;                      /* borrowed: decode must NOT close it */
    sp_arm_ring2_register(&be);

    knobs_off(); knobs_ring_common();
    knob("SP_RECALL_B", "64");
    knob("SP_RING2", "1"); knob("SP_RING2_DISK", "1");
    int32_t got[PTOT];
    SP_CHECK_EQ_I64(run_decode(got), PTOT, "registered-backend decode completes");
    if (!seq_equal(g_base, got)) dump_seq("registered-be", got);
    SP_CHECK(seq_equal(g_base, got),
             "registered platform backend, identity budget == baseline");
    SP_CHECK(g_cbe.writes  > 0, "registered backend received the spill writes");
    SP_CHECK(g_cbe.batches > 0, "registered backend served reads via read_batch");
    SP_CHECK(g_cbe.allocs == 2 && g_cbe.frees == 2,
             "direct-I/O staging came from the backend allocator (2 alloc / 2 free)");
    fprintf(stderr, "    [counting-be] writes=%ld reads=%ld batches=%ld allocs=%ld frees=%ld\n",
            g_cbe.writes, g_cbe.reads, g_cbe.batches, g_cbe.allocs, g_cbe.frees);

    sp_arm_ring2_register(NULL);          /* unregister: decode falls back to stdio */
    knobs_off();
    free(g_cbe.mem[0]); free(g_cbe.mem[1]);
}

static void T_GENKV_NTT_TOP1(void) {
    knobs_off();
    knob("SP_ENGINE_NTT_ATTN", "1");    /* HD=8 -> Bluestein poly-ring decode score */
    int32_t got[PTOT];
    SP_CHECK_EQ_I64(run_decode(got), PTOT, "NTT-decode completes");
    if (!seq_equal(g_base, got)) dump_seq("ntt-decode", got);
    SP_CHECK(seq_equal(g_base, got),
             "NTT decode attention top-1 sequence == f32 baseline (decode-side NTT gate)");
    knobs_off();
}


/* ── NTT FUSION run-gates (Bluestein-unlocked): the HD=8 fixture engages the
 * keystore via the Bluestein arm, so the fusion architecture is validated
 * NATIVELY in-tree — plain, two-ring, backend, and the compact-and-spill
 * fusion (the named success metric). All identity-budget == baseline. */
static void T_GENKV_FUSION_PLAIN(void) {
    knobs_off(); knob("SP_NTT_KV", "1");
    int32_t got[PTOT];
    SP_CHECK_EQ_I64(run_decode(got), PTOT, "fusion plain decode completes (Bluestein HD=8)");
    if (!seq_equal(g_base, got)) dump_seq("fusion-plain", got);
    SP_CHECK(seq_equal(g_base, got), "fusion (residue K cache) == baseline sequence");
    knobs_off(); knob("SP_NTT_KV", "0");
}
static void T_GENKV_FUSION_MOCK_RING(void) {
    knobs_off(); knobs_ring_common();
    knob("SP_NTT_KV", "1"); knob("SP_RECALL_B", "64"); knob("SP_RING2", "1");
    int32_t got[PTOT];
    SP_CHECK_EQ_I64(run_decode(got), PTOT, "fusion x mock two-ring completes");
    if (!seq_equal(g_base, got)) dump_seq("fusion-mock", got);
    SP_CHECK(seq_equal(g_base, got), "fusion x mock Ring-2 (residue spill/fetch) == baseline");
    knobs_off(); knob("SP_NTT_KV", "0");
}
static void T_GENKV_FUSION_BACKEND(void) {
    knobs_off(); knobs_ring_common();
    knob("SP_NTT_KV", "1"); knob("SP_RECALL_B", "64");
    knob("SP_RING2", "1"); knob("SP_RING2_DISK", "1"); knob("SP_RING2_DIR", ".");
    int32_t got[PTOT];
    SP_CHECK_EQ_I64(run_decode(got), PTOT, "fusion x stdio backend completes");
    if (!seq_equal(g_base, got)) dump_seq("fusion-backend", got);
    SP_CHECK(seq_equal(g_base, got), "fusion x Ring-2 backend (residue blocks on disk) == baseline");
    knobs_off(); knob("SP_NTT_KV", "0");
    remove("sp_arm_ring2_k.bin"); remove("sp_arm_ring2_v.bin");
}
static void T_GENKV_FUSION_FUSE(void) {
    knobs_off(); knobs_ring_common();
    knob("SP_NTT_KV", "1"); knob("SP_RECALL_B", "64");
    knob("SP_RING2", "1"); knob("SP_RING2_DISK", "1"); knob("SP_RING2_DIR", ".");
    knob("SP_RECALL_FUSE", "1");
    int32_t got[PTOT];
    SP_CHECK_EQ_I64(run_decode(got), PTOT, "fusion x compact-and-spill completes");
    if (!seq_equal(g_base, got)) dump_seq("fusion-fuse", got);
    SP_CHECK(seq_equal(g_base, got),
             "fusion x compact-and-spill (residue prefill buffer + boundary bulk-spill) == baseline");
    knobs_off(); knob("SP_NTT_KV", "0");
    remove("sp_arm_ring2_k.bin"); remove("sp_arm_ring2_v.bin");
}

/* ── C1L.0b: replay-decode null (XBAR curator load seam at the MODEL layer) ──
 * Capture a full episode to a private dir via the ring2 spill, then re-decode
 * with SP_REPLAY overwriting every position's post-RoPE K/V from the loaded
 * (bit-identical) episode read back through the non-truncating
 * sp_arm_ring2_stdio_open_ro path. Three checks:
 *   (1) replay of the INTACT episode == baseline sequence (faithful round-trip
 *       — the loaded K/V flow through projk + cache + attention losslessly);
 *   (2) a ZEROED episode must DIVERGE (proves the loaded bytes actually drive
 *       attention — replay is not a silent no-op);
 *   (3) replay-off is the baseline itself (the bit-exact floor; every other
 *       gate in this file runs with SP_REPLAY=0). */
static void zero_file(const char *path) {
    FILE *f = fopen(path, "r+b");
    if (!f) return;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char zb[4096]; memset(zb, 0, sizeof zb);
    while (n > 0) { size_t c = (n < (long)sizeof zb) ? (size_t)n : sizeof zb;
                    fwrite(zb, 1, c, f); n -= (long)c; }
    fclose(f);
}
static void T_GENKV_REPLAY_NULL(void) {
    sp_mkdir("epcap");
    /* CAPTURE: spill the full episode to epcap/ (identity budget => == baseline). */
    knobs_off(); knobs_ring_common();
    knob("SP_RECALL_B", "64");
    knob("SP_RING2", "1"); knob("SP_RING2_DISK", "1"); knob("SP_RING2_DIR", "epcap");
    int32_t cap[PTOT];
    SP_CHECK_EQ_I64(run_decode(cap), PTOT, "capture decode completes (episode -> epcap/)");
    SP_CHECK(seq_equal(g_base, cap), "capture (ring2 spill) == baseline");

    /* REPLAY-INTACT: ring/recall OFF; full-context f32 cache; overwrite all
     * positions' K/V from the loaded episode via the _ro path. */
    knobs_off();
    char np[16]; snprintf(np, sizeof np, "%d", PTOT);
    knob("SP_REPLAY", "1"); knob("SP_REPLAY_DIR", "epcap"); knob("SP_REPLAY_NPOS", np);
    int32_t rep[PTOT];
    SP_CHECK_EQ_I64(run_decode(rep), PTOT, "replay decode completes (loaded episode as history)");
    if (!seq_equal(g_base, rep)) dump_seq("replay", rep);
    SP_CHECK(seq_equal(g_base, rep),
             "replay of INTACT episode == baseline (faithful K/V round-trip through _ro load)");

    /* NEGATIVE CONTROL: zero the episode bins, replay again -> must DIVERGE. */
    zero_file("epcap/sp_arm_ring2_k.bin");
    zero_file("epcap/sp_arm_ring2_v.bin");
    int32_t neg[PTOT];
    SP_CHECK_EQ_I64(run_decode(neg), PTOT, "zeroed-episode replay completes");
    SP_CHECK(!seq_equal(g_base, neg),
             "ZEROED episode DIVERGES from baseline (loaded K/V bytes drive attention)");
    dump_seq("replay-zeroed", neg);

    knobs_off();
    remove("epcap/sp_arm_ring2_k.bin"); remove("epcap/sp_arm_ring2_v.bin");
    sp_rmdir("epcap");
}

/* ── C1L.2 Step 1: recall-hit telemetry (the LRU / association signal) ───────
 * Attach a per-position counter; the content-selected top-k positions are
 * counted (sinks + window excluded, being structural). Proves: (a) attaching
 * the counter does NOT change the decoded sequence (telemetry-null), and (b)
 * the hit histogram is non-uniform — some positions are recalled by content
 * (hot), others never (cold) — which is the signal cold-evict (Step 2) acts on. */
static void T_GENKV_RECALL_HITS(void) {
    knobs_off(); knobs_ring_common();
    knob("SP_RECALL_B", "8");           /* sparse: 2 sinks + 2 top-k + 4 window */
    knob("SP_RING2", "1");
    int32_t ref[PTOT];
    SP_CHECK_EQ_I64(run_decode(ref), PTOT, "sparse decode (no telemetry) completes");

    int hits[PTOT]; memset(hits, 0, sizeof hits);
    sp_arm_hits_attach(hits, PTOT);
    int32_t got[PTOT];
    SP_CHECK_EQ_I64(run_decode(got), PTOT, "sparse decode (telemetry attached) completes");
    sp_arm_hits_detach();

    SP_CHECK(seq_equal(ref, got),
             "telemetry-null: attached counters do NOT change the decoded sequence");

    int total = 0, maxh = 0, hot = 0, cold = 0;
    for (int i = 0; i < PTOT; i++) {
        total += hits[i]; if (hits[i] > maxh) maxh = hits[i];
        if (hits[i] > 0) hot++; else cold++;
    }
    fprintf(stderr, "    [hits] total=%d max=%d hot=%d cold=%d  hist:", total, maxh, hot, cold);
    for (int i = 0; i < PTOT; i++) fprintf(stderr, " %d", hits[i]);
    fprintf(stderr, "\n");
    SP_CHECK(total > 0, "content selections were recorded (the recall signal exists)");
    SP_CHECK(hot > 0 && cold > 0,
             "hit histogram is non-uniform — cold positions are identifiable (the evict signal)");
    knobs_off();
}

int main(void) {
    if (load_model()) { fprintf(stderr, "FATAL: fixture model load failed\n"); return 1; }
    SP_RUN(T_GENKV_DETERMINISM);
    SP_RUN(T_GENKV_ARM_MOCK_PARITY);
    SP_RUN(T_GENKV_ARM_BACKEND_PARITY);
    SP_RUN(T_GENKV_ARM_FUSE_PARITY);
    SP_RUN(T_GENKV_ARM_SPARSE_RUNS);
    SP_RUN(T_GENKV_REGISTERED_BACKEND);
    SP_RUN(T_GENKV_NTT_TOP1);
    SP_RUN(T_GENKV_FUSION_PLAIN);
    SP_RUN(T_GENKV_FUSION_MOCK_RING);
    SP_RUN(T_GENKV_FUSION_BACKEND);
    SP_RUN(T_GENKV_FUSION_FUSE);
    SP_RUN(T_GENKV_REPLAY_NULL);
    SP_RUN(T_GENKV_RECALL_HITS);
    qwen3_free(g_qm);
    sp_model_unload(g_sm);
    remove("fx_arm.spm"); remove("fx_arm.spt");
    return SP_DONE();
}
