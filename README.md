# shannon-prime-system

The **math core** of the [shannon-prime-lattice](https://github.com/nihilistau/shannon-prime-lattice)
project: a clean from-scratch C library (`libshannonprime`) that exposes a
frozen Layer-1 C ABI for inference + the discrete-algebra primitives every
engine backend shares.

This library is consumed by [shannon-prime-system-engine](https://github.com/nihilistau/shannon-prime-system-engine)
(as a Git submodule under `lib/shannon-prime-system/`). Every accelerated
backend — CPU AVX2/AVX-512, CUDA PTX, Vulkan, Hexagon HVX — registers
against this library's L1 ABI; the scalar reference forward defined here
is the bit-exact correctness anchor each backend gates against.

The project's public, receipts-first results (the two-ring long-context
memory envelope, the reducing loader, the quantization story) live at
**[Position Is Arithmetic](https://github.com/nihilistau/Position_Is_Arithmetic)**
(live site: https://nihilistau.github.io/Position_Is_Arithmetic/). The
developer umbrella is [shannon-prime-lattice](https://github.com/nihilistau/shannon-prime-lattice).

License: MIT. See `LICENSE`.

---

## 1. What's in here

The scalar reference forward here is the **bit-exact correctness anchor** — a discrete `Z_q` forward proven **argmax bit-exact to llama.cpp** on five architecture families: Qwen3-0.6B, Qwen2.5-Coder-0.5B, Gemma3-1B, **Gemma4-E2B**, and **Qwen3.6-35B-A3B MoE (Gated DeltaNet)**. The reducing `.sp-model` codec (`OK_Q4` / `OK_Q8`, body ≤ source quant) is output-lossless (C1, top-1 == oracle). Everything below is a primitive every accelerated backend shares and gates against.

| Module | Header | Purpose |
|--------|--------|---------|
| **L1 ABI** | `include/sp/sp_l1.h` | The frozen session + forward + arch-query surface (PPT-LAT-L1-ABI-v0). Two-function forward (`sp_prefill_chunk` / `sp_decode_step`), clone/rewind, deterministic-mode anchor. |
| **Status** | `include/sp/sp_status.h` | `sp_status` enum (SP_OK, SP_ENOMEM, SP_ECANCEL, ..., SP_EHVX) + thread-local `sp_last_error()`. Frozen — values must not be renumbered. |
| **Model loader** | `include/sp/sp_model.h` | `.sp-model` / `.sp-tokenizer` on-disk format. `sp_model_load` is pure `mmap` + header parse + tensor table pointer setup; zero malloc proportional to tensor data. |
| **NTT-CRT** | `include/sp/ntt_crt.h` | Dual-prime negacyclic NTT over `Z_q[x]/(x^N+1)` with Barrett reduction. Frozen primes `q_1=1073738753`, `q_2=1073732609`, `M=q_1·q_2≈2^60`. `N ∈ {128, 256, 512}`. |
| **Polynomial-ring attention** | `include/sp/poly_ring.h` | `sp_pr_init` + `sp_pr_inner` + `sp_pr_mul` + `sp_pr_attention`. Attention `⟨q,k⟩` = coefficient 0 of negacyclic `q ⊗ k*`, exact in `Z` when `|⟨q,k⟩| < M/2`. |
| **Frobenius lift** | `include/sp/frobenius_lift.h` | Per-row int8 codes + per-row fp32 scale. Symmetric `[-127,127]` range; round-half-away-from-zero deterministic across FP modes. 4× compression vs fp32. |
| **Packed-weight arena** | `include/sp/arena.h` | The named collection of `sp_frob_packed_tensor`s the backends read at matmul time. Single in-RAM layout for Q8 and Q4 mixed-precision. |
| **Spinor block** | `include/sp/spinor_block.h` | Frozen 63-byte VHT2 + Möbius KV-cache record + CRC-8 trailer. Wire format is the struct (no hidden padding). |
| **KSTE encoder** | `include/sp/kste.h` | Deterministic 64-byte packed-tree fingerprint for K-vectors of int32 components. `T_{60,3}` family. Tier-0/Tier-1 componentwise dominance. |
| **Forward dispatch** | `include/sp/forward_dispatch.h` | `sp_matmul`, `sp_embed_row`, `sp_as_f32` — the model-coupled weight-access layer. Honors `SP_ENGINE_FROB`, `SP_ENGINE_F16_ACT`, `SP_Q4_PROMOTE` knobs. |
| **Forward kernels** | `include/sp/forward_kernels.h` | `sp_dot_f32`, `sp_rmsnorm`, `sp_rope_neox`, `sp_attn_head` — portable scalar reference primitives every backend gates against. |
| **Weight dtype** | `include/sp/weight_dtype.h` | GGUF/GGML on-disk dequant: `sp_f16_to_f32`, `sp_f32_to_f16`, `sp_dequant_row` (F32 / F16 / Q8_0). |
| **GGUF parser** | `include/sp/gguf.h` | GGUF v3 mmap parser → backend-agnostic descriptor. Used by `sp_transcode` to produce `.sp-model` files. |
| **Hash primitives** | `include/sp/sp_hash.h` | CRC-32, SHA-256, XXH64 for `.sp-model` tensor-name hashing + integrity. |
| **Sieve** | `include/sp/sp_sieve.h` | Friedman-sieve dominance frontier (Phase 5+ feature, off by default). |
| **OK-ring arithmetic** | `include/sp/ok_int.h` | `O_K = Z[(1+√-163)/2]` arithmetic primitives (Phase 1A). |
| **Test harness** | `include/sp/sp_test.h` | Shared test macros used by every `core/<module>/<module>_test.c`. |

The L1 ABI in `sp_l1.h` is **frozen** (tag `lat-phase2-contract-frozen`).
Other public headers are frozen for already-shipped sub-phases; growth
discipline for `sp_arch_info` is in the §2 comment block of `sp_l1.h`
(append fields in the reserved 256-byte arch-struct tail; loader memcpies
`min(arch_struct_size, sizeof(sp_arch_info))` so old files leave new fields
zero — the "unspecified" sentinel).

---

## 2. Current status

Honest snapshot, 2026-06-03. This main reflects the standalone repo state.
The engine-submoduled copy of math-core (under
`shannon-prime-system-engine/lib/shannon-prime-system/`) carries the most
recent sprint output including the WIRE-HEX §6 forward-backend hook —
that submodule's commit will land back on this repo's main in the next
sync wave.

**Headline (what the math-core now proves).** The discrete forward is bit-exact
on **5 arch families** (through the 35B-A3B Gated-DeltaNet MoE); the reducing
`.sp-model` codec is **output-lossless and smaller than source** (C1); the
NTT-CRT / Frobenius / Spinor / KSTE primitives are all shipped + gated. These
primitives feed the engine's measured envelope — the two-ring memory (910× @32k,
7.57 µs/read off Optane) and the WIRE-CPU integer pipe (0.84 → 39.52 tok/s, 47×)
are realized in the engine repo on top of this core. The open headline remains
the **Spinor per-vector KV codec ratio at bit-exact** (lossy 29/31 today) — see
`shannon-prime-lattice/papers/PPT-LAT-STATE.md`.

| Component | Status |
|-----------|--------|
| `core/forward_kernels` — `sp_dot_f32`, `sp_rmsnorm`, `sp_rope_neox`, `sp_attn_head` | **shipped** (scalar reference) |
| `core/forward_dispatch` — `sp_matmul`, `sp_embed_row`, `sp_as_f32` | **shipped** |
| `core/forward` — Qwen3 / Gemma3 / Gemma4 / Qwen3.6-35B-A3B MoE (Gated DeltaNet) prefill + persistent-KV decode | **shipped** |
| `core/session` — `sp_session_create`, `sp_prefill_chunk`, `sp_decode_step`, clone/rewind, KV-mode flags | **shipped** |
| `core/io_format` — `.sp-model` v0 loader (`sp_model_load`/`sp_model_unload`, `sp_model_arch`) | **shipped** |
| `core/io_format` — **reducing codec** (`OK_Q4`/`OK_Q8` body ≤ source quant) | **shipped (C1)** — output-lossless top-1 (gemma4 + qwen35moe); qwen35moe 16.3 < 19.7 GB, Qwen3-0.6B-f16 1,439 → 720 MB (50%) |
| `core/io_hash` — CRC-32 / SHA-256 / XXH64 | **shipped** |
| `core/weight_dtype` — F16↔F32 + per-row dequant (F32/F16/Q8_0) | **shipped** |
| `core/gguf` — GGUF v3 mmap parser | **shipped** |
| `core/ntt_crt` — dual-prime CRT-NTT, `N ∈ {128,256,512}` | **shipped** |
| `core/poly_ring` — R_q attention via NTT | **shipped** |
| `core/poly_ring_bluestein` — arbitrary-N power-of-2 via Bluestein chirp-z | **shipped** (engine submodule) |
| `core/frobenius` — per-row Q8/Q4 codec | **shipped** |
| `core/arena` — packed-weight arena | **shipped** |
| `core/vht2` — Spinor 63-byte block + Möbius reorder + CRC-8 | **shipped** |
| `core/kste` — encoder + Tier-0/Tier-1 dominance | **shipped** |
| `core/model` — Qwen3 / Qwen2.5 / Gemma3 / Gemma4 / Qwen3.6-35B-A3B MoE representation + GGUF load/free | **shipped** |
| `core/ok_arith` — `O_K = Z[(1+√-163)/2]` integer arithmetic | **shipped** |
| `core/arm` — Algebraic Resonance Memory (HRR binding in R_q) | **in progress** |
| `core/sieve` — Friedman-Kruskal dominance sieve | **in progress** |
| `core/dominance` — componentwise dominance helpers | **shipped** |
| `core/sp_channel` — TailSlayer channel oracle integration | **shipped** (offline cache pattern) |
| L1 ABI §6 forward-backend hook (`sp_session_register_forward_backend`) | **shipped on engine submodule** (sprint WIRE-HEX); will sync to this repo's main in next bump |

**Frozen primes & constants** (mirrored in `include/sp/ntt_crt.h`):

| Constant | Value |
|----------|------:|
| `SP_NTT_Q1` | `1073738753` |
| `SP_NTT_Q2` | `1073732609` |
| `SP_NTT_M` (= `q_1·q_2`) | `1152908312643096577` |
| Garner `Q1_INV_MOD_Q2` | `894602413` |
| Admissible `N` (direct NTT) | `{128, 256, 512}` |
| Admissible `N` (Bluestein) | all powers of 2 ≤ 512 |

The frozen primes both have 2-adic valuation `v_2(q-1) = 10`, so `N > 512`
is mathematically impossible with the current dual-prime pair. Long-context
NTT (ctx ≥ 1024) is done via **tiled `N=512` transforms** — asymptotic
`O(N log N)` decoupling still holds because of tiling. See
`shannon-prime-lattice/papers/PPT-LAT-Roadmap.md` §4-NTT for the
tiling rationale. A third prime to extend the admissible N range is filed
as `Phase 4-NTT-PRIME-EXTENSION` (cascades across Garner constants,
L1 ABI, every cross-backend bit-identity gate).

---

## 3. Build

### 3.1 Whole repo (default tier — Windows MinGW gcc 15.2 + Ninja)

```bash
cmake -B build -G Ninja -DCMAKE_C_COMPILER=gcc
cmake --build build
ctest --test-dir build --output-on-failure
```

### 3.2 Other compilers

| Tier | Toolchain | Notes |
|------|-----------|-------|
| 1 | Windows MinGW gcc 15.2 | Closes in-session; primary CI target |
| 2 | Linux gcc 11+ / clang 14+ | `.github/workflows/ci.yml` |
| 3 | Windows MSVC (VS 2019 BT) | Follow-up wave; `cmake -B build-msvc -G "Visual Studio 16 2019"` |
| Cross | `aarch64-linux-android` (NDK r25+) | Engine-side; see `shannon-prime-system-engine/tools/sp_daemon/build-android-libs.bat` |

### 3.3 Single-module fast iteration

Each `core/<module>/` is self-contained with its own `CMakeLists.txt`:

```bash
cmake -B core/ntt_crt/build -S core/ntt_crt -G Ninja -DCMAKE_C_COMPILER=gcc
cmake --build core/ntt_crt/build
ctest --test-dir core/ntt_crt/build --output-on-failure
```

Run once with `-DSP_UBSAN=ON` before closing a module. The build
auto-falls-back to `-fsanitize-undefined-trap-on-error` where libubsan
is unavailable (e.g. MinGW-Builds).

### 3.4 Module conventions

Read `CONVENTIONS.md` before adding a Phase-1 module. The convention
matters because the root build is **EXISTS-guarded** — `SP_MODULES` lists
every module, but each line is skipped if its directory isn't present.
That lets parallel agents drop in `core/<m>/CMakeLists.txt` without ever
editing the root.

Each module owns:
- `core/<m>/CMakeLists.txt` (`sp_module(<m> SOURCES ... TESTS ...)`)
- `core/<m>/<m>.c` + headers
- `core/<m>/<m>_test.c` using `sp_test.h` macros
- Public header at `include/sp/<m>.h`

---

## 4. Professional API reference

### 4.1 Lifetimes and ownership (`sp_l1.h`)

Two opaque types cross the FFI:

```c
typedef struct sp_model    sp_model;     // read-only after load; many sessions per model
typedef struct sp_session  sp_session;   // single-thread state: KV + ARM + sieve + arch scratch
```

- **Caller-allocates on the hot path.** Logits buffers are caller-owned
  and sized from `sp_arch_info.vocab_size`. L1 mallocs nothing L2 has to
  free. There is no `sp_alloc`/`sp_free` pair.
- **Model immutable after load.** Many sessions per model. The L2 wrapper
  is `Send + Sync`.
- **Session single-thread mutable.** `Send` but NOT `Sync` — no two
  threads inside one forward call. Rust bindings enforce with `&mut self`.
- **Cancellation inverted to an L2-owned atomic flag.** `cancel_flag`
  pointer must stay valid until `sp_session_destroy` returns. L1 reads
  it relaxed at layer boundaries; on cancel, unwinds to the last
  completed boundary and returns `SP_ECANCEL`.
- **Determinism fixed at create.** `sp_session_config.deterministic = 1`
  → serial reductions, single stream, bit-exact reference behaviour.

### 4.2 Model load / unload (`sp_model.h`)

```c
sp_status sp_model_load   (const char *sp_model_path,
                           const char *sp_tokenizer_path,
                           /*out*/ sp_model **out);
void      sp_model_unload (sp_model *m);
sp_status sp_model_arch   (const sp_model *m, sp_arch_info *out);
```

`sp_model_load` is pure `mmap` + header parse + tensor table pointer
setup. Zero malloc proportional to tensor data; the file IS the
in-memory layout. `sp_model_arch` populates a caller-stack-allocated
`sp_arch_info` from the `.sp-model` header's reserved 256-byte arch
struct.

### 4.3 Session lifecycle (`sp_l1.h` §1, §6)

```c
typedef struct {
    size_t   max_context;     /* 0 = arch default */
    int      deterministic;   /* nonzero = bit-exact */
    uint32_t arm_bank_kb;     /* Phase 9+; 0 = arch default */
    uint32_t sieve_capacity;  /* Phase 5+; 0 = arch default */
    uint32_t flags;           /* SP_KV_SPINOR | SP_KV_SPINOR_REF */
    uint32_t precision_override; /* sp_precision (FP16 / QF32) */
} sp_session_config;

sp_status sp_session_create (const sp_model *m,
                             const sp_session_config *cfg,
                             volatile int *cancel_flag,    /* L2-owned, may be NULL */
                             sp_session **out_session);
void      sp_session_destroy(sp_session *s);
uint32_t  sp_session_precision(const sp_session *s);
```

KV-mode flag selection precedence:

| Flag | Effect |
|------|--------|
| `0` (default) | Persistent f32 KV cache (no compression) |
| `SP_KV_SPINOR` | Persistent compressed KV: VHT2 + Möbius 63-byte Spinor blocks, decoded inline on read |
| `SP_KV_SPINOR_REF` | Parity reference: f32 cache + in-place Spinor round-trip (decode-from-block ≡ this, by codec identity) |

Precision selection (FP16 sub-phase):
`cfg->precision_override` > `arch_info.preferred_precision` > `SP_PRECISION_F32`.
Math-core reference forward ignores this and stays f32; backend dispatch
reads it to select the FP16 vs F32 vs QF32 kernel path.

### 4.4 Two-function forward (`sp_l1.h` §3)

```c
sp_status sp_prefill_chunk(sp_session *s,
                           const int32_t *tokens, size_t n_tokens,
                           float *logits_last, size_t logits_capacity);

sp_status sp_decode_step  (sp_session *s,
                           int32_t token,
                           float *logits, size_t logits_capacity);
```

`prefill` consumes `n_tokens`, advances position by `n_tokens`, writes
ONLY the last token's logits. `decode` consumes one token, advances by
one, writes that token's logits. `logits_capacity < vocab_size` returns
`SP_EBADARG`. On a tripped cancel flag the call unwinds to the last
completed layer and returns `SP_ECANCEL`.

### 4.5 Speculative-decoding primitives (`sp_l1.h` §4)

```c
sp_status sp_session_clone   (const sp_session *s,
                              volatile int *cancel_flag,
                              sp_session **out);
sp_status sp_session_rewind  (sp_session *s, size_t n_tokens);
sp_status sp_session_position(const sp_session *s, size_t *pos_out);
```

`clone` deep-copies KV/ARM/sieve into an independent session (the
spec-decode fork). `rewind` rolls back `n_tokens` of accepted state
(the reject primitive; ARM writes are journaled so rewind is exact).
`position` reports the current sequence index.

### 4.6 Status codes (`sp_status.h`)

```c
typedef enum {
    SP_OK                =   0,

    /* Generic */
    SP_ENOMEM            =  -1,
    SP_ECANCEL           =  -2,
    SP_EBADARG           =  -3,
    SP_EBADSTATE         =  -4,
    SP_EUNSUPPORTED      =  -5,
    SP_EIO               =  -6,

    /* Model load / arch */
    SP_EBADFORMAT        = -10,
    SP_EBADARCH          = -11,
    SP_ETOKENIZER_HASH   = -12,
    SP_EVOCAB            = -13,

    /* Discrete-algebra layer */
    SP_ESPINOR_BADBLOCK  = -20,
    SP_EVHT2_DOMAIN      = -21,
    SP_EMOBIUS_PERM      = -22,
    SP_EOK_NORM          = -23,
    SP_EFROBENIUS_QUANT  = -24,
    SP_ENTT_OVERFLOW     = -25,
    SP_ERING_DEGREE      = -26,

    /* Lattice features */
    SP_ESIEVE_FULL       = -30,
    SP_EARM_BANK_FULL    = -31,
    SP_EDOMINANCE_CYCLE  = -32,
    SP_ECONTEXT_FULL     = -33,

    /* Backend */
    SP_ECUDA             = -40,
    SP_EVULKAN           = -41,
    SP_EHVX              = -42,
    SP_EBACKEND_OOM      = -43
} sp_status;

const char *sp_last_error(void);   /* thread-local detail string */
```

Values **must not be renumbered** — frozen ABI. The
discrete-algebra codes (-20..-26) are the "lost the algebraic
invariant" surface: a tripped theorem maps to a status rather than
collapsing to `SP_EBADSTATE`. The CUDA/Vulkan/Hexagon backends wrap
their native error into the matching `SP_E*` code and stash the
human-readable detail in the thread-local string.

### 4.7 NTT-CRT primitive (`ntt_crt.h`)

```c
#define SP_NTT_Q1   ((uint32_t)1073738753u)
#define SP_NTT_Q2   ((uint32_t)1073732609u)
#define SP_NTT_M    ((int64_t)1152908312643096577)

typedef struct ntt_ctx ntt_ctx;

ntt_ctx *ntt_init(uint32_t N);      /* N ∈ {128,256,512}; else NULL */
void     ntt_free(ntt_ctx *ctx);

void ntt_forward(const ntt_ctx *ctx, const int32_t *in,
                 uint32_t *out1, uint32_t *out2);     /* residue mod q1, mod q2 */

void ntt_pointwise_mul(const ntt_ctx *ctx,
                       const uint32_t *a1, const uint32_t *b1, uint32_t *out1,
                       const uint32_t *a2, const uint32_t *b2, uint32_t *out2);

void ntt_inverse(const ntt_ctx *ctx, const uint32_t *in1, const uint32_t *in2,
                 int64_t *out);     /* CRT-recombined signed centered (-M/2, M/2] */
```

Production path uses **no 128-bit type** — 30-bit residues × 30-bit
residues fit in `uint64_t`; Barrett reduction keeps every intermediate
< `2^64`. Configure-time guard `T_NTT_5` enforces this in
`core/ntt_crt/CMakeLists.txt`. A test-only parity oracle
(`ntt_ref_int128.c`) uses `__int128` to cross-check the production path.
This header is MSVC-clean (no GNU extensions); compiles on every tier.

### 4.8 Polynomial-ring attention (`poly_ring.h`)

```c
typedef struct sp_pr_ctx sp_pr_ctx;

sp_pr_ctx *sp_pr_init(uint32_t N);         /* N ∈ {128,256,512}; else NULL */
void       sp_pr_free(sp_pr_ctx *ctx);
uint32_t   sp_pr_degree(const sp_pr_ctx *ctx);

void  sp_pr_mul(sp_pr_ctx *ctx, const int32_t *a, const int32_t *b,
                int64_t *out);             /* out = a ⊗ b in R_q, N coeffs */

int64_t sp_pr_inner(sp_pr_ctx *ctx, const int32_t *q, const int32_t *k);
        /* ⟨q,k⟩ = coefficient 0 of negacyclic q ⊗ k*; exact in Z when |⟨q,k⟩| < M/2 */

void sp_pr_attention(sp_pr_ctx *ctx, const int32_t *q,
                     const int32_t *const *keys, int n_keys,
                     double *probs_out);   /* softmax over recovered int scores */
```

The negacyclic involution `k*` (`k*_0 = k_0`, `k*_j = -k_{N-j}` for `j>0`)
turns the attention inner product into one coefficient of a polynomial
product computed exactly via NTT. Bit-identical to scalar dot for any
coefficient range that fits `|⟨q,k⟩| < M/2` (e.g. `|coeff| < 2^23`
gives `|⟨q,k⟩| < 2^55` — room to spare).

For arbitrary power-of-2 `N` ≤ 256 (e.g. `HD = 64` for Qwen3 / Qwen2.5-Coder),
use `poly_ring_bluestein.h` (engine submodule):

```c
sp_pr_bluestein_ctx *sp_pr_bluestein_init(uint32_t N);   /* N power-of-2 ≤ 256 */
void                 sp_pr_bluestein_free(sp_pr_bluestein_ctx *);
int64_t              sp_pr_bluestein_inner(sp_pr_bluestein_ctx *, const int32_t *q, const int32_t *k);
void                 sp_pr_bluestein_mul(sp_pr_bluestein_ctx *, const int32_t *a, const int32_t *b, int64_t *out);

/* Override the inner NTT-CRT dispatch (engine backends register here). */
typedef int (*sp_compute_ntt_dispatch_fn)(void *handle, int q_idx, int N,
                                          const int32_t *in, uint32_t *out_residue);
void sp_pr_bluestein_set_backend(sp_pr_bluestein_ctx *, void *handle,
                                 sp_compute_ntt_dispatch_fn fwd,
                                 sp_compute_ntt_dispatch_fn inv);
```

Non-power-of-2 head-dims with odd factors (96, 288, 384) are not
admissible — the frozen primes' `odd_part(q-1)` is not divisible by 3
so mixed-radix is mathematically invalid. For those head-dims the
"boring but correct" answer is the direct integer dot product with
Barrett (also shipped). See the lattice memory entry
`reference-ntt-bluestein-arbitrary-n-escape`.

### 4.9 Forward dispatch (`forward_dispatch.h`)

```c
int sp_matmul(const qwen3_model *m, const gguf_tensor *W,
              const float *X, int n_tok, int in, int out, float *Y);
int sp_embed_row(const qwen3_model *m, int32_t tok, int E, float *dst);
const float *sp_as_f32(const qwen3_model *m, const gguf_tensor *t);
void sp_kernels_read_env(void);   /* refresh SP_ENGINE_FROB / SP_ENGINE_F16_ACT / SP_Q4_PROMOTE */
```

Honors the packed-weight arena when built (inline lift, no per-matmul
re-quantization) and the runtime weight-path gate knobs:

| Knob | Values | Effect |
|------|--------|--------|
| `SP_ENGINE_FROB` | `0` / `1` / `2` / `3` / `4` | 0 = pure f32 reference; 1 = Q8 inline lift; 2 = Q8 with dequant arena; 3 = Q4 inline; 4 = Q4 with mixed-precision promotion |
| `SP_ENGINE_F16_ACT` | `0` / `1` | Round matmul activations to F16 (ggml-faithful path for cross-validation) |
| `SP_Q4_PROMOTE` | `float` | Q4 rows whose round-trip rel-error exceeds this threshold get promoted to Q8 |
| `SP_CPU_SCALAR` | `0` / `1` | Force scalar reduction (disable AVX vectorization) |
| `SP_ARENA_RELEASE` | `0` / `1` | Release the GGUF mapping after arena pack (~50% RAM cut) |
| `SP_ARENA_EMBED` | `0` / `1` | Include the token embedding in the arena pack |
| `SP_ENGINE_NTT_ATTN` | `0` / `1` | Enable polynomial-ring NTT attention overlay |
| `SP_KV_SPINOR` | `0` / `1` | Compressed Spinor-block KV cache |
| `SP_KV_SPINOR_REF` | `0` / `1` | Parity reference: f32 cache + in-place Spinor round-trip |

### 4.10 Forward kernels (`forward_kernels.h`)

```c
float sp_dot_f32(const float *a, const float *b, int n);
void  sp_rmsnorm(const float *x, const float *w, int n, float eps, float *out);
void  sp_rmsnorm_head(float *v, const float *w, int d, float eps);
void  sp_rope_neox(float *v, int d, int p, float base);
void  sp_attn_head(const float *qh, const float *KC, const float *VC,
                   int pos, int KVD, int kvh, int HD, float ascale, int win,
                   float *sc, float *out);
```

These are the **portable scalar reference**. Vectorised backends reorder
the reduction and gate to these for correctness. Sum-of-squares in
`sp_rmsnorm` accumulates in `double` — that's the reference precision
behaviour.

### 4.11 Frobenius lift (`frobenius_lift.h`)

Per-row scaling is the load-bearing choice. A single per-tensor scale
collapses the dynamic range of every row onto the same int8 grid; rows
whose magnitudes are small relative to the tensor max degrade to noise.

```c
#define SP_FROB_QMAX 127   /* symmetric range, NOT -128 */

typedef struct {
    int      rows, cols;
    int8_t  *packed;       /* rows*cols codes, row-major */
    float   *row_scale;    /* fp32 per-row scale */
} sp_frob_tensor;

float   sp_frob_row_scale(const float *row, int cols);     /* max_c |row[c]| */
int8_t  sp_frob_quantize  (float v, float s);              /* ceiling-shift round-half-away-from-zero */
float   sp_frob_dequantize(int8_t q, float s);             /* q * (s / 127) */
```

Quantisation rule: `q = copysignf(floorf(fabsf(x) + 0.5f), x)` — symmetric
round-half-away-from-zero, deterministic across FP rounding modes. The
symmetric `[-127, 127]` range (NOT `-128`) is required for `v_hat = q·(s/127)`
to reproduce `v = +s` exactly at code `+127`.

A zero row (`s == 0`) is encoded as all-zero codes with `row_scale = 0` and
dequantises back to all zeros.

### 4.12 Spinor block (`spinor_block.h`) — FROZEN

The 63-byte container, the 7/55/1 split, `SP_SPINOR_LAYOUT_VERSION`, and
the CRC-8 trailer are FROZEN per roadmap S4.5/S7.9. Any field move /
resize / reorder / semantic change requires a version bump and a
migration note. Tests `T_VHT_3` (size), `T_VHT_5` (header byte image),
`T_VHT_6` (version constant) guard the contract.

```c
typedef struct {
    uint8_t vht2_header[7];  /* bytes 0-3 scale fp32, byte 4 exponent i8, byte 5 basis_sel, byte 6 reserved */
    uint8_t mobius_body[55]; /* 55 int8 quantized anchor coefficients at Möbius-permuted positions */
    uint8_t checksum;        /* CRC-8/SMBus over bytes [0..62] */
} sp_spinor_block_t;
```

Möbius permutation `i → (17·i) mod n`: pure index permutation, no data
dependence. 17 is coprime to 55 (= 5·11). VHT2 anchors (v1 canonical
basis) map anchor `i` to canonical coordinate `i`; longer K-vectors
truncate, shorter pad with zeros.

### 4.13 KSTE encoder (`kste.h`)

```c
#define SP_KSTE_LAYOUT_VERSION 1
#define SP_KSTE_BRANCHING   3
#define SP_KSTE_DEPTH       3
#define SP_KSTE_LABEL_DIM   6

typedef union { uint8_t bytes[64]; } sp_kste_tree;
```

Maps an int32 K-vector into a fixed 64-byte packed tree in `T_{60,3}`:
1 root + 3 first-level children + 9 grandchildren = 13 nodes. Each node
carries a 6-component label of quantised order statistics (min, 20th,
40th, 60th, 80th percentile sample, max) over the K-vector slice that
node covers — six independent sample points, so the componentwise order
has genuine incomparable pairs.

Input must be **int32**; floats can't be made byte-identical across
platforms. Callers holding floats quantize at the door.

Children at each level are sorted into canonical lexicographic order at
encode time, so "same first-level child multiset" is byte-detectable
and Tier-1 dominance is well-defined.

**Known gotcha** (memory entry `reference-kste-quantize-i16-clamp`):
`label_of` clamps int32 inputs to int16 range before order-statistics.
Token IDs > 32k saturate. Callers in the token-ID domain should fold
to i16 via XOR-low/XOR-high (or SplitMix64 + fold for clustered IDs).

### 4.14 Packed-weight arena (`arena.h`)

```c
sp_arena *sp_arena_build(const struct qwen3_model *m, int precision,
                         float q4_promote, int include_embed);
sp_arena *sp_arena_from_packed(const sp_arena_tensor *ts, int n, int precision);
void      sp_arena_free(sp_arena *a);

const sp_arena_tensor *sp_arena_find(const sp_arena *a, const char *name);
int  sp_arena_dequant_row(const sp_arena_tensor *at, int r, float *dst);

size_t sp_arena_bytes(const sp_arena *a);
int    sp_arena_precision(const sp_arena *a);   /* 8 or 4 */
long   sp_arena_promoted(const sp_arena *a);    /* Q4 rows promoted to Q8 */
```

Phase 1a covers matmul weights only (attn q/k/v/o, ffn gate/up/down,
LM-head `output`); embedding and norms stay f32 from the GGUF mapping.
Phase 1b folds the embedding in and releases the mapping (≈50% RAM cut).

`sp_arena_from_packed` is the `.sp-model` load path — the on-disk Q8
codes are byte-identical to what `sp_arena_build` would produce from the
GGUF source, so the loader reconstructs a bit-identical arena via a
single `memcpy` per tensor.

---

## 5. How math-core relates to the backends

Math-core's forward path is the **reference** — bit-exact, scalar where
possible, deterministic. Every backend in
`shannon-prime-system-engine/src/backends/` is a *replacement* for some
slice of that reference, validated against it via a byte-exact test
gate (`T_*_BIT_EXACT`). Two registration shapes are supported:

| Shape | API | Use when |
|-------|-----|----------|
| **Full forward** | `sp_session_register_forward_backend(s, handle, fn)` | Backend owns the entire forward pass (e.g. `gemma3_forward_hexagon` uploads the Q8 weight blob to cDSP once and runs the whole forward there). Activated for **prefill** only; decode keeps the math-core reference (no persistent-KV API across backends). |
| **NTT dispatch** | `sp_pr_bluestein_set_backend(ctx, handle, fwd_fn, inv_fn)` | Backend owns just the inner residue NTTs (e.g. cDSP runs `ntt_hvx_vtcm_oracle` + `intt_hvx_oracle` via FastRPC). Routes through the polynomial-ring attention overlay. |

The L1 ABI §6 hook landed via sprint WIRE-HEX (2026-05-31). See
`shannon-prime-system-engine/tools/sp_compute_skel/docs/CLOSURE-WIRE-HEX.md`
for the canonical wiring template — CUDA + Vulkan daemon ports are
symmetric sprints under the same hook.

---

## 6. License

MIT. See `LICENSE`.
