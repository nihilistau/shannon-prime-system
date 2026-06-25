# shannon-prime-system — the math core

> **The clean, engine-free discrete-inference substrate of [Shannon-Prime](https://github.com/nihilistau/shannon-prime-lattice).**
> A from-scratch C library exposing a **frozen Layer-1 C ABI** plus the exact-integer discrete-algebra
> primitives (`O_K = Z[(1+√−163)/2]`, dual-prime CRT-NTT, Frobenius lift, Spinor block, KSTE, the
> two-ring KV memory) that every engine backend shares.

## What Shannon-Prime is

Shannon-Prime is a **fully local, byte-exact, auditable language-model organism**. It serves Google's
**Gemma-4-12B** (OK_Q4B quant) on a single **RTX 2060**, through **our own** inference engine, on an
**exact-integer arithmetic substrate** (`O_K = Z[(1+√−163)/2]`, dual-prime negacyclic CRT-NTT), with
a working memory it owns: it learns facts from conversation, recalls them, **forgets / supersedes /
merges** them on its own judgement, calls tools and runs code, stores whole conversations both
complete and summarized, and consolidates them on a heartbeat. Every mechanism is a flag that is a
**strict no-op when unset** (the "null floor"); every number has a reproducing command and a gate.

**The thesis (one line):** *position is arithmetic.* An LLM's container can be made **exact
arithmetic** — cross-machine-deterministic, auditable — without losing quality. **Byte-exact = exact
arithmetic / cross-machine determinism / auditability — NOT compression.** Every structure-on-content
compression lever is a measured honest-negative ([§7](#7-boundary-thesis--honest-negatives)).

> **Read this first:** lattice [`papers/PPT-LAT-KEYSTONE.md`](https://github.com/nihilistau/shannon-prime-lattice/blob/main/papers/PPT-LAT-KEYSTONE.md) —
> the canonical, current, complete description of the whole system (the five repos, the architecture
> diagram, the memory model, the gate index). This README is this repo's specifics.

## The five repositories

| Repo | Role | This core's relation |
|---|---|---|
| **shannon-prime-lattice** | umbrella: papers, contracts, RFC, the KEYSTONE map | foundation docs |
| **shannon-prime-system** (this) | the math core: O_K, NTT-CRT, exact islands, ARM two-ring, the frozen L1 ABI | the substrate everything rides on |
| **shannon-prime-system-engine** | the inference engine + backends + the resident daemon + memory agency | consumes this core via the `lib/shannon-prime-system` submodule |
| **shannon-prime-harness** | the agent harness: tool calling, conversation memory, the agency loop | calls the engine daemon over HTTP |
| **Position_Is_Arithmetic** | the public face: receipts-first papers + `LEDGER.md` | public results |

This repo is **consumed by the engine as the `lib/shannon-prime-system` submodule** — every
accelerated backend (CPU AVX2/AVX-512, CUDA, Vulkan, Hexagon HVX) registers against this library's
L1 ABI, and the scalar reference forward here is the bit-exact correctness anchor each backend gates
against. The standalone clone can sit *behind* the submodule pin: `git fetch` + check before building.

## Where this core sits in the stack

```
 ┌───────────────────────────────────┐
 │  MATH CORE  (this repo)            │
 │  O_K = Z[(1+√−163)/2]              │   ← the exact-integer container everything rides on
 │  dual-prime NTT-CRT (q1,q2≈2^60)   │
 │  core/exact_islands (RMS/softmax/  │   ← the 4 nonlinear fp32 islands as exact-integer refs
 │     GELU/RoPE, CORDIC, no libm)    │      (anchor for the engine's SP_BYTEEXACT forward)
 │  core/arm  two-ring KV memory      │   ← the episodic store the recall organism rides
 │  core/frobenius + core/arena (OK_Q4B)
 │  L1 ABI: forward + kvdecode verbs  │   ← registered by the engine's CUDA/CPU backends
 └───────────────┬───────────────────┘
                 │ registers via L1 §6 (forward) / §6b (kvdecode)
                 ▼
   ENGINE backends (CUDA/CPU/Vulkan/Hexagon)  ──►  sp-daemon (served 12B chat + W_c recall)
```

The recall-relevance problem the ARM contract posed — *which stored episode is load-bearing for this
query?* — is **SOLVED**, but **the live selector itself lives host-side in the engine daemon**
(`recall.rs`/`routes.rs`); **NO frozen-ABI change and NO `.sp-model` format change**. This core owns
the episode store (`core/arm/`) + the exact-integer substrate it rides on; the selector is an
engine-side rider ([§5](#5-the-recall-organism-where-the-pieces-live)).

## 1. The frozen L1 ABI (`include/sp/sp_l1.h`)

The L1 ABI is **frozen** (tag `lat-phase2-contract-frozen`). Growth is **append-only**. Core verbs:

```c
sp_status sp_prefill_chunk (sp_session*, const int32_t *toks, size_t n, float *logits_last, size_t cap);
sp_status sp_decode_step   (sp_session*, int32_t token, float *logits, size_t cap);
sp_status sp_session_clone (const sp_session*, volatile int *cancel, sp_session **out);   // spec-decode fork
sp_status sp_session_rewind(sp_session*, size_t n_tokens);                                 // O(1) reject (journaled)
sp_status sp_session_position(const sp_session*, size_t *pos_out);
```

Backend registration hooks:

| § | Verb | What it owns |
|---|---|---|
| **§6** | `sp_session_register_forward_backend` | A full forward pass (PREFILL). |
| **§6b** *(additive)* | `sp_session_register_kvdecode_backend` | A **stateful, session-resident KV decode** (`open/prefill/decode_step/rewind/position/close`). When registered, `sp_decode_step` routes the single-token call to it. The engine's CUDA `gemma4_kv_decode_logits` registers here so the daemon drives the 12B token-by-token (`G-WIRE-CUDA-DECODE-GEMMA4`: 32/32 == oracle, VRAM O(1)). Append-only — no frozen surface renumbered. |

Other frozen surfaces: `exact_islands.h` (the 4 islands; gate `T_EXACT_ISLANDS`), `sp_status.h`,
`sp_model.h` (the `.sp-model` mmap format), `ntt_crt.h` / `poly_ring.h` (dual-prime NTT),
`frobenius_lift.h`, `arena.h` (OK_Q4B), `spinor_block.h` (the frozen 63-byte VHT2 record), `kste.h`,
`arm.h` (the two-ring recall router), `ok_int.h` (`O_K` arithmetic).

### Frozen primes & constants (`include/sp/ntt_crt.h`)

| Constant | Value |
|---|---:|
| `SP_NTT_Q1` | `1073738753` |
| `SP_NTT_Q2` | `1073732609` |
| `SP_NTT_M` (= `q1·q2`) | `1152908312643096577` |
| Garner `Q1_INV_MOD_Q2` | `894602413` |
| Admissible `N` (direct NTT) | `{128, 256, 512}` |

`M ≈ 2^60` fits `uint64_t` → **no `__int128`**. Long-context NTT is done via tiled `N=512` transforms.

## 2. What lives in `core/`

The scalar reference forward here is the **bit-exact correctness anchor** — a discrete `Z_q` forward
proven **argmax bit-exact to llama.cpp** on five arch families: Qwen3-0.6B, Qwen2.5-Coder-0.5B,
Gemma3-1B, Gemma4-E2B, and Qwen3.6-35B-A3B MoE (Gated DeltaNet).

| Module | Status | Notes |
|---|---|---|
| `core/forward/decode.c` — the **only** decode in the tree | `[PROVEN]` | two-ring KV + NTT-KV fusion + the `SP_REPLAY` episode-replay seam (off-path bit-exact, `T_GENKV_REPLAY_NULL` 34/34). `core/forward/gemma4.c` is forward-only. |
| `core/arm/` — the two-ring KV memory + ARM recall router | `[PROVEN]` | ±1 Rademacher projection + bit-packed signature scan, recall-hit telemetry, cold-evict mask, abstract Ring-2 backend ABI. Gates `T_ARM`, `T_ARM_SIG`, `T_ARM_GEOM`, `T_ARM_GENKV`. |
| `core/exact_islands/` — the 4 nonlinear fp32 islands as exact-integer references | `[PROVEN]` | RMSNorm / softmax / GELU / RoPE (RoPE via fixed-point CORDIC, no libm). Gate `T_EXACT_ISLANDS`. The math-core anchor for the engine's `SP_BYTEEXACT` byte-exact forward — the **auditability** axis, NOT compression. |
| `core/ntt_crt/` + `core/poly_ring/` — dual-prime CRT-NTT + R_q attention | `[PROVEN]` | byte-exact negacyclic NTT, production path uses **no 128-bit type** (`T_NTT_5` guard). |
| `core/ring3/` — native-C Ring-3 VSA bind/unbind + NIGHTSHIFT consolidation | `[PROVEN]` | gate `T_RING3_NATIVE` GREEN. |
| `core/frobenius/` + `core/arena/` — Frobenius-lift codec + arena layout v2 | `[PROVEN]` | the **OK_Q4B** per-32-block-scaled codec; carries the gemma-4-12B artifact. |
| `core/vht2/` — the **frozen** 63-byte Spinor block | `[PROVEN]` | VHT2 + Möbius reorder + CRC-8; `0xA5` sentinel. |
| `core/kste/` · `core/ok_arith/` · `core/sieve/` | `[PROVEN]` / `[WIRED]` | KSTE fingerprint; `O_K` integer arithmetic; the dominance sieve. |

## 3. Headline status (verify, don't assume)

**`[PROVEN]` — the forward + substrate.** Argmax bit-exact on 5 arch families; the reducing
`.sp-model` codec is output-lossless and smaller than source. Citable engine envelope (lattice
`LEDGER.md`): gemma-4-12B **26.1 tok/s @ wikitext PPL 5.12 on one RTX 2060-12GB**; the gemma-4 GGUF
ecosystem ships broken weights (gold forward PPL 4.68 vs GGUF 192–506) → **safetensors-direct is the
only trusted weight path**; two-ring **910× resident-KV shrink @32k**. `[HONEST-NEGATIVE]` the 32k
NIAH MISSed — kept on the record.

**`[gated-GREEN] / default-off` — the byte-exact forward, anchored here.** Byte-exact = **exact
arithmetic / cross-machine determinism** (auditability), NOT compression. The new piece in this core
is `core/exact_islands/` + the L1 §6b kvdecode verb; on the engine side these carry the default-off
`SP_BYTEEXACT` device-integer forward — `G-BYTEEXACT-FORWARD-12B`: OFF = PPL 4.6665 byte-identical /
ON = 4.6569 parity / run-to-run bit-identical. The one remaining item is EXTERNAL — a bit-identical
logit check across two **physical** GPUs.

**`[gated-GREEN]` — XBAR memory unified onto this repo's exact-integer O_K substrate.** Ring-3 VSA
bind **256/256 bit-identical** to native `sp_pr_mul`/`ntt` and reduction-order-immune
(`G-R3-BIND-on-O_K`); a Frobenius π^k integer Ring-2 episode store (`G-R2-FROB`); the full
real-episode organism loop (`G-XBAR-ORGANISM-FULL`).

## 4. Build

Build truth is pinned in the engine repo (`shannon-prime-system-engine/docs/BUILD-ENV.md`). The
canonical CPU build is **MinGW gcc 15.2** in `build/` (Ninja); **MSVC cannot build the CPU tree**.

```bash
cmake -B build -G Ninja -DCMAKE_C_COMPILER=gcc
cmake --build build
ctest --test-dir build --output-on-failure
```

**Sync discipline (binding):** this repo is also carried as the engine's `lib/shannon-prime-system`
submodule, so the two checkouts can diverge. `git fetch` + check `git rev-list --count HEAD..origin/main`
**before** any build or commit; every standalone commit is followed by a submodule bump in the engine.

## 5. The recall organism — where the pieces live

The "which episode is load-bearing?" relevance problem is **SOLVED**, split cleanly:

- **This core owns** the episode store (`core/arm/` Ring-2 + the ARM router) and the exact-integer
  substrate it rides on.
- **The engine owns** the live selector (host-side, default-off, null-floor): a **curator**, a
  **teacher-forced ablation labeler** (`SP_B3_SECRET` cudaMemset-ablates the secret's source KV rows
  and re-scores — novel needle collapse **−33.56** vs parametric **−0.15**, pinned **TAU=−8.0**, the
  official ADMISSION oracle), and a **learned `W_c` head** (the live RECALL selector). And, beyond
  recall, the **memory agency** (FORGET / DECIDE / MERGE) — the model curating its own store. Engine
  gates `G-CHAT-B3-WC-DEPLOY`, `G-FORGET`, `G-DECIDE`, `G-MERGE`.

This rides entirely on this repo's two-ring substrate but is an **engine-side rider** — the L1 §6b
verb and the OK_Q4 `.sp-model` container are untouched.

## 6. How math-core relates to the backends

Math-core's forward path is the **reference** — bit-exact, scalar where possible, deterministic. Every
backend in the engine is a *replacement* for a slice of it, validated via a byte-exact gate
(`T_*_BIT_EXACT`). Three registration shapes: **full forward** (§6),
**persistent-KV decode** (§6b), **NTT dispatch** (`sp_pr_bluestein_set_backend`, inner residue NTTs).

## 7. Boundary thesis — honest negatives

`O_K` wins on **exact arithmetic (the container)**; every structure-on-*content* lever was
measured-inert and is kept as an `[HONEST-NEGATIVE]` (do not re-litigate):

- Split-prime `O_K` Dirichlet carriers (`d7d96fe`) — operationally inert.
- Möbius-on-M (`1e70763`) — sheds memories 1.000 → 0.969 @ N=32.
- Entropy-coding the Frobenius codes (`e6d17bb`) — 1.02× dead weight.
- T2-Möbius on the real 12B embedding (`ac76c8e`) — recon cos 0.032 == random.
- The compression reading of "byte-exact" — convicted redundant against the existing per-32-block
  OK_Q4B at gold PPL 4.6665. Byte-exact is the **auditability** axis only.

## Navigation

| You want… | Read |
|---|---|
| **The whole system, current + complete** | lattice `papers/PPT-LAT-KEYSTONE.md` |
| Agent entry + read-order + pre-flight | `AGENTS.md` |
| This repo's specifics + non-negotiables | `CLAUDE.md` |
| Module conventions (before adding a module) | `CONVENTIONS.md` |
| The proven record / the math | lattice `papers/PPT-LAT-STATE.md` / `papers/PPT-LAT-Theory.md` |
| Public results + reproduce commands | [Position Is Arithmetic](https://github.com/nihilistau/Position_Is_Arithmetic) `LEDGER.md` |

## License

MIT. See `LICENSE`.
