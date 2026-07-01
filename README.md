---
type: reference
title: "shannon-prime-system — the exact-integer math core + frozen L1 C ABI"
description: "What this repo is (the discrete-algebra substrate: O_K ring, dual-prime CRT-NTT, Frobenius codec, the 4 exact nonlinear islands, ARM two-ring KV, C2 router, Ring-3 VSA) vs the other four Shannon-Prime repos; the primitive inventory; the boundary thesis; build one-liner + toolchain note; pointers to lattice canon."
tags: [reference, math-core, l1-abi, ntt-crt, o_k, exact-islands, frobenius, arm, ring3, byte-exact, boundary-thesis]
timestamp: 2026-07-01T00:00:00Z
resource: shannon-prime-system/README.md
sp_status: GREEN
sp_gate: G-BYTEEXACT-FORWARD-12B
sp_commit: TBD
sp_repro: none
---

# shannon-prime-system — the math core

> **The exact-integer discrete-algebra substrate of [Shannon-Prime](https://github.com/nihilistau/shannon-prime-lattice).**
> A from-scratch C library that exposes a **frozen Layer-1 (L1) C ABI** plus the exact-integer
> primitives every engine backend rides on. This is the substrate under everything.
> It does **not** serve chat (that is the engine) and it does **not** hold the docs (those live in
> the lattice repo). It is the container: exact arithmetic, cross-machine determinism, auditability.

## 1. What this repo IS (and what the other four are)

`shannon-prime-system` is **THE MATH CORE** — the discrete-algebra primitives and the frozen L1 C
ABI. Everything else in the project is built on top of it or documents it.

| Repo | Role | Relation to this core |
|---|---|---|
| **shannon-prime-lattice** | umbrella: papers, contracts, the KEYSTONE map, the SCOREBOARD, the RFCs — **the docs and the status source of truth** | documents this core; canon lives there |
| **shannon-prime-system** (this) | **the math core:** O_K, dual-prime CRT-NTT, Frobenius codec, the 4 exact islands, ARM two-ring KV, C2 router, Ring-3 VSA, the **frozen L1 C ABI** | the substrate everything rides on |
| **shannon-prime-system-engine** | the inference engine + accelerated backends (CUDA/CPU/Vulkan/Hexagon) + the resident daemon + memory agency; **serves the 12B chat** | consumes this core via the `lib/shannon-prime-system` submodule; registers backends against this L1 ABI |
| **shannon-prime-harness** | the agent harness: tool calling, conversation memory, the agency loop | calls the engine daemon over HTTP |
| **Position_Is_Arithmetic** | the public face: receipts-first papers + `LEDGER.md` | public results |

This core is **consumed by the engine as the `lib/shannon-prime-system` submodule.** Every
accelerated backend registers against this library's L1 ABI, and the scalar reference forward here is
the **bit-exact correctness anchor** each backend gates against.

> **Status source of truth is the lattice repo**, not this README:
> `papers/VERIFIED-SCOREBOARD.md`, `papers/PPT-LAT-KEYSTONE.md`, `papers/PPT-LAT-Theory.md`,
> `papers/PPT-LAT-L1-ABI-v0.md`, `papers/PPT-LAT-FINDINGS-LEDGER.md`. Read those before trusting any
> "current edge" prose; this README describes the substrate, not the day-to-day campaign.

## 2. Where this core sits in the stack

```
 ┌─────────────────────────────────────────────┐
 │  MATH CORE  (this repo)                       │
 │  O_K = Z[(1+√−163)/2]  exact-integer ring     │  ← the container everything rides on
 │  dual-prime CRT-NTT  (q1,q2 ≈ 2^30, M ≈ 2^60) │
 │  Frobenius π^k codec  (OK_Q4B packed weights) │
 │  4 exact nonlinear islands  (RMS/softmax/     │  ← anchor for the engine's SP_BYTEEXACT forward
 │     GELU/CORDIC-RoPE, no libm)                 │
 │  ARM two-ring Spinor KV  (episodic store)      │
 │  C2 256-bit signature router · Ring-3 VSA D=1024 │
 │  ── frozen L1 C ABI (include/sp/sp_l1.h) ──    │  ← the stable surface below
 └───────────────────────┬─────────────────────┘
                         │  L1 §6 (forward) / §6b (kvdecode) registration
                         ▼
   ENGINE backends  (CUDA / CPU / Vulkan / Hexagon)
                         │
                         ▼
   sp-daemon  (served 12B chat + W_c recall + memory agency)
                         │  HTTP
                         ▼
   HARNESS  (tool calling · conversation memory · agency loop)

   docs / canon / status  ──►  LATTICE repo (papers/)
```

The math core owns the **container and the primitives**; the engine owns the **live behavior**
(the W_c recall selector, the memory agency, the served chat) as a host-side rider — **no frozen-ABI
change and no `.sp-model` format change**.

## 3. The frozen L1 ABI (`include/sp/sp_l1.h`)

The L1 ABI is **frozen** (tag `lat-phase2-contract-frozen`). Growth is **append-only** — never
renumber an `sp_status` value, a `§`, or a dispatch-table row. Core verbs:

```c
sp_status sp_prefill_chunk  (sp_session*, const int32_t *toks, size_t n, float *logits_last, size_t cap);
sp_status sp_decode_step    (sp_session*, int32_t token, float *logits, size_t cap);
sp_status sp_session_clone  (const sp_session*, volatile int *cancel, sp_session **out); // spec-decode fork
sp_status sp_session_rewind (sp_session*, size_t n_tokens);                              // O(1) reject (journaled)
sp_status sp_session_position(const sp_session*, size_t *pos_out);
```

Backend registration hooks:

| § | Verb | What it owns |
|---|---|---|
| **§6** | `sp_session_register_forward_backend` | A full forward pass (PREFILL). |
| **§6b** *(additive)* | `sp_session_register_kvdecode_backend` | A **stateful, session-resident KV decode** (`open/prefill/decode_step/rewind/position/close`). When registered, `sp_decode_step` routes the single-token call to it. The engine's CUDA `gemma4_kv_decode_logits` registers here so the daemon drives the 12B token-by-token. Append-only — no frozen surface renumbered. |

Other frozen surfaces: `exact_islands.h` (the 4 islands; gate `T_EXACT_ISLANDS`), `sp_status.h`,
`sp_model.h` (the `.sp-model` mmap format), `ntt_crt.h` / `poly_ring.h` (dual-prime NTT),
`frobenius_lift.h`, `arena.h` (OK_Q4B), `spinor_block.h` (the frozen 63-byte VHT2 record), `kste.h`,
`arm.h` (the two-ring recall router), `ok_int.h` (`O_K` arithmetic), `ring3.h` (Ring-3 VSA).
The authoritative call-surface reference is lattice `papers/PPT-LAT-L1-ABI-v0.md`.

### Frozen primes & constants (`include/sp/ntt_crt.h`)

| Constant | Value |
|---|---:|
| `SP_NTT_Q1` | `1073738753` |
| `SP_NTT_Q2` | `1073732609` |
| `SP_NTT_M` (= q1·q2) | `1152908312643096577` |
| Garner `Q1_INV_MOD_Q2` | `894602413` |
| Admissible `N` (direct NTT) | `{128, 256, 512}` |

`M ≈ 2^60` fits `uint64_t` → **no `__int128` in the NTT path** (long-context NTT is tiled `N=512`).
The `__int128` in the tree is confined to `core/exact_islands/exact_islands.c` — see the build note.

## 4. Primitive inventory (`core/` + `include/sp/`)

The scalar reference forward here is the **bit-exact correctness anchor** — a discrete `Z_q` forward
proven **argmax bit-exact to llama.cpp** across multiple architecture families.

| Primitive | Module | Notes |
|---|---|---|
| **O_K arithmetic** — `O_K = Z[(1+√−163)/2]` | `core/ok_arith/` (`ok_int.h`) | the exact-integer ring; the algebraic container the whole substrate rides in. |
| **Dual-prime CRT-NTT** — negacyclic, R_q attention | `core/ntt_crt/` + `core/poly_ring/` | frozen primes q1=1073738753, q2=1073732609, M=1152908312643096577; byte-exact, production path uses **no 128-bit type**. |
| **Frobenius π^k codec** — the OK_Q4B packed-weight codec | `core/frobenius/` + `core/arena/` | per-32-block-scaled Frobenius-lift codec; carries the gemma-4-12B artifact (arena layout v2). |
| **The 4 exact nonlinear islands** | `core/exact_islands/` (`exact_islands.h`) | RMSNorm / softmax / GELU / **RoPE via fixed-point CORDIC (no libm)** as exact-integer references; gate `T_EXACT_ISLANDS`. The math-core anchor for the engine's `SP_BYTEEXACT` byte-exact forward. |
| **ARM two-ring Spinor KV** — the episodic store | `core/arm/` (`arm.h`) + `core/vht2/` (the frozen 63-byte Spinor block) | ±1 Rademacher recall router + bit-packed signature scan, recall-hit telemetry, cold-evict mask, abstract Ring-2 backend ABI. Gates `T_ARM`, `T_ARM_SIG`, `T_ARM_GEOM`, `T_ARM_GENKV`. |
| **C2 256-bit signature router** | (content-hash signatures; period-6 rebase to gemma4 global layers) | the content-addressed cue used to route Ring-3 superpositions; Hamming-verify accept/reject. |
| **Ring-3 VSA** — bind/unbind, D=1024 | `core/ring3/` (`ring3.h`) | native-C VSA bind/unbind + NIGHTSHIFT consolidation; runs on native `sp_pr_mul`/`ntt` bit-identically. Gate `T_RING3_NATIVE`. |
| **The single decode** | `core/forward/decode.c` | the **only** decode in the tree — two-ring KV + NTT-KV fusion + the `SP_REPLAY` episode-replay seam (off-path bit-exact). `core/forward/gemma4.c` is forward-only. |
| **KSTE fingerprint · dominance sieve** | `core/kste/`, `core/kste_md/`, `core/sieve/`, `core/dominance/` | KSTE content fingerprint; the dominance sieve. |

## 5. Headline status (verify against lattice, don't assume)

- **`[PROVEN]` — the forward + substrate.** Argmax bit-exact on multiple arch families; the reducing
  `.sp-model` codec is output-lossless and smaller than source. Citable engine envelope lives in
  lattice `LEDGER.md` / `VERIFIED-SCOREBOARD.md`.
- **`[PROVEN]` — the byte-exact 12B forward** (gate `G-BYTEEXACT-FORWARD-12B`). Byte-exact =
  **exact arithmetic / cross-machine determinism / AUDITABILITY — NOT compression.** The new piece in
  this core is `core/exact_islands/` + the L1 §6b kvdecode verb; on the engine side these carry the
  default-off `SP_BYTEEXACT` device-integer forward (OFF = PPL 4.6665 byte-identical null floor / ON =
  4.6569 parity / run-to-run bit-identical). The one remaining external item is a bit-identical logit
  check across two **physical** GPUs.
- **`[gated-GREEN]` — XBAR memory unified onto this repo's exact-integer O_K substrate.** Ring-3 VSA
  bind **256/256 bit-identical** to native `sp_pr_mul`/`ntt` and reduction-order-immune
  (`G-R3-BIND-on-O_K`); a Frobenius π^k integer Ring-2 episode store (`G-R2-FROB`); the full
  real-episode organism loop (`G-XBAR-ORGANISM-FULL`).
- **Content-addressing → a primary forward axis.** The byte-exact SHA content-address insight now
  feeds a **primary forward axis: the SP-SWARM / DHT memory mesh** — a content-addressed memory
  distributed across nodes, keyed by byte-exact SHA addresses (byte-exact arithmetic is what makes a
  content address *portable and verifiable* across machines). The design lives in the lattice repo;
  this core supplies the exact-integer substrate and the content-hash primitives it is built on.

## 6. The boundary thesis — honest negatives (do not re-litigate)

`O_K` wins on **exact arithmetic (the container)**; every structure-on-*content* compression lever
was measured-inert and is kept as an `[HONEST-NEGATIVE]`:

- Split-prime `O_K` Dirichlet carriers — operationally inert.
- Möbius-on-M — sheds memories 1.000 → 0.969 @ N=32.
- Entropy-coding the Frobenius codes — 1.02× dead weight.
- T2 / T4-Möbius on the real 12B embedding / weights — recon cos ≈ random.
- The KSTE / Friedman magnitude-depth router on real last-token global-Q — collapses to 1.00× (input
  is the wall).
- The compression reading of "byte-exact" — convicted redundant against the existing per-32-block
  OK_Q4B at gold PPL 4.6665. Byte-exact is the **auditability** axis only.

The full ledger of levers + constants is lattice `papers/PPT-LAT-FINDINGS-LEDGER.md`.

## 7. Build

The math core builds under the engine's clean-build chain; the authoritative toolchain doc is lattice
**`papers/BUILD-ENV-TOOLCHAIN.md`** (verified GREEN 2026-06-28).

```bash
cmake -B build -G Ninja && cmake --build build
ctest --test-dir build --output-on-failure
```

**Toolchain note (binding).** The CPU / math-core build uses **`clang-cl`**, **NOT `cl.exe`**.
`core/exact_islands/exact_islands.c` uses `__int128` (plus C11 `<stdatomic.h>` and other Clang/GCC
constructs); `cl.exe` rejects `__int128` (`error C4235`). `clang-cl` supports them **and** emits
**MSVC-ABI** `.lib` archives that the `x86_64-pc-windows-msvc` cargo daemon links. A prior session
flipped the env to `cl.exe` and that single drift made clean-build RED — **do not revert it.** Full
toolchain map + every gotcha (gate `G-CLEAN-BUILD`): lattice `papers/BUILD-ENV-TOOLCHAIN.md`.

**Sync discipline (binding):** this repo is also carried as the engine's `lib/shannon-prime-system`
submodule, so the two checkouts can diverge. `git fetch` + check
`git rev-list --count HEAD..origin/main` **before** any build or commit; every standalone commit is
followed by a submodule bump in the engine.

## 8. Navigation

| You want… | Read |
|---|---|
| **What is built / what is open (receipts-checked)** | lattice `papers/VERIFIED-SCOREBOARD.md` |
| **The whole system, current + complete** | lattice `papers/PPT-LAT-KEYSTONE.md` |
| The proven record / the math (T1–T8, O_K, CRT primes) | lattice `papers/PPT-LAT-Theory.md` |
| The L1 call-surface reference | lattice `papers/PPT-LAT-L1-ABI-v0.md` |
| The ledger of levers + honest negatives | lattice `papers/PPT-LAT-FINDINGS-LEDGER.md` |
| Agent entry + read-order + pre-flight | `AGENTS.md` |
| This repo's specifics + non-negotiables | `CLAUDE.md` |
| Module conventions (before adding a module) | `CONVENTIONS.md` |
| Public results + reproduce commands | [Position Is Arithmetic](https://github.com/nihilistau/Position_Is_Arithmetic) `LEDGER.md` |

## License

MIT. See `LICENSE`. GitHub: [nihilistau/shannon-prime-system](https://github.com/nihilistau/shannon-prime-system).
