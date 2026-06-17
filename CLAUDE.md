# CLAUDE.md — shannon-prime-system (the math core)

**This is Shannon-Prime's math core. The canonical session bootstrap is `D:\F\shannon-prime-repos\shannon-prime-lattice\prompt.md` — read it first** (project, current state, methodology, machine, doc map, operator). This file is the short version + this repo's specifics.

**Repo role:** the clean math core, no engine deps. Key surfaces:
- `core/arm/` — **ARM, the two-ring KV memory** (`sp_arm_*`: ±1 Rademacher recall router, quickselect select, Ring-1 slot map, the abstract Ring-2 backend + stdio reference, recall-hit telemetry `sp_arm_hits_*`, cold-evict mask `sp_arm_evict_*`).
- `core/forward/decode.c` — **the ONLY decode in the tree** (`generate_kv_impl`, shared by `qwen3_generate_kv` / `qwen3_ppl_decode`): the two-ring, NTT-KV fusion, and the `SP_REPLAY` episode-replay seam (C1L.0b). `core/forward/gemma4.c` is forward-only (no decode/ring — that's P3).
- `core/vht2/` (63-byte Spinor block, frozen, 0xA5 sentinel) · `core/ntt_crt/` + `core/poly_ring/` (dual-prime NTT) · `core/frobenius/` + `core/arena/` (Frobenius lift, OK_Q4/**OK_Q4B** packed weights) · `core/kste/`, `core/sieve/`, `core/dominance/`.
- The gate harness for the two-ring + replay + cold-evict lives in `core/session/arm_genkv_gate.c` (`T_ARM_GENKV`, `T_GENKV_REPLAY_NULL`, `T_GENKV_RECALL_HITS`, `T_GENKV_COLD_EVICT`).

**Status (2026-06-17; engine-side, no math-core change):** XBAR P3 is CLOSED end-to-end (P3.0→P3.4) — this repo's `SP_REPLAY` replay seam + `core/xbar/xbar_episode.c` owner-map manifest are now proven on the Exec/gemma4-CUDA path as **P3.3** (`G-P3-SHARED` 3-leg PASS on 12B + E2B) + **P3.4** recall-quality (`G-P3-PPL` +1.38% < 2% gate). The GNA "EAR" line is CLOSED on physical silicon (GNA_HW 0.877 == emu == FP32). Both engine-side; recorded for cross-repo state. Detail: lattice `papers/CONTRACT-XBAR-P3` + `CONTRACT-KAIROS-K0-K1` §7.4–§7.6.

**Build:** canonical CPU backend = **MinGW gcc 15.2, `build/` dir, ninja** (MSVC cannot build the CPU tree — Tier-3-deferred). Build a gate: `cd build && ninja test_arm_genkv && core/session/test_arm_genkv.exe`. Standalone gcc needs `-D_POSIX_C_SOURCE=200809L -D_FILE_OFFSET_BITS=64` and to link `arm.c`+`arm_scan.c`.

**Git (binding lesson):** this repo is ALSO carried as a submodule (`lib/shannon-prime-system`) inside the engine, so the standalone copy can sit **behind** `origin/main`. **`git fetch` + check `git rev-list --count HEAD..origin/main` BEFORE building/committing**; rebase if behind.

**Non-negotiables:** receipts-first (no number without a command); bit-exact-when-off (the `SP_*` overlays are strict no-ops by default — verify it); no silent gate revision (surface upstream); check the code + commits + `git fetch` before trusting memory; verify Gemini's claims against the tree; drive by default. Full detail in lattice `prompt.md`.

**Environment & credentials (2026-06-11):** compute lanes, shells/traps, storage law → lattice `ENVIRONMENT.md`. Current state/queue → lattice `SESSION-HANDOFF.md`. Secrets → `archive\notes_and_stuff\creds\claude-credentials.txt` (outside all repos; reference paths, never values).
