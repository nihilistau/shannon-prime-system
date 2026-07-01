# AGENTS.md — shannon-prime-system (the math core)

Agent entry point + navigation for **the clean, engine-free math core**. If you are an agent
landing in this repo, read this first, then follow the read-order below before touching anything.

---

## 0. Read order (do this before any work)

1. **[`README.md`](README.md)** — what this repo is, the frozen L1 ABI, the `core/` modules + their tiers, the substrate diagram.
2. **[`CLAUDE.md`](CLAUDE.md)** — this repo's specifics + the non-negotiables (short form).
3. **Lattice canon** (`D:\F\shannon-prime-repos\shannon-prime-lattice\`):
   - `prompt.md` — the canonical session bootstrap (project, machine, doc map, operator).
   - `papers/VERIFIED-SCOREBOARD.md` — **the status source of truth (receipts-checked): what is built, what is open.** Trust this over any "current edge" prose.
   - `papers/PPT-LAT-KEYSTONE.md` — the canonical, current, complete system map (the five repos, architecture, memory model, gate index).
   - `papers/PPT-LAT-Theory.md` — the math (the PPT substitution, `O_K`/`Q(√−163)`, CRT primes, the frozen Spinor + KSTE formats, theorems T1–T8). **Read it — skipping it has caused real drift.**
   - `papers/PPT-LAT-L1-ABI-v0.md` — the L1 call-surface reference. `papers/PPT-LAT-FINDINGS-LEDGER.md` — the ledger of levers + honest negatives.
4. **[`HISTORY.md`](HISTORY.md)** — the hashed Tier-0 LUT of this repo's milestones (the git short-hash IS the content address; dig with `git show <hash>`). Regenerate with `python <lattice>/tools/okf_history.py gen --repo . --out HISTORY.md`.
5. The active contract for whatever you're touching (lattice `papers/CONTRACT-*.md`).

---

## 1. The MEM-OKF lookup pre-flight (BINDING — before building anything)

This project has rebuilt the same subsystems 20+ times. A new file for a capability that already
exists is a **defect**. Before building ANY subsystem:

```bash
python D:\F\shannon-prime-repos\shannon-prime-lattice\tools\okf_mem.py lookup \
    --root D:\F\shannon-prime-repos\shannon-prime-lattice\memory-okf <keyword>
```

…then `grep` the tree. The content-addressed LUT→summary→full store + the rule live in lattice
`papers/MEMORY-OKF-PROFILE.md` (`prompt.md` §0/§8). At session end, bank durable
"X already exists — don't rebuild" facts via `okf_mem.py add`.

---

## 2. The frozen-ABI rule (do not violate)

`include/sp/sp_l1.h` is **FROZEN** (tag `lat-phase2-contract-frozen`). Growth is **append-only**:

- **Never renumber** an `sp_status` value, a `§`, or a dispatch-table row. Existing consumers depend on the numbers.
- **Append, never edit:** new arch fields go in the reserved 256-byte `sp_arch_info` tail (loader memcpies `min(...)` so old files leave new fields zero). New backend knobs register in this header **first** (the §6c/§6d/§6e pattern: registered here, implemented engine-side), then grow the §6b dispatch struct append-only if they must become generic.
- **An unregistered session must be byte-compatible** with every existing consumer (calloc-zero default = null floor). Verify it.

Surfaces frozen for shipped sub-phases: the 63-byte Spinor block (`spinor_block.h`, `T_VHT_3/5/6`),
the KSTE layout (`kste.h`), the frozen NTT primes (`ntt_crt.h`).

---

## 3. Non-negotiables (receipts-first, bit-exact-when-off)

- **Receipts-first.** No number without a reproducing command + a gate/commit. Every headline figure carries its scope.
- **Bit-exact-when-off.** Every `SP_*` overlay is a strict no-op by default (null floor). **Verify it** — `OFF == baseline byte-identical` is the floor, not the headline.
- **No silent gate revision.** If a gate would need to move, surface it upstream — don't quietly re-baseline.
- **Check code + commits + `git fetch` before trusting memory or a summary.** This repo is also the engine's `lib/shannon-prime-system` submodule, so the standalone copy can sit **behind** `origin/main`: `git fetch` + `git rev-list --count HEAD..origin/main` before building/committing; rebase if behind; bump the submodule in the engine after each standalone commit.
- **Verify Gemini's claims** against the actual tree/paper before adopting.
- **Honest negatives stay attached** (see README §7). Don't re-litigate a measured negative.
- **Anti-overclaim:** use the exact tier vocabulary (`[PROVEN]` / `[WIRED]` / `[gated-GREEN]` / `[DESIGN]` / `[HONEST-NEGATIVE]`). Default-off ≠ live.

---

## 4. Build quick-ref

The CPU / math-core build uses **`clang-cl`** (MSVC-ABI), **NOT `cl.exe`** — `core/exact_islands/exact_islands.c`
uses `__int128` (+ C11 `<stdatomic.h>`), which `cl.exe` rejects (`C4235`); `clang-cl` supports them and
emits MSVC-ABI `.lib` archives the `x86_64-pc-windows-msvc` cargo daemon links. **Do not revert the env
to `cl.exe`** (that drift = clean-build RED, a recurring restart cause).

```bash
cmake -B build -G Ninja && cmake --build build
ctest --test-dir build --output-on-failure
```

A single gate, e.g. the two-ring harness: `cd build && ninja test_arm_genkv && core/session/test_arm_genkv.exe`.
Authoritative from-clean chain + every gotcha (gate `G-CLEAN-BUILD`): lattice
`papers/BUILD-ENV-TOOLCHAIN.md` (verified GREEN 2026-06-28). Conventions: `CONVENTIONS.md`.

---

## 5. OKF knowledge discipline

Any knowledge `.md` you create/touch carries the SP-OKF frontmatter
(`type` + `title/description/tags/timestamp/resource` + `sp_status/sp_gate/sp_commit/sp_repro`).
New `type`s register in lattice `papers/SP-OKF-PROFILE.md` §2 first. Validate a touched bundle before
commit: `python <lattice>/tools/okf_validate.py <bundle-dir>` (gate `G-OKF-CONFORM`).
