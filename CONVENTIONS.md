---
type: convention
title: shannon-prime-system — module conventions
description: Read this before adding a Phase-1 module.
tags: [convention]
timestamp: 2026-06-06T08:11:43Z
resource: ./CONVENTIONS.md
sp_status: ACTIVE
sp_gate: none
sp_commit: TBD
sp_repro: none
---

# shannon-prime-system — module conventions

Read this before adding a Phase-1 module. Every `core/<module>/` looks the
same so parallel agents never collide and the integrated build stays trivial.

## Anti-contamination (binding)

Do **not** read, copy, or vendor code from
`D:\F\shannon-prime-repos\shannon-prime\` or `…\shannon-prime-engine\`. Math
papers under `…\papers\PPT-ARM\` are conceptual reference only — read theorem
statements, re-derive the implementation from scratch. See
`../shannon-prime-lattice/papers/PPT-LAT-Roadmap.md` §3.1.

## Files a module owns (and nothing else)

A module agent for module `<m>` may create/edit ONLY:

- `core/<m>/*.c` — implementation + test sources
- `core/<m>/*.h` — module-PRIVATE headers (internal helpers shared between
  the module's own `.c` files; not part of the public surface)
- `core/<m>/CMakeLists.txt` — from the template below
- `include/sp/<header>.h` — that module's PUBLIC header(s) (basename matches
  your API/source name, not necessarily the module dir)
- `core/<m>/build*/` — its own build dir(s)

**Forbidden — do not touch (shared scaffold):**
`CMakeLists.txt` (root), `cmake/sp_module.cmake`, `include/sp/sp_test.h`,
`CONVENTIONS.md`, `README.md`, `.github/`, and any other module's
`core/<other>/` directory or `include/sp/<other>*.h`. The root CMake already
references every module behind an `EXISTS` guard — you do not register
yourself anywhere.

Module agents do **not** run git. Integration commits are done by the
coordinator after the module's tests are green.

## CMakeLists.txt template

Each `core/<m>/CMakeLists.txt` is both standalone-buildable and
add_subdirectory-able. Copy this, fill in sources:

```cmake
cmake_minimum_required(VERSION 3.20)
# When built directly (cmake -S core/<m>) we are the top-level project.
# When add_subdirectory()'d by the root, the parent already did project().
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_<m> C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

sp_add_module(<m>
    SOURCES   <impl>.c
    TEST      <impl>_test.c
    TEST_NAME T_<TAG>)        # e.g. T_OK, T_NTT — matches the roadmap prefix
```

`sp_add_module` builds static lib `sp_<m>`, test exe `test_<m>`, and registers
a ctest entry named `T_<TAG>`. The public header root `include/` is wired onto
both targets automatically — include your header as `#include "sp/<header>.h"`.
The header basename matches your API/source name, **not** necessarily the
module dir: module `ok_arith` ships `include/sp/ok_int.h` and `ok_int.c`.

For a parity oracle / fixture compiled only into the test (not the production
lib), pass it via `TEST_SOURCES`:

```cmake
sp_add_module(ntt_crt
    SOURCES      ntt_crt.c
    TEST         ntt_crt_test.c
    TEST_SOURCES ntt_ref_int128.c   # __int128 oracle; test-only, never in sp_ntt_crt
    TEST_NAME    T_NTT)
```

## Test harness

Use `include/sp/sp_test.h`. One executable per module; inside it, one
`static void T_<TAG>_<n>(void)` per roadmap-named test, driven by `SP_RUN`:

```c
#include "sp/sp_test.h"
#include "sp/<m>.h"

static void T_OK_1(void) { SP_CHECK(/* … */ 1, "UFD on 256 norms"); }
/* … T_OK_2 … T_OK_6 … */

int main(void) {
    SP_RUN(T_OK_1); SP_RUN(T_OK_2); SP_RUN(T_OK_3);
    SP_RUN(T_OK_4); SP_RUN(T_OK_5); SP_RUN(T_OK_6);
    return SP_DONE();
}
```

The exe prints `[T_OK_1] PASS/FAIL` per case and exits nonzero if any check
failed, so `ctest` names the failing contract item.

`sp_test.h` ships only `SP_CHECK(cond, label)` and `SP_CHECK_EQ_I64(got, want,
label)`. For domain types (struct equality, "equal up to a unit", float
tolerance, byte-for-byte), write a small local predicate and wrap it:
`SP_CHECK(sp_ok_eq(x, y), "round-trip")`. Do not edit `sp_test.h`.

## TDD (house style)

Write the test file first with the named `T_*` cases, build it, watch it fail,
then implement until green. See `superpowers:test-driven-development`.

## Build + test loop

From the repo root, with MinGW gcc + Ninja on PATH:

```bash
# standalone single module (fast iteration):
cmake -B core/<m>/build -S core/<m> -G Ninja -DCMAKE_C_COMPILER=gcc
cmake --build core/<m>/build
ctest --test-dir core/<m>/build --output-on-failure

# whole repo (coordinator integration run):
cmake -B build -G Ninja -DCMAKE_C_COMPILER=gcc
cmake --build build
ctest --test-dir build --output-on-failure
```

Run with `-DSP_UBSAN=ON` once before declaring a module done (gcc/clang only).

## GPU / accelerator benchmarking (binding when reporting tok/s or GB/s)

Added 2026-06-06 after a Stage-Beta speedup was reported as `12.65×` and turned
out to be three stacked measurement artifacts. Before ANY GPU throughput number
ships in a doc, gate, or commit message:

1. **Warm up.** Run the timed path ≥1–2× untimed first. CUDA lazy module load +
   cuBLAS JIT/heuristic selection happen on the FIRST kernel launch of a process
   (~13× first-decode penalty at 0.6B). Cold-vs-warm comparisons are invalid.
2. **Long window.** `n_gen ≥ 256` (or enough iters that wall-clock ≫ timer
   jitter). Sub-second windows swing wildly (32/88/92 tok/s for one path).
3. **Pin BOTH clocks.** `nvidia-smi -lgc <sm>,<sm>` locks ONLY the SM clock. A
   weight-reading GEMV is **memory-bound**, so its throughput tracks the GDDR6
   clock — which must be at full speed too (it auto-boosts under sustained load;
   consumer GeForce `-lmc` is flaky/deprecated). Reset with `-rgc` after.
4. **Confirm the kernel is on the binding bottleneck (Amdahl).** At 0.6B the
   decode is *overhead*-bound (launches + attention + 150k-vocab argmax), not
   weight-bandwidth-bound — a faster weight-GEMV ties f32 there. The win must be
   measured where it binds (large model, or an isolated matmul sweep).
5. **Trust within-run ratios over absolutes.** Clock/thermal drift moves the
   absolute even under a lock; the A-vs-B ratio measured back-to-back is stable.
6. **Isolated bench validates kernel MATH; production gate validates the
   DATA-STRUCTURE handoff.** Both are required — a uniform-synthetic bench passed
   while the production Q4 path returned 0/256 (a K-quant mixed-precision arena
   read the Q8 head as Q4). Gate against the real artifact in the real loop.

These are reusable across CUDA / Vulkan / Hexagon. Full rationale: engine
`tests/bench_gemv_int8.cu` + lattice `papers/SESSION-CLOSED-stage-beta-speed.md`.

## Platform-gate policy

This session closes the **Windows MinGW-gcc** tier (roadmap §3.7). Linux gcc
is verified by `.github/workflows/ci.yml` on push. Windows MSVC is a separate
follow-up wave — keep code MSVC-clean (no `__int128` outside test-only parity
oracles, no GNU-only extensions in production sources).
