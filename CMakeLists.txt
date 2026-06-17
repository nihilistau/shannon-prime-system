cmake_minimum_required(VERSION 3.20)
project(shannon_prime_system
    VERSION 0.0.1
    DESCRIPTION "Math core for shannon-prime-lattice"
    LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(SP_SYSTEM_BUILD_TESTS "Build unit tests" ON)

# ──────────────────────────────────────────────────────────────────────
# Phase 1 layout (see ../shannon-prime-lattice/papers/PPT-LAT-Roadmap.md §7
# and CONVENTIONS.md):
#   core/<module>/   — one Phase-1 primitive each, self-contained, with its
#                      own CMakeLists.txt, sources, and test.
#   include/sp/      — public headers + sp_test.h (shared test harness).
#   cmake/           — sp_module.cmake helper.
#
# Each module is referenced below but guarded by EXISTS, so the tree
# configures green even while a module is still empty. This is what lets
# parallel Phase-1 agents drop in their own core/<module>/CMakeLists.txt
# without ever touching this file.
# ──────────────────────────────────────────────────────────────────────

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/sp_module.cmake)

if(SP_SYSTEM_BUILD_TESTS)
    enable_testing()
endif()

set(SP_MODULES
    forward_kernels # L1 -- portable scalar reference forward kernels (dot/rmsnorm/rope/attn)
    exact_islands   # BYTE-EXACT -- exact-integer fp32-island replacements (rmsnorm/softmax/gelu); reduction-order-immune
    io_hash         # L1 -- .sp-model hash primitives (CRC-32 / SHA-256 / XXH64)
    io_format       # L1 -- .sp-model mmap loader + accessors + L1 status-string impl
    weight_dtype    # L1 -- GGUF weight-dtype dequant (F16<->F32, F32/F16/Q8_0)
    gguf            # L1 -- GGUF v3 on-disk model parser (mmap -> backend-agnostic descriptor)
    ok_arith        # 1A — O_K arithmetic over Q(sqrt-163)
    ntt_crt         # 1B — dual-prime CRT-NTT
    poly_ring       # 1C — R_q polynomial-ring attention
    vht2            # 1D — VHT2 + Mobius reorder + 63-byte Spinor block
    frobenius       # 1E — Frobenius lift for Q8 weight storage
    kste            # 1F — KSTE encoder + Tier-0/Tier-1 dominance
    arena           # L1 -- packed-weight arena (named sp_frob_packed_tensor set); listed after its 1E frobenius codec dependency
    model           # L1 -- Qwen3/Gemma3 model representation + GGUF load/free/release lifecycle (on arena + gguf)
    forward_dispatch # L1 -- model-coupled weight-access kernels (matmul/embed/norm-accessor) on forward_kernels + arena
    arm             # C2.1 -- ARM two-ring KV memory: ±1 recall router + Ring-1 sink/window map + Ring-2 abstract backend (forward depends on it; listed before forward)
    forward         # L1 -- Qwen3 forward orchestration: prefill + persistent-KV decode (on forward_dispatch + the overlay codecs poly_ring/kste/vht2)
    session         # L1 -- sp_session ABI (create/prefill/decode) + sp_model_to_qwen3 bridge (on io_format + forward + model + arena + frobenius)
    sp_channel      # TS.MAP -- GF(2) memory-channel oracle (hedge-read timing; perf overlay only, never a correctness dependency)
    dominance       # later — dominance order / signature dominance
    sieve)          # later — Friedman sieve

foreach(_m IN LISTS SP_MODULES)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/core/${_m}/CMakeLists.txt)
        add_subdirectory(core/${_m})
        message(STATUS "shannon-prime-system: enabled module ${_m}")
    endif()
endforeach()

message(STATUS "shannon-prime-system ${PROJECT_VERSION} configured (Phase 1)")
