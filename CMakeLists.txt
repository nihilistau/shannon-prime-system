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
    ok_arith        # 1A — O_K arithmetic over Q(sqrt-163)
    ntt_crt         # 1B — dual-prime CRT-NTT
    poly_ring       # 1C — R_q polynomial-ring attention
    vht2            # 1D — VHT2 + Mobius reorder + 63-byte Spinor block
    frobenius       # 1E — Frobenius lift for Q8 weight storage
    kste            # 1F — KSTE encoder + Tier-0/Tier-1 dominance
    dominance       # later — dominance order / signature dominance
    sieve           # later — Friedman sieve
    arm)            # later — Algebraic Resonance Memory

foreach(_m IN LISTS SP_MODULES)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/core/${_m}/CMakeLists.txt)
        add_subdirectory(core/${_m})
        message(STATUS "shannon-prime-system: enabled module ${_m}")
    endif()
endforeach()

message(STATUS "shannon-prime-system ${PROJECT_VERSION} configured (Phase 1)")
