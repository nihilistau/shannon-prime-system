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
# Phase 0 stub.  Source layout is:
#   core/        — KSTE, dominance/sieve, ARM, NTT primitives (Phase 1+)
#   include/sp/  — public headers
#   tests/       — unit + integration tests
#
# Each phase in shannon-prime-lattice/papers/PPT-LAT-Roadmap.md
# names the deliverable files; they get added to the build here.
# Phase 1 lands core/sp_kste.{c,h} and tests/test_sp_kste.cpp.
# ──────────────────────────────────────────────────────────────────────

# Sub-targets enabled per phase:
# add_subdirectory(core)
# add_subdirectory(include)

if(SP_SYSTEM_BUILD_TESTS)
    enable_testing()
    # add_subdirectory(tests)  # enabled in Phase 1
endif()

message(STATUS "shannon-prime-system ${PROJECT_VERSION} configured (Phase 0)")
