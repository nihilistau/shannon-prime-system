cmake_minimum_required(VERSION 3.20)
# When built directly (cmake -S core/arm) we are the top-level project.
# When add_subdirectory()'d by the root, the parent already did project().
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_arm C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

# ARM (Algebraic Resonance Memory) — the two-ring KV memory core promoted from
# the engine's proven C2.1 production implementation: frozen ±1 recall router +
# quickselect, Ring-1 sink+window slot map, Ring-2 abstract backend + portable
# stdio reference store. Self-contained (no module dependencies); the forward
# consumes it, platform Ring-2 stores register through the backend interface.
sp_add_module(arm
    SOURCES   arm.c arm_scan.c
    TEST      arm_test.c
    TEST_NAME T_ARM)

# T_ARM_GEOM — per-layer-class geometry router gate (G-P3-GEOM substrate,
# CONTRACT-XBAR-C1-lite §3b): heterogeneous {nkv, head_dim} descriptors in the
# gemma4 shape (period-6 GLOBAL nkv=1/hd=512 vs SWA nkv=2/hd=256), brute-force
# f32 selection oracle per layer, uniform-null bit-parity vs the legacy path,
# and the R-stride/sizing corruption guard. Same SP_SYSTEM_BUILD_TESTS gate as
# sp_add_module (see cmake/sp_module.cmake).
if(NOT DEFINED SP_SYSTEM_BUILD_TESTS OR SP_SYSTEM_BUILD_TESTS)
    add_executable(test_arm_geom arm_geom_test.c)
    target_link_libraries(test_arm_geom PRIVATE sp_arm)
    target_include_directories(test_arm_geom PRIVATE ${SP_INCLUDE_DIR})
    target_compile_options(test_arm_geom PRIVATE ${SP_C_FLAGS} ${SP_UBSAN_C_OPT})
    if(SP_UBSAN_L_OPT)
        target_link_options(test_arm_geom PRIVATE ${SP_UBSAN_L_OPT})
    endif()
    add_test(NAME T_ARM_GEOM COMMAND test_arm_geom)
endif()
