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
