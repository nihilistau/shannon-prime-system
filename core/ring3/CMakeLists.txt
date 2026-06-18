cmake_minimum_required(VERSION 3.20)
# Standalone (cmake -S core/ring3) vs add_subdirectory()'d by the root.
# This module is the native-C Ring-3 VSA + NIGHTSHIFT consolidation, built on
# the exact-integer negacyclic CRT-NTT (sp_poly_ring -> sp_ntt_crt). sp_add_module
# pulls the sibling deps in for a standalone build and just links them in the root.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_ring3 C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

sp_add_module(ring3
    SOURCES   ring3.c
    TEST      ring3_gate.c
    TEST_NAME T_RING3_NATIVE
    DEPENDS   sp_poly_ring)
