cmake_minimum_required(VERSION 3.20)
# When built directly (cmake -S core/poly_ring) we are the top-level project.
# When add_subdirectory()'d by the root, the parent already did project().
# This module links sp_ntt_crt (Phase 1B) via DEPENDS; sp_add_module pulls the
# sibling in for a standalone build and just links it in the root build.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_poly_ring C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

sp_add_module(poly_ring
    SOURCES   poly_ring.c poly_ring_bluestein.c resdot.c
    TEST      poly_ring_test.c
    TEST_NAME T_PR
    DEPENDS   sp_ntt_crt)
