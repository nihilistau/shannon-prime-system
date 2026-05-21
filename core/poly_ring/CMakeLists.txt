cmake_minimum_required(VERSION 3.20)
# When built directly (cmake -S core/poly_ring) we are the top-level project.
# When add_subdirectory()'d by the root, the parent already did project().
# NOTE: this module links sp_ntt_crt (Phase 1B), which only exists in the
# whole-repo build. Build/test via the ROOT build, not standalone:
#     cmake -B build -G Ninja -DCMAKE_C_COMPILER=gcc
#     cmake --build build
#     ctest --test-dir build -R T_PR --output-on-failure
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_poly_ring C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

sp_add_module(poly_ring
    SOURCES   poly_ring.c
    TEST      poly_ring_test.c
    TEST_NAME T_PR)

# poly_ring is built on the Phase-1B CRT-NTT kernel. The root build orders
# ntt_crt before poly_ring, so target sp_ntt_crt already exists here. PUBLIC so
# the sp/ntt_crt.h include dir propagates to test_poly_ring and any future
# consumer of sp_poly_ring.
target_link_libraries(sp_poly_ring PUBLIC sp_ntt_crt)
