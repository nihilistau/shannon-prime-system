cmake_minimum_required(VERSION 3.20)
# When built directly (cmake -S core/ntt_crt) we are the top-level project.
# When add_subdirectory()'d by the root, the parent already did project().
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_ntt_crt C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

# T_NTT_5 (enforcing gate): the production path must contain no 128-bit type.
# This fails configure if "__int128" appears anywhere in ntt_crt.c. The
# test-only parity oracle ntt_ref_int128.c is exempt (it is never compiled into
# the production library sp_ntt_crt).
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/ntt_crt.c _sp_ntt_crt_src)
if(_sp_ntt_crt_src MATCHES "__int128")
    message(FATAL_ERROR
        "[T_NTT_5] __int128 found in production source ntt_crt.c — the "
        "dual-prime kernel must use Barrett uint64 arithmetic only. Move any "
        "128-bit code into the test-only oracle ntt_ref_int128.c.")
endif()

# The __int128 parity oracle is gcc/clang-only (MSVC has no __int128). On MSVC
# it is left out of the test exe; T_NTT_2 (live oracle) self-skips there and
# T_NTT_3 gates MSVC against the gcc-pre-generated ntt_ref_vectors.h fixture.
set(_ntt_test_extra "")
if(NOT MSVC)
    list(APPEND _ntt_test_extra ntt_ref_int128.c)
endif()

sp_add_module(ntt_crt
    SOURCES      ntt_crt.c
    TEST         ntt_crt_test.c
    TEST_SOURCES ${_ntt_test_extra}
    TEST_NAME    T_NTT)
