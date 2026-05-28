cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_sieve C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

sp_add_module(sieve
    SOURCES   sp_sieve.c
    TEST      sieve_test.c
    TEST_NAME T_SIEVE
    DEPENDS   sp_kste sp_io_hash)
