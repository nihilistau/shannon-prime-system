cmake_minimum_required(VERSION 3.20)
# Standalone (cmake -S core/io_hash) vs add_subdirectory()'d by the root.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_io_hash C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

sp_add_module(io_hash
    SOURCES   sp_hash.c
    TEST      sp_hash_test.c
    TEST_NAME T_IO_HASH)
