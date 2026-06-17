cmake_minimum_required(VERSION 3.20)
# Standalone (cmake -S core/exact_islands) vs add_subdirectory()'d by the root.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_exact_islands C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

sp_add_module(exact_islands
    SOURCES   exact_islands.c
    TEST      exact_islands_test.c
    TEST_NAME T_EXACT_ISLANDS)
