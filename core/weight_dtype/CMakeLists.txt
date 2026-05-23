cmake_minimum_required(VERSION 3.20)
# Standalone (cmake -S core/weight_dtype) vs add_subdirectory()'d by the root.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_weight_dtype C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

sp_add_module(weight_dtype
    SOURCES   weight_dtype.c
    TEST      weight_dtype_test.c
    TEST_NAME T_WDTYPE)
