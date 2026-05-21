cmake_minimum_required(VERSION 3.20)
# When built directly (cmake -S core/frobenius) we are the top-level project.
# When add_subdirectory()'d by the root, the parent already did project().
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_frobenius C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

sp_add_module(frobenius
    SOURCES   frobenius_lift.c
    TEST      frobenius_test.c
    TEST_NAME T_FRO)
