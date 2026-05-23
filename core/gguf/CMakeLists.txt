cmake_minimum_required(VERSION 3.20)
# Standalone (cmake -S core/gguf) vs add_subdirectory()'d by the root.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_gguf C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

# GGUF v3 on-disk model parser: mmap + header/metadata/tensor-table parse into a
# backend-agnostic descriptor. Self-contained (stdlib + platform mmap only); the
# weight binding that consumes it (qwen3_model / qwen3_load) stays engine-side until
# the model-representation increment above this.
sp_add_module(gguf
    SOURCES   gguf.c
    TEST      gguf_test.c
    TEST_NAME T_GGUF)
