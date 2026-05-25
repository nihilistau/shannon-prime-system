cmake_minimum_required(VERSION 3.20)
# Standalone (cmake -S core/io_format) vs add_subdirectory()'d by the root.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_io_format C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

# The mmap .sp-model loader + accessors and the L1 status-string impl (sp_error).
# The loader verifies header CRC-32 and the tokenizer SHA-256 via the hash module.
# sp_model_to_qwen3 (the runtime-model adapter) is NOT here -- it stays engine-side
# until the model representation migrates; its declaration in sp/sp_model.h is inert.
sp_add_module(io_format
    SOURCES      sp_model_load.c sp_model_arch.c sp_error.c
    TEST         io_format_test.c
    TEST_SOURCES sp_model_fixture.c
    TEST_NAME    T_IO_FORMAT
    DEPENDS      sp_io_hash)
