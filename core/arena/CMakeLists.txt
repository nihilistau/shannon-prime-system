cmake_minimum_required(VERSION 3.20)
# Standalone (cmake -S core/arena) vs add_subdirectory()'d by the root.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_arena C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

# Packed-weight arena: the named collection of per-row Frobenius Q8/Q4 packed
# tensors (sp/arena.h). The builder consumes a loaded qwen3_model (sp/model.h),
# reads GGUF source rows (gguf accessors), dequants them (sp_dequant_row), and
# packs via the Frobenius codec. The .sp-model "adopt already-packed" path and the
# inline-lift accessors round out the ABI every backend reads at matmul time.
# DEPENDS: frobenius (the codec + byte layout), weight_dtype (source-row dequant),
# gguf (tensor-data accessor). The model representation header (sp/model.h) is
# declaration-only — pulled in by include, no library to link.
sp_add_module(arena
    SOURCES   arena.c
    TEST      arena_test.c
    TEST_NAME T_ARENA
    DEPENDS   sp_frobenius sp_weight_dtype sp_gguf)
