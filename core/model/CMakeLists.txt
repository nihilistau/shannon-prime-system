cmake_minimum_required(VERSION 3.20)
# Standalone (cmake -S core/model) vs add_subdirectory()'d by the root.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_model C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

# Qwen3/Gemma3 model representation lifecycle (sp/model.h): open a GGUF, read the
# architecture config, bind every weight tensor, optionally build the packed arena,
# and the Phase-1b source-release + free paths. The forward pass and generation
# (also declared in sp/model.h) are a later increment and are not implemented here;
# this library defines only the load/free/release symbols. DEPENDS: arena (packed
# weights + the .sp-model adopt path), gguf (parser + accessors), weight_dtype (the
# release-path norm dequant) — gguf/weight_dtype also reach the binding transitively
# through arena's PUBLIC link, listed for clarity.
sp_add_module(model
    SOURCES   model.c
    TEST      model_test.c
    TEST_NAME T_MODEL
    DEPENDS   sp_arena sp_gguf sp_weight_dtype)
