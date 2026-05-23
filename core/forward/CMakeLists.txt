cmake_minimum_required(VERSION 3.20)
# Standalone (cmake -S core/forward) vs add_subdirectory()'d by the root.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_forward C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

# Qwen3 L1 forward orchestration (definitions of qwen3_forward / qwen3_forward_ex /
# qwen3_generate_kv, declared in sp/model.h): the 13-step transformer prefill plus the
# persistent-KV O(n) decode, composing the migrated kernel stack. DEPENDS its building
# blocks directly: forward_dispatch (sp_matmul / sp_embed_row / sp_as_f32, which also
# carries the model representation header + arena/gguf/weight_dtype/frobenius), forward_kernels
# (the elementary primitives), and the overlay codecs poly_ring (NTT-attention sp_pr_*),
# kste (KSTE KV sp_kste_encode), and vht2 (Spinor KV sp_spinor_*).
sp_add_module(forward
    SOURCES   forward.c
    TEST      forward_test.c
    TEST_NAME T_FORWARD
    DEPENDS   sp_forward_dispatch sp_forward_kernels sp_poly_ring sp_kste sp_vht2)
