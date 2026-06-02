cmake_minimum_required(VERSION 3.20)
# Standalone (cmake -S core/forward) vs add_subdirectory()'d by the root.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_forward C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

# L1 reference forward orchestration (definitions declared in sp/model.h): the Qwen3 path
# (forward.c — the 13-step prefill qwen3_forward/qwen3_forward_ex plus the persistent-KV O(n)
# decode qwen3_generate_kv), the Gemma3 path (gemma3.c — gemma3_forward, the same prefill shape
# with the Gemma deltas), and the greedy O(n^2) generate (generate.c — qwen3_generate, a thin
# re-prefill loop over qwen3_forward). All compose the migrated kernel stack. DEPENDS the building
# blocks: forward_dispatch (sp_matmul / sp_embed_row / sp_as_f32, which also carries the model
# representation header + arena/gguf/weight_dtype/frobenius) and forward_kernels (the elementary
# primitives); the overlay codecs poly_ring (NTT-attention sp_pr_*), kste (KSTE KV sp_kste_encode),
# and vht2 (Spinor KV sp_spinor_*) are needed only by the Qwen3 forward_ex/decode overlay paths.
sp_add_module(forward
    SOURCES   forward.c gemma3.c gemma4.c qwen25.c qwen36.c generate.c
    TEST      forward_test.c
    TEST_NAME T_FORWARD
    DEPENDS   sp_forward_dispatch sp_forward_kernels sp_poly_ring sp_kste sp_vht2)

# M_QWEN36 — Qwen3.6-35B-A3B (qwen35moe) forward top-1 correctness gate (GGUF-direct).
# SLOW / model-gated: skips (exit 0) when the 19.7 GB model is absent. Asserts the
# greedy top-1 sequence is bit-exact to the llama.cpp oracle (5444 8 198).
if(NOT DEFINED SP_SYSTEM_BUILD_TESTS OR SP_SYSTEM_BUILD_TESTS)
    add_executable(test_qwen36_gate qwen36_gate.c)
    target_link_libraries(test_qwen36_gate PRIVATE sp_forward sp_model)
    target_include_directories(test_qwen36_gate PRIVATE ${SP_INCLUDE_DIR})
    target_compile_options(test_qwen36_gate PRIVATE ${SP_C_FLAGS})
    add_test(NAME M_QWEN36 COMMAND test_qwen36_gate)
    set_tests_properties(M_QWEN36 PROPERTIES LABELS "SLOW" TIMEOUT 3600)
endif()
