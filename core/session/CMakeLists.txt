cmake_minimum_required(VERSION 3.20)
# Standalone (cmake -S core/session) vs add_subdirectory()'d by the root. NOTE: this
# module has a wide dependency fan-in (forward + model + arena + frobenius + io_format),
# so it is intended to be built from the ROOT tree (which mirrors CI); a bare standalone
# configure may double-add a shared sibling. The root build is the supported path.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_session C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

# L1 session ABI (sp/sp_l1.h §1/§3): sp_session_create/destroy/position + sp_prefill_chunk
# (series 1) over the reconstructed runnable model. sp_model_bridge.c is sp_model_to_qwen3 —
# the const sp_model* -> qwen3_model reconstruction (the documented adapter path, NOT a veneer
# over a GGUF-loaded model). DEPENDS: io_format (sp_model accessors + sp_model_arch + sp_set_error),
# forward (qwen3_forward + transitively forward_dispatch/model/arena/frobenius), and explicitly
# model/arena/frobenius (direct callees of the bridge/session); io_hash for the test fixture.
sp_add_module(session
    SOURCES      sp_session.c sp_model_bridge.c
    TEST         session_test.c
    TEST_SOURCES qwen3_fixture.c gemma3_fixture.c
    TEST_NAME    T_SESSION
    DEPENDS      sp_io_format sp_forward sp_model sp_arena sp_frobenius sp_io_hash)
