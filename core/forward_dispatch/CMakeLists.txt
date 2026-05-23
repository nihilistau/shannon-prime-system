cmake_minimum_required(VERSION 3.20)
# Standalone (cmake -S core/forward_dispatch) vs add_subdirectory()'d by the root.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_forward_dispatch C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

# Model-coupled weight-access kernels (sp/forward_dispatch.h): the matmul that reads a
# model's weight tensor (arena inline-lift or GGUF dequant-on-demand), the embedding
# lookup, and the norm accessor. Sits on forward_kernels (the scalar reference dot) and
# arena (the packed-weight accessors). arena's PUBLIC link transitively supplies the
# frobenius / weight_dtype / gguf symbols this layer also calls (sp_frob_*, sp_dequant_row,
# sp_f16<->f32, gguf_tensor_data); the model representation header is on the include path.
sp_add_module(forward_dispatch
    SOURCES   forward_dispatch.c
    TEST      forward_dispatch_test.c
    TEST_NAME T_FWD_DISPATCH
    DEPENDS   sp_forward_kernels sp_arena)
