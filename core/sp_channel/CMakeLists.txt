cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sp_channel C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    enable_testing()
endif()
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/sp_module.cmake)

sp_add_module(channel
    SOURCES   sp_channel_map.c sp_channel_probe.c sp_hedge.c
    TEST      sp_channel_test.c
    TEST_NAME T_CHANNEL
    DEPENDS   sp_io_format)

# sp_channel_probe.c uses pthreads on POSIX for hedge-read threading.
if(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(sp_channel PUBLIC Threads::Threads)
    if(TARGET test_channel)
        target_link_libraries(test_channel PRIVATE Threads::Threads)
    endif()
endif()

# The test needs the internal header from the module source directory.
if(TARGET test_channel)
    target_include_directories(test_channel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# M_TS_HEDGE oracle bench — separate executable (sp_add_module allows only one TEST target).
add_executable(bench_ts_hedge bench_ts_hedge.c)
target_link_libraries(bench_ts_hedge PRIVATE sp_channel)
target_include_directories(bench_ts_hedge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include)
if(UNIX)
    target_link_libraries(bench_ts_hedge PRIVATE Threads::Threads)
endif()

# §16.3 TS.HEDGE correctness tests (T_HEDGE_*).
add_executable(test_sp_hedge test_sp_hedge.c)
target_link_libraries(test_sp_hedge PRIVATE sp_channel)
target_include_directories(test_sp_hedge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include)
add_test(NAME T_HEDGE COMMAND test_sp_hedge)

# §16.3 TS.HEDGE production bench (M_TS_HEDGE_PROD).
add_executable(bench_sp_hedge bench_sp_hedge.c)
target_link_libraries(bench_sp_hedge PRIVATE sp_channel)
target_include_directories(bench_sp_hedge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include)
if(UNIX)
    target_link_libraries(bench_sp_hedge PRIVATE Threads::Threads m)
endif()
