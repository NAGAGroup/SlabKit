# One test file per public header, plus the two-device test that only has
# something to say when the environment enumerates more than one device.
function(add_sycl_executable TARGET_NAME)
    add_executable(${TARGET_NAME} ${ARGN})
    add_sycl_to_target(TARGET ${TARGET_NAME} SOURCES ${ARGN})
endfunction()

add_sycl_executable(slabkit_tests
    test_capabilities.cpp
    test_context.cpp
    test_queue.cpp
    test_handles.cpp
    test_arena.cpp
    test_pool.cpp
    test_malloc.cpp
    test_make.cpp
    test_make_shared.cpp
    test_cross_device.cpp
)

target_link_libraries(slabkit_tests PRIVATE
    naga::slabkit
    Catch2::Catch2WithMain
)

include(CTest)
include(Catch)
catch_discover_tests(slabkit_tests)
