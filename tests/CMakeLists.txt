# Helper function to create SYCL executables
# Usage: add_sycl_executable(target_name source1.cpp source2.cpp ...)
function(add_sycl_executable TARGET_NAME)
    # ${ARGN} contains all arguments after TARGET_NAME
    add_executable(${TARGET_NAME} ${ARGN})
    add_sycl_to_target(TARGET ${TARGET_NAME} SOURCES ${ARGN})
endfunction()


add_sycl_executable(slabkit_tests
    test_memory.cpp
    test_array.cpp
    test_context_array.cpp
    test_context.cpp
    test_api.cpp
    test_transfers.cpp
)


target_link_libraries(slabkit_tests PRIVATE
    naga::slabkit
    Catch2::Catch2WithMain
)

include(CTest)
include(Catch)
catch_discover_tests(slabkit_tests)
