add_executable(slabkit_tests
    test_memory.cpp
    test_array.cpp
)

target_link_libraries(slabkit_tests PRIVATE
    slabkit
    Catch2::Catch2WithMain
)

include(CTest)
include(Catch)
catch_discover_tests(slabkit_tests)
