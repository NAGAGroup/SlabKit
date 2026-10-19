cmake_minimum_required(VERSION 3.23)
project(slabkit-tests VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The Windows toolchain's activation puts -std=c++17 in CXXFLAGS, hence in CMAKE_CXX_FLAGS, and acpp refuses two -std options
# ("Multiple c++ standards defined"). CMake adds the one CMAKE_CXX_STANDARD asks for, so drop the inherited one.
string(REGEX REPLACE "(^| )-std=[^ ]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

find_package(AdaptiveCpp REQUIRED)
find_package(slabkit REQUIRED)
find_package(Catch2 3 REQUIRED)

# One test file per public header, plus the two-device test that only has
# something to say when the environment enumerates more than one device.
set(SLABKIT_TEST_SOURCES
    test_capabilities.cpp test_context.cpp test_queue.cpp test_host_work.cpp
    test_kinds.cpp test_memory.cpp test_allocator.cpp test_usm_vector.cpp
    test_handles.cpp test_arena.cpp test_pool.cpp test_malloc.cpp
    test_make.cpp test_make_shared.cpp test_cross_device.cpp test_unordered_queue.cpp
)
add_executable(slabkit-tests ${SLABKIT_TEST_SOURCES})
add_sycl_to_target(TARGET slabkit-tests SOURCES ${SLABKIT_TEST_SOURCES})
target_link_libraries(slabkit-tests PRIVATE naga::slabkit Catch2::Catch2WithMain)

# On Windows the Catch2 library comes from a build that does not define StringMaker<std::byte>, while this clang build sees C++17
# and declares it, which leaves an undefined symbol at link time. Tell the header not to declare it.
if(WIN32)
    target_compile_definitions(slabkit-tests PRIVATE CATCH_CONFIG_NO_CPP17_BYTE)
endif()

include(GNUInstallDirs)
install(TARGETS slabkit-tests RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
