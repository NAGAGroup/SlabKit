cmake_minimum_required(VERSION 3.23)
project(
    slabkit
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Thin RAII wrappers and memory pools over SYCL USM"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(AdaptiveCpp REQUIRED)
find_package(Catch2 3 REQUIRED)

add_subdirectory(ext/memory)

# Header-only interface target
add_library(slabkit INTERFACE)
target_include_directories(slabkit INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(slabkit INTERFACE foonathan_memory)

enable_testing()
add_subdirectory(tests)

