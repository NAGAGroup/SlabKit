cmake_minimum_required(VERSION 3.23)
project(
    slabkit
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "USM device memory with owning handles, for SYCL"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The Windows toolchain's activation puts -std=c++17 in CXXFLAGS, hence in CMAKE_CXX_FLAGS, and acpp refuses two -std options
# ("Multiple c++ standards defined"). CMake adds the one CMAKE_CXX_STANDARD asks for, so drop the inherited one.
string(REGEX REPLACE "(^| )-std=[^ ]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The library carries compiled translation units, and some of them contain SYCL
# kernels, so the SYCL toolchain is required to build it -- not only its tests.
find_package(AdaptiveCpp REQUIRED)

# The pool's free list is mutex-protected.
find_package(Threads REQUIRED)

set(SLABKIT_SOURCES
    src/capabilities.cpp
    src/context.cpp
    src/queue.cpp
    src/unordered_queue.cpp
    src/deleter.cpp
    src/arena.cpp
    src/pool.cpp
    src/site.cpp
    src/allocator.cpp
)

add_library(naga-slabkit SHARED ${SLABKIT_SOURCES})
add_library(naga::slabkit ALIAS naga-slabkit)

target_include_directories(naga-slabkit
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(naga-slabkit PUBLIC cxx_std_20)
target_link_libraries(naga-slabkit PUBLIC Threads::Threads)
set_target_properties(naga-slabkit PROPERTIES
    EXPORT_NAME slabkit
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

add_sycl_to_target(TARGET naga-slabkit SOURCES ${SLABKIT_SOURCES})

# ------------------------------------------------------------------------------
# Installation / package export
# ------------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS naga-slabkit
    EXPORT slabkitTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT slabkitTargets
    FILE slabkitTargets.cmake
    NAMESPACE naga::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/slabkit
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/slabkitConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Consumers link a library whose headers include <sycl/sycl.hpp>, so they need
# the SYCL toolchain found before the imported target is used.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/slabkitConfig.cmake"
"include(CMakeFindDependencyMacro)
find_dependency(AdaptiveCpp)
find_dependency(Threads)
include(\"\${CMAKE_CURRENT_LIST_DIR}/slabkitTargets.cmake\")
")

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/slabkitConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/slabkitConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/slabkit
)
