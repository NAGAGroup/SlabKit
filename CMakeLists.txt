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

# Header-only interface target
add_library(slabkit INTERFACE)
add_library(slabkit::slabkit ALIAS slabkit)
target_include_directories(slabkit INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(slabkit INTERFACE cxx_std_20)

option(SLABKIT_BUILD_TESTS "Build Catch2 test suite" OFF)

if(SLABKIT_BUILD_TESTS)
    # The SYCL toolchain is only required for compiling tests; the installed
    # package is plain headers (consumers bring their own SYCL implementation).
    find_package(AdaptiveCpp REQUIRED)
    find_package(Catch2 3 REQUIRED)
    enable_testing()
    add_subdirectory(tests)
endif()

# ------------------------------------------------------------------------------
# Installation / package export (needed for pixi-build-cmake packaging)
# ------------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS slabkit EXPORT slabkitTargets)

install(EXPORT slabkitTargets
    FILE slabkitTargets.cmake
    NAMESPACE slabkit::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/slabkit
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/slabkitConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT
)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/slabkitConfig.cmake"
"include(\"\${CMAKE_CURRENT_LIST_DIR}/slabkitTargets.cmake\")
")

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/slabkitConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/slabkitConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/slabkit
)
