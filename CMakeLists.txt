cmake_minimum_required(VERSION 3.23)
project(
    slabkit
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "NAGA: Nonlocal Acoustic GPU Accelerated LBM"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(AdaptiveCpp REQUIRED)

