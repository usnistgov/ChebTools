cmake_minimum_required(VERSION 3.0)

project(ChebTools)

# Set the standard for C++ to c++11
set(CMAKE_CXX_STANDARD 11)

include_directories("${CMAKE_SOURCE_DIR}/externals/Eigen")

if (CHEBTOOLS_PYBIND11)
    # Build pybind11 python module
    add_subdirectory("${CMAKE_SOURCE_DIR}/externals/pybind11")
    add_definitions(-DPYBIND11)
    pybind11_add_module(ChebTools "${CMAKE_SOURCE_DIR}/main.cpp")
elseif (CHEBTOOLS_MONO)
    # Build monolithic exe
    add_executable(ChebTools "${CMAKE_SOURCE_DIR}/main.cpp")
else()
    message(FATAL_ERROR "Must select a module: CHEBTOOLS_PYBIND11,CHEBTOOLS_MONO")
endif()

