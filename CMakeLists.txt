cmake_minimum_required(VERSION 3.0)

project(ChebTools)

# Set the standard for C++ to c++11
set(CMAKE_CXX_STANDARD 11)

# Add root as include directory
include_directories("${CMAKE_SOURCE_DIR}/include")

if (CHEBTOOLS_PYBIND11)
    # Build pybind11 python module
    add_subdirectory("${CMAKE_SOURCE_DIR}/externals/pybind11")
    add_definitions(-DPYBIND11)
    pybind11_add_module(ChebTools ${APP_SOURCES} "${CMAKE_SOURCE_DIR}/main.cpp")
else()
    message(FATAL_ERROR "Must select a module: CHEBTOOLS_PYBIND11")
endif()

