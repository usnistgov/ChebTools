cmake_minimum_required(VERSION 3.0)

project(ChebTools)

# Set the standard for C++ to c++11
set(CMAKE_CXX_STANDARD 11)

include_directories("${CMAKE_SOURCE_DIR}/externals/Eigen")
include_directories("${CMAKE_SOURCE_DIR}/include")

set(SOURCES "${CMAKE_SOURCE_DIR}/src/ChebTools.cpp" "${CMAKE_SOURCE_DIR}/src/speed_tests.cpp")

if (CHEBTOOLS_PYBIND11)
    # Build pybind11 python module
    add_subdirectory("${CMAKE_SOURCE_DIR}/externals/pybind11")
    pybind11_add_module(ChebTools "${CMAKE_SOURCE_DIR}/src/pybind11_wrapper.cpp" ${SOURCES})
    target_compile_definitions(ChebTools PUBLIC -DPYBIND11)

    # Also build monolithic exe
    add_executable(ChebToolsMonolith "${CMAKE_SOURCE_DIR}/main.cpp" ${SOURCES})

elseif (CHEBTOOLS_MONO)
    # Build monolithic exe
    add_executable(ChebToolsMonolith "${CMAKE_SOURCE_DIR}/main.cpp" ${SOURCES})
else()
    message(FATAL_ERROR "Must select a module: CHEBTOOLS_PYBIND11,CHEBTOOLS_MONO")
endif()

