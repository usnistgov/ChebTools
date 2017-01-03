cmake_minimum_required(VERSION 2.8)

project(ChebTools)

# Set the standard for C++ to c++11
set(CMAKE_CXX_STANDARD 11)

include_directories("${CMAKE_SOURCE_DIR}/externals/Eigen")
include_directories("${CMAKE_SOURCE_DIR}/include")

set(SOURCES "${CMAKE_SOURCE_DIR}/src/ChebTools.cpp" "${CMAKE_SOURCE_DIR}/src/speed_tests.cpp")

# Build pybind11 python module
add_subdirectory("${CMAKE_SOURCE_DIR}/externals/pybind11")
pybind11_add_module(ChebTools "${CMAKE_SOURCE_DIR}/src/pybind11_wrapper.cpp" ${SOURCES})
target_compile_definitions(ChebTools PUBLIC -DPYBIND11)

# Also build monolithic exe
add_executable(ChebToolsMonolith "${CMAKE_SOURCE_DIR}/main.cpp" ${SOURCES})

# Also build Catch testing module
include_directories("${CMAKE_SOURCE_DIR}/externals/Catch/single_include")
add_executable(ChebToolsCatchTests "${CMAKE_SOURCE_DIR}/tests/tests.cpp" ${SOURCES})