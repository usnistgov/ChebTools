cmake_minimum_required(VERSION 3.0)

project(ChebTools)

# Set the standard for C++ to c++11
set(CMAKE_CXX_STANDARD 11)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

set(SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/ChebTools.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/speed_tests.cpp")

if (MSVC)
    list(APPEND SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
endif()

# Check for the existence of OpenMP and enable it as needed
# see also http://stackoverflow.com/a/12404666/1360263
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if (CHEBTOOLS_STATIC_LIBRARY OR NISTFIT_PYBIND11)
    add_library(ChebTools STATIC ${SOURCES})
    # Add target include directories for easy linking with other applications
    target_include_directories(ChebTools PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
else()
    if (NOT CHEBTOOLS_NO_PYBIND11)
        # Build pybind11 python module
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/pybind11")
        pybind11_add_module(ChebTools "${CMAKE_CURRENT_SOURCE_DIR}/src/pybind11_wrapper.cpp" ${SOURCES})
        target_compile_definitions(ChebTools PUBLIC -DPYBIND11)
    endif()

    # Also build monolithic exe
    add_executable(ChebToolsMonolith "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp" ${SOURCES})
    
    # Also build Catch testing module
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Catch/single_include")
    add_executable(ChebToolsCatchTests "${CMAKE_CURRENT_SOURCE_DIR}/tests/tests.cpp" ${SOURCES})
endif()