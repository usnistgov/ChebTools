cmake_minimum_required(VERSION 3.0)

project(ChebTools)

# Set the standard for C++ to c++17
set(CMAKE_CXX_STANDARD 17)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

set(SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/ChebTools.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/speed_tests.cpp")

if (MSVC)
    list(APPEND SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
endif()

# Check for the existence of OpenMP and enable it as needed
# see also http://stackoverflow.com/a/12404666/1360263
find_package(OpenMP)

if (CHEBTOOLS_STATIC_LIBRARY OR NISTFIT_PYBIND11)
    add_library(ChebTools STATIC ${SOURCES})
    # Add target include directories for easy linking with other applications
    target_include_directories(ChebTools PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(ChebTools PUBLIC OpenMP::OpenMP_CXX)
else()
    if (NOT CHEBTOOLS_NO_PYBIND11)
        # Build pybind11 python module
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/pybind11")
        pybind11_add_module(ChebTools "${CMAKE_CURRENT_SOURCE_DIR}/src/pybind11_wrapper.cpp" ${SOURCES})
        target_compile_definitions(ChebTools PUBLIC -DPYBIND11)
        target_link_libraries(ChebTools PUBLIC OpenMP::OpenMP_CXX)
    endif()

    # Also build monolithic exe
    add_executable(ChebToolsMonolith "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp" ${SOURCES})
    target_link_libraries(ChebToolsMonolith PUBLIC OpenMP::OpenMP_CXX)

    # Build monolithic exe for testing 2D expansions
    add_executable(ChebToolsMonolith2D "${CMAKE_CURRENT_SOURCE_DIR}/src/test_2D.cpp")
    target_link_libraries(ChebToolsMonolith2D PUBLIC OpenMP::OpenMP_CXX)
    
    # Also build Catch testing module
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Catch/single_include")
    add_executable(ChebToolsCatchTests "${CMAKE_CURRENT_SOURCE_DIR}/tests/tests.cpp" ${SOURCES})
    target_link_libraries(ChebToolsCatchTests PUBLIC OpenMP::OpenMP_CXX)
    
endif()
