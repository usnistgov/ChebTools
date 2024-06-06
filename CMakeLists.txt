set(CMAKE_OSX_ARCHITECTURES x86_64;arm64 CACHE INTERNAL "archs for osx")
cmake_minimum_required(VERSION 3.16)

project(ChebTools)

# Set the standard for C++ to c++17
set(CMAKE_CXX_STANDARD 17)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/Catch")

set(SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/ChebTools.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/speed_tests.cpp")

if (MSVC)
    list(APPEND SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
endif()

option(CHEBTOOLS_NANOBIND False "")
option(CHEBTOOLS_PYBIND11 False "")

set(OPENMP_NEEDED False)
if (OPENMP_NEEDED)
    # Check for the existence of OpenMP and enable it as needed
    # see also http://stackoverflow.com/a/12404666/1360263
    find_package(OpenMP)
endif()

# Single-source the version, either from scikit, or from parsing the pyproject.toml
if (SKBUILD)
    add_definitions("-DCHEBTOOLSVERSION=\"${SKBUILD_PROJECT_VERSION_FULL}\"")
else()
    file(READ "pyproject.toml" TOML_CONTENT)
    set(REG "version = \"([0-9]+\\.[0-9]+\\.[0-9]+)\"")
    string(REGEX MATCH "${REG}" VERSION_MATCH "${TOML_CONTENT}")
    if (NOT VERSION_MATCH)
        message(FATAL_ERROR "Can't parse the version")
    else()
        string(REGEX REPLACE "${REG}" "\\1" PROJECT_VERSION_FULL "${VERSION_MATCH}")
        message(STATUS "Version: ${PROJECT_VERSION_FULL}")
        add_definitions("-DCHEBTOOLSVERSION=\"${PROJECT_VERSION_FULL}\"")
    endif()
endif()

if (CHEBTOOLS_STATIC_LIBRARY OR NISTFIT_PYBIND11)
    add_library(ChebTools STATIC ${SOURCES})
    # Add target include directories for easy linking with other applications
    target_include_directories(ChebTools PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    if (OPENMP_NEEDED)
        target_link_libraries(ChebTools PUBLIC OpenMP::OpenMP_CXX)
    endif()
else()
    if (CHEBTOOLS_PYBIND11)
        if (CHEBTOOLS_NANOBIND)
            message(FATAL_ERROR "only one python binding can be enabled")
        endif()
        # Build pybind11 python module
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/pybind11")
        pybind11_add_module(ChebTools "${CMAKE_CURRENT_SOURCE_DIR}/src/pybind11_wrapper.cpp" ${SOURCES})
        target_compile_definitions(ChebTools PUBLIC -DPYBIND11)
        if (OPENMP_NEEDED)
            target_link_libraries(ChebTools PUBLIC OpenMP::OpenMP_CXX)
        endif()
    endif()

    if (CHEBTOOLS_NANOBIND)
        find_package(Python COMPONENTS Interpreter Development.Module REQUIRED) # from https://github.com/pypa/cibuildwheel/issues/639#issuecomment-1397443143
        if (CHEBTOOLS_PYBIND11)
            message(FATAL_ERROR "only one python binding can be enabled")
        endif()
        # Build nanobind python module
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/externals/nanobind")
        nanobind_add_module(ChebTools "${CMAKE_CURRENT_SOURCE_DIR}/src/nanobind_wrapper.cpp" ${SOURCES})
        set(NAME ChebTools)
        nanobind_add_stub(
          ChebTools_stub
          MODULE ChebTools
          OUTPUT "${CMAKE_SOURCE_DIR}/src/ChebTools/ChebTools.pyi"
          PYTHON_PATH $<TARGET_FILE_DIR:ChebTools>
          DEPENDS ChebTools
          MARKER_FILE py.typed
          VERBOSE
        )
        # Install directive for scikit-build-core for the stubs
        install(FILES "${CMAKE_SOURCE_DIR}/src/ChebTools/ChebTools.pyi" DESTINATION ${NAME})
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/py.typed" DESTINATION ${NAME})
        unset(NAME)

        if (OPENMP_NEEDED)
            target_link_libraries(ChebTools PUBLIC OpenMP::OpenMP_CXX)
        endif()
        message(STATUS "Added nanobind interface")
        install(TARGETS ChebTools LIBRARY DESTINATION ChebTools)
    endif()

    if (NOT CHEBTOOLS_NO_MONOLITH)
        # Also build monolithic exe
        add_executable(ChebToolsMonolith "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp" ${SOURCES})
        if (OPENMP_NEEDED)
            target_link_libraries(ChebToolsMonolith PUBLIC OpenMP::OpenMP_CXX)
        endif()
    endif()
    
    if (NOT CHEBTOOLS_NO_CATCH)
        # Also build Catch testing module
        include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Catch")
        add_executable(ChebToolsCatchTests "${CMAKE_CURRENT_SOURCE_DIR}/tests/tests.cpp" ${SOURCES})
        if (OPENMP_NEEDED)
            target_link_libraries(ChebToolsCatchTests PUBLIC OpenMP::OpenMP_CXX)
        endif()
        target_link_libraries(ChebToolsCatchTests PUBLIC Catch2WithMain)
    endif()
    
endif()

add_subdirectory("scripts/Basu")
