cmake_minimum_required(VERSION 3.16)
project(tree_packing VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings with pybind11" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(ENABLE_SIMD "Enable SIMD optimizations (AVX2)" ON)
option(ENABLE_OPENMP "Enable OpenMP parallelization" OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG)
    endif()
    if(ENABLE_SIMD)
        add_compile_options(-march=native)
    endif()
    # Disable FMA contraction to ensure deterministic floating-point results
    # FMA has different rounding than separate multiply+add, causing inconsistencies
    add_compile_options(-ffp-contract=off)
endif()

# OpenMP
if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found, enabling parallel execution")
    else()
        message(WARNING "OpenMP not found, disabling parallel execution")
        set(ENABLE_OPENMP OFF)
    endif()
endif()

# Create the main library
add_library(tree_packing STATIC
    src/core/types.cpp
    src/core/tree.cpp
    src/core/solution.cpp
    src/core/problem.cpp
    src/core/global_state.cpp
    src/core/update_stack.cpp
    src/spatial/grid2d.cpp
    src/geometry/sat.cpp
    src/geometry/intersection.cpp
    src/constraints/intersection.cpp
    src/constraints/bounds.cpp
    src/optimizers/optimizer.cpp
    src/optimizers/ruin.cpp
    src/optimizers/recreate.cpp
    src/optimizers/alns.cpp
    src/optimizers/sa.cpp
    src/optimizers/combine.cpp
    src/optimizers/noise.cpp
    src/random/rng.cpp
)
set_target_properties(tree_packing PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(tree_packing PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(tree_packing PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(tree_packing PUBLIC ENABLE_OPENMP)
endif()

if(ENABLE_SIMD)
    target_compile_definitions(tree_packing PUBLIC ENABLE_SIMD)
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(tree_packing_cpp python/bindings.cpp)
    target_link_libraries(tree_packing_cpp PRIVATE tree_packing)

    # Install the Python module to the python package directory
    install(TARGETS tree_packing_cpp
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/python/tree_packing_cpp
    )
endif()

# Tests
if(BUILD_TESTS)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/Catch2/CMakeLists.txt)
        add_subdirectory(third_party/Catch2)
    else()
        include(FetchContent)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG        v3.4.0
        )
        FetchContent_MakeAvailable(Catch2)
    endif()

    add_executable(test_tree_packing
        tests/test_types.cpp
        tests/test_tree.cpp
        tests/test_sat.cpp
        tests/test_grid2d.cpp
        tests/test_solution.cpp
        tests/test_optimizers.cpp
        tests/test_incremental.cpp
    )
    target_link_libraries(test_tree_packing PRIVATE
        tree_packing
        Catch2::Catch2WithMain
        Catch2::Catch2
    )
    target_link_options(test_tree_packing PRIVATE "-Wl,--start-group" "-Wl,--end-group")

    include(CTest)
    include(Catch)
    catch_discover_tests(test_tree_packing)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_executable(example_basic examples/basic.cpp)
    target_link_libraries(example_basic PRIVATE tree_packing)

    add_executable(example_profile_basic examples/profile_basic.cpp)
    target_link_libraries(example_profile_basic PRIVATE tree_packing)

    add_executable(benchmark examples/benchmark.cpp)
    target_link_libraries(benchmark PRIVATE tree_packing)
endif()
