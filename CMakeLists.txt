############################################################
# BlockSp - A templated C++ library for block sparse linear algebra
# CMakeLists.txt, version 1.0
#
# BlockSp is a header-only library. This CMake file builds:
#   - The library
#   - The main executable (tests/examples)
#   - Optionally, the comparison tests to the Eigen linear algebra library
#
# Usage:
#   mkdir build && cd build
#   cmake .. 
#   cmake --build . --config Release
#
# Options:
#   -DBLOCKSP_BUILD_EIGEN_COMPARE=ON   Enable Eigen comparison tests (requires Eigen3)
#   -DBLOCKSP_STACK_SIZE_MB=10         Stack size in MB (default: 10)
#   -DCMAKE_BUILD_TYPE=Release         Build type, typically Debug or Release
#
############################################################

cmake_minimum_required(VERSION 3.16)
project(BlockSp
    VERSION 1.0
    DESCRIPTION "A templated C++ library for block sparse linear algebra"
    LANGUAGES CXX
)

############################################################
# Options
############################################################

option(BLOCKSP_BUILD_EIGEN_COMPARE "Build Eigen comparison tests" OFF)
set(BLOCKSP_STACK_SIZE_MB "10" CACHE STRING "Stack size in MB")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

############################################################
# C++ Standard
############################################################

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

############################################################
# Find MKL
############################################################

# Method 1: Use MKL's CMake config (preferred, available in oneAPI 2021+)
# Set MKL_ROOT or MKLROOT environment variable if not found automatically.
#
# Method 2: If MKL's CMake config is not available, fall back to manual
#
# MKL threading: use Intel OpenMP (iomp5) for best performance with MKL.

# Try MKL's own CMake config first
find_package(MKL CONFIG QUIET)

if(MKL_FOUND)
    message(STATUS "Found MKL via MKL CMake config")
else()
    # Fall back to environment-based detection
    # Users should set MKLROOT or MKL_ROOT
    if(DEFINED ENV{MKLROOT})
        set(MKL_ROOT_DIR "$ENV{MKLROOT}")
    elseif(DEFINED ENV{MKL_ROOT})
        set(MKL_ROOT_DIR "$ENV{MKL_ROOT}")
    else()
        # Common default locations
        if(WIN32)
            file(GLOB MKL_ROOT_CANDIDATES
                "C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
                "C:/Program Files/Intel/oneAPI/mkl/latest"
            )
        else()
            file(GLOB MKL_ROOT_CANDIDATES
                "/opt/intel/oneapi/mkl/latest"
                "/opt/intel/mkl"
            )
        endif()
        if(MKL_ROOT_CANDIDATES)
            list(GET MKL_ROOT_CANDIDATES 0 MKL_ROOT_DIR)
        endif()
    endif()

    if(MKL_ROOT_DIR)
        message(STATUS "Using MKL from: ${MKL_ROOT_DIR}")

        # Include directories
        set(MKL_INCLUDE_DIR "${MKL_ROOT_DIR}/include")
        if(NOT EXISTS "${MKL_INCLUDE_DIR}")
            message(FATAL_ERROR "MKL include directory not found: ${MKL_INCLUDE_DIR}")
        endif()

        # Library directory
        if(WIN32)
            set(MKL_LIB_DIR "${MKL_ROOT_DIR}/lib")
        else()
            set(MKL_LIB_DIR "${MKL_ROOT_DIR}/lib/intel64")
            if(NOT EXISTS "${MKL_LIB_DIR}")
                set(MKL_LIB_DIR "${MKL_ROOT_DIR}/lib")
            endif()
        endif()

        # Find MKL libraries
        find_library(MKL_INTEL_LP64 NAMES mkl_intel_lp64 PATHS ${MKL_LIB_DIR} NO_DEFAULT_PATH)
        find_library(MKL_INTEL_THREAD NAMES mkl_intel_thread PATHS ${MKL_LIB_DIR} NO_DEFAULT_PATH)
        find_library(MKL_CORE NAMES mkl_core PATHS ${MKL_LIB_DIR} NO_DEFAULT_PATH)

        if(MKL_INTEL_LP64 AND MKL_INTEL_THREAD AND MKL_CORE)
            set(MKL_FOUND TRUE)
            set(MKL_LIBRARIES ${MKL_INTEL_LP64} ${MKL_INTEL_THREAD} ${MKL_CORE})
            set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
            message(STATUS "Found MKL libraries manually")
        else()
            message(FATAL_ERROR
                "MKL libraries not found in ${MKL_LIB_DIR}.\n"
                "Please set MKLROOT or MKL_ROOT environment variable to your MKL installation.\n"
                "  Example: cmake .. -DMKLROOT=/opt/intel/oneapi/mkl/latest"
            )
        endif()
    else()
        message(FATAL_ERROR
            "Intel MKL not found. BlockSp requires MKL for dense/block linear algebra.\n"
            "Please install Intel oneAPI MKL and set MKLROOT:\n"
            "  https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html\n"
            "  Example: cmake .. -DMKLROOT=/opt/intel/oneapi/mkl/latest"
        )
    endif()
endif()

# Find Intel OpenMP (iomp5) - required for MKL threaded builds
if(NOT MKL_FOUND OR NOT TARGET MKL::MKL)
    # Find iomp5 for manual MKL setup
    if(DEFINED ENV{ONEAPI_ROOT})
        set(COMPILER_LIB_DIR "$ENV{ONEAPI_ROOT}/compiler/latest/lib")
    elseif(DEFINED ENV{CMPLR_ROOT})
        set(COMPILER_LIB_DIR "$ENV{CMPLR_ROOT}/lib")
    else()
        if(WIN32)
            file(GLOB COMPILER_LIB_CANDIDATES
                "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/lib"
                "C:/Program Files/Intel/oneAPI/compiler/latest/lib"
            )
        else()
            file(GLOB COMPILER_LIB_CANDIDATES
                "/opt/intel/oneapi/compiler/latest/lib"
                "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin"
            )
        endif()
        if(COMPILER_LIB_CANDIDATES)
            list(GET COMPILER_LIB_CANDIDATES 0 COMPILER_LIB_DIR)
        endif()
    endif()

    if(WIN32)
        find_library(IOMP5_LIB NAMES libiomp5md iomp5md PATHS ${COMPILER_LIB_DIR} NO_DEFAULT_PATH)
    else()
        find_library(IOMP5_LIB NAMES iomp5 PATHS ${COMPILER_LIB_DIR} NO_DEFAULT_PATH)
    endif()

    if(NOT IOMP5_LIB)
        # Fall back to system OpenMP
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            set(IOMP5_LIB OpenMP::OpenMP_CXX)
            message(STATUS "Using system OpenMP (Intel iomp5 not found)")
        else()
            message(WARNING "Neither Intel OpenMP (iomp5) nor system OpenMP found. MKL may not thread correctly.")
        endif()
    else()
        message(STATUS "Found Intel OpenMP: ${IOMP5_LIB}")
    endif()
endif()

############################################################
# Find blitz++
############################################################

# blitz++ does not always ship a CMake config.
# Look for the header and set include path.
#
# Users can set BLITZ_ROOT to the blitz installation directory.
# e.g., cmake .. -DBLITZ_ROOT=/path/to/blitz

set(BLITZ_ROOT "" CACHE PATH "Path to blitz++ installation directory (containing blitz/ headers)")

find_path(BLITZ_INCLUDE_DIR
    NAMES blitz/array.h
    PATHS
        ${BLITZ_ROOT}
        $ENV{BLITZ_ROOT}
        /usr/include
        /usr/include/blitz
        /usr/local/include
        /usr/local/include/blitz
        /opt/local/include
        /opt/local/include/blitz
    PATH_SUFFIXES
        include
        ""
)

if(BLITZ_INCLUDE_DIR)
    message(STATUS "Found blitz++ headers: ${BLITZ_INCLUDE_DIR}")
else()
    message(FATAL_ERROR
        "blitz++ headers not found (looking for blitz/array.h).\n"
        "Please install blitz++ and set BLITZ_ROOT:\n"
        "  https://github.com/blitzpp/blitz\n"
        "  Example: cmake .. -DBLITZ_ROOT=/path/to/blitz"
    )
endif()

############################################################
# Create interface library target for BlockSp headers
############################################################

add_library(blocksp INTERFACE)
target_include_directories(blocksp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(blocksp INTERFACE cxx_std_20)

# Link MKL
if(TARGET MKL::MKL)
    target_link_libraries(blocksp INTERFACE MKL::MKL)
else()
    target_include_directories(blocksp INTERFACE ${MKL_INCLUDE_DIRS})
    target_link_libraries(blocksp INTERFACE ${MKL_LIBRARIES})
    if(IOMP5_LIB)
        target_link_libraries(blocksp INTERFACE ${IOMP5_LIB})
    endif()
    # pthreads and math on Linux
    if(UNIX AND NOT APPLE)
        target_link_libraries(blocksp INTERFACE pthread m dl)
    endif()
endif()

# Link blitz++
target_include_directories(blocksp INTERFACE ${BLITZ_INCLUDE_DIR})

############################################################
# Main executable
############################################################

add_executable(blocksp_main main.cpp)
target_link_libraries(blocksp_main PRIVATE blocksp)

# Platform-specific settings
if(MSVC)
    target_compile_options(blocksp_main PRIVATE
        /W3             # Warning level 3
        /sdl-           # Disable SDL checks
        /std:c++latest  # Latest C++ standard
        /bigobj         # Large object file support (heavily templated code)
    )
    # Increase stack size (MKL + TinyMatrices can use significant stack)
    math(EXPR STACK_SIZE_BYTES "${BLOCKSP_STACK_SIZE_MB} * 1024 * 1024")
    target_link_options(blocksp_main PRIVATE /STACK:${STACK_SIZE_BYTES})
else()
    target_compile_options(blocksp_main PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter    # blitz++ triggers these
        -Wno-sign-compare        # blitz++ triggers these
        -fpermissive
        -funroll-loops
        -fstrict-aliasing
        -fomit-frame-pointer
        -ffast-math
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(blocksp_main PRIVATE
            -Wno-deprecated-declarations  # blitz++ may use deprecated features
        )
    endif()
    # Increase stack size on Unix
    math(EXPR STACK_SIZE_BYTES "${BLOCKSP_STACK_SIZE_MB} * 1024 * 1024")
    if(APPLE)
        target_link_options(blocksp_main PRIVATE -Wl,-stack_size,0x${STACK_SIZE_BYTES})
    else()
        target_link_options(blocksp_main PRIVATE -Wl,-z,stacksize=${STACK_SIZE_BYTES})
    endif()
endif()

############################################################
# Eigen comparison tests (optional)
############################################################

if(BLOCKSP_BUILD_EIGEN_COMPARE)
    find_package(Eigen3 REQUIRED NO_MODULE)

    if(Eigen3_FOUND)
        message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")

        # The Eigen comparison is included via main.cpp when the
        # blocksp_eigen_compare.hpp header is present.
        target_compile_definitions(blocksp_main PRIVATE BLOCKSP_USE_EIGEN)
        target_link_libraries(blocksp_main PRIVATE Eigen3::Eigen)
    else()
        message(WARNING "Eigen3 not found. Eigen comparison tests disabled.")
    endif()
endif()

############################################################
# Summary
############################################################

message(STATUS "")
message(STATUS "=== BlockSp Configuration ===")
message(STATUS "  Version:          ${PROJECT_VERSION}")
message(STATUS "  C++ Standard:     C++20")
message(STATUS "  Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  MKL:              Found")
message(STATUS "  blitz++:          ${BLITZ_INCLUDE_DIR}")
message(STATUS "  Stack Size:       ${BLOCKSP_STACK_SIZE_MB} MB")
if(BLOCKSP_BUILD_EIGEN_COMPARE)
    message(STATUS "  Eigen Compare:    ON")
else()
    message(STATUS "  Eigen Compare:    OFF (use -DBLOCKSP_BUILD_EIGEN_COMPARE=ON)")
endif()
message(STATUS "=============================")
message(STATUS "")
