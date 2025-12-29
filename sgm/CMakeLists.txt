cmake_minimum_required(VERSION 3.0)

project(sgm_wrapper VERSION 0.1.0)

message(STATUS "======================  !Flightmare!  ======================")

################################################################################
# Options
################################################################################
option(BUILD_TESTS "Building the tests" ON)
option(BUILD_UNITY_BRIDGE_TESTS "Building the Unity Bridge tests" ON)
option(BUILD_BENCH "Building the benchmark." OFF)
option(ENABLE_FAST "Build with optimizations for speed" ON)
option(ENABLE_BLAS "Build using BLAS and LAPACK libraries" OFF)
option(ENABLE_PARALLEL "Build using openmp parallelization" ON)
option(EIGEN_FROM_SYSTTEM "Use the system-provided Eigen" ON)
set(
  EIGEN_ALTERNATIVE "" CACHE STRING
  "Path to alternative Eigen, autodownload if blank"
)

# Including dependencies 
include(cmake/pybind11.cmake)

message(STATUS "======> Setup Dependencies")
if(EIGEN_FROM_SYSTTEM)
  find_package(Eigen3 3.3.4 QUIET)
  if(EIGEN3_FOUND)
    message(STATUS "Using system provided Eigen.")
    message(${EIGEN3_INCLUDE_DIR})
  else()
    message(STATUS "No sufficient Eigen version (3.3.4) found.")
    message(STATUS "Restoring to download Eigen sources.")
    include(cmake/eigen.cmake)
  endif()
elseif(EIGEN_ALTERNATIVE STREQUAL "")
  include(cmake/eigen.cmake)
else()
  set(EIGEN_INCLUDE_DIR ${EIGEN_ALTERNATIVE})
endif()
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3
        -gencode=arch=compute_52,code=sm_52
        -gencode=arch=compute_61,code=sm_61
        -gencode=arch=compute_62,code=sm_62
        -gencode=arch=compute_75,code=sm_75
        -gencode=arch=compute_86,code=sm_86
        )

message("CUDA Libs: ${CUDA_LIBRARIES}")
message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
message("CUDA curand library: ${CUDA_curand_LIBRARY}")

# Including dependencies 
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

################################################################################
# Setup Compilation
################################################################################
message(STATUS "======> Setup Compilation")

# Set default build type
if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

# Add c++ flags 
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -fPIC -Wall -DNDEBUG  -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -fPIC -Wall -g -fopenmp")
set(CMAKE_CXX_STANDARD 17)

# Architectural flags
if("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "armv7l")
  message(STATUS "Using ARMv7 optimized flags!")
  set(CMAKE_CXX_ARCH_FLAGS " -Wno-psabi -march=armv7-a -mfpu=neon -mfloat-abi=hard -funsafe-math-optimizations")
elseif("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
  message(STATUS "Using ARM aarch64 optimized flags!")
  set(CMAKE_CXX_ARCH_FLAGS " -Wno-psabi -march=armv8-a+crypto -mcpu=cortex-a57+crypto")
else()
  set(CMAKE_CXX_ARCH_FLAGS " -march=native")
endif()

# Optimized flags
if(ENABLE_FAST)
  message(STATUS "Enabling fast optimization flags!")
  set(CMAKE_CXX_FAST_FLAGS " -Ofast")
else()
  set(CMAKE_CXX_FAST_FLAGS " -O0")
endif()

# BLAS Flags
if(BLAS_FOUND AND LAPACK_FOUND)
  message(STATUS "Enabling BLAS and LAPACK")
  set(CMAKE_CXX_BLAS_FLAGS " -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACK -DEIGEN_USE_LAPACKE")
else()
  set(CMAKE_CXX_BLAS_FLAGS "")
endif()

# Summarize Flags
set(CMAKE_CXX_FLAGS_RELEASE
  "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FAST_FLAGS} ${CMAKE_CXX_ARCH_FLAGS} ${CMAKE_CXX_PAR_FLAGS}")
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "The activated CXX RELEASE configuration is:\n ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "The activated CXX DEBUG configuration is:\n ${CMAKE_CXX_FLAGS_DEBUG}")

################################################################################
# Specify Build Resources
################################################################################
message(STATUS "======>  Setup Build  ")
# Library and Executables
include_directories(include ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
  
cuda_add_library(sgm_gpu
      src/sgm_gpu/sgm_gpu.cu
      src/sgm_gpu/costs.cu
      src/sgm_gpu/hamming_cost.cu
      src/sgm_gpu/left_right_consistency.cu
      src/sgm_gpu/median_filter.cu)

pybind11_add_module(sgm_wrapper MODULE
    src/wrapper/sgm_wrapper.cpp
  )
  
if(EIGEN3_FOUND)
    target_include_directories(sgm_wrapper PRIVATE
      # ${PROJECT_SOURCE_DIR}/externals/pybind11-src/include
      ${PYBIND11_INCLUDE_DIR}
      ${PROJECT_SOURCE_DIR}/include
      ${EIGEN3_INCLUDE_DIR})
else()
    target_include_directories(sgm_wrapper PRIVATE
      # ${PROJECT_SOURCE_DIR}/externals/pybind11-src/include
      ${PYBIND11_INCLUDE_DIR}
      ${PROJECT_SOURCE_DIR}/externals/eigen/eigen3 
      ${PROJECT_SOURCE_DIR}/include)
endif()
target_link_libraries(sgm_wrapper PRIVATE sgm_gpu ${OpenCV_LIBRARIES})
