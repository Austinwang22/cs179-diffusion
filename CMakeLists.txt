cmake_minimum_required(VERSION 3.27)
project(diffusion LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# ---------------------------------------------------------------------------
# CUDA toolkit + arch
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED COMPONENTS
  cudart cublas cublasLt cufft cusolver nvtx3 curand)

# native = highest arch of the host GPU
set(CMAKE_CUDA_ARCHITECTURES native)

# ---------------------------------------------------------------------------
# cuDNN (not covered by FindCUDAToolkit on some distros)
# ---------------------------------------------------------------------------
find_library(CUDNN_LIBRARY NAMES cudnn
             HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/lib /usr/local/lib
             REQUIRED)
find_path(CUDNN_INCLUDE_DIR cudnn.h
          HINTS ${CUDAToolkit_INCLUDE_DIRS} /usr/include /usr/local/include)

# ---------------------------------------------------------------------------
# Common compile flags
# ---------------------------------------------------------------------------
add_compile_options(
  -Wno-changes-meaning
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)

# ---------------------------------------------------------------------------
# diffusion_lib – core library with all CUDA/C++ sources & headers
# ---------------------------------------------------------------------------
add_library(diffusion_lib
        src/vendor/argparse.hpp
        src/vendor/safetensors.hh
        src/vendor/safetensors_impl.cpp
        src/CudaBuffer.cuh
        src/CudaBuffer.cu
        src/ErrorCheck.h
        src/HostBuffer.cpp
        src/HostBuffer.h

        src/diffusion/DiffusionConfig.h
        src/diffusion/DiffusionLoader.h
        src/diffusion/DiffusionLoader.cu
        src/diffusion/DiffusionWeights.h

        src/diffusion/DiffusionUNet.cuh
        src/diffusion/DiffusionLayers.cuh
        src/diffusion/DiffusionHelper.cuh
        src/diffusion/DiffusionKernels.cuh
        src/diffusion/DiffusionEDMPrecond.cuh)

target_include_directories(diffusion_lib
    PUBLIC ${CUDAToolkit_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})

target_link_libraries(diffusion_lib
    PUBLIC
      CUDA::cudart
      CUDA::cublas      CUDA::cublasLt
      CUDA::cufft       CUDA::cusolver
      CUDA::curand
      CUDA::nvtx3
      ${CUDNN_LIBRARY})

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------

# 1. original main demo
add_executable(diffusion src/vendor/argparse.hpp src/main.cu)
target_link_libraries(diffusion PRIVATE diffusion_lib)

# 2. tests
add_executable(tests src/tests.cu)
target_link_libraries(tests PRIVATE diffusion_lib)

# 3. **NEW** Euler sampler (src/sample.cu)
add_executable(sample src/sample.cu)
target_link_libraries(sample PRIVATE diffusion_lib)
