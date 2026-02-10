cmake_minimum_required(VERSION 3.12)
project(matmul-playground LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# A100
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

file(GLOB SOURCES "src/kernel*.cu")
add_library(kernels ${SOURCES})

# Set CUDA compilation options
target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>) # print register use
target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -dlcm=cg>) # turn off L1 cache
target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>)
target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Uncomment to enable device debug info
# target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)

# Uncomment to keep the generated cubin and ptx files
# target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--keep>)

add_library(cublas_launch src/cublas_benchmark.cu)
target_link_libraries(cublas_launch CUDA::cublas)

add_executable(runner src/main.cu)
target_link_libraries(runner PUBLIC kernels cublas_launch)