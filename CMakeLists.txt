cmake_minimum_required(VERSION 3.5)
project(sycl-bench)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake Build Type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build flags
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG -fno-omit-frame-pointer" CACHE STRING "Flags used by the C++ compiler during debug builds." FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math" CACHE STRING "Flags used by the C++ compiler during release builds." FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -DNDEBUG -march=native -ffast-math -g -fno-omit-frame-pointer" CACHE STRING "Flags used by the C++ compiler during release builds with debug info." FORCE)


if(CMAKE_GENERATOR STREQUAL "Ninja")
	set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -fdiagnostics-color=always")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
	set(COMPUTECPP_USER_FLAGS "${COMPUTECPP_USER_FLAGS} -fdiagnostics-color=always")
endif()

if(SYCL_BENCH_ENABLE_QUEUE_PROFILING)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSYCL_BENCH_ENABLE_QUEUE_PROFILING")
endif()

include(InstallRequiredSystemLibraries)
include(CPack)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/polybench/common)

set(supported_implementations
	AdaptiveCpp
	dpcpp
	triSYCL
)

list(FIND supported_implementations ${SYCL_IMPL} impl_idx)

if(NOT SYCL_IMPL OR impl_idx EQUAL -1)
	message(FATAL_ERROR "Please specify SYCL_IMPL (one of: ${supported_implementations})")
endif()

if(SYCL_IMPL STREQUAL "AdaptiveCpp")
	find_package(AdaptiveCpp REQUIRED)
elseif(SYCL_IMPL STREQUAL "dpcpp")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

	if(DPCPP_WITH_CUDA_BACKEND)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda")

		set(CUDA_ARCH "" CACHE STRING "CUDA device architecture e.g. sm_70")

		if(NOT CUDA_ARCH STREQUAL "")
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend --cuda-gpu-arch=${CUDA_ARCH}")
		endif()
	endif()

	if(DPCPP_WITH_ROCM_BACKEND)
		set(ROCM_ARCH "" CACHE STRING "ROCm device architecture e.g. gfx908")

		if(NOT ROCM_ARCH STREQUAL "")
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=amd_gpu_${ROCM_ARCH}")
		endif()
	endif()

	if(DPCPP_WITH_LZ_BACKEND)
		set(LZ_ARCH "" CACHE STRING "Level Zero device architecture e.g. acm-g10")

		if(NOT LZ_ARCH STREQUAL "")
		  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=${LZ_ARCH}")
		endif()
	endif()

elseif(SYCL_IMPL STREQUAL "triSYCL")
	find_package(TriSYCL MODULE REQUIRED)
endif()

# Check if SYCL implementation implements the required SYCL features
include(HasFeatures)

set(benchmarks
  micro/arith.cpp
  micro/DRAM.cpp
  micro/host_device_bandwidth.cpp
  micro/pattern_L2.cpp
  micro/sf.cpp
  micro/local_mem.cpp
  single-kernel/median.cpp
  single-kernel/scalar_prod.cpp
  single-kernel/sobel.cpp
  single-kernel/sobel5.cpp
  single-kernel/sobel7.cpp
  single-kernel/vec_add.cpp
  single-kernel/lin_reg_error.cpp
  single-kernel/lin_reg_coeff.cpp
  single-kernel/kmeans.cpp
  single-kernel/mol_dyn.cpp
  single-kernel/nbody.cpp
  pattern/segmentedreduction.cpp
  pattern/reduction.cpp
  runtime/dag_task_throughput_sequential.cpp
  runtime/dag_task_throughput_independent.cpp
  runtime/blocked_transform.cpp
  runtime/matmulchain.cpp
  polybench/2DConvolution.cpp
  polybench/2mm.cpp
  polybench/3DConvolution.cpp
  polybench/3mm.cpp
  polybench/atax.cpp
  polybench/bicg.cpp
  polybench/correlation.cpp
  polybench/covariance.cpp
  polybench/fdtd2d.cpp
  polybench/gemm.cpp
  polybench/gesummv.cpp
  polybench/gramschmidt.cpp
  polybench/mvt.cpp
  polybench/syr2k.cpp
  polybench/syrk.cpp

  # compiletime/compiletime.cpp
  sycl2020/atomics/atomic_reduction.cpp
  sycl2020/USM/usm_accessors_latency.cpp
  sycl2020/USM/usm_instr_mix.cpp
  sycl2020/USM/usm_pinned_overhead.cpp
  sycl2020/USM/usm_allocation_latency.cpp
)
# Selectively add benchmarks based on some SYCL 2020 features
if (SYCL_BENCH_HAS_SPEC_CONSTANTS)
  list(APPEND benchmarks sycl2020/spec_constants/spec_constant_convolution.cpp)
endif()
if (SYCL_BENCH_HAS_KERNEL_REDUCTIONS)
  list(APPEND benchmarks sycl2020/kernel_reduction/kernel_reduction.cpp)
endif()
if (SYCL_BENCH_HAS_GROUP_ALGORITHMS)
  list(APPEND benchmarks sycl2020/group_algorithms/reduce_over_group.cpp)
endif()

# Setting variables
add_compile_definitions(SYCL_BENCH_HAS_FP64_SUPPORT=$<BOOL:${SYCL_BENCH_HAS_FP64_SUPPORT}>)

foreach(benchmark IN LISTS benchmarks)
	get_filename_component(target ${benchmark} NAME_WE)

	add_executable(${target} ${benchmark})

	if(SYCL_IMPL STREQUAL "AdaptiveCpp")
		add_sycl_to_target(TARGET ${target} SOURCES ${benchmark})
	endif()

	if(SYCL_IMPL STREQUAL "dpcpp")
		target_compile_definitions(${target} PRIVATE __DPCPP__)
	endif()

  if(SYCL_IMPL STREQUAL "triSYCL")
    add_sycl_to_target(${target})
    target_compile_definitions(${target} PRIVATE __TRISYCL__)
  endif()
  
  if(ENABLE_TIME_EVENT_PROFILING)
    target_compile_definitions(${target} PUBLIC SYCL_BENCH_ENABLE_QUEUE_PROFILING=1)
  endif()
  
  install(TARGETS ${target} RUNTIME DESTINATION bin/benchmarks/)
  get_filename_component(dir ${benchmark} DIRECTORY)
  set_property(TARGET ${target} PROPERTY FOLDER ${dir})
endforeach(benchmark)

# The "compiletime" target should only be used in the context of the compile time evaluation script
# set_target_properties(compiletime PROPERTIES EXCLUDE_FROM_ALL 1)
install(PROGRAMS bin/run-suite DESTINATION bin/)
install(FILES ${PROJECT_SOURCE_DIR}/Brommy.bmp DESTINATION share/)
