cmake_minimum_required (VERSION 3.5)
project(sycl-bench)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake Build Type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(hipSYCL CONFIG)
if(NOT hipSYCL_FOUND)
  find_package(ComputeCpp REQUIRED)
endif()

set(benchmarks
  micro/pattern_arith.cpp
  micro/pattern_DRAM.cpp
  micro/pattern_L2.cpp
  micro/pattern_sf.cpp
  micro/pattern_shared.cpp
  single-kernel/median.cpp
  single-kernel/scalar_prod.cpp
  single-kernel/sobel.cpp
  single-kernel/sobel5.cpp
  single-kernel/sobel7.cpp
  single-kernel/vec_add.cpp
  single-kernel/lin_reg.cpp
  single-kernel/kmeans.cpp
  single-kernel/mol_dyn.cpp
  pattern/segmentedreduction.cpp
  pattern/reduction.cpp
  runtime/dag_task_throughput.cpp
  runtime/blocked_transform.cpp
)

foreach(benchmark IN LISTS benchmarks)
  get_filename_component(target ${benchmark} NAME_WE)

  add_executable(${target} ${benchmark})
  add_sycl_to_target(TARGET ${target} SOURCES ${benchmark})

  install(TARGETS ${target} RUNTIME DESTINATION bin/)
endforeach(benchmark)

add_subdirectory(polybench)

