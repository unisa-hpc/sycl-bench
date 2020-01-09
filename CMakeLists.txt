cmake_minimum_required (VERSION 3.5)
project(sycl-bench)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(pattern_arith  micro/pattern_arith.cpp)
add_executable(pattern_DRAM   micro/pattern_DRAM.cpp)
add_executable(pattern_L2     micro/pattern_L2.cpp)
add_executable(pattern_sf     micro/pattern_sf.cpp)
add_executable(pattern_shared micro/pattern_shared.cpp)
add_executable(median         single-kernel/median.cpp)
add_executable(scalar_prod    single-kernel/scalar_prod.cpp)
add_executable(sobel          single-kernel/sobel.cpp)
add_executable(sobel5         single-kernel/sobel5.cpp)
add_executable(sobel7         single-kernel/sobel7.cpp)
add_executable(vec_add        single-kernel/vec_add.cpp)

install(TARGETS 
  pattern_arith pattern_DRAM pattern_L2 pattern_sf pattern_shared 
  median scalar_prod sobel sobel5 sobel7 vec_add
  RUNTIME DESTINATION bin/)
