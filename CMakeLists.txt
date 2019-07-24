cmake_minimum_required(VERSION 3.9)

project(celerity-rsim CXX)

find_package(Celerity 0.0.0 REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 14)

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

add_library(include INTERFACE)
target_include_directories(include INTERFACE include)

set(HEADER_FILES
)
source_group("Headers" FILES ${HEADER_FILES})

add_executable(
  sobel
  sobel/sobel.cpp
)
target_link_libraries(sobel PRIVATE include)

set(CMAKE_CXX_COMPILER ${DEVICE_COMPILER})
set(CMAKE_CXX_FLAGS "--restrict-device-header-path=")
add_celerity_to_target(TARGET sobel SOURCES sobel/sobel.cpp)
