cmake_minimum_required(VERSION 3.25) # TODO: No idea with one is the actual minimum

macro(check_feature VAR FILENAME)
    if(NOT DEFINED SYCL_BENCH_HAS_${VAR})
        try_compile(SYCL_BENCH_HAS_${VAR} ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/has-features/src/${FILENAME} 
            CMAKE_FLAGS ${CMAKE_CXX_FLAGS}
            OUTPUT_VARIABLE OUTPUT_VAR
        )
    endif()

    if (SYCL_BENCH_HAS_${VAR})
        set(RES ON)
    else()
        set(RES OFF)
    endif()
    message(STATUS "${VAR}: ${RES}")    

endmacro()

message(STATUS "Checking for SYCL features....")
check_feature(KERNEL_REDUCTION kernel_reduction_dummy.cpp)
check_feature(SPEC_CONSTANTS spec_constants_dummy.cpp)
check_feature(GROUP_ALGORITHMS group_algorithms_dummy.cpp)
check_feature(FP64_SUPPORT fp64_support_dummy.cpp)