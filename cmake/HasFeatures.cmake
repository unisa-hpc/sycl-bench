macro(check_feature VAR FILENAME)
    if(NOT DEFINED RUN_RES_${VAR})
            try_run(RUN_RES_${VAR} COMPILE_RES_${VAR} ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/has-features/src/${FILENAME} 
                CMAKE_FLAGS ${CMAKE_CXX_FLAGS}
                COMPILE_OUTPUT_VARIABLE OUTPUT_VAR
                RUN_OUTPUT_VARIABLE RUN_VAR 
            )
    endif()

    if (COMPILE_RES_${VAR} AND RUN_RES_${VAR} EQUAL 0)
        set(RES ON)
    else()
        set(RES OFF)
    endif()
    message(STATUS "${VAR}: ${RES}")
endmacro()

message(STATUS "Checking for SYCL features....")
check_feature(KERNEL_REDUCTIONS kernel_reduction_dummy.cpp)
check_feature(SPEC_CONSTANTS spec_constants_dummy.cpp)
check_feature(GROUP_ALGORITHMS group_algorithms_dummy.cpp)
check_feature(FP64_SUPPORT fp64_support_dummy.cpp)