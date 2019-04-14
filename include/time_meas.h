#pragma once 

// time measurement
#include "common.h"

#include <iostream>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

class TimeMeasurement : public BenchmarkHook
{
private:
    std::chrono::time_point<std::chrono::high_resolution_clock>  t1;
    std::chrono::time_point<std::chrono::high_resolution_clock>  t2;

public:
    TimeMeasurement() : BenchmarkHook() {}
    
    virtual void atInit() {}    
    virtual void preSetup() {} 
    virtual void postSetup() {}

    virtual void preKernel() {
        t1 = Clock::now();
    }

    virtual void postKernel() {
        t2 = Clock::now();
        std::cout << "time " 
            << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
            << " nanoseconds" 
            << std::endl;        
    }
};
