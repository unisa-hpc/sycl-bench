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

    std::size_t nanoseconds;
public:
    TimeMeasurement() : BenchmarkHook() {}
    
    virtual void atInit() override {}    
    virtual void preSetup() override {} 
    virtual void postSetup() override {}

    virtual void preKernel() override {
        nanoseconds = 0;
        t1 = Clock::now();
    }

    virtual void postKernel() override {
        t2 = Clock::now();
        nanoseconds = 
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();     
    }

    virtual void emitResults(ResultConsumer& consumer) override {
        consumer.consumeResult("kernel-run-time", 
            std::to_string(this->nanoseconds), 
            " [ns]");
    }
};
