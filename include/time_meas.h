#pragma once 

// time measurement
#include "benchmark_hook.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>

typedef std::chrono::high_resolution_clock Clock;

class TimeMeasurement : public BenchmarkHook
{
private:
    std::chrono::time_point<std::chrono::high_resolution_clock>  t1;
    std::chrono::time_point<std::chrono::high_resolution_clock>  t2;

    std::vector<double> seconds;
public:
    TimeMeasurement() : BenchmarkHook() {}
    
    virtual void atInit() override {}    
    virtual void preSetup() override {} 
    virtual void postSetup() override {}

    virtual void preKernel() override {
        t1 = Clock::now();
    }

    virtual void postKernel() override {
        t2 = Clock::now();
        seconds.push_back( 
            static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count())
            * 1.e-9);    
    }

    virtual void emitResults(ResultConsumer& consumer) override {

        double total_sec = std::accumulate(seconds.begin(), seconds.end(), 0.);

        consumer.consumeResult("kernel-run-time", 
            std::to_string(total_sec / 
                static_cast<double>(seconds.size())), 
            " [s]");
        // TODO: We could / should also calculate stddev here
    }
};
