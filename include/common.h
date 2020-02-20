#pragma once 
#include <CL/sycl.hpp>

#include <string>
#include <iostream>
#include <cassert>
#include <sstream>
#include <memory>
#include <algorithm> // for std::min
#include <type_traits>
#include <unordered_set>
#include <optional>

#include "command_line.h"
#include "result_consumer.h"
#include "type_traits.h"

  
#include "benchmark_hook.h"
#include "benchmark_traits.h"
#include "prefetched_buffer.h"
#include "time_metrics.h"

#ifdef NV_ENERGY_MEAS    
  #include "nv_energy_meas.h"
#endif



template<class Benchmark>
class BenchmarkManager
{
public:
  BenchmarkManager(const BenchmarkArgs &_args) : args(_args)  {}

  void addHook(BenchmarkHook &h)
  {
    hooks.push_back(&h);
  }

  template<typename... Args>
  void run(Args&&... additionalArgs)
  {
    args.result_consumer->proceedToBenchmark(Benchmark{args, additionalArgs...}.getBenchmarkName());

    args.result_consumer->consumeResult(
      "problem-size", std::to_string(args.problem_size));
    args.result_consumer->consumeResult(
      "local-size", std::to_string(args.local_size));
    args.result_consumer->consumeResult(
      "device-name", args.device_queue.get_device()
                           .template get_info<cl::sycl::info::device::name>());
    args.result_consumer->consumeResult(
      "sycl-implementation", this->getSyclImplementation());

    TimeMetricsProcessor<Benchmark> time_metrics(args);

    for(auto h : hooks) h->atInit();

    bool all_runs_pass = true;
    try {
      // Run until we have as many runs as requested or until
      // verification fails
      for(std::size_t run = 0; run < args.num_runs && all_runs_pass; ++run) {
        Benchmark b(args, additionalArgs...);

        for(auto h : hooks) h->preSetup();

        b.setup();

        args.device_queue.wait_and_throw();
        for(auto h : hooks) h->postSetup();

        std::vector<cl::sycl::event> run_events;
        run_events.reserve(1024); // Make sure we don't need to resize during benchmarking.

        // Performance critical measurement section starts here
        for(auto h : hooks) h->preKernel();
        const auto before = std::chrono::high_resolution_clock::now();
        if constexpr(detail::BenchmarkTraits<Benchmark>::supportsQueueProfiling) {
          b.run(run_events);
        } else {
          b.run();
        }
        args.device_queue.wait_and_throw();
        const auto after = std::chrono::high_resolution_clock::now();
        for(auto h : hooks) h->postKernel();
        // Performance critical measurement section ends here

        time_metrics.addTimingResult("run-time", std::chrono::duration_cast<std::chrono::nanoseconds>(after - before));

        if(detail::BenchmarkTraits<Benchmark>::supportsQueueProfiling) {
#if defined(SYCL_BENCH_ENABLE_QUEUE_PROFILING)
          // TODO: We might also want to consider the "command_submit" time.
          std::chrono::nanoseconds total_time{0};
          for(auto& e : run_events) {
            const auto start = e.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
            const auto end = e.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
            total_time += std::chrono::nanoseconds(end - start);
          }
          time_metrics.addTimingResult("kernel-time", total_time);
#else
          time_metrics.markAsUnavailable("kernel-time");
#endif
        } else {
          time_metrics.markAsUnavailable("kernel-time");
        }

        if constexpr(detail::BenchmarkTraits<Benchmark>::hasVerify) {
          if(args.verification.range.size() > 0) {
            if(args.verification.enabled) {
              if(!b.verify(args.verification)) {
                all_runs_pass = false;
              }
            }
          }
        }
      }
    } catch(...) {
      args.result_consumer->discard();
      std::rethrow_exception(std::current_exception());
    }

    time_metrics.emitResults(*args.result_consumer);

    for (auto h : hooks) {
      // Extract results from the hooks
      h->emitResults(*args.result_consumer);
    }

    if(args.verification.range.size() == 0 || !args.verification.enabled ||
        !detail::BenchmarkTraits<Benchmark>::hasVerify) {
      args.result_consumer->consumeResult("Verification", "N/A");
    }
    else if(!all_runs_pass){
      // error
      args.result_consumer->consumeResult("Verification", "FAIL");
    }
    else {
      // pass
      args.result_consumer->consumeResult("Verification", "PASS");
    }        
    
    args.result_consumer->flush();
    
  }

private:
  BenchmarkArgs args;  
  std::vector<BenchmarkHook*> hooks;

  std::string getSyclImplementation() const {
#if defined(__HIPSYCL__)
    return "hipSYCL";
#elif defined(__COMPUTECPP__)
    return "ComputeCpp";
#elif defined(__LLVM_SYCL__)
    return "LLVM (Intel DPC++)";
#elif defined(__LLVM_SYCL_CUDA__)
    return "LLVM CUDA (Codeplay)";
#elif defined(__TRISYCL__)
    return "triSYCL";
#else
    return "UNKNOWN";
#endif
  }
};


class BenchmarkApp
{
  BenchmarkArgs args;  
  cl::sycl::queue device_queue;
  std::unordered_set<std::string> benchmark_names;
  
public:  
  BenchmarkApp(int argc, char** argv)
  {
    try{
      args = BenchmarkCommandLine{argc, argv}.getBenchmarkArgs();
    }
    catch(std::exception& e){
      std::cerr << "Error while parsing command lines: " << e.what() << std::endl;
    }
  }

  const BenchmarkArgs& getArgs() const
  { return args; }

  bool shouldRunNDRangeKernels() const
  {
    return !args.cli.isFlagSet("--no-ndrange-kernels");
  }

  template<class Benchmark, typename... AdditionalArgs>
  void run(AdditionalArgs&&... additional_args)
  {
    try {
      const auto name = Benchmark{args, additional_args...}.getBenchmarkName();
      if(benchmark_names.count(name) == 0) {
        benchmark_names.insert(name);
      } else {
        std::cerr << "Benchmark with name '" << name << "' has already been run\n";
        throw std::runtime_error("Duplicate benchmark name");
      }

      BenchmarkManager<Benchmark> mgr(args);

#ifdef NV_ENERGY_MEAS
      NVEnergyMeasurement nvem;
      mgr.addHook(nvem);
#endif

      mgr.run(additional_args...);
    }
    catch(cl::sycl::exception& e){
      std::cerr << "SYCL error: " << e.what() << std::endl;
    }
    catch(std::exception& e){
      std::cerr << "Error: " << e.what() << std::endl;
    }
  }
};
