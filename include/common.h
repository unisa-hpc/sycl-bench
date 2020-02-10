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

#include "command_line.h"
#include "result_consumer.h"
#include "type_traits.h"

  
#include "benchmark_hook.h"
#include "time_meas.h"

#ifdef NV_ENERGY_MEAS    
  #include "nv_energy_meas.h"
#endif

namespace detail {

template <typename T, typename = void>
struct VerificationDispatcher {
  static constexpr bool canVerify = false;
};

template <typename T>
struct VerificationDispatcher<T, std::void_t<decltype(&T::verify)>> {
  static constexpr bool canVerify = true;
};

} // namespace detail

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


    for(auto h : hooks) h->atInit();

    bool all_runs_pass = true;
    // Run until we have as many runs as requested or until
    // verification fails
    for(std::size_t run = 0; run < args.num_runs && all_runs_pass; ++run) {
      Benchmark b(args, additionalArgs...);

      for(auto h : hooks) h->preSetup();    
      b.setup();
      args.device_queue.wait_and_throw();
      for(auto h: hooks)  h->postSetup();
      
      for(auto h: hooks)  h->preKernel();
      b.run();
      // Make sure work has actually completed,
      // otherwise we may end up measuring incorrect
      // runtimes!
      args.device_queue.wait_and_throw();
      for (auto h : hooks) h->postKernel();

      if constexpr(detail::VerificationDispatcher<Benchmark>::canVerify) {
        if(args.verification.range.size() > 0) {
          if(args.verification.enabled) {
            if(!b.verify(args.verification)) {
              all_runs_pass = false;
            }
          }
        }
      }
    }

    for (auto h : hooks) {
      // Extract results from the hooks
      h->emitResults(*args.result_consumer);
    }

    if(args.verification.range.size() == 0 || !args.verification.enabled ||
        !detail::VerificationDispatcher<Benchmark>::canVerify) {
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

  std::string getSyclImplementation() const
  {
#if defined(__HIPSYCL__)
    return "hipSYCL";
#elif defined(__COMPUTECPP__)
    return "ComputeCpp";
#else
    // ToDo: Find out how they can be distinguished
    return "triSYCL or Intel SYCL";
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

      // Add hooks to benchmark manager, perhaps depending on command line
      // arguments?

      TimeMeasurement tm;
      if (!args.cli.isFlagSet("--no-runtime-measurement"))
        mgr.addHook(tm);

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
