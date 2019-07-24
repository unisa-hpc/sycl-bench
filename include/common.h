#pragma once 
#include <celerity/celerity.h>
//#include <CL/sycl.hpp>



#include <string>
#include <iostream>
#include <cassert>
#include <sstream>
#include <memory>
#include <algorithm> // for std::min
#include "command_line.h"
#include "result_consumer.h"

class BenchmarkHook
{
public:
  virtual void atInit() = 0;
  virtual void preSetup() = 0;
  virtual void postSetup()= 0;
  virtual void preKernel() = 0;
  virtual void postKernel() = 0;
  virtual void emitResults(ResultConsumer&) {}
};
  
#include "time_meas.h"


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

  void run()
  {
    args.result_consumer->proceedToBenchmark(
      Benchmark::getBenchmarkName());

    args.result_consumer->consumeResult(
      "problem-size", std::to_string(args.problem_size));
    args.result_consumer->consumeResult(
      "local-size", std::to_string(args.local_size));

    for(auto h : hooks) h->atInit();

    Benchmark b(args);    

    for(auto h : hooks) h->preSetup();    
    b.setup();
    for(auto h: hooks)  h->postSetup();
    
    for(auto h: hooks)  h->preKernel();
    b.run();
    // Make sure work has actually completed,
    // otherwise we may end up measuring incorrect
    // runtimes!
    //args.device_queue.wait_and_throw();

    // HACK alert!
    // we need to do this by deleting the queue in Celerity currently
    delete args.device_queue;

    for (auto h : hooks) {
      h->postKernel();
      // Extract results from the hooks
      h->emitResults(*args.result_consumer);
    }

    if(args.verification.range.size() > 0)
    {
      if(!b.verify(args.verification)){
        // error
        args.result_consumer->consumeResult("Verification", "FAIL");
      }
      else {
        // pass
        args.result_consumer->consumeResult("Verification", "PASS");
      }
    }
    args.result_consumer->flush();
    
  }

private:
  BenchmarkArgs args;  
  std::vector<BenchmarkHook*> hooks;
};


class BenchmarkApp
{
  BenchmarkArgs args;
  
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

  template<class Benchmark>
  void run()
  {
    try {
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

      mgr.run();
    }
    catch(cl::sycl::exception& e){
      std::cerr << "SYCL error: " << e.what() << std::endl;
    }
    catch(std::exception& e){
      std::cerr << "Error: " << e.what() << std::endl;
    }
  }
};
