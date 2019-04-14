#pragma once 
#include <CL/sycl.hpp>

#include <string>
#include <iostream>
#include <sstream>
#include <algorithm> // for std::min
#include "command_line.h"


class BenchmarkHook
{
public:
  virtual void atInit() = 0;
  virtual void preSetup() = 0;
  virtual void postSetup()= 0;
  virtual void preKernel() = 0;
  virtual void postKernel() = 0;
};

#ifdef TIME_MEAS    
  #include "time_meas.h"
  //class TimeMeasurement;
#endif
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
    args.device_queue.wait_and_throw();
    for(auto h: hooks)  h->postKernel();
    
    if(args.verification.range.size() > 0)
    {
      if(!b.verify(args.verification)){
        // error
        std::cerr << "Verification ERROR" << std::endl;
      }
      else {
        // pass
        std::cout << "Verification successful" << std::endl;
      }        
    }
    
  }

private:
  BenchmarkArgs args;  
  std::vector<BenchmarkHook*> hooks;
};


class BenchmarkApp
{
  BenchmarkArgs args;  
  cl::sycl::queue device_queue;

public:  
  BenchmarkApp(int argc, char** argv)
  {
    BenchmarkCommandLine cli{argc, argv};
    args = cli.getBenchmarkArgs();
  }

  template<class Benchmark>
  void run()
  {
    BenchmarkManager<Benchmark> mgr(args);
    
    std::cout << "Running with problem size " << args.problem_size 
      << " and local size " << args.local_size 
      <<  std::endl;

    // Add hooks to benchmark manager, perhaps depending on command line arguments?

    #ifdef TIME_MEAS
      TimeMeasurement tm;
      mgr.addHook(tm);
    #endif

    #ifdef NV_ENERGY_MEAS
      NVEnergyMeasurement nvem;
      mgr.addHook(nvem);
    #endif

    mgr.run();
  }
};
