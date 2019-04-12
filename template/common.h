#pragma once 
#include <CL/sycl.hpp>

#include <string>
#include <iostream>
#include <sstream>
#include <algorithm> // for std::min


using namespace std;

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



struct VerificationSetting
{
  cl::sycl::id<3> begin;
  cl::sycl::range<3> range;
};

struct BenchmarkArgs
{
  size_t problem_size;
  size_t local_size; 
  cl::sycl::queue *device_queue;
  VerificationSetting verification;
};


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
    // TODO so far the first CPU is selected
    cl::sycl::cpu_selector selector;
    device_queue = selector.select_device();

    // Parses command line arguments
    vector <std::string> sources;
    size_t problem_size = 1024;
    size_t local_size = 256;     
    //bool verification = true;

    for (int i = 1; i < argc; ++i) {
      if (string(argv[i]) == "-size" && (i+1 < argc) ) {
        istringstream iss(argv[i++]);
        iss >> problem_size; 
      } 
      else 
      if (string(argv[i]) == "-local" && (i+1 < argc) ) {
        istringstream iss(argv[i++]);
        iss >> local_size;
      }
    }           

    // TODO at the moment it cheks no more than 2048 elements    
    size_t range_max = std::min<size_t>(2048, problem_size);
    VerificationSetting defaultVerSetting = { {0,0,0}, {range_max,range_max,range_max} };
    args = {problem_size, local_size, &device_queue, defaultVerSetting};    
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
