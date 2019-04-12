#include <CL/sycl.hpp>

#include <string>
#include <iostream>
#include <sstream>

#ifdef TIME_MEAS    
  #include "time_meas.h"
#endif

#ifdef NV_ENERGY_MEAS    
  #include "nv_energy_meas.h"
#endif



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
        std::cerr << "Verification error" << std::endl;
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
  
  cl::sycl::queue device_queue; // default queue selection

public:  
  BenchmarkApp(int argc, char** argv)
  {
    // Parses command line arguments
    vector <std::string> sources;
    size_t problem_size, local_size;     

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

    // FIXME queue selection parameters 

    args = {problem_size, local_size, &device_queue /*, verification*/};    
  }

    template<class Benchmark>
  void run()
  {
    BenchmarkManager<Benchmark> mgr(args);

    // Add hooks to benchmark manager, perhaps depending on command line arguments?

    #ifdef TIME_MEAS    
      mgr.addHook(std::make_unique<TimeMeasurement>());
    #endif

    #ifdef NV_ENERGY_MEAS
      mgr.addHook(std::make_unique<NVEnergyMeasurement>());
    #endif

    mgr.run();
  }
};
