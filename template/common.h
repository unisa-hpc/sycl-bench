#include <CL/sycl.hpp>

#include <string>
#include <sstream>

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

struct VerificationSettings
{
  cl::sycl::id<3> verificationBegin;
  cl::sycl::range<3> verificationRange;
};

struct BenchmarkArgs
{
  size_t problem_size;
  size_t local_size; 
  cl::sycl::queue *device_queue;

  //VerificationSettings vSettings;
  //BenchmarkArgs(std::size_t problemSize, std::size_t localSize, VerificationSettings vSettings);  
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
    //auto b = new Benchmark(this)

    for(auto h : hooks) h->preSetup();    
    b.setup();
    for(auto h: hooks)  h->postSetup();
    
    for(auto h: hooks)  h->preKernel();
    b.run();
    for(auto h: hooks)  h->postKernel();

    /*
    if(args.verificationSettings.verificationRange.size() > 0)
    {
      if(!h->verify(_args.verificationBegin, _args.verificationRange))
        //Error
      else
        //Pass
    }
    */
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

    // default queue selection   

    args = {problem_size, local_size, &device_queue /*, verification*/};    
  }

    template<class Benchmark>
  void run()
  {
    BenchmarkManager<Benchmark> mgr(args);

    // Add hooks to benchmark manager, perhaps depending on command line arguments?
//    mgr.addHook(std::make_unique<EnergyMeasurement>());
//    mgr.addHook(std::make_unique<TimeMeasurement>());

    mgr.run();
  }
};
