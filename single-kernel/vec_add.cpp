#include "common.h"

#include <iostream>

// Opening cl::sycl namespace is unsupported on hipSYCL 
// (mainly due to CUDA/HIP design issues), better 
// avoid it
//using namespace cl::sycl;
namespace s = cl::sycl;
class VecAddKernel;

class VecAddBench
{
protected:    
    std::vector<int> input1;
    std::vector<int> input2;
    std::vector<int> output;
    BenchmarkArgs args;

public:
  VecAddBench(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {      
    //std::cout << args.problem_size << std::endl;
      // host memory allocation
    // TODO Fill data with values?
    input1.resize(args.problem_size, 1);
    input2.resize(args.problem_size, 2);
    output.resize(args.problem_size, 0);
  }

  void run() {    
    s::buffer<int, 1> input1_buf(input1.data(), s::range<1>(args.problem_size));
    s::buffer<int, 1> input2_buf(input2.data(), s::range<1>(args.problem_size));
    s::buffer<int, 1> output_buf(output.data(), s::range<1>(args.problem_size));

    args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {
      auto in1 = input1_buf.get_access<s::access::mode::read>(cgh);
      auto in2 = input2_buf.get_access<s::access::mode::read>(cgh);
      // Use discard_write here, otherwise the content of the host
      // buffer must first be copied to device
      auto out = output_buf.get_access<s::access::mode::discard_write>(cgh);
      cl::sycl::range<1> ndrange {args.problem_size};

      cgh.parallel_for<class VecAddKernel>(ndrange,
        [=](cl::sycl::id<1> gid) 
        {
            out[gid] = in1[gid] + in2[gid];
        });
    });

  }

  bool verify(VerificationSetting &ver) { 
    bool pass = true;
    for(size_t i=ver.begin[0]; i<ver.begin[0]+ver.range[0]; i++){
        int expected = input1[i] + input2[i];
        //std::cout << i << ") " << output[i] << " : " << expected << std::endl;
        if(expected != output[i]){
            pass = false;
            break;
        }
      }    
    return pass;
  }
  
  static std::string getBenchmarkName() {
    return "VectorAddition";
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<VecAddBench>();  
  return 0;
}
