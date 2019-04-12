#include "common.h"

#include <iostream>

using namespace cl::sycl;
class VecAddKernel;

class VecAddBench
{
protected:    
    std::vector<int> input1;
    std::vector<int> input2;
    std::vector<int> output;
    const BenchmarkArgs &args;

public:
  VecAddBench(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {      
    //std::cout << args.problem_size << std::endl;
      // host memory allocation
     input1.resize(args.problem_size, 1);
     input2.resize(args.problem_size, 2);
     output.resize(args.problem_size, 0);
  }

  void run() {    
    buffer<int, 1> input1_buf(&input1[0], range<1>(args.problem_size));
    buffer<int, 1> input2_buf(&input2[0], range<1>(args.problem_size));
    buffer<int, 1> output_buf(&output[0], range<1>(args.problem_size));

    args.device_queue->submit(
        [&](cl::sycl::handler& cgh) {
      auto in1 = input1_buf.get_access<access::mode::read>(cgh);
      auto in2 = input2_buf.get_access<access::mode::read>(cgh);
      auto out = output_buf.get_access<access::mode::write>(cgh);
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
  
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<VecAddBench>();  
  return 0;
}
