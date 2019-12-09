#include "common.h"

#include <iostream>

//using namespace cl::sycl;
namespace s = cl::sycl;
class ScalarProdkernel;

class ScalarProdBench
{
protected:    
    std::vector<int> input1;
    std::vector<int> input2;
    std::vector<int> temp_output;
    std::vector<int> output;
    BenchmarkArgs args;

public:
  ScalarProdBench(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {      
    //std::cout << args.problem_size << std::endl;
      // host memory allocation
    // TODO Fill data with values?
    input1.resize(args.problem_size, 1);
    input2.resize(args.problem_size, 1);
    temp_output.resize(args.problem_size, 0);
    // Vector product result
    output.resize(1, 0);
  }

  void run() {    
    s::buffer<int, 1> input1_buf(input1.data(), s::range<1>(args.problem_size));
    s::buffer<int, 1> input2_buf(input2.data(), s::range<1>(args.problem_size));
    s::buffer<int, 1> temp_output_buf(temp_output.data(), s::range<1>(args.problem_size));
    s::buffer<int, 1> output_buf(output.data(), s::range<1>(1));

    args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {
      auto in1 = input1_buf.get_access<s::access::mode::read>(cgh);
      auto in2 = input2_buf.get_access<s::access::mode::read>(cgh);
      // Use discard_write here, otherwise the content of the host
      // buffer must first be copied to device
      auto temp_out = temp_output_buf.get_access<s::access::mode::discard_write>(cgh);
      auto out = output_buf.get_access<s::access::mode::discard_write>(cgh);
      cl::sycl::nd_range<1> ndrange (args.problem_size, 252);

      cgh.parallel_for<class ScalarProdKernel>(ndrange,
        [=](cl::sycl::nd_item<1> item_id) 
        {
          size_t gid= item_id.get_global_linear_id();
          temp_out[gid] = in1[gid] * in2[gid];

          item_id.barrier(s::access::fence_space::global_space);

          if (gid == 0) {
            for(auto i = 0; i < args.problem_size; ++i)
              out[0] += temp_out[i];
          }
        });
    });
  }

  bool verify(VerificationSetting &ver) { 
    bool pass = true;
    int expected = 0;
    int test = 0;
    for(size_t i = 0; i < args.problem_size; i++) {
        expected += input1[i] * input2[i];
        std::cout << temp_output[i];
        test += temp_output[i];
    }

    //TODO: Cleanup after barrier fix
    std::cout << "Scalar product CPU =" << expected << std::endl;
    std::cout << "Scalar product on Device =" << output[0] << std::endl;
    std::cout << "Test =" << test << std::endl;
    
    if(expected != output[0]) {
      pass = false;
    }
    return pass;
  }
  
  static std::string getBenchmarkName() {
    return "ScalarProduct";
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<ScalarProdBench>();  
  return 0;
}
