#include "common.h"
#include <iostream>

//using namespace cl::sycl;
namespace s = cl::sycl;
class LinearRegressionKernel;

class LinearRegressionBench
{
protected:    
    std::vector<float> input1;
    std::vector<float> input2;
    std::vector<float> alpha;
    std::vector<float> beta;
    std::vector<float> output;
    BenchmarkArgs args;

public:
  LinearRegressionBench(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {      
    // host memory allocation and initialization
    input1.resize(args.problem_size);
    input2.resize(args.problem_size);
    alpha.resize(args.problem_size);
    beta.resize(args.problem_size);
    output.resize(args.problem_size, 0);

    for (size_t i =0; i < args.problem_size; i++) {
      input1[i] = i;
      input2[i] = i;
    }
  }

  void run() {    
    s::buffer<float, 1> input1_buf(input1.data(), s::range<1>(args.problem_size));
    s::buffer<float, 1> input2_buf(input2.data(), s::range<1>(args.problem_size));
    s::buffer<float, 1> alpha_buf(alpha.data(), s::range<1>(args.problem_size));
    s::buffer<float, 1> beta_buf(beta.data(), s::range<1>(args.problem_size));
    s::buffer<float, 1> output_buf(output.data(), s::range<1>(args.problem_size));
    
    args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {
      auto in1 = input1_buf.get_access<s::access::mode::read>(cgh);
      auto in2 = input2_buf.get_access<s::access::mode::read>(cgh);
      auto alpha = alpha_buf.get_access<s::access::mode::read>(cgh);
      auto beta = beta_buf.get_access<s::access::mode::read>(cgh);
      // Use discard_write here, otherwise the content of the host buffer must first be copied to device
      auto output = output_buf.get_access<s::access::mode::discard_write>(cgh);

      cl::sycl::range<1> ndrange (args.problem_size);

      cgh.parallel_for<class LinearRegressionKernel>(ndrange,
        [=](cl::sycl::nd_item<1> item) 
        {
          size_t gid= item.get_global_linear_id();

          auto error = 0;
          if (gid < args.problem_size) {
              auto e = (alpha[gid]*in1[gid] + beta[gid]) - in2[gid];
              output[gid] = e*e;

              // Use parallel reduction to add errors
              for (size_t i =0; i < args.problem_size; i++)
                error += error + output[i];
          }
          output[gid] = error;
        });
    });
  }

  bool verify(VerificationSetting &ver) { 
    bool pass = true;
    float expected_error = 0;

    for(size_t i = 0; i < args.problem_size; i++) {
        float e = (alpha[i] * input1[i] + beta[i]) - input2[i];
        e = e*e;
        expected_error += e;
    }

    //std::cout << "Error on CPU =" << expected_error << std::endl;
    //std::cout << "Error on Device =" << output[0] << std::endl;
    
    if(expected_error != output[0]) {
      pass = false;
    }
    return pass;
  }
  
  static std::string getBenchmarkName() {
    return "LinearRegression";
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<LinearRegressionBench>();  
  return 0;
}
