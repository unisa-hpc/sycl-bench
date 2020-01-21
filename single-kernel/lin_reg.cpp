#include "common.h"
#include <iostream>

//using namespace cl::sycl;
namespace s = cl::sycl;
template <typename T> class LinearRegressionKernel;

template <typename T>
class LinearRegressionBench
{
protected:    
    std::vector<T> input1;
    std::vector<T> input2;
    std::vector<T> alpha;
    std::vector<T> beta;
    std::vector<T> output;
    std::vector<T> expected_output;
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
    expected_output.resize(args.problem_size, 0);

    for (size_t i = 0; i < args.problem_size; i++) {
      input1[i] = static_cast <T> (rand()) / static_cast <T> (RAND_MAX);
      input2[i] = static_cast <T> (rand()) / static_cast <T> (RAND_MAX);
      alpha[i] = static_cast <T> (rand()) / static_cast <T> (RAND_MAX);
      beta[i] = static_cast <T> (rand()) / static_cast <T> (RAND_MAX);
    }
  }

  void run() {    
    s::buffer<T, 1> input1_buf(input1.data(), s::range<1>(args.problem_size));
    s::buffer<T, 1> input2_buf(input2.data(), s::range<1>(args.problem_size));
    s::buffer<T, 1> alpha_buf(alpha.data(), s::range<1>(args.problem_size));
    s::buffer<T, 1> beta_buf(beta.data(), s::range<1>(args.problem_size));
    s::buffer<T, 1> output_buf(output.data(), s::range<1>(args.problem_size));
    
    args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {
      auto in1 = input1_buf.template get_access<s::access::mode::read>(cgh);
      auto in2 = input2_buf.template get_access<s::access::mode::read>(cgh);
      auto alpha = alpha_buf.template get_access<s::access::mode::read>(cgh);
      auto beta = beta_buf.template get_access<s::access::mode::read>(cgh);
      // Use discard_write here, otherwise the content of the host buffer must first be copied to device
      auto output = output_buf.template get_access<s::access::mode::discard_write>(cgh);

      cl::sycl::range<1> ndrange (args.problem_size);

      cgh.parallel_for<class LinearRegressionKernel<T>>(ndrange,
        [=](cl::sycl::id<1> idx) 
        {
          size_t gid= idx[0];
          T a = alpha[gid];
          T b = beta[gid];
          T error = 0.0;
          if (gid < args.problem_size) {
              // Use parallel reduction to add errors
              for (size_t i = 0; i < args.problem_size; i++) {
                T e = (a*in1[i] + b) - in2[i];
                error += e*e;
              }
          }
          output[gid] = error;
        });
    });
  }

  bool compare(std::vector<T> expected_output, std::vector<T> output, const int length, const T epsilon) {
      T error = 0.0f;
      T ref = 0.0f;

      for(size_t i = 0; i < length; ++i) {
          T diff = expected_output[i] - output[i];
          error += diff * diff;
          ref += expected_output[i] * expected_output[i];
      }

      T normRef = sqrtf((T) ref);
      if (fabs(ref) < 1e-7f) {
          return false;
      }

      T normError = sqrtf((T) error);
      error = normError / normRef;

      std::cout << "error =" << error << "epsilon =" << epsilon;

      return error < epsilon;
  }

  bool verify(VerificationSetting &ver) { 
    bool pass = true;

    for (size_t i = 0; i < args.problem_size; i ++) {
      T error = 0.0;
      for(size_t j = 0; j < args.problem_size; j++) {
        T e = (alpha[i] * input1[j] + beta[i]) - input2[j];
        error += e*e;
      }
      expected_output[i] = error; 
    }

    //for (size_t i = 0; i < args.problem_size; i++)
      //std::cout << "Expected= " << expected_output[i] << "," << "output = " << output[i] << std::endl;

    bool equal = compare(expected_output, output, args.problem_size, 0.000001);
    
    if(!equal) {
      pass = false;
    }
    return pass;
  }
  
  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "LinearRegression_";
    name << ReadableTypename<T>::name;
    return name.str();     
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<LinearRegressionBench<float>>();
  app.run<LinearRegressionBench<double>>();   
  return 0;
}
