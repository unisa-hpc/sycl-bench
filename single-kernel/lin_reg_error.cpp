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

    PrefetchedBuffer<T, 1> input1_buf;
    PrefetchedBuffer<T, 1> input2_buf;
    PrefetchedBuffer<T, 1> alpha_buf;
    PrefetchedBuffer<T, 1> beta_buf;
    PrefetchedBuffer<T, 1> output_buf;

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

    input1_buf.initialize(args.device_queue, input1.data(), s::range<1>(args.problem_size));
    input2_buf.initialize(args.device_queue, input2.data(), s::range<1>(args.problem_size));
    alpha_buf. initialize(args.device_queue, alpha.data(), s::range<1>(args.problem_size));
    beta_buf.  initialize(args.device_queue, beta.data(), s::range<1>(args.problem_size));
    output_buf.initialize(args.device_queue, output.data(), s::range<1>(args.problem_size));
  }

  void run(std::vector<cl::sycl::event>& events) {
    
    events.push_back(args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {
      auto in1 = input1_buf.template get_access<s::access::mode::read>(cgh);
      auto in2 = input2_buf.template get_access<s::access::mode::read>(cgh);
      auto alpha = alpha_buf.template get_access<s::access::mode::read>(cgh);
      auto beta = beta_buf.template get_access<s::access::mode::read>(cgh);
      // Use discard_write here, otherwise the content of the host buffer must first be copied to device
      auto output = output_buf.template get_access<s::access::mode::discard_write>(cgh);

      cl::sycl::range<1> ndrange (args.problem_size);

      cgh.parallel_for<class LinearRegressionKernel<T>>(ndrange,
        [=, problem_size = args.problem_size](cl::sycl::id<1> idx)
        {
          size_t gid= idx[0];
          T a = alpha[gid];
          T b = beta[gid];
          T error = 0.0;
          if (gid < problem_size) {
              // Use parallel reduction to add errors
              for (size_t i = 0; i < problem_size; i++) {
                T e = (a*in1[i] + b) - in2[i];
                error += e*e;
              }
          }
          output[gid] = error;
        });
    }));
  }

  bool compare(const std::vector<T>& expected_output, const int length, const T epsilon) {
      T error = 0.0f;
      T ref = 0.0f;

      auto output = output_buf.template get_access<s::access::mode::read>();

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

      //std::cout << "error =" << error << "epsilon =" << epsilon;

      return error < epsilon;
  }

  bool verify(VerificationSetting &ver) { 

    for (size_t i = 0; i < args.problem_size; i ++) {
      T error = 0.0;
      for(size_t j = 0; j < args.problem_size; j++) {
        T e = (alpha[i] * input1[j] + beta[i]) - input2[j];
        error += e*e;
      }
      expected_output[i] = error; 
    }

    return compare(expected_output, args.problem_size, 0.000001);
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
