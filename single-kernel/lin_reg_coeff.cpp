#include "common.h"
#include <iostream>

//using namespace cl::sycl;
namespace s = cl::sycl;

class VecProductKernel;
class VecReduceKernel;

class LinearRegressionBench
{
protected:    
    std::vector<float> input1;
    std::vector<float> input2;
    std::vector<float> output;
    BenchmarkArgs args;

public:
  LinearRegressionBench(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {      
    // host memory allocation and initialization
    input1.resize(args.problem_size);
    input2.resize(args.problem_size);
    output.resize(args.problem_size, 0);

    for (size_t i = 0; i < args.problem_size; i++) {
      input1[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      input2[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      // input1[i] = 1.0;
      // input2[i] = 2.0;
    }
  }

  void vec_product(s::buffer<float, 1> &input1_buf, s::buffer<float, 1> &input2_buf, s::buffer<float, 1> &output_buf) {
    args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {
      auto in1 = input1_buf.get_access<s::access::mode::read>(cgh);
      auto in2 = input2_buf.get_access<s::access::mode::read>(cgh);
 
       // Use discard_write here, otherwise the content of the host buffer must first be copied to device
      auto intermediate_product = output_buf.get_access<s::access::mode::discard_write>(cgh);

      cl::sycl::nd_range<1> ndrange (args.problem_size, args.local_size);

      cgh.parallel_for<class VecProductKernel>(ndrange,
        [=](cl::sycl::nd_item<1> item) 
        {
          size_t gid= item.get_global_linear_id();
          intermediate_product[gid] = in1[gid] * in2[gid];
        });
    });
  }

float reduce(s::buffer<float, 1> &input_buf) {
  auto array_size = args.problem_size;
    auto wgroup_size = args.local_size;
    // Not yet tested with more than 2
    auto elements_per_thread = 2;

    while (array_size!= 1) {
      auto n_wgroups = (array_size + wgroup_size*elements_per_thread - 1)/(wgroup_size*elements_per_thread); // two threads per work item

      args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {

          auto global_mem = input_buf.get_access<s::access::mode::read_write>(cgh);
      
          // local memory for reduction
          auto local_mem = s::accessor <int, 1, s::access::mode::read_write, s::access::target::local> {s::range<1>(wgroup_size), cgh};
          cl::sycl::nd_range<1> ndrange (n_wgroups*wgroup_size, wgroup_size);
    
          cgh.parallel_for<class VecReduceKernel>(ndrange,
          [=](cl::sycl::nd_item<1> item) 
            {
              size_t gid= item.get_global_linear_id();
              size_t lid = item.get_local_linear_id();

              // initialize local memory to 0
              local_mem[lid] = 0; 

              if ((elements_per_thread * gid) < array_size) {
                  local_mem[lid] = global_mem[elements_per_thread*gid] + global_mem[elements_per_thread*gid + 1];
              }

              item.barrier(s::access::fence_space::local_space);

              for (size_t stride = 1; stride < wgroup_size; stride *= elements_per_thread) {
                auto local_mem_index = elements_per_thread * stride * lid;
                if (local_mem_index < wgroup_size) {
                    local_mem[local_mem_index] = local_mem[local_mem_index] + local_mem[local_mem_index + stride];
                }

                item.barrier(s::access::fence_space::local_space);
              }

              // Only one work-item per work group writes to global memory 
              if (lid == 0) {
                global_mem[item.get_group_linear_id()] = local_mem[0];
              }
            });
        });

      array_size = n_wgroups;
    }
    auto reduced_value = input_buf.get_access<s::access::mode::read>();
    return(reduced_value[0]);

}

  void run() {    
    s::buffer<float, 1> input1_buf(input1.data(), s::range<1>(args.problem_size));
    s::buffer<float, 1> input2_buf(input2.data(), s::range<1>(args.problem_size));
    s::buffer<float, 1> output_buf(output.data(), s::range<1>(args.problem_size));

    vec_product(input1_buf, input2_buf, output_buf);

    float ss_xy = reduce(output_buf);

    vec_product(input1_buf, input1_buf, output_buf);

    float ss_xx = reduce(output_buf);

    float mean_x = reduce(input1_buf)/args.problem_size;

    float mean_y = reduce(input2_buf)/args.problem_size;

    ss_xy = ss_xy - mean_x*mean_y;

    ss_xx = ss_xx - mean_x*mean_x;

    float coeff_b1 = ss_xy/ss_xx;

    float coeff_b0 = mean_y-coeff_b1*mean_x;

    //std::cout << "ss_xy = " << ss_xy << "ss_xx = " << ss_xx << std::endl;
    //std::cout << "Mean_x = " << mean_x << "Mean_y = " << mean_y << std::endl;
    std::cout << "Coeff_b1 = " << coeff_b1 << ", " << "Coeff_b0 = " << coeff_b0 << std::endl;
  }

  bool verify(VerificationSetting &ver) { 
     bool pass = true;

    // for (size_t i = 0; i < args.problem_size; i ++) {
    //   float error = 0.0;
    //   for(size_t j = 0; j < args.problem_size; j++) {
    //     float e = (alpha[i] * input1[j] + beta[i]) - input2[j];
    //     error += e*e;
    //   }
    //   expected_output[i] = error; 
    // }

    // for (size_t i = 0; i < args.problem_size; i++)
    //   std::cout << "Expected= " << expected_output[i] << "," << "output = " << output[i] << std::endl;

    // bool equal = compare(expected_output, output, args.problem_size, 0.000001);
    
    // if(!equal) {
    //   pass = false;
    // }
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
