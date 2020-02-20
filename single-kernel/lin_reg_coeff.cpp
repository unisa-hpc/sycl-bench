#include "common.h"
#include <iostream>

//using namespace cl::sycl;
namespace s = cl::sycl;

template <typename T> class VecProductKernel;
template <typename T> class VecReduceKernel;

template <typename T>
class LinearRegressionCoeffBench
{
protected:    
    std::vector<T> input1;
    std::vector<T> input2;
    std::vector<T> output;
    T coeff_b1;
    T coeff_b0;

    // Only needed for verification as reduction is done inplace which modifies the input
    std::vector<T> input1ver;
    std::vector<T> input2ver;
    BenchmarkArgs args;

    PrefetchedBuffer<T, 1> input1_buf;
    PrefetchedBuffer<T, 1> input2_buf;
    PrefetchedBuffer<T, 1> output_buf;

public:
  LinearRegressionCoeffBench(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {      
    // host memory allocation and initialization
    input1.resize(args.problem_size);
    input2.resize(args.problem_size);
    output.resize(args.problem_size, 0);

    input1ver.resize(args.problem_size);
    input2ver.resize(args.problem_size);

    for (size_t i = 0; i < args.problem_size; i++) {
       input1ver[i] = input1[i] = 1.0;
       input2ver[i] = input2[i] = 2.0;
    }

    input1_buf.initialize(args.device_queue, input1.data(), s::range<1>(args.problem_size));
    input2_buf.initialize(args.device_queue, input2.data(), s::range<1>(args.problem_size));
    output_buf.initialize(args.device_queue, output.data(), s::range<1>(args.problem_size));
  }

  void vec_product(std::vector<cl::sycl::event>& events, s::buffer<T, 1> &input1_buf, s::buffer<T, 1> &input2_buf, s::buffer<T, 1> &output_buf) {
    events.push_back(args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {
      auto in1 = input1_buf.template get_access<s::access::mode::read>(cgh);
      auto in2 = input2_buf.template get_access<s::access::mode::read>(cgh);
 
       // Use discard_write here, otherwise the content of the host buffer must first be copied to device
      auto intermediate_product = output_buf.template get_access<s::access::mode::discard_write>(cgh);

      cl::sycl::nd_range<1> ndrange (args.problem_size, args.local_size);

      cgh.parallel_for<class VecProductKernel<T>>(ndrange,
        [=](cl::sycl::nd_item<1> item) 
        {
          size_t gid= item.get_global_linear_id();
          intermediate_product[gid] = in1[gid] * in2[gid];
        });
    }));
  }

T reduce(std::vector<cl::sycl::event>& events, s::buffer<T, 1> &input_buf) {
  auto array_size = args.problem_size;
  auto wgroup_size = args.local_size;
  // Not yet tested with more than 2
  auto elements_per_thread = 2;

  while (array_size!= 1) {
    auto n_wgroups = (array_size + wgroup_size*elements_per_thread - 1)/(wgroup_size*elements_per_thread); // two threads per work item

    events.push_back(args.device_queue.submit(
      [&](cl::sycl::handler& cgh) {

        auto global_mem = input_buf.template get_access<s::access::mode::read_write>(cgh);
    
        // local memory for reduction
        auto local_mem = s::accessor <T, 1, s::access::mode::read_write, s::access::target::local> {s::range<1>(wgroup_size), cgh};
        cl::sycl::nd_range<1> ndrange (n_wgroups*wgroup_size, wgroup_size);
  
        cgh.parallel_for<class VecReduceKernel<T>>(ndrange,
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
      }));
    array_size = n_wgroups;
  }
  auto reduced_value = input_buf.template get_access<s::access::mode::read>();
  return(reduced_value[0]);
}

  void run(std::vector<cl::sycl::event>& events) {

    vec_product(events, input1_buf.get(), input2_buf.get(), output_buf.get());

    T ss_xy = reduce(events, output_buf.get());

    vec_product(events, input1_buf.get(), input1_buf.get(), output_buf.get());

    T ss_xx = reduce(events, output_buf.get());

    T mean_x = reduce(events, input1_buf.get())/args.problem_size;
    T mean_y = reduce(events, input2_buf.get())/args.problem_size;

    ss_xy = ss_xy - mean_x*mean_y;
    ss_xx = ss_xx - mean_x*mean_x;

    coeff_b1 = ss_xy/ss_xx;
    coeff_b0 = mean_y - coeff_b1*mean_x;

    //std::cout << "ss_xy = " << ss_xy << "ss_xx = " << ss_xx << std::endl;
    //std::cout << "Mean_x = " << mean_x << "Mean_y = " << mean_y << std::endl;
    //std::cout << "Coeff_b1 = " << coeff_b1 << ", " << "Coeff_b0 = " << coeff_b0 << std::endl;
  }

  bool verify(VerificationSetting &ver) { 
     bool pass = true;
    
    T sum_of_vec1 = 0;
    T sum_of_vec2 = 0;
    for (size_t i = 0; i < args.problem_size; i++) {
      sum_of_vec1 += input1ver[i];
      sum_of_vec2 += input2ver[i];
    }

    T mean_x = sum_of_vec1/args.problem_size;
    T mean_y = sum_of_vec2/args.problem_size;

    T ss_xy = 0;
    T ss_xx = 0;
    for (size_t i = 0; i < args.problem_size; i++) {
      ss_xy += input1ver[i]*input2ver[i];
      ss_xx += input1ver[i]*input1ver[i];
    }

    ss_xy = ss_xy - mean_x*mean_y;
    ss_xx = ss_xx - mean_x*mean_x;

    T expected_coeff_b1 = ss_xy/ss_xx;
    T expected_coeff_b0 = mean_y - expected_coeff_b1*mean_x;

    //std::cout << "Coeff_b1 = " << coeff_b1 << ", " << "Coeff_b0 = " << coeff_b0 << std::endl;
    //std::cout << "Expected Coeff_b1 = " << expected_coeff_b1 << ", " << "Expected Coeff_b0 = " << expected_coeff_b0 << std::endl;

    const T tolerance = 0.00001;
    if ((fabs(expected_coeff_b0 - coeff_b0) > tolerance) || (fabs(expected_coeff_b1 - coeff_b1) > tolerance))
      pass = false;

    return pass;
  }
  
  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "LinearRegressionCoeff_";
    name << ReadableTypename<T>::name;
    return name.str();     
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  if(app.shouldRunNDRangeKernels()){
    app.run<LinearRegressionCoeffBench<float>>();
    app.run<LinearRegressionCoeffBench<double>>();   
  }
  return 0;
}
