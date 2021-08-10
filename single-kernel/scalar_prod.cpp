#include "common.h"

#include <iostream>
#include <type_traits>
#include <iomanip>

//using namespace cl::sycl;
namespace s = cl::sycl;

template<typename T, bool>
class ScalarProdKernel;
template<typename T, bool>
class ScalarProdKernelHierarchical;

template<typename T, bool>
class ScalarProdReduction;
template<typename T, bool>
class ScalarProdReductionHierarchical;

class ScalarProdGatherKernel;

template<typename T, bool Use_ndrange = true>
class ScalarProdBench
{
protected:    
    std::vector<T> input1;
    std::vector<T> input2;
    std::vector<T> output;
    BenchmarkArgs args;

    PrefetchedBuffer<T, 1> input1_buf;
    PrefetchedBuffer<T, 1> input2_buf;
    PrefetchedBuffer<T, 1> output_buf;

public:
  ScalarProdBench(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {      
    // host memory allocation and initialization
    input1.resize(args.problem_size);
    input2.resize(args.problem_size);
    output.resize(args.problem_size);

    for (size_t i = 0; i < args.problem_size; i++) {
      input1[i] = static_cast<T>(1);
      input2[i] = static_cast<T>(2);
      output[i] = static_cast<T>(0);
    }

    input1_buf.initialize(args.device_queue, input1.data(), s::range<1>(args.problem_size));
    input2_buf.initialize(args.device_queue, input2.data(), s::range<1>(args.problem_size));
    output_buf.initialize(args.device_queue, output.data(), s::range<1>(args.problem_size));
  }

  void run(std::vector<cl::sycl::event>& events) {
    
    events.push_back(args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {
      auto in1 = input1_buf.template get_access<s::access::mode::read>(cgh);
      auto in2 = input2_buf.template get_access<s::access::mode::read>(cgh);
      // Use discard_write here, otherwise the content of the hostbuffer must first be copied to device
      auto intermediate_product = output_buf.template get_access<s::access::mode::discard_write>(cgh);

      if(Use_ndrange){
        cl::sycl::nd_range<1> ndrange (args.problem_size, args.local_size);

        cgh.parallel_for<class ScalarProdKernel<T, Use_ndrange>>(ndrange,
          [=](cl::sycl::nd_item<1> item) 
          {
            size_t gid= item.get_global_linear_id();
            intermediate_product[gid] = in1[gid] * in2[gid];
          });
      }
      else {
        cgh.parallel_for_work_group<class ScalarProdKernelHierarchical<T, Use_ndrange>>(
          cl::sycl::range<1>{args.problem_size / args.local_size},
          cl::sycl::range<1>{args.local_size},
          [=](cl::sycl::group<1> grp){
            grp.parallel_for_work_item([&](cl::sycl::h_item<1> idx){
              size_t gid = idx.get_global_id(0);
              intermediate_product[gid] = in1[gid] * in2[gid];
            });
          });
      }
    }));

    // std::cout << "Multiplication of vectors completed" << std::endl;

    auto array_size = args.problem_size;
    auto wgroup_size = args.local_size;
    // Not yet tested with more than 2
    auto elements_per_thread = 2;

    while (array_size!= 1) {
      auto n_wgroups = (array_size + wgroup_size*elements_per_thread - 1)/(wgroup_size*elements_per_thread); // two threads per work item

      events.push_back(args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {

          auto global_mem = output_buf.template get_access<s::access::mode::read_write>(cgh);
      
          // local memory for reduction
          auto local_mem = s::accessor <T, 1, s::access::mode::read_write, s::access::target::local> {s::range<1>(wgroup_size), cgh};
          cl::sycl::nd_range<1> ndrange (n_wgroups*wgroup_size, wgroup_size);
    
          if(Use_ndrange) {
            cgh.parallel_for<class ScalarProdReduction<T, Use_ndrange>>(ndrange,
            [=](cl::sycl::nd_item<1> item) 
              {
                size_t gid= item.get_global_linear_id();
                size_t lid = item.get_local_linear_id();

                // initialize local memory to 0
                local_mem[lid] = 0; 

                for(int i = 0; i < elements_per_thread; ++i) {
                  int input_element = gid + i * n_wgroups * wgroup_size;
                  
                  if(input_element < array_size)
                    local_mem[lid] += global_mem[input_element];
                }

                item.barrier(s::access::fence_space::local_space);

                for(size_t stride = wgroup_size/elements_per_thread; stride >= 1; stride /= elements_per_thread) {
                  if(lid < stride) {
                    for(int i = 0; i < elements_per_thread-1; ++i){
                      local_mem[lid] += local_mem[lid + stride + i];
                    }
                  }
                  item.barrier(s::access::fence_space::local_space);
                }
                
                // Only one work-item per work group writes to global memory 
                if (lid == 0) {
                  global_mem[item.get_global_id()] = local_mem[0];
                }
              });
          }
          else {
            cgh.parallel_for_work_group<class ScalarProdReductionHierarchical<T, Use_ndrange>>(
              cl::sycl::range<1>{n_wgroups}, cl::sycl::range<1>{wgroup_size},
              [=](cl::sycl::group<1> grp){
                
                grp.parallel_for_work_item([&](cl::sycl::h_item<1> idx){
                  const size_t gid = idx.get_global_id(0);
                  const size_t lid = idx.get_local_id(0);

                  // initialize local memory to 0
                  local_mem[lid] = 0; 

                  for(int i = 0; i < elements_per_thread; ++i) {
                    int input_element = gid + i * n_wgroups * wgroup_size;
                  
                    if(input_element < array_size)
                      local_mem[lid] += global_mem[input_element];
                  }
                });

                for(size_t stride = wgroup_size/elements_per_thread; stride >= 1; stride /= elements_per_thread) {
                  grp.parallel_for_work_item([&](cl::sycl::h_item<1> idx){
                  
                    const size_t lid = idx.get_local_id(0);
                    
                    if(lid < stride) {
                      for(int i = 0; i < elements_per_thread-1; ++i){
                        local_mem[lid] += local_mem[lid + stride + i];
                      }
                    }
                  });
                }
                grp.parallel_for_work_item([&](cl::sycl::h_item<1> idx){
                  const size_t lid = idx.get_local_id(0);
                  if(lid == 0)
                    global_mem[grp.get_id(0) * grp.get_local_range(0)] = local_mem[0];
                });
              });
          }
        }));
      
      events.push_back(args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {

          auto global_mem = output_buf.template get_access<s::access::mode::read_write>(cgh);
      
          cgh.parallel_for<ScalarProdGatherKernel>(cl::sycl::range<1>{n_wgroups},
                                                   [=](cl::sycl::id<1> idx){
            global_mem[idx] = global_mem[idx * wgroup_size];
          });
        }));
      array_size = n_wgroups;
    }
  }

  bool verify(VerificationSetting &ver) { 
    bool pass = true;
    auto expected = static_cast <T>(0);

    auto output_acc = output_buf.template get_access<s::access::mode::read>();

    for(size_t i = 0; i < args.problem_size; i++) {
        expected += input1[i] * input2[i];
    }

    //std::cout << "Scalar product on CPU =" << expected << std::endl;
    //std::cout << "Scalar product on Device =" << output[0] << std::endl;

    // Todo: update to type-specific test (Template specialization?)
    const auto tolerance = 0.00001f;
    if(std::fabs(expected - output_acc[0]) > tolerance) {
      pass = false;
    }

    return pass;
  }
  
  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "ScalarProduct_";
    name << (Use_ndrange ? "NDRange_" : "Hierarchical_");
    name << ReadableTypename<T>::name;
    return name.str();
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  if(app.shouldRunNDRangeKernels()) {
    app.run<ScalarProdBench<int, true>>();
    app.run<ScalarProdBench<long long, true>>();
    app.run<ScalarProdBench<float, true>>();
    app.run<ScalarProdBench<double, true>>();
  }

  app.run<ScalarProdBench<int, false>>();
  app.run<ScalarProdBench<long long, false>>();
  app.run<ScalarProdBench<float, false>>();
  app.run<ScalarProdBench<double, false>>();  

  return 0;
}
