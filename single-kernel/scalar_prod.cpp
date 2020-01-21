#include "common.h"

#include <iostream>

//using namespace cl::sycl;
namespace s = cl::sycl;

template<bool>
class ScalarProdKernel;
template<bool>
class ScalarProdKernelHierarchical;

template<bool>
class ScalarProdReduction;
template<bool>
class ScalarProdReductionHierarchical;

template<bool Use_ndrange = true>
class ScalarProdBench
{
protected:    
    std::vector<int> input1;
    std::vector<int> input2;
    std::vector<int> output;
    BenchmarkArgs args;

public:
  ScalarProdBench(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {      
    // host memory allocation and initialization
    input1.resize(args.problem_size);
    input2.resize(args.problem_size);
    output.resize(args.problem_size, 0);

    for (size_t i =0; i < args.problem_size; i++) {
      input1[i] = i;
      input2[i] = i;
    }

  }

  void run() {    
    s::buffer<int, 1> input1_buf(input1.data(), s::range<1>(args.problem_size));
    s::buffer<int, 1> input2_buf(input2.data(), s::range<1>(args.problem_size));
    s::buffer<int, 1> output_buf(output.data(), s::range<1>(args.problem_size));
    
    args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {
      auto in1 = input1_buf.get_access<s::access::mode::read>(cgh);
      auto in2 = input2_buf.get_access<s::access::mode::read>(cgh);
      // Use discard_write here, otherwise the content of the hostbuffer must first be copied to device
      auto intermediate_product = output_buf.get_access<s::access::mode::discard_write>(cgh);

      if(Use_ndrange){
        cl::sycl::nd_range<1> ndrange (args.problem_size, args.local_size);

        cgh.parallel_for<class ScalarProdKernel<Use_ndrange>>(ndrange,
          [=](cl::sycl::nd_item<1> item) 
          {
            size_t gid= item.get_global_linear_id();
            intermediate_product[gid] = in1[gid] * in2[gid];
          });
      }
      else {
        cgh.parallel_for_work_group<class ScalarProdKernelHierarchical<Use_ndrange>>(
          cl::sycl::range<1>{args.problem_size / args.local_size},
          cl::sycl::range<1>{args.local_size},
          [=](cl::sycl::group<1> grp){
            grp.parallel_for_work_item([&](cl::sycl::h_item<1> idx){
              size_t gid = idx.get_global_id(0);
              intermediate_product[gid] = in1[gid] * in2[gid];
            });
          });
      }
    });

    // std::cout << "Multiplication of vectors completed" << std::endl;

    auto array_size = args.problem_size;
    auto wgroup_size = args.local_size;
    // Not yet tested with more than 2
    auto elements_per_thread = 2;

    while (array_size!= 1) {
      auto n_wgroups = (array_size + wgroup_size*elements_per_thread - 1)/(wgroup_size*elements_per_thread); // two threads per work item

      args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {

          auto global_mem = output_buf.get_access<s::access::mode::read_write>(cgh);
      
          // local memory for reduction
          auto local_mem = s::accessor <int, 1, s::access::mode::read_write, s::access::target::local> {s::range<1>(wgroup_size), cgh};
          cl::sycl::nd_range<1> ndrange (n_wgroups*wgroup_size, wgroup_size);
    
          if(Use_ndrange) {
            cgh.parallel_for<class ScalarProdReduction<Use_ndrange>>(ndrange,
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
          }
          else {
            cgh.parallel_for_work_group<class ScalarProdReductionHierarchical<Use_ndrange>>(
              cl::sycl::range<1>{n_wgroups}, cl::sycl::range<1>{wgroup_size},
              [=](cl::sycl::group<1> grp){
                grp.parallel_for_work_item([&](cl::sycl::h_item<1> idx){
                  const size_t gid = idx.get_global_id(0);
                  const size_t lid = idx.get_local_id(0);

                  // initialize local memory to 0
                  local_mem[lid] = 0; 

                  if ((elements_per_thread * gid) < array_size) {
                    local_mem[lid] = global_mem[elements_per_thread * gid] +
                                     global_mem[elements_per_thread * gid + 1];
                  }
                });
                for (size_t stride = 1; stride < wgroup_size; stride *= elements_per_thread) {
                  grp.parallel_for_work_item([&](cl::sycl::h_item<1> idx){
                  
                    const size_t lid = idx.get_local_id(0);
                    auto local_mem_index = elements_per_thread * stride * lid;

                    if (local_mem_index < wgroup_size) {
                      local_mem[local_mem_index] =
                          local_mem[local_mem_index] +
                          local_mem[local_mem_index + stride];
                    }
                  });
                }
                grp.parallel_for_work_item([&](cl::sycl::h_item<1> idx){
                  const size_t lid = idx.get_local_id(0);
                  global_mem[grp.get_id(0)] = local_mem[0];
                });
              });
          }
        });

      array_size = n_wgroups;
    }
  }

  bool verify(VerificationSetting &ver) { 
    bool pass = true;
    int expected = 0;

    for(size_t i = 0; i < args.problem_size; i++) {
        expected += input1[i] * input2[i];
    }

    std::cout << "Scalar product on CPU =" << expected << std::endl;
    std::cout << "Scalar product on Device =" << output[0] << std::endl;
    
    if(expected != output[0]) {
      pass = false;
    }
    return pass;
  }
  
  static std::string getBenchmarkName() {
    if(Use_ndrange)
      return "ScalarProduct_NDRange";
    else
      return "ScalarProduct_Hierarchical";
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  if(app.shouldRunNDRangeKernels())
    app.run<ScalarProdBench<true>>();
  
  app.run<ScalarProdBench<false>>();
  return 0;
}
