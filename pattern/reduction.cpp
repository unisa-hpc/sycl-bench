
#include "common.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <cassert>

using namespace cl;

template <typename T> class ReductionKernelNDRange;
template <typename T> class ReductionKernelHierarchical;

template <typename T>
class Reduction
{
protected:
    std::vector<T> _input;
    BenchmarkArgs _args;

    PrefetchedBuffer<T, 1> _input_buff;
    PrefetchedBuffer<T, 1> _output_buff;
    sycl::buffer<T, 1>* _final_output_buff;
    T _result;
public:
    Reduction(const BenchmarkArgs &args)
      : _args{args}
  {  
    assert(_args.problem_size % _args.local_size == 0);
  }

  void generate_input(std::vector<T>& out)
  {
    out.resize(_args.problem_size);
    for(std::size_t i = 0; i < out.size(); ++i)
      out[i] = static_cast<T>(i);
  }

  void setup() {
    generate_input(_input);

    _input_buff.initialize(_args.device_queue, static_cast<const T*>(_input.data()), sycl::range<1>(_args.problem_size));
    _output_buff.initialize(_args.device_queue, sycl::range<1>{_args.problem_size});
  }


  void submit_ndrange(std::vector<cl::sycl::event>& events){
    this->submit([this, &events](sycl::buffer<T, 1> *input, sycl::buffer<T, 1> *output,
                        const size_t reduction_size, const size_t num_groups) {
      events.push_back(this->local_reduce_ndrange(input, output, reduction_size, num_groups));
    });
  }

  void submit_hierarchical(std::vector<cl::sycl::event>& events){
    this->submit([this, &events](sycl::buffer<T, 1> *input, sycl::buffer<T, 1> *output,
                        const size_t reduction_size, const size_t num_groups) {
      events.push_back(this->local_reduce_hierarchical(input, output, reduction_size,
                                      num_groups));
    });
  }

  bool verify(VerificationSetting &ver) {
    T result = _final_output_buff->template get_access<sycl::access::mode::read>(
        sycl::range<1>{0}, sycl::id<1>{1})[0];

    // Calculate CPU result in fp64 to avoid obtaining a wrong verification result
    std::vector<double> input_fp64(_input.size());
    for(std::size_t i = 0; i < _input.size(); ++i)
      input_fp64[i] = static_cast<double>(_input[i]);

    double delta =
        static_cast<double>(result) - std::accumulate(input_fp64.begin(), input_fp64.end(), T{});
    
    return std::abs(delta) < 1.e-5;
  }
private:
  template<class Kernel_invocation_function>
  void submit(Kernel_invocation_function kernel)
  {
    sycl::buffer<T, 1>* input_buff = &_input_buff.get();
    sycl::buffer<T, 1>* output_buff = &_output_buff.get();

    size_t current_reduction_size = _args.problem_size;
    size_t current_num_groups = _args.problem_size / _args.local_size;

    do {
      // invoke local reduction
      kernel(input_buff, output_buff, current_reduction_size,
             current_num_groups);

      current_reduction_size = current_num_groups;
      if(current_num_groups > 1)
        current_num_groups = 
          (current_reduction_size + _args.local_size - 1) / _args.local_size;
      else
        // This was the final iteration
        current_num_groups = 0;

      // Only swap if it's not the final iteration
      if(current_num_groups > 0)
        std::swap(input_buff, output_buff);

    } while(current_num_groups > 0);
    
    _final_output_buff = output_buff;
  }

  sycl::event local_reduce_ndrange(
    sycl::buffer<T,1>* input, sycl::buffer<T,1>* output,
    const size_t reduction_size, const std::size_t num_groups)
  {
    return _args.device_queue.submit([&](sycl::handler &cgh) {

      sycl::nd_range<1> ndrange{num_groups * _args.local_size,
                                _args.local_size};

      using namespace cl::sycl::access;

      auto acc     = input->template get_access<mode::read>(cgh);
      auto acc_out = output->template get_access<mode::discard_write>(cgh);
      auto scratch = sycl::accessor<T, 1, mode::read_write, target::local>
        {_args.local_size, cgh};

      const int group_size = _args.local_size;

      cgh.parallel_for<ReductionKernelNDRange<T>>(
        ndrange,
        [=](sycl::nd_item<1> item) {
          
          const int lid = item.get_local_id(0);
          const auto gid = item.get_global_id();

          scratch[lid] = (gid[0] < reduction_size) ? acc[gid] : 0;
          
          for(int i = group_size/2; i > 0; i /= 2) {

            item.barrier();
            if(lid < i) 
              scratch[lid] += scratch[lid + i];

          }
          if(lid == 0)
            acc_out[item.get_group(0)] = scratch[0];
        });
    }); // submit
  }

  sycl::event local_reduce_hierarchical(
    sycl::buffer<T,1>* input, sycl::buffer<T,1>* output, 
    const size_t reduction_size, const std::size_t num_groups)
  {
    return _args.device_queue.submit(
        [&](sycl::handler& cgh) {

      using namespace sycl::access;

      auto acc     = input->template get_access<mode::read>(cgh);
      auto acc_out = output->template get_access<mode::discard_write>(cgh);

      auto scratch = sycl::accessor<T, 1, mode::read_write, target::local>
        {_args.local_size, cgh};

      const int group_size = _args.local_size;

      cgh.parallel_for_work_group<ReductionKernelHierarchical<T>>(
        sycl::range<1>{num_groups},
        sycl::range<1>{_args.local_size},
        [=](sycl::group<1> grp) {

          grp.parallel_for_work_item([&](sycl::h_item<1> idx){
            const int lid = idx.get_local_id(0);
            const auto gid = idx.get_global_id();

            scratch[lid] = (gid[0] < reduction_size) ? acc[gid] : 0;
          });
        
          for(int i = group_size/2; i > 0; i /= 2) {
            grp.parallel_for_work_item([&](sycl::h_item<1> idx){
              const int lid = idx.get_local_id(0);

              if (lid < i) 
                scratch[lid] += scratch[lid + i];
            });
          }

          // Spawn another parallel_for_work_item to work around
          // limitations in hipSYCL device implementation of
          // hierarchical parallel for
          grp.parallel_for_work_item([&](sycl::h_item<1> idx){
            if(idx.get_local_id(0) == 0)
              acc_out[grp.get_id(0)] = scratch[0];
          });

        });
    }); // submit
  }
 
};

template<class T>
class ReductionNDRange : public Reduction<T>
{
public:
  ReductionNDRange(const BenchmarkArgs &args)
  : Reduction<T>{args}
  {}

  void run(std::vector<cl::sycl::event>& events){
    this->submit_ndrange(events);
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "Pattern_Reduction_NDRange_";
    name << ReadableTypename<T>::name;
    return name.str();
  }
};

template<class T>
class ReductionHierarchical : public Reduction<T>
{
public:
  ReductionHierarchical(const BenchmarkArgs &args)
  : Reduction<T>{args}
  {}

  void run(std::vector<cl::sycl::event>& events){
    this->submit_hierarchical(events);
    // Waiting is not necessary as the BenchmarkManager will already call
    // wait_and_throw() here
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "Pattern_Reduction_Hierarchical_";
    name << ReadableTypename<T>::name;
    return name.str();
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);

  // Using short will lead to overflow even for
  // small problem sizes
  //app.run< ReductionNDRange<short>>();
  if(app.shouldRunNDRangeKernels()){
    app.run< ReductionNDRange<int>>();
    app.run< ReductionNDRange<long long>>();
    app.run< ReductionNDRange<float>>();
    app.run< ReductionNDRange<double>>();
  }
  //app.run< ReductionHierarchical<short>>();
  app.run< ReductionHierarchical<int>>();
  app.run< ReductionHierarchical<long long>>();
  app.run< ReductionHierarchical<float>>();
  app.run< ReductionHierarchical<double>>();

  return 0;
}



