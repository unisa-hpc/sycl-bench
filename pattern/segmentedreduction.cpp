
#include "common.h"

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

#ifdef SYCL_BENCH_ALLOW_ROCPRIM
#if defined(HIPSYCL_PLATFORM_HCC)
#define ENABLE_ROCPRIM
#endif
#endif

#ifdef ENABLE_ROCPRIM
#include <rocprim/rocprim.hpp>

template<class T, int blocksize>
__host__ __device__
T rocprim_local_reduction(T value)
{
  T output{};
#ifdef SYCL_DEVICE_ONLY
  rocprim::block_reduce<T,blocksize>{}.reduce(value, output);
#endif
  return output;
}

#endif

using namespace cl;

template <typename T> class ReductionKernelNDRange;
template <typename T> class ReductionKernelHierarchical;

template <typename T>
class SegmentedReduction
{
protected:
    std::vector<T> _input;
    BenchmarkArgs _args;
    PrefetchedBuffer<T, 1> _buff;
public:
  SegmentedReduction(const BenchmarkArgs &args)
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
    _buff.initialize(_args.device_queue,_input.data(), sycl::range<1>(_args.problem_size));
  }

  void submit_ndrange(std::vector<cl::sycl::event>& events){
    
    events.push_back(_args.device_queue.submit(
        [&](sycl::handler& cgh) {

      sycl::nd_range<1> ndrange {_args.problem_size, _args.local_size};

      using namespace cl::sycl::access;

      auto acc = _buff.template get_access<mode::read_write>(cgh);

#ifdef ENABLE_ROCPRIM
      cgh.parallel_for<ReductionKernelNDRange<T>>(
		      ndrange, [=](sycl::nd_item<1> item){
          T value = acc[item.get_global_id(0)];
          
	  const int blocksize = item.get_local_range(0);
	  T result {};
	  if     (blocksize==32)   result = rocprim_local_reduction<T,32>(value);
	  else if(blocksize==64)   result = rocprim_local_reduction<T,64>(value);
	  else if(blocksize==128)  result = rocprim_local_reduction<T,128>(value);
	  else if(blocksize==256)  result = rocprim_local_reduction<T,256>(value);
	  else if(blocksize==512)  result = rocprim_local_reduction<T,512>(value);
	  else if(blocksize==1024) result = rocprim_local_reduction<T,1024>(value);
          
	  if(item.get_local_id(0)==0)
	    acc[item.get_global_id(0)] = result;
        });
#else
      auto scratch = sycl::accessor<T, 1, mode::read_write, target::local>
        {_args.local_size, cgh};

      const int group_size = _args.local_size;

      cgh.parallel_for<ReductionKernelNDRange<T>>(
        ndrange,
        [=](sycl::nd_item<1> item) {
          
          const int lid = item.get_local_id(0);
          const auto gid = item.get_global_id();

          scratch[lid] = acc[gid];

          for(int i = group_size/2; i > 0; i /= 2) {

            item.barrier();
            if(lid < i) 
              scratch[lid] += scratch[lid + i];

          }
          if(lid == 0) 
            acc[gid] = scratch[0];
        });
#endif
    })); // submit
  }

  void submit_hierarchical(std::vector<cl::sycl::event>& events){

    events.push_back(_args.device_queue.submit(
        [&](sycl::handler& cgh) {

      using namespace sycl::access;

      auto acc = _buff.template get_access<mode::read_write>(cgh);
      auto scratch = sycl::accessor<T, 1, mode::read_write, target::local>
        {_args.local_size, cgh};

      const int group_size = _args.local_size;

      cgh.parallel_for_work_group<ReductionKernelHierarchical<T>>(
        sycl::range<1>{_args.problem_size / _args.local_size},
        sycl::range<1>{_args.local_size},
        [=](sycl::group<1> grp) {

          grp.parallel_for_work_item([&](sycl::h_item<1> idx){
            const int lid = idx.get_local_id(0);
            const auto gid = idx.get_global_id();

            scratch[lid] = acc[gid];
          });
        
          for(int i = group_size/2; i > 0; i /= 2) {
            grp.parallel_for_work_item([&](sycl::h_item<1> idx){
              const int lid = idx.get_local_id(0);

              if (lid < i) 
                scratch[lid] += scratch[lid + i];
            });
          }

          grp.parallel_for_work_item([&](sycl::h_item<1> idx){
            if(idx.get_local_id(0) == 0)
              acc[idx.get_global_id()] = scratch[0];
          });
        });
    })); // submit
  }

  bool verify(VerificationSetting &ver) {
    std::vector<T> original_input;
    generate_input(original_input);

    auto acc = _buff.template get_access<sycl::access::mode::read>();
    size_t num_groups = _args.problem_size / _args.local_size;

    for(size_t group = 0; group < num_groups; ++group) {
      
      size_t group_offset = group * _args.local_size;
      T sum = 0;

      for(size_t local_id = 0; local_id < _args.local_size; ++local_id) {
        sum += original_input[group_offset + local_id];
      }
      for(size_t local_id = 0; local_id < _args.local_size; ++local_id) {
        if(local_id == 0) {
          double delta = std::abs(acc[group_offset+local_id]-sum);
          if(delta > 1.e-4){
            //return false;
          }
        } else {
          if(acc[group_offset + local_id] != original_input[group_offset + local_id])
            return false;
        }
      }
    }

    return true;
  }

 
};

template<class T>
class SegmentedReductionNDRange : public SegmentedReduction<T>
{
public:
  SegmentedReductionNDRange(const BenchmarkArgs &args)
  : SegmentedReduction<T>{args}
  {}

  void run(std::vector<cl::sycl::event>& events){
    this->submit_ndrange(events);
    // Waiting is not necessary as the BenchmarkManager will already call
    // wait_and_throw() here
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "Pattern_SegmentedReduction_NDRange_";
    name << ReadableTypename<T>::name;
    return name.str();
  }
};

template<class T>
class SegmentedReductionHierarchical : public SegmentedReduction<T>
{
public:
  SegmentedReductionHierarchical(const BenchmarkArgs &args)
  : SegmentedReduction<T>{args}
  {}

  void run(std::vector<cl::sycl::event>& events){
    this->submit_hierarchical(events);
    // Waiting is not necessary as the BenchmarkManager will already call
    // wait_and_throw() here
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "Pattern_SegmentedReduction_Hierarchical_";
    name << ReadableTypename<T>::name;
    return name.str();
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);

  if(app.shouldRunNDRangeKernels()) {
    app.run< SegmentedReductionNDRange<short>>();
    app.run< SegmentedReductionNDRange<int>>();
    app.run< SegmentedReductionNDRange<long long>>();
    app.run< SegmentedReductionNDRange<float>>();
    app.run< SegmentedReductionNDRange<double>>();
  }

  app.run< SegmentedReductionHierarchical<short>>();
  app.run< SegmentedReductionHierarchical<int>>();
  app.run< SegmentedReductionHierarchical<long long>>();
  app.run< SegmentedReductionHierarchical<float>>();
  app.run< SegmentedReductionHierarchical<double>>();

  return 0;
}



