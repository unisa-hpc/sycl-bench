
#include "common.h"

#include <cassert>
#include <iostream>
#include <vector>

using namespace sycl;

template <typename T>
class ReductionKernelNDRange;
template <typename T>
class ReductionKernelHierarchical;

template <typename T>
class SegmentedReduction {
protected:
  std::vector<T> _input;
  BenchmarkArgs _args;
  PrefetchedBuffer<T, 1> _buff;

public:
  SegmentedReduction(const BenchmarkArgs& args) : _args{args} { assert(_args.problem_size % _args.local_size == 0); }

  void generate_input(std::vector<T>& out) {
    out.resize(_args.problem_size);
    for(std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<T>(i);
  }

  void setup() {
    generate_input(_input);
    _buff.initialize(_args.device_queue, _input.data(), sycl::range<1>(_args.problem_size));
  }

  void submit_ndrange(std::vector<sycl::event>& events) {
    events.push_back(_args.device_queue.submit([&](sycl::handler& cgh) {
      sycl::nd_range<1> ndrange{_args.problem_size, _args.local_size};

      using namespace sycl::access;

      auto acc = _buff.template get_access<mode::read_write>(cgh);
      auto scratch = sycl::local_accessor<T, 1>{_args.local_size, cgh};

      const int group_size = _args.local_size;

      cgh.parallel_for<ReductionKernelNDRange<T>>(ndrange, [=](sycl::nd_item<1> item) {
        const int lid = item.get_local_id(0);
        const auto gid = item.get_global_id();

        scratch[lid] = acc[gid];

        for(int i = group_size / 2; i > 0; i /= 2) {
          sycl::group_barrier(item.get_group());
          if(lid < i)
            scratch[lid] += scratch[lid + i];
        }
        if(lid == 0)
          acc[gid] = scratch[0];
      });
    })); // submit
  }

  void submit_hierarchical(std::vector<sycl::event>& events) {
    events.push_back(_args.device_queue.submit([&](sycl::handler& cgh) {
      using namespace sycl::access;

      auto acc = _buff.template get_access<mode::read_write>(cgh);
      auto scratch = sycl::local_accessor<T, 1>{_args.local_size, cgh};

      const int group_size = _args.local_size;

      cgh.parallel_for_work_group<ReductionKernelHierarchical<T>>(sycl::range<1>{_args.problem_size / _args.local_size},
          sycl::range<1>{_args.local_size}, [=](sycl::group<1> grp) {
            grp.parallel_for_work_item([&](sycl::h_item<1> idx) {
              const int lid = idx.get_local_id(0);
              const auto gid = idx.get_global_id();

              scratch[lid] = acc[gid];
            });

            for(int i = group_size / 2; i > 0; i /= 2) {
              grp.parallel_for_work_item([&](sycl::h_item<1> idx) {
                const int lid = idx.get_local_id(0);

                if(lid < i)
                  scratch[lid] += scratch[lid + i];
              });
            }

            grp.parallel_for_work_item([&](sycl::h_item<1> idx) {
              if(idx.get_local_id(0) == 0)
                acc[idx.get_global_id()] = scratch[0];
            });
          });
    })); // submit
  }

  bool verify(VerificationSetting& ver) {
    std::vector<T> original_input;
    generate_input(original_input);

    auto acc = _buff.get_host_access();
    size_t num_groups = _args.problem_size / _args.local_size;

    for(size_t group = 0; group < num_groups; ++group) {
      size_t group_offset = group * _args.local_size;
      T sum = 0;

      for(size_t local_id = 0; local_id < _args.local_size; ++local_id) {
        sum += original_input[group_offset + local_id];
      }
      for(size_t local_id = 0; local_id < _args.local_size; ++local_id) {
        if(local_id == 0) {
          if(acc[group_offset + local_id] != sum)
            return false;
        } else {
          if(acc[group_offset + local_id] != original_input[group_offset + local_id])
            return false;
        }
      }
    }

    return true;
  }
};

template <class T>
class SegmentedReductionNDRange : public SegmentedReduction<T> {
public:
  SegmentedReductionNDRange(const BenchmarkArgs& args) : SegmentedReduction<T>{args} {}

  void run(std::vector<sycl::event>& events) {
    this->submit_ndrange(events);
    // Waiting is not necessary as the BenchmarkManager will already call
    // wait_and_throw() here
  }

  static std::string getBenchmarkName(BenchmarkArgs& args) {
    std::stringstream name;
    name << "Pattern_SegmentedReduction_NDRange_";
    name << ReadableTypename<T>::name;
    return name.str();
  }
};

template <class T>
class SegmentedReductionHierarchical : public SegmentedReduction<T> {
public:
  SegmentedReductionHierarchical(const BenchmarkArgs& args) : SegmentedReduction<T>{args} {}

  void run(std::vector<sycl::event>& events) {
    this->submit_hierarchical(events);
    // Waiting is not necessary as the BenchmarkManager will already call
    // wait_and_throw() here
  }

  static std::string getBenchmarkName(BenchmarkArgs& args) {
    std::stringstream name;
    name << "Pattern_SegmentedReduction_Hierarchical_";
    name << ReadableTypename<T>::name;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  if(app.shouldRunNDRangeKernels()) {
    app.run<SegmentedReductionNDRange<short>>();
    app.run<SegmentedReductionNDRange<int>>();
    app.run<SegmentedReductionNDRange<long long>>();
    app.run<SegmentedReductionNDRange<float>>();
    if constexpr(SYCL_BENCH_ENABLE_FP64_BENCHMARKS) {
      app.run<SegmentedReductionNDRange<double>>();
    }
  }

  app.run<SegmentedReductionHierarchical<short>>();
  app.run<SegmentedReductionHierarchical<int>>();
  app.run<SegmentedReductionHierarchical<long long>>();
  app.run<SegmentedReductionHierarchical<float>>();
  if constexpr(SYCL_BENCH_ENABLE_FP64_BENCHMARKS) {
    app.run<SegmentedReductionHierarchical<double>>();
  }
  return 0;
}
