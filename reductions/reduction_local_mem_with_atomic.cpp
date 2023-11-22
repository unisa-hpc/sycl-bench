#include "common.h"
#include <iostream>

namespace s = sycl;

template <typename T>
class ReductionLocalMem {
  int problem_size = 1;
  int local_size = 1;
  BenchmarkArgs args;
  PrefetchedBuffer<T, 1> in_buf;
  PrefetchedBuffer<T, 1> out_buf;
  std::vector<T> in_vec;
  T reduction_results;

public:
  ReductionLocalMem(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    problem_size = args.problem_size;
    local_size = args.local_size;
    in_vec.resize(problem_size);
    std::fill(in_vec.begin(), in_vec.end(), 1);
    reduction_results = 0;
    in_buf.initialize(args.device_queue, in_vec.data(), s::range<1>{in_vec.size()});
    out_buf.initialize(args.device_queue, &reduction_results, s::range<1>{1});
  }
  void run(std::vector<s::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
      auto in_acc = in_buf.template get_access<s::access_mode::read>(cgh);
      auto out_acc = out_buf.template get_access<s::access_mode::write>(cgh);
      auto scratch = sycl::local_accessor<T, 1>{local_size, cgh};
      const int group_size = local_size;

      auto ndrange = s::nd_range<1>{problem_size, local_size};

      cgh.parallel_for<ReductionLocalMem<T>>(ndrange, [=](sycl::nd_item<1> item) {
        const int lid = item.get_local_id(0);
        const auto gid = item.get_global_id();

        scratch[lid] = (gid[0] < item.get_global_range(0)) ? in_acc[gid] : 0;

        for(int i = group_size / 2; i > 0; i /= 2) {
          sycl::group_barrier(item.get_group());
          if(lid < i)
            scratch[lid] += scratch[lid + i];
        }

        // Implement atomic reduction: each WI0 in a WG should write in the out_acc using atomic operation
        s::atomic_ref<T, s::memory_order::relaxed, s::memory_scope::device, s::access::address_space::global_space> atm(
            out_acc[0]);
        if(lid == 0) {
          atm.fetch_add(scratch[0]);
        }
      });
    }));
  }
  bool verify(VerificationSetting& ver) {
    auto results = out_buf.get_host_access();
    T verified_results = 0;
    verified_results = std::reduce(in_vec.data(), in_vec.data() + problem_size, 0, std::plus<T>());

    if(results[0] == verified_results)
      return true;
    else
      return false;
  }


  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "ReductionLocalMem_";
    name << ReadableTypename<T>::name;

    return name.str();
  }
};


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<ReductionLocalMem<int>>();
  app.run<ReductionLocalMem<long long>>();
  app.run<ReductionLocalMem<float>>();
  if(app.deviceSupportsFP64())
    app.run<ReductionLocalMem<double>>();
  return 0;
}
