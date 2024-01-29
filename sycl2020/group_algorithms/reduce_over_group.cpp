#include "common.h"
#include "polybenchUtilFuncts.h"
#include <iostream>

namespace s = sycl;

template <typename T>
class ReduceGroupAlgorithm {
  int problem_size = 1;
  int local_size = 1;
  BenchmarkArgs args;
  PrefetchedBuffer<T, 1> in_buf;
  PrefetchedBuffer<T, 1> out_buf;
  std::vector<T> in_vec;
  T reduction_results;

public:
  ReduceGroupAlgorithm(const BenchmarkArgs& _args) : args(_args) {}

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
      const int group_size = local_size;

      auto ndrange = s::nd_range<1>{problem_size, local_size};

      cgh.parallel_for<ReduceGroupAlgorithm<T>>(ndrange, [=](sycl::nd_item<1> item) {
        const int lid = item.get_local_id(0);
        const auto gid = item.get_global_id();

        T partial_sum = s::reduce_over_group(item.get_group(), in_acc[gid], s::plus<T>());

        s::atomic_ref<T, s::memory_order::relaxed, s::memory_scope::device, s::access::address_space::global_space> atm(
            out_acc[0]);
        if(lid == 0) {
          atm.fetch_add(partial_sum);
        }
      });
    }));
  }
  bool verify(VerificationSetting& ver) {
    auto results = out_buf.get_host_access();
    constexpr auto ERROR_THRESHOLD = 0.05;

    T verified_results = problem_size;

    if(percentDiff(results[0], verified_results) > ERROR_THRESHOLD) {
      std::cerr << "output: " << results[0] << " correct output: " << verified_results << std::endl;
      return false;
    } else
      return true;
  }


  static std::string getBenchmarkName(BenchmarkArgs& args) {
    std::stringstream name;
    name << "ReduceGroupAlgorithm_";
    name << ReadableTypename<T>::name;

    return name.str();
  }
};


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<ReduceGroupAlgorithm<int>>();
  app.run<ReduceGroupAlgorithm<long long>>();
  app.run<ReduceGroupAlgorithm<float>>();
  if(app.deviceSupportsFP64())
    app.run<ReduceGroupAlgorithm<double>>();
  return 0;
}
