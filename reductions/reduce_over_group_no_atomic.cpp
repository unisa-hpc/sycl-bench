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
  std::vector<T> out_vec;

public:
  ReduceGroupAlgorithm(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    problem_size = args.problem_size;
    local_size = args.local_size;
    in_vec.resize(problem_size);
    out_vec.resize(problem_size / local_size);

    std::fill(in_vec.begin(), in_vec.end(), 1);
    in_buf.initialize(args.device_queue, in_vec.data(), s::range<1>{in_vec.size()});
    out_buf.initialize(args.device_queue, out_vec.data(), s::range<1>{out_vec.size()});
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
        const auto wg_id = item.get_group().get_group_id();
        T partial_sum = s::reduce_over_group(item.get_group(), in_acc[gid], s::plus<T>());

        if(lid == 0)
          out_acc[wg_id] = partial_sum;
      });
    }));
  }
  bool verify(VerificationSetting& ver) {
    auto results = out_buf.get_host_access();
    constexpr auto ERROR_THRESHOLD = 0.05;

    T verified_results = local_size;
    for(int i = 0; i < problem_size / local_size; i++) {
      T diff = percentDiff(results[i], verified_results);
      if(diff > ERROR_THRESHOLD)
        return false;
    }
    return true;
  }


  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "ReduceGroupAlgorithmNoAtomic_";
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
