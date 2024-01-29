#include "common.h"
#include "polybenchUtilFuncts.h"
#include <iostream>

namespace s = sycl;

template <typename T>
class ReductionAtomic {
  int problem_size = 1;
  BenchmarkArgs args;
  PrefetchedBuffer<T, 1> in_buf;
  PrefetchedBuffer<T, 1> out_buf;
  std::vector<T> in_vec;
  T reduction_results;

public:
  ReductionAtomic(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    problem_size = args.problem_size;
    in_vec.resize(problem_size);

    std::fill(in_vec.begin(), in_vec.end(), 1);

    reduction_results = 0.f;
    in_buf.initialize(args.device_queue, in_vec.data(), s::range<1>{in_vec.size()});
    out_buf.initialize(args.device_queue, &reduction_results, s::range<1>{1});
  }
  void run(std::vector<s::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
      auto in_acc = in_buf.template get_access<s::access_mode::read>(cgh);
      auto out_acc = out_buf.template get_access<s::access_mode::write>(cgh);
      auto ndrange = s::nd_range<1>{problem_size, args.local_size};
      cgh.parallel_for<ReductionAtomic<T>>(ndrange, [=](sycl::nd_item<1> it) {
        const auto gid = it.get_global_id();

        s::atomic_ref<T, s::memory_order::relaxed, s::memory_scope::device, s::access::address_space::global_space> atm(
            out_acc[0]);

        atm.fetch_add(in_acc[gid]);
      });
    }));
  }
  bool verify(VerificationSetting& ver) {
    auto results = out_buf.get_host_access();
    constexpr auto ERROR_THRESHOLD = 0.05f;

    T verified_results = 0;
    for(int i = 0; i < in_vec.size(); i++) verified_results += in_vec[i];

    if(percentDiff(results[0], verified_results) > ERROR_THRESHOLD) {
      return false;
    } else
      return true;
  }


  static std::string getBenchmarkName(BenchmarkArgs& args) {
    std::stringstream name;
    name << "ReductionAtomic_";
    name << ReadableTypename<T>::name;

    return name.str();
  }
};


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<ReductionAtomic<int>>();
  app.run<ReductionAtomic<long long>>();
  app.run<ReductionAtomic<float>>();
  if(app.deviceSupportsFP64())
    app.run<ReductionAtomic<double>>();
  return 0;
}
