#include "common.h"
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
    reduction_results = 0;
    in_buf.initialize(args.device_queue, in_vec.data(), s::range<1>{in_vec.size()});
    out_buf.initialize(args.device_queue, &reduction_results, s::range<1>{1});
  }
  void run(std::vector<s::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
      auto in_acc = in_buf.template get_access<s::access_mode::read>(cgh);
      auto out_acc = out_buf.template get_access<s::access_mode::write>(cgh);


      cgh.parallel_for<ReductionAtomic<T>>(problem_size, [=](sycl::id<1> id) {
        const auto gid = id;

        // Implement atomic reduction: each WI0 in a WG should write in the out_acc using atomic operation
        s::atomic_ref<T, s::memory_order::relaxed, s::memory_scope::device, s::access::address_space::global_space> atm(
            out_acc[0]);

        atm.fetch_add(in_acc[gid]);
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
