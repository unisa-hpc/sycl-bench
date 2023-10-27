#include "common.h"
#include <iostream>

namespace s = sycl;

template <typename T, typename Op>
class KernelReductionBench {
  int problem_size = 1;
  BenchmarkArgs args;
  PrefetchedBuffer<T, 1> in_buf;
  PrefetchedBuffer<T, 1> out_buf;
  std::vector<T> in_vec;
  T reduction_results;

public:
  KernelReductionBench(const BenchmarkArgs& _args) : args(_args) {}

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
// Update reduction variables to SYCL 2020 spec behavior #578: https://github.com/AdaptiveCpp/AdaptiveCpp/pull/578
#ifdef __ACPP__
      auto r = s::reduction(out_buf.template get_access<s::access_mode::read_write>(cgh), Op());
#else
      auto r = s::reduction(out_buf.get(), cgh, Op());
#endif
      auto in_acc = in_buf.template get_access<s::access_mode::read>(cgh);
      cgh.parallel_for(s::range<1>{problem_size}, r, [=](s::id<1> idx, auto& op) { op.combine(in_acc[idx]); });
    }));
  }
  bool verify(VerificationSetting& ver) {
    auto results = out_buf.get_host_access();
    T verified_results = 0;
    verified_results = std::reduce(in_vec.data(), in_vec.data() + problem_size, 0, Op());

    if(results[0] == verified_results)
      return true;
    else
      return false;
  }


  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "KernelReduction_";
    name << ReadableTypename<T>::name;
    if constexpr(std::is_same<Op, sycl::plus<T>>::value) {
      name << "_plus";
    } else {
      name << "_multiplies";
    }
    return name.str();
  }
};

template <typename T>
void runOperators(BenchmarkApp& app) {
  app.run<KernelReductionBench<T, sycl::plus<T>>>();
  app.run<KernelReductionBench<T, sycl::multiplies<T>>>();
}

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  runOperators<int>(app);
  runOperators<long long>(app);
  runOperators<float>(app);
  if(app.deviceSupportsFP64())
    runOperators<double>(app);
  return 0;
}
