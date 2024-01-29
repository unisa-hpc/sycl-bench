#include "common.h"

namespace s = sycl;
#include <iostream>

#define KERNEL_LAUNCHES_DEFAULT 50000

template <typename DATA_TYPE, bool in_order = false, bool synch = false>
class accessor_latency_kernel;
template <typename DATA_TYPE, bool in_order = false, bool synch = false>
class usm_latency_kernel;
/**
Measure Accessors latency compared to USM
The benchmark submits multiple small kernels which stress SYCL dependency tracking.
 */


template <typename DATA_TYPE, bool in_order>
class LatencyBenchmark {
protected:
  BenchmarkArgs args;
  size_t kernel_launches_num;

  LatencyBenchmark(const BenchmarkArgs& args, const size_t kernel_launches_num)
      : args(args), kernel_launches_num(kernel_launches_num) {}

  s::range<1> getRange() const { return s::range<1>{args.problem_size}; }

  s::nd_range<1> getNDRange() const {
    return s::nd_range<1>{args.problem_size, args.problem_size > 1024 ? 1024 : args.problem_size};
  }

  sycl::queue& get_queue() {
    if constexpr(in_order) {
      return args.device_queue_in_order;
    } else {
      return args.device_queue;
    }
  }
};

template <typename DATA_TYPE, bool in_order = false, bool synch = false>
class AccessorLatency : LatencyBenchmark<DATA_TYPE, in_order> {
protected:
  PrefetchedBuffer<DATA_TYPE> buff_A;
  PrefetchedBuffer<DATA_TYPE> buff_B;
  PrefetchedBuffer<DATA_TYPE> buff_C;

public:
  using base = LatencyBenchmark<DATA_TYPE, in_order>;
  using base::args;
  using base::base;
  using base::get_queue;
  using base::getNDRange;
  using base::getRange;
  using base::kernel_launches_num;

  AccessorLatency(const BenchmarkArgs& args, const size_t kernel_launches_num) : base(args, kernel_launches_num) {}

  // TODO: Problem size?
  void setup() {
    const auto range = getRange();
    buff_A.initialize(args.device_queue, range);
    buff_B.initialize(args.device_queue, range);
    buff_C.initialize(args.device_queue, range);
  }

  void run(std::vector<sycl::event>& events) {
    auto& queue = get_queue();
    for(int i = 0; i < kernel_launches_num; i++) {
      auto event = queue.submit([&](s::handler& cgh) {
        auto acc_A = buff_A.template get_access<s::access::mode::read>(cgh, buff_A.get_range());
        auto acc_B = buff_B.template get_access<s::access::mode::read>(cgh, buff_B.get_range());
        auto acc_C = buff_C.template get_access<s::access::mode::write>(cgh, buff_C.get_range());

        cgh.parallel_for<class accessor_latency_kernel<DATA_TYPE, in_order, synch>>(
            getNDRange(), [=](s::nd_item<1> item) {
              const auto id = item.get_global_linear_id();
              acc_C[id] = acc_A[id] + acc_B[id];
            });
      });
      if constexpr(synch) {
        queue.wait();
      }
      events.push_back(event);
      // swap buffers
      std::swap(buff_A, buff_B);
      std::swap(buff_A, buff_C);
    }
  }


  static std::string getBenchmarkName(BenchmarkArgs& args) {
    std::stringstream name;
    name << "SYCL2020_Accessors_Latency_";
    name << ReadableTypename<DATA_TYPE>::name << "_";
    name << (in_order ? "in_order" : "out_of_order") << "_";
    name << (synch ? "synch" : "") << "_";
    return name.str();
  }
};

template <typename DATA_TYPE, bool in_order = false, bool synch = false>
class USMLatency : LatencyBenchmark<DATA_TYPE, in_order> {
protected:
  USMBuffer<DATA_TYPE> buff_A;
  USMBuffer<DATA_TYPE> buff_B;
  USMBuffer<DATA_TYPE> buff_C;

  using base = LatencyBenchmark<DATA_TYPE, in_order>;
  using base::args;
  using base::base;
  using base::get_queue;
  using base::getNDRange;
  using base::getRange;
  using base::kernel_launches_num;

public:
  USMLatency(const BenchmarkArgs& args, const size_t kernel_launches_num) : base(args, kernel_launches_num) {}

  // TODO: Problem size?
  void setup() {
    buff_A.initialize(args.device_queue, getRange());
    buff_B.initialize(args.device_queue, getRange());
    buff_C.initialize(args.device_queue, getRange());
  }

  void run(std::vector<sycl::event>& events) {
    auto& queue = get_queue();
    sycl::event event;
    auto* acc_A = buff_A.get();
    auto* acc_B = buff_B.get();
    auto* acc_C = buff_C.get();
    for(int i = 0; i < kernel_launches_num; i++) {
      event = queue.submit([&](s::handler& cgh) {
        // Disable kernel dependencies build when queue is in_order
        if constexpr(!in_order && !synch) {
          cgh.depends_on(event);
        }
        cgh.parallel_for<class usm_latency_kernel<DATA_TYPE, in_order, synch>>(getNDRange(), [=](s::nd_item<1> item) {
          const auto id = item.get_global_linear_id();
          acc_C[id] = acc_A[id] + acc_B[id];
        });
      });
      if constexpr(synch) {
        queue.wait();
      }
      // Add kernel event to kernel's list
      events.push_back(event);
      // swap buffers to
      std::swap(buff_A, buff_B);
      std::swap(buff_A, buff_C);
    }
  }


  static std::string getBenchmarkName(BenchmarkArgs& args) {
    std::stringstream name;
    name << "USM_Latency_";
    name << ReadableTypename<DATA_TYPE>::name << "_";
    name << (in_order ? "in_order" : "out_of_order") << "_";
    name << (synch ? "synch" : "") << "_";
    return name.str();
  }
};

template <template <typename DATA_TYPE, bool in_order = false, bool synch = false> typename latency_kernel>
void launchBenchmarks(BenchmarkApp& app, const size_t kernel_launches_num) {
  app.run<latency_kernel<float>>(kernel_launches_num);       // out-of-order, no synch
  app.run<latency_kernel<float, true>>(kernel_launches_num); // in-order, no synch
  if constexpr(SYCL_BENCH_ENABLE_FP64_BENCHMARKS) {
    if(app.deviceSupportsFP64()) {
      app.run<latency_kernel<double>>(kernel_launches_num);       // out-of-order, no synch
      app.run<latency_kernel<double, true>>(kernel_launches_num); // in-order, no synch
    }
  }
}

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  const size_t kernel_launches_num = app.getArgs().cli.getOrDefault("--num-launches", KERNEL_LAUNCHES_DEFAULT);

  launchBenchmarks<AccessorLatency>(app, kernel_launches_num);
  launchBenchmarks<USMLatency>(app, kernel_launches_num);
}
