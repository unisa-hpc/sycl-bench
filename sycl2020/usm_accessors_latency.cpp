#include "common.h"

#include <iostream>

namespace s = cl::sycl;

// TODO: As kernel parameter
#define NUM_KERNELS 50000

/**
Measure Accessors latency compared to USM
The benchmark submits multiple small kernels which stress SYCL dependency tracking.
 */


template <typename DATA_TYPE, std::size_t dim>
class LatencyBenchmark {
protected:
  BenchmarkArgs args;

  LatencyBenchmark(const BenchmarkArgs& args) : args(args) {}

  s::range<dim> getRange() const {
    if constexpr(dim == 1) {
      return s::range<1>{args.problem_size};
    } else if constexpr(dim == 2) {
      return s::range<2>{args.problem_size, 1};
    } else if constexpr(dim == 3) {
      return s::range<3>{args.problem_size, 1, 1};
    } else {
      // Not reachable
      throw std::invalid_argument("Illegal dim provided");
    }
  }
};

template <typename DATA_TYPE, int dim = 1, bool use_id = true>
class AccessorLatency : LatencyBenchmark<DATA_TYPE, dim> {
protected:
  PrefetchedBuffer<DATA_TYPE, dim> buff_A;
  PrefetchedBuffer<DATA_TYPE, dim> buff_B;
  PrefetchedBuffer<DATA_TYPE, dim> buff_C;

public:
  using base = LatencyBenchmark<DATA_TYPE, dim>;
  using base::args;
  using base::base;
  using base::getRange;

  AccessorLatency(const BenchmarkArgs& args) : base(args) {}

  // TODO: Problem size?
  void setup() {
    const auto range = getRange();
    buff_A.initialize(args.device_queue, range);
    buff_B.initialize(args.device_queue, range);
    buff_C.initialize(args.device_queue, range);
  }

  void run() {
    auto& queue = args.device_queue;
    for(int i = 0; i < NUM_KERNELS; i++) {
      queue.submit([&](s::handler& cgh) {
        auto acc_A = buff_A.template get_access<s::access::mode::read>(cgh, buff_A.get_range());
        auto acc_B = buff_A.template get_access<s::access::mode::read>(cgh, buff_A.get_range());
        auto acc_C = buff_A.template get_access<s::access::mode::write>(cgh, buff_A.get_range());

        cgh.parallel_for(getRange(), [=](s::item<dim> item) {
          if constexpr(use_id) {
            acc_C[item] = acc_A[item] + acc_B[item];
          } else {
            // Manual unroll, ugly but it works
            if constexpr(dim == 1) {
              acc_C[item[0]] = acc_A[item[0]] + acc_B[item[0]];
            }
            if constexpr(dim == 2) {
              acc_C[item[0]][item[1]] = acc_A[item[0]][item[1]] + acc_B[item[0]][item[1]];
            }
            if constexpr(dim == 3) {
              acc_C[item[0]][item[1]][item[2]] = acc_A[item[0]][item[1]][item[2]] + acc_B[item[0]][item[1]][item[2]];
            }
          }
        });
      });
      // swap buffers
      std::swap(buff_A, buff_B);
      std::swap(buff_A, buff_C);
    }
  }

  bool verify(VerificationSetting& settings) {
    // TODO
    return true;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "SYCL2020_Accessors_Latency_";
    name << ReadableTypename<DATA_TYPE>::name << "_";
    name << dim << "dim_";
    if constexpr (!use_id){
      name << "no_id_index_";
    }
    name << NUM_KERNELS;
    return name.str();
  }
};

template <typename DATA_TYPE, std::size_t dim>
class USMLatency : LatencyBenchmark<DATA_TYPE, dim> {
protected:
  USMBuffer<DATA_TYPE, dim> buff_A;
  USMBuffer<DATA_TYPE, dim> buff_B;
  USMBuffer<DATA_TYPE, dim> buff_C;

  using base = LatencyBenchmark<DATA_TYPE, dim>;
  using base::args;
  using base::base;
  using base::getRange;

public:

  USMLatency(const BenchmarkArgs& args) : base(args) {}

  // TODO: Problem size?
  void setup() {
    buff_A.initialize(args.device_queue, getRange());
    buff_B.initialize(args.device_queue, getRange());
    buff_C.initialize(args.device_queue, getRange());
  }

  void run() {
    auto& queue = args.device_queue;
    cl::sycl::event event;
    for(int i = 0; i < NUM_KERNELS; i++) {
      event = queue.submit([&](s::handler& cgh) {
        cgh.depends_on(event);
        cgh.parallel_for(
            getRange(), [acc_A = buff_A.get(), acc_B = buff_B.get(), acc_C = buff_C.get()](s::item<dim> item) {
              const auto id = item.get_linear_id();
              acc_C[id] = acc_A[id] + acc_B[id];
            });
      });
      // swap buffers to
      std::swap(buff_A, buff_B);
      std::swap(buff_A, buff_C);
    }
  }

  bool verify(VerificationSetting& settings) {
    // TODO
    return true;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "SYCL2020_USM_Latency_";
    name << ReadableTypename<DATA_TYPE>::name << "_";
    name << dim << "dim_";
    name << NUM_KERNELS;
    return name.str();
  }
};

void launchAccessorBenchmarks(BenchmarkApp& app) {
  auto& args = app.getArgs();
  if(args.cli.isFlagSet("-disable-id-index")) {
    app.run<AccessorLatency<float, 1, false>>();
    app.run<AccessorLatency<float, 2, false>>();
    app.run<AccessorLatency<float, 3, false>>();
    if(app.deviceSupportsFP64()) {
      app.run<AccessorLatency<double, 1, false>>();
      app.run<AccessorLatency<double, 2, false>>();
      app.run<AccessorLatency<double, 3, false>>();
    }
  } else {
    app.run<AccessorLatency<float, 1, true>>();
    app.run<AccessorLatency<float, 2, true>>();
    app.run<AccessorLatency<float, 3, true>>();
    if(app.deviceSupportsFP64()) {
      app.run<AccessorLatency<double, 1, true>>();
      app.run<AccessorLatency<double, 2, true>>();
      app.run<AccessorLatency<double, 3, true>>();
    }
  }
}

void launchUSMBenchmarks(BenchmarkApp& app) {
  app.run<USMLatency<float, 1>>();
  app.run<USMLatency<float, 2>>();
  app.run<USMLatency<float, 3>>();
  if(app.deviceSupportsFP64()) {
    app.run<USMLatency<double, 1>>();
    app.run<USMLatency<double, 2>>();
    app.run<USMLatency<double, 3>>();
  }
}

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  launchAccessorBenchmarks(app);
  launchUSMBenchmarks(app);
}
