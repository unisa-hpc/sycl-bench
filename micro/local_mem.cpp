#include "common.h"

#include <iostream>

namespace s = cl::sycl;

template <typename DATA_TYPE, int COMP_ITERS>
class MicroBenchLocalMemoryKernel;

/* Microbenchmark stressing the local memory. */
template <typename DATA_TYPE, int COMP_ITERS>
class MicroBenchLocalMemory {
protected:
  std::vector<DATA_TYPE> input;
  BenchmarkArgs args;

  PrefetchedBuffer<DATA_TYPE, 1> input_buf;
  PrefetchedBuffer<DATA_TYPE, 1> output_buf;

public:
  MicroBenchLocalMemory(const BenchmarkArgs& _args) : args(_args) {
    assert(args.problem_size % args.local_size == 0 && "Invalid problem_size/local_size combination.");
  }

  void setup() {
    // buffers initialized to a default value
    input.resize(args.problem_size, 42);

    input_buf.initialize(args.device_queue, input.data(), s::range<1>(args.problem_size));
    output_buf.initialize(args.device_queue, s::range<1>(args.problem_size));
  }

  void run(std::vector<cl::sycl::event>& events) {
    events.push_back(args.device_queue.submit([&](cl::sycl::handler& cgh) {
      auto in = input_buf.template get_access<s::access::mode::read>(cgh);
      auto out = output_buf.template get_access<s::access::mode::discard_write>(cgh);
      // local memory definition
      s::accessor<DATA_TYPE, 1, s::access::mode::read_write, s::access::target::local> local_mem(args.local_size, cgh);

      s::nd_range<1> ndrange{{args.problem_size}, {args.local_size}};

      cgh.parallel_for<MicroBenchLocalMemoryKernel<DATA_TYPE, COMP_ITERS>>(ndrange, [=](s::nd_item<1> item) {
        DATA_TYPE r0;
        int gid = item.get_global_id(0);
        int lid = item.get_local_id(0);
        int lid2 = (item.get_local_id(0)+1) % item.get_local_range()[0];

        local_mem[lid] = in[gid];

        item.barrier(s::access::fence_space::local_space);

        // Note: this is dangerous, as a compiler could in principle be smart enough to figure out that it can just drop this
        //       so far, we haven't encountered such a compiler, and all options to make it "safer" 
        //       introduce overhead on at least some platform / data type combinations
        for(int i = 0; i < COMP_ITERS; i++) {
          local_mem[lid2] = local_mem[lid];
        }

        item.barrier(s::access::fence_space::local_space);

        out[gid] = local_mem[lid];
      });
    })); // submit
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "MicroBench_LocalMem_";
    name << ReadableTypename<DATA_TYPE>::name << "_";
    name << COMP_ITERS;
    return name.str();
  }

  bool verify(VerificationSetting& ver) {
    auto result = output_buf.template get_access<s::access::mode::read>();
    for(size_t i = 0; i < args.problem_size; ++i) {
      if(result[i] != 42) {
        return false;
      }
    }
    return true;
  }

  static ThroughputMetric getThroughputMetric(const BenchmarkArgs& args) {
    const double throughput = (args.problem_size * sizeof(DATA_TYPE) * COMP_ITERS * 2) / 1024 / 1024 / 1024;
    return {throughput, "GiB"};
  }
};

int main(int argc, char** argv) {
  constexpr int compute_iters = 1024 * 4;

  BenchmarkApp app(argc, argv);

  // int
  app.run<MicroBenchLocalMemory<int, compute_iters>>();

  // single precision
  app.run<MicroBenchLocalMemory<float, compute_iters>>();

  // double precision
  if(app.deviceSupportsFP64())
    app.run<MicroBenchLocalMemory<double, compute_iters>>();

  return 0;
}
