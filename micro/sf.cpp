#include "common.h"

namespace s = cl::sycl;

template <typename DataT, int N>
class MicroBenchSpecialFuncKernel;

/**
 * Microbenchmark stressing the special function units.
 */
template <typename DataT, int Iterations = 16>
class MicroBenchSpecialFunc {
protected:
  std::vector<DataT> input;
  BenchmarkArgs args;

  PrefetchedBuffer<DataT, 1> input_buf;
  PrefetchedBuffer<DataT, 1> output_buf;

public:
  MicroBenchSpecialFunc(const BenchmarkArgs& args) : args(args) {}

  void setup() {
    input.resize(args.problem_size, DataT{3.14});

    input_buf.initialize(args.device_queue, input.data(), s::range<1>(args.problem_size));
    output_buf.initialize(args.device_queue, s::range<1>(args.problem_size));
  }

  void run(std::vector<cl::sycl::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
      auto in = input_buf.template get_access<s::access::mode::read>(cgh);
      auto out = output_buf.template get_access<s::access::mode::discard_write>(cgh);

      cgh.parallel_for<MicroBenchSpecialFuncKernel<DataT, Iterations>>(
          s::range<1>{args.problem_size}, [=](s::id<1> gid) {
            DataT r0, v1, v2;
            r0 = in[gid];
            v1 = v2 = r0;
            for(int i = 0; i < Iterations; ++i) {
              r0 = s::cos(v1);
              v1 = s::sin(v2);
              v2 = s::tan(r0);
            }
            out[gid] = v2;
          });
    }));
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "MicroBench_sf_";
    name << ReadableTypename<DataT>::name << "_";
    name << Iterations;
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<MicroBenchSpecialFunc<float>>();
  app.run<MicroBenchSpecialFunc<double>>();

  return 0;
}
