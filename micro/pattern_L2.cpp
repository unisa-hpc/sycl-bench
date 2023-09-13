#include "common.h"

#include <iostream>

namespace s = cl::sycl;

template <typename DATA_TYPE, int COMP_ITERS> class MicroBenchL2Kernel;

/* Microbenchmark stressing the main arithmetic units. */
template <typename DATA_TYPE, int COMP_ITERS>
class MicroBenchL2
{
protected:
    std::vector<DATA_TYPE> input;
    BenchmarkArgs args;

    PrefetchedBuffer<DATA_TYPE, 1> input_buf;
    PrefetchedBuffer<DATA_TYPE, 1> output_buf;
public:
  MicroBenchL2(const BenchmarkArgs &_args) : args(_args) {}

  void setup() {
    // buffers initialized to a default value 
    input. resize(args.problem_size, 10);

    input_buf.initialize(args.device_queue, input.data(), s::range<1>(args.problem_size));
    output_buf.initialize(args.device_queue, s::range<1>(args.problem_size));
  }

  void run(std::vector<cl::sycl::event>& events){

    events.push_back(args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {
      auto in  =  input_buf.template get_access<s::access::mode::read>(cgh);
      auto out = output_buf.template get_access<s::access::mode::discard_write>(cgh);
      cl::sycl::range<1> ndrange {args.problem_size};

      cgh.parallel_for<MicroBenchL2Kernel<DATA_TYPE,COMP_ITERS>>(ndrange,
        [=](cl::sycl::id<1> gid)
      {
        DATA_TYPE r0;
        for (int i=0;i<COMP_ITERS;i++) {
            r0 = in[gid];
            out[gid] = r0; 
        }
        out[gid] = r0;
      });
    })); // submit
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "MicroBench_L2_";
    name << ReadableTypename<DATA_TYPE>::name << "_";
    name << COMP_ITERS;
    return name.str();
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);

  // int
  app.run< MicroBenchL2<int,1> >();
  app.run< MicroBenchL2<int,2> >();
  app.run< MicroBenchL2<int,4> >();
  app.run< MicroBenchL2<int,8> >();
  app.run< MicroBenchL2<int,16> >();

  // single precision  
  app.run< MicroBenchL2<float,1> >();
  app.run< MicroBenchL2<float,2> >();
  app.run< MicroBenchL2<float,4> >();
  app.run< MicroBenchL2<float,8> >();
  app.run< MicroBenchL2<float,16> >();

  // double precision
  if(app.deviceSupportsFP64()) {
    app.run<MicroBenchL2<double, 1>>();
    app.run<MicroBenchL2<double, 2>>();
    app.run<MicroBenchL2<double, 4>>();
    app.run<MicroBenchL2<double, 8>>();
    app.run<MicroBenchL2<double, 16>>();
  }

  return 0;
}



