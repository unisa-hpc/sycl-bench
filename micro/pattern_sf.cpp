#include "common.h"

#include <iostream>

namespace s = cl::sycl;

template <typename DATA_TYPE, int N> class MicroBenchSpecialFuncKernel;

/* Microbenchmark stressing the special function units. */
template <typename DATA_TYPE, int N>
class MicroBenchSpecialFunc
{
protected:
    std::vector<DATA_TYPE> input;
    std::vector<DATA_TYPE> output;
    BenchmarkArgs args;

    PrefetchedBuffer<DATA_TYPE, 1> input_buf;
    PrefetchedBuffer<DATA_TYPE, 1> output_buf;
public:
  MicroBenchSpecialFunc(const BenchmarkArgs &_args) : args(_args) {}

  void setup() {
    // buffers initialized to a default value
    input. resize(args.problem_size, 10);
    output.resize(args.problem_size, 42);

    input_buf.initialize(args.device_queue, input.data(), s::range<1>(args.problem_size));
    output_buf.initialize(args.device_queue, output.data(), s::range<1>(args.problem_size));
  }

  void run(){

    args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {
      auto in  =  input_buf.template get_access<s::access::mode::read>(cgh);
      auto out = output_buf.template get_access<s::access::mode::discard_write>(cgh);
      cl::sycl::range<1> ndrange {args.problem_size};

      cgh.parallel_for<MicroBenchSpecialFuncKernel<DATA_TYPE,N>>(ndrange,
        [=](cl::sycl::id<1> gid)
      {
        DATA_TYPE r0, r1, r2, r3;
        r0 = in[gid];
        r1 = r2 = r3 = r0;
        for (int i=0;i<N;i++) {
            r0 = s::log(r1);
            r1 = s::cos(r2);
            r2 = s::log(r3);
            r3 = s::sin(r0);
        }
        out[gid] = r0;
      });
    }); // submit
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "MicroBench_sf_";
    name << ReadableTypename<DATA_TYPE>::name << "_";
    name << N;
    return name.str();
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);

  // int
  // TODO Special functions are only defined for floats, using them with ints
  // without explicitly casting to float/double will cause ambiguous overloaded function call
  /* 
  app.run< MicroBenchSpecialFunc<int,1> >();
  app.run< MicroBenchSpecialFunc<int,2> >();
  app.run< MicroBenchSpecialFunc<int,4> >();
  app.run< MicroBenchSpecialFunc<int,8> >();
  app.run< MicroBenchSpecialFunc<int,16> >();
  */

  // single precision  
  app.run< MicroBenchSpecialFunc<float,1> >();
  app.run< MicroBenchSpecialFunc<float,2> >();
  app.run< MicroBenchSpecialFunc<float,4> >();
  app.run< MicroBenchSpecialFunc<float,8> >();
  app.run< MicroBenchSpecialFunc<float,16> >();

  // double precision
  app.run< MicroBenchSpecialFunc<double,1> >();
  app.run< MicroBenchSpecialFunc<double,2> >();
  app.run< MicroBenchSpecialFunc<double,4> >();
  app.run< MicroBenchSpecialFunc<double,8> >();
  app.run< MicroBenchSpecialFunc<double,16> >();

  return 0;
}

