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
    std::vector<DATA_TYPE> output;
    BenchmarkArgs args;

public:
  MicroBenchL2(const BenchmarkArgs &_args) : args(_args) {}

  void setup() {
    // buffers initialized to a default value 
    input. resize(args.problem_size, 10);
    output.resize(args.problem_size, 42);
  }

  void run(){
    s::buffer<DATA_TYPE, 1>  input_buf (input.data(), s::range<1>(args.problem_size));
    s::buffer<DATA_TYPE, 1> output_buf(output.data(), s::range<1>(args.problem_size));

    args.device_queue.submit(
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
    }); // submit
    args.device_queue.wait_and_throw();
  }

  bool verify(VerificationSetting &ver) {
    bool pass = true;
    std::cout << "No verification available" << std::endl;
    return pass;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "MicroBench_";
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
  app.run< MicroBenchL2<double,1> >();
  app.run< MicroBenchL2<double,2> >();
  app.run< MicroBenchL2<double,4> >();
  app.run< MicroBenchL2<double,8> >();
  app.run< MicroBenchL2<double,16> >();

  return 0;
}



