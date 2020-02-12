#include "common.h"

#include <iostream>

namespace s = cl::sycl;

template <typename DATA_TYPE, int N> class MicroBenchArithmeticKernel;

/* Microbenchmark stressing the main arithmetic units. */
template <typename DATA_TYPE, int N>
class MicroBenchArithmetic
{
protected:    
    std::vector<DATA_TYPE> input;
    std::vector<DATA_TYPE> output;
    BenchmarkArgs args;

public:
  MicroBenchArithmetic(const BenchmarkArgs &_args) : args(_args) {}
  
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
      
      cgh.parallel_for<MicroBenchArithmeticKernel<DATA_TYPE, N>>(ndrange,
        [=](cl::sycl::id<1> gid)
      {
        DATA_TYPE r0, r1, r2, r3;
        r0 = in[gid];
        r1 = r2 = r3 = r0;
        for (int i=0;i<N;i++) {
            r0 = r0 * r0 + r1;
            r1 = r1 * r1 + r2;
            r2 = r2 * r2 + r3;
            r3 = r3 * r3 + r0;
        }
        out[gid] = r0;
      });  
    }); // submit
    args.device_queue.wait_and_throw();
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "MicroBench_Arith_";
    name << ReadableTypename<DATA_TYPE>::name << "_";
    name << N;
    return name.str();
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);

  // int
  app.run< MicroBenchArithmetic<int,1> >();  
  app.run< MicroBenchArithmetic<int,2> >();  
  app.run< MicroBenchArithmetic<int,4> >();
  app.run< MicroBenchArithmetic<int,8> >();  
  app.run< MicroBenchArithmetic<int,16> >();  

  // single precision  
  app.run< MicroBenchArithmetic<float,1> >();
  app.run< MicroBenchArithmetic<float,2> >();  
  app.run< MicroBenchArithmetic<float,4> >();  
  app.run< MicroBenchArithmetic<float,8> >();  
  app.run< MicroBenchArithmetic<float,16> >();  

  // double precision
  app.run< MicroBenchArithmetic<double,1> >();
  app.run< MicroBenchArithmetic<double,2> >();  
  app.run< MicroBenchArithmetic<double,4> >();  
  app.run< MicroBenchArithmetic<double,8> >();  
  app.run< MicroBenchArithmetic<double,16> >();  

  return 0;
}


