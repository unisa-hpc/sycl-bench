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

    PrefetchedBuffer<DATA_TYPE, 1> input_buf;
    PrefetchedBuffer<DATA_TYPE, 1> output_buf;
public:
  MicroBenchArithmetic(const BenchmarkArgs &_args) 
  : args(_args){}
  
  void setup() {     
    // buffers initialized to a default value 
    input. resize(args.problem_size, 10);
    output.resize(args.problem_size, 42);

    input_buf.initialize(args.device_queue, input.data(), s::range<1>(args.problem_size));
    output_buf.initialize(args.device_queue, output.data(), s::range<1>(args.problem_size));
  }

  void run(std::vector<cl::sycl::event>& events){

    events.push_back(args.device_queue.submit(
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
    })); // submit
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


