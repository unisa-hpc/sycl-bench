#include "common.h"

#include <iostream>

namespace s = cl::sycl;

template <typename DATA_TYPE, int COMP_ITERS, int TILE_DIM> class MicroBenchLocalMemoryKernel;

/* Microbenchmark stressing the local memory. */
template <typename DATA_TYPE, int COMP_ITERS, int TILE_DIM>
class MicroBenchLocalMemory
{
protected:
    std::vector<DATA_TYPE> input;
    std::vector<DATA_TYPE> output;
    BenchmarkArgs args;

    PrefetchedBuffer<DATA_TYPE, 1> input_buf;
    PrefetchedBuffer<DATA_TYPE, 1> output_buf;
public:
  MicroBenchLocalMemory(const BenchmarkArgs &_args) : args(_args) {}

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
      // local memory definition
      s::accessor<DATA_TYPE, 1,s::access::mode::read_write, s::access::target::local> 
        local_mem(TILE_DIM, cgh);

      s::range<1> ndrange {args.problem_size};

      cgh.parallel_for<MicroBenchLocalMemoryKernel<DATA_TYPE,COMP_ITERS,TILE_DIM>>(ndrange,
        [=](cl::sycl::id<1> gid)
      {
        DATA_TYPE r0;
        int lid = gid[0] % TILE_DIM; 

        for (int i=0;i<COMP_ITERS;i++) {
	    r0 = local_mem[lid];
            local_mem[TILE_DIM - lid - 1] = r0;
        }
      });
    }); // submit
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "MicroBench_Shared_";
    name << ReadableTypename<DATA_TYPE>::name << "_";
    name << COMP_ITERS;
    name << "_";
    name << TILE_DIM;
    return name.str();
  }
};

int main(int argc, char** argv)
{
  const int tile_dim = 1024;
  const int compute_iters = 8192;

  BenchmarkApp app(argc, argv);

  // int
  app.run< MicroBenchLocalMemory<int,compute_iters,tile_dim> >();

  // single precision  
  app.run< MicroBenchLocalMemory<float,compute_iters,tile_dim> >();

  // double precision
  app.run< MicroBenchLocalMemory<double,compute_iters,tile_dim> >();

  return 0;
}


