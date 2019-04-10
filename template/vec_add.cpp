#include "common.h"

using namespace cl::sycl;

class VecAddBench
{
protected:
    int* input1 = nullptr;
    int* input2 = nullptr;
    int* output = nullptr;        
    const BenchmarkArgs &args;

public:
  VecAddBench(const BenchmarkArgs &_args) : args(_args) {}

  
  void setup() {      
      // host memory allocation
      input1 = new int[args.problem_size];
      input2 = new int[args.problem_size];
      output = new int[args.problem_size];
      // input initialization
      for(int i=0; i<args.problem_size; i++){
          input1[i] = 1;
          input2[i] = 2;
          output[i] = 0;
      }

  }

  void run() {
    
    buffer<int, 1> input1_buf(input1, range<1>(args.problem_size));
    buffer<int, 1> input2_buf(input2, range<1>(args.problem_size));
    buffer<int, 1> output_buf(output, range<1>(args.problem_size));

    args.queue.submit([&](cl::sycl::handler& cgh) {
      auto in1 = input1_buf.get_access<access::mode::read>(cgh);
      auto in2 = input2_buf.get_access<access::mode::read>(cgh);
      auto out = output_buf.get_access<access::mode::write>(cgh);

      cgh.parallel_for<class VecAddKernel>(args.problem_size,
        [=] (cl::sycl::id<1> gid) {
            out[gid] = in1[gid] + in2[gid];
        }
        );        
    }
  }


  bool verify(cl::sycl::id<3> verificationBegin, cl::sycl::range<3> verificationRange) { 
    std::cout << output[0] <<  output[10] << endl;
  }

  
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<VecAddBench>();  
}



