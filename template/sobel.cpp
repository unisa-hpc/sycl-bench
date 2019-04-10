#include "common.h"
#include <iostream>

using BenchmarkArguments = std::unordered_map<std::string, std::string>;

class SobelBench
{
protected:


public:
  SobelBench(MandatoryBenchmarkArgs args, const BenchmarkArguments& additionalArgs)
  {

  }

  virtual ~SobelBench(){}

  void setup(std::size_t problemSize) {
    /* We define and initialize data to be copied to the device. */
    int data[nElems] = {0};

  }

  void run(std::size_t localSize) {
    



  try {
    default_selector selector;
    queue myQueue(selector, [](exception_list l) {
      for (auto ep : l) {
        try {
          std::rethrow_exception(ep);
        } catch (const exception& e) {
          std::cout << "Asynchronous exception caught:\n" << e.what();
        }
      }
    });

    buffer<int, 1> buf(data, range<1>(nElems));

    myQueue.submit([&](handler& cgh) {
      auto ptr = buf.get_access<access::mode::read_write>(cgh);

      auto myRange = nd_range<1>(range<1>(nElems), range<1>(nElems / 4));

      auto myKernel = ([=](nd_item<1> item) {
        ptr[item.get_global_id()] = item.get_global_id()[0];
      });

      cgh.parallel_for<assign_elements>(myRange, myKernel);
    });

  } catch (const exception& e) {
    std::cout << "Synchronous exception caught:\n" << e.what();
    return 2;
  }

  }

  bool verify(cl::sycl::id<3> verificationBegin, cl::sycl::range<3> verificationRange) { 
    

  }

  
};




int main(int argc, char** argv){

  BenchmarkApp benchmark(argc,argv);  
  Sobel sobel_bench();  
  benchmark.addHook(sobel_bench);   


}


/* We define the number of work items to enqueue. */
const int nElems = 64u;

class assign_elements;



  /* Check the result is correct. */
  int result = 0;
  for (int i = 0; i < nElems; i++) {
    if (data[i] != i) {
      std::cout << "The results are incorrect (element " << i << " is "
                << data[i] << ")!\n";
      result = 1;
    }
  }
  if (result != 1) {
    std::cout << "The results are correct." << std::endl;
  }
  return result;
}
