#include <CL/sycl.hpp>
#include <iostream>

#include "common.h"
#include "bitmap.h"


namespace s = cl::sycl;
class MedianFilterBenchKernel; // kernel forward declaration

using cl::sycl::float4;

/*
  A median filter with a windows of 3 pixels (3x3).
  Input and output are two-dimensional buffers of floats.     
 */
class MedianFilterBench
{
protected:
    std::vector<float4> input;
    std::vector<float4> output;

    size_t w, h; // size of the input picture
    size_t size; // user-defined size (input and output will be size x size)
    BenchmarkArgs args;

public:
  MedianFilterBench(const BenchmarkArgs &_args) : args(_args) {}

  void setup() {
    size = args.problem_size; // input size defined by the user
    input.resize(size * size); 
    load_bitmap_mirrored("../Brommy.bmp", size, input);
    output.resize(size * size);
  }

  inline void swap(float4 A[], int i, int j){    
    /*if(A[i] > A[j]) {
      float temp = A[i];
      A[i] = A[j];
      A[j] = temp;
    }*/    
    A[i] = fmin(A[i], A[j]);
    A[j] = fmax(A[i], A[j]);
  }

  void run() {    
    s::buffer<float4, 2>  input_buf( input.data(), s::range<2>(size, size));    
    s::buffer<float4, 2> output_buf(output.data(), s::range<2>(size, size));

    args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {
      auto in  = input_buf .get_access<s::access::mode::read>(cgh);
      auto out = output_buf.get_access<s::access::mode::discard_write>(cgh);
      cl::sycl::range<2> ndrange {size, size};

      cgh.parallel_for<class MedianFilterBenchKernel>(ndrange,
        [=](cl::sycl::id<2> gid) 
        {
          int x = gid[0];
          int y = gid[1];

          // Optimization note: this array can be prefetched in local memory, TODO
	  float4 window[9];
          int k = 0;
          for(int i = -1; i<2; i++)
            for(int j = -1; j<2; j++) {
              uint xs = fmin(fmax(x+j, 0), size-1); // borders are handled here with extended values
              uint ys = fmin(fmax(y+i, 0), size-1);
              window[k] =in[ {xs,ys} ];
              k++;
            }
          
          // (channel-wise) median selection using bitonic sorting
          // the following network is used (Bose-Nelson algorithm):
          // [[0,1],[2,3],[4,5],[7,8]]
          // [[0,2],[1,3],[6,8]]
          // [[1,2],[6,7],[5,8]]
          // [[4,7],[3,8]]
          // [[4,6],[5,7]]
          // [[5,6],[2,7]]
          // [[0,5],[1,6],[3,7]]
          // [[0,4],[1,5],[3,6]]
          // [[1,4],[2,5]]
          // [[2,4],[3,5]]
          // [[3,4]]
          // se also http://pages.ripco.net/~jgamble/nw.html
	  swap(window, 0, 1);
          swap(window, 2, 3);
          swap(window, 0, 2);
          swap(window, 1, 3);
          swap(window, 1, 2);
          swap(window, 4, 5);
          swap(window, 7, 8);
          swap(window, 6, 8);
          swap(window, 6, 7);
          swap(window, 4, 7);
          swap(window, 4, 6);
          swap(window, 5, 8);
          swap(window, 5, 7);
          swap(window, 5, 6);
          swap(window, 0, 5);
          swap(window, 0, 4);
          swap(window, 1, 6);
          swap(window, 1, 5);
          swap(window, 1, 4);
          swap(window, 2, 7);
          swap(window, 3, 8);
          swap(window, 3, 7);
          swap(window, 2, 5);
          swap(window, 2, 4);
          swap(window, 3, 6);
          swap(window, 3, 5);
          swap(window, 3, 4);

	  out[gid] = window[4];
       }
       );
     });
     
     args.device_queue.wait_and_throw();
   }


  bool verify(VerificationSetting &ver) {  
    save_bitmap("median.bmp", size, output);

    bool pass = true;
    for(size_t i=ver.begin[0]; i<ver.begin[0]+ver.range[0]; i++){
      int x = i % size;
      int y = i / size;
      float4 window[9];
      int k = 0;
      for(int i = -1; i<2; i++)
        for(int j = -1; j<2; j++) {
          uint xs = fmin(fmax(x+j, 0), size-1); // borders are handled here with extended values
          uint ys = fmin(fmax(y+i, 0), size-1);
          window[k] =input[xs + ys*size ];
          k++;
        }
      swap(window, 0, 1);
      swap(window, 2, 3);
      swap(window, 0, 2);
      swap(window, 1, 3);
      swap(window, 1, 2);
      swap(window, 4, 5);
      swap(window, 7, 8);
      swap(window, 6, 8);
      swap(window, 6, 7);
      swap(window, 4, 7);
      swap(window, 4, 6);
      swap(window, 5, 8);
      swap(window, 5, 7);
      swap(window, 5, 6);
      swap(window, 0, 5);
      swap(window, 0, 4);
      swap(window, 1, 6);
      swap(window, 1, 5);
      swap(window, 1, 4);
      swap(window, 2, 7);
      swap(window, 3, 8);
      swap(window, 3, 7);
      swap(window, 2, 5);
      swap(window, 2, 4);
      swap(window, 3, 6);
      swap(window, 3, 5);
      swap(window, 3, 4);
      float4 expected = window[4];
      float4 dif = fdim(output[i], expected);
      float length = cl::sycl::length(dif);
      if(length > 0.01f)
      {
        pass = false;
        break;
      }
    }    
    return pass;
}


static std::string getBenchmarkName() {
    return "MedianFilter";
  }

}; // MedianFilterBench class


int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<MedianFilterBench>();  
  return 0;
}


