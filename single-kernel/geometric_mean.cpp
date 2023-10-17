#include <sycl/sycl.hpp>
#include <iostream>

#include "common.h"

namespace s = sycl;


class GeometricMeanKernel; // kernel forward declaration


class GeometricMean
{
protected:
    size_t size; 
    size_t local_size;
    int chunkSize;
    std::vector<s::float16> input;
    std::vector<float> output;

    BenchmarkArgs args;


    PrefetchedBuffer<s::float16, 1> buf_input;    
    PrefetchedBuffer<float, 1> buf_output;    


    void fillrandom_float(float* arrayPtr, int width, int height, float rangeMin, float rangeMax){
      if(!arrayPtr) {
        fprintf(stderr, "Cannot fill array: NULL pointer.\n");
        return;
      }
      srand(7);
      double range = (double)(rangeMax - rangeMin);     
      for(int i = 0; i < height; i++)
          for(int j = 0; j < width; j++) {
              int index = i*width + j;
              arrayPtr[index] = rangeMin + (float)(range*((float)rand() / RAND_MAX)); 	
          }    
    }   

public:
  GeometricMean(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {
    size = args.problem_size; // input size defined by the user
    local_size = args.local_size; // set local work_group size
    
    chunkSize = 16;
    input.resize(size);
    output.resize(size);
    
    fillrandom_float((float*)input.data(),size, chunkSize, 0.001f ,100000.f);
    

    // init buffer
    buf_input.initialize(args.device_queue, input.data(), s::range<1>(size));
    buf_output.initialize(args.device_queue, output.data(), s::range<1>(size));
  }

  void run(std::vector<s::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
    auto input_acc = buf_input.get_access<s::access::mode::read>(cgh);
    auto output_acc = buf_output.get_access<s::access::mode::write>(cgh);
   
   
    s::range<1> ndrange{size};

      cgh.parallel_for<class GeometricMeanKernel>(ndrange, [input_acc, output_acc, chunkSize_= chunkSize, size_ = size](s::id<1> id) {
          int gid = id[0];
	
            if(gid >= size_)
                return;

            s::float16 val = input_acc[gid];
            
            float mean = s::log(val.s0()) + s::log(val.s1()) + s::log(val.s2()) + s::log(val.s3()) + s::log(val.s4()) + s::log(val.s5()) + s::log(val.s6()) + s::log(val.s7()) + 
                s::log(val.s8()) + s::log(val.s9()) + s::log(val.sA()) + s::log(val.sB()) + s::log(val.sC()) + s::log(val.sD()) + s::log(val.sE()) + s::log(val.sF());
            mean /= chunkSize_;
            
            float euler = 2.718281828459045235f;
        
            output_acc[gid] = s::pow(euler, mean);   
      });
    }));
   }
    
  bool verify(VerificationSetting& ver) {
        buf_input.reset();
        buf_output.reset();
        unsigned int check = 1;
        float host_mean = 0.0f;
        float* testInput = (float*)input.data();
        for(unsigned int i = 0; i < size*chunkSize; ++i) 
          host_mean = host_mean + log(testInput[i]);
        host_mean /= size*chunkSize;
        host_mean = pow(2.718281828459045235f, host_mean);	
        
        printf("Host mean is %f\n", host_mean);

        float device_mean = 0.0f;
        for(unsigned int i = 0; i < size; ++i) 
          device_mean = device_mean + log(output[i]);

        device_mean /= size;
        device_mean = pow(2.718281828459045235f, device_mean);			
        printf("Device mean is %f\n", device_mean);		

        return fabs(device_mean - host_mean) < 1.0f ? true : false;    
  }


  static std::string getBenchmarkName() { return "Geometric mean"; }

}; // GeometricMean class


int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<GeometricMean>();  
  return 0;
}


