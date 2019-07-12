#include "../sobel/sobel.h"

class Sobel5Bench {
public:
    Sobel5Bench(const BenchmarkArgs &_args) : SobelBench(_args) {} 

    virtual void run() {
      s::buffer<float4, 2>  input_buf( input.data(), s::range<2>(size, size));
      s::buffer<float4, 2> output_buf(output.data(), s::range<2>(size, size));

      args.device_queue.submit(
          [&](cl::sycl::handler& cgh) {
        auto in  = input_buf .get_access<s::access::mode::read>(cgh);
        auto out = output_buf.get_access<s::access::mode::discard_write>(cgh);
        cl::sycl::range<2> ndrange {size, size};

        // Sobel kernel 5x5
        const float kernel[] =
        { 
	  1,  2, 0,  -2, -1,
          4,  8, 0,  -8, -4,
	  6, 12, 0, -12, -6,
	  4,  8, 0,  -8, -4,
	  1,  2, 0,  -2, -1
	};

      cgh.parallel_for<class SobelBenchKernel>(ndrange,
        [=](cl::sycl::id<2> gid)
        {
            int x = gid[0];
            int y = gid[1];
            float4 Gx = float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 Gy = float4(0.0f, 0.0f, 0.0f, 0.0f);
            const int radius = 5;

            // constant-size loops in [-2,0,+2]
            for(uint x_shift = -2; x_shift<=2; x_shift++)
                for(uint y_shift = -2; y_shift<=2; y_shift++)
                {
                  // sample position
                  uint xs = x + x_shift;
                  uint ys = y + y_shift;
                  // for the same pixel, convolution is always 0  
                  if(x==xs && y==ys)  continue;
                  // boundary check
                  if(xs < 0 || xs > size || ys < 0 || ys > size) continue;

                  // sample color
                  float4 sample = in[ {xs,ys} ];

                  // convolution calculation
                  int offset_x = x_shift + y_shift * radius;
                  int offset_y = y_shift + x_shift * radius;

                  float conv_x = kernel[offset_x];
                  float4 conv4_x = (float4)(conv_x);
                  Gx += conv4_x * sample;

                  float conv_y = kernel[offset_y];
                  float4 conv4_y = (float4)(conv_y);
                  Gy += conv4_y * sample;
               }

          // taking root of sums of squares of Gx and Gy        
          float4 color = hypot(Gx, Gy);
          out[gid] = clamp(color, float4(0.0), float4(1.0));
        });
  });
}

  virtual bool verify(VerificationSetting &ver) {
      saveOutput();
      bool pass = true;
      return pass;
  }

};


int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<SobelBench>();
  return 0;
}





