#include "sobel.h"


void SobelBench::run() {    
    s::buffer<float4, 2>  input_buf( input.data(), s::range<2>(size, size));    
    s::buffer<float4, 2> output_buf(output.data(), s::range<2>(size, size));

    args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {
      auto in  = input_buf .get_access<s::access::mode::read>(cgh);
      auto out = output_buf.get_access<s::access::mode::discard_write>(cgh);
      cl::sycl::range<2> ndrange {size, size};

      // Sobel kernel 3x3
      const float kernel[] =
      { 1, 0, -1,
        2, 0, -2,
        1, 0, -1
      };


      cgh.parallel_for<class SobelBenchKernel>(ndrange,
        [=](cl::sycl::id<2> gid) 
        {
            int x = gid[0];
            int y = gid[1];

            float4 Gx = float4(0.0f, 0.0f, 0.0f, 0.0f);
       	    float4 Gy = float4(0.0f, 0.0f, 0.0f, 0.0f);
            const int half_radius = 1;
            const int radius = 3;

            // constant-size loops
            for(uint x_shift = 0; x_shift<3; x_shift++)
                for(uint y_shift = 0; y_shift<3; y_shift++)
		{
                  // sample position
		  uint xs = x - half_radius + x_shift;
		  uint ys = y - half_radius + y_shift;
                  // for the same pixel, convolution is always 0  
                  if(x==xs && y==ys) 
                    continue; 
                  // boundary check
                  if(xs < 0 || xs > size || ys < 0 || ys > size)
		    continue;
                    
	          // sample color
	      //int c = xs + ys * width;
	      //float4 sample = convert_float4(in[c]);
              //float4 sample = in[xs][ys];
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
	        //const int index = x + y * ndrange[0]; //size; //width;
	  float4 color = hypot(Gx, Gy)/(float4)(2);
	  out[gid] = clamp(color, float4(0.0), float4(1.0));
//	  out[gid] = hypot(Gx,Gy) ; 
//	  out[gid] = in[gid]; 
//          out[gid] = float4(1.0,0.5,0.5,0.5);
        });

  });

}


bool SobelBench::verify(VerificationSetting &ver) { 
  
    saveOutput();
    bool pass = true;
    /*
    for(size_t i=ver.begin[0]; i<ver.begin[0]+ver.range[0]; i++){
        int expected = input1[i] + input2[i];
        //std::cout << i << ") " << output[i] << " : " << expected << std::endl;
        if(expected != output[i]){
            pass = false;
            break;
        }
      }    
    */

    return pass;
}



int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<SobelBench>();  
  return 0;
}

