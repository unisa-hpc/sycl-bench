#include <CL/sycl.hpp>
#include <iostream>

#include "common.h"
#include "bitmap.h"


// Opening cl::sycl namespace is unsupported on hipSYCL 
// (mainly due to CUDA/HIP design issues), better 
// avoid it
//using namespace cl::sycl;
namespace s = cl::sycl;
class SobelBenchKernel; // kernel forward declaration

using cl::sycl::float4;

/*
    Classic sobel filter with a convolution matrix 3x3.
    Input and output are two dimensional buffer of floats.     
 */
class SobelBench
{
protected:    
    std::vector<float4> input_host;    
    std::vector<float4> output_host;
    
    size_t w, h; // size of the input picture
    size_t size; // used defined size (output will be size x size)
    BenchmarkArgs args;     

public:
  SobelBench(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {      
    // load input image
    Bitmap input_image;
    input_image.open("../Brommy.bmp");
    PixelMatrix pixels = input_image.toPixelMatrix();
    w = pixels.size();
    if(w>0)
        h = pixels[0].size();
    else    
        h = 0;
    // input size defined by the user  
    size = args.problem_size;
    std::cout << "w: " << w << ", h: " << h << ", size: " << size << std::endl;
    
    // prepare the input buffer (similar to a GL_MIRRORED_REPEAT of the input picture)
    input_host.resize(size * size);    
    for(size_t i=0; i<size; i++)
        for(size_t j=0; j<size; j++){            
            Pixel pixel = pixels[i%h][j%w]; // mirror repeat
            float4 color = float4(pixel.red, pixel.green, pixel.blue, 0); // cast int-to-float
            // write            
            size_t target = (i % w) / (j % h);
            input_host[j + i * size] = color;
        }
    
    output_host.resize(size * size);
  }

  void run() {    
    s::buffer<float4, 2> input_buf (& input_host[0], s::range<2>(size, size));    
    s::buffer<float4, 2> output_buf(&output_host[0], s::range<2>(size, size));

    args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {
      auto in  = input_buf.get_access<s::access::mode::read>(cgh);
      auto out = output_buf.get_access<s::access::mode::discard_write>(cgh);
      //cl::sycl::range<2> ndrange {w, h};
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
                    
              if(x==xs && y==ys) continue; // for the same pixel, convolution is always 0
                    
				      // sample color
				      //int c = xs + ys * width;
				      //float4 sample = convert_float4(in[c]);
              float4 sample = in[xs][ys];

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

	        /* taking root of sums of squares of Gx and Gy */	
	        const int index = x + y * ndrange[0]; //size; //width;
	        out[gid] = hypot(Gx, Gy)/(float4)(2);
        });
    });

  }

  bool verify(VerificationSetting &ver) { 
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

    // write the output picture
    std::cout << "writing the output" << std::endl;
    Bitmap output_image;
    
    PixelMatrix pixels;
    pixels.resize(size);
    pixels[0].resize(size);
    /*
    w = pixels.size();
    if(w>0)
        h = pixels[0].size();
    else    
        h = 0
    */
        
    for(size_t i=0; i<size; i++)
        for(size_t j=0; j<size; j++){ 
          float4 color = output_host[i*size + j];
          pixels[i][j].red   = (int) color.x();
          pixels[i][j].green = (int) color.y();
          pixels[i][j].blue  = (int) color.z();
        }

    output_image.fromPixelMatrix(pixels);
    output_image.save("./output.bmp");
    return pass;
  }
  
  static std::string getBenchmarkName() {
    return "Sobel";
  }
};


int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<SobelBench>();  
  return 0;
}

