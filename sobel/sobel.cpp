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
    std::vector<float4> input;    
    std::vector<float4> output;    
    size_t w, h; // size of the input picture
    size_t size; // used defined size (output will be size x size)
    BenchmarkArgs args;     

public:
  SobelBench(const BenchmarkArgs &_args) : args(_args) {}
  
  void setup() {      
    // load input image
    Bitmap input_image;
    input_image.open("../Brommy.bmp");
    std::cout << "input image loaded" << std::endl;
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
    input.resize(size * size);    
    for(size_t i=0; i<size; i++)
        for(size_t j=0; j<size; j++){            
            Pixel pixel = pixels[i%h][j%w]; // mirror repeat
	//	std::cout << pixel.red << std::endl;
            float4 color = float4(pixel.red / 255.0f, pixel.green / 255.0f, pixel.blue / 255.0f, 1.0f); // cast int-to-float  
            input[j + i * size] = color; // write to input buffer
        }
    std::cout << "image resized to match the provided input size" << std::endl;
    output.resize(size * size);

  }


  void run() {    
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

      //std::cout << "parallel_for -:" << std::endl;

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
    
args.device_queue.wait_and_throw();
     
    std::cout << "Computed *"<< std::endl;   
 //   saveOutput(); 
  }

  // When verification is enable, the program also writes the output in a output.bmp file.
  bool verify(VerificationSetting &ver) { 
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
  

  bool saveOutput(){

    // write the output picture
    std::cout << "writing the output" << std::endl;
    Bitmap output_image;    
    PixelMatrix pixels;
    pixels.resize(size);
    for(size_t i=0; i<size; i++){
        pixels[i].resize(size);
        for(size_t j=0; j<size; j++){ 
          float4 color = output[i * size + j] * 255.f;
          pixels[i][j].red   = (int) color.x();
          pixels[i][j].green = (int) color.y();
          pixels[i][j].blue  = (int) color.z();
        }
    }
    std::cout << "buffer" << std::endl;
    output_image.fromPixelMatrix(pixels);
    std::cout << output_image.isImage() << std::endl;

    output_image.save("./output.bmp");
    
    return true;
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

