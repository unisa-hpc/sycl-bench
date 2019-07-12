#pragma once

#include <CL/sycl.hpp>
#include <iostream>

#include "common.h"
#include "bitmap.h"


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


  virtual void run();
  virtual bool verify(VerificationSetting &);

  static std::string getBenchmarkName() {
    return "Sobel";
  }


};

