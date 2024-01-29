#include <iostream>
#include <sycl/sycl.hpp>

#include "bitmap.h"
#include "common.h"


namespace s = sycl;
class Sobel7BenchKernel; // kernel forward declaration


/*
  A Sobel filter with a convolution matrix 7x7.
  Input and output are two-dimensional buffers of floats.
 */
class Sobel7Bench {
protected:
  std::vector<sycl::float4> input;
  std::vector<sycl::float4> output;

  size_t w, h; // size of the input picture
  size_t size; // user-defined size (input and output will be size x size)
  BenchmarkArgs args;

  PrefetchedBuffer<sycl::float4, 2> input_buf;
  PrefetchedBuffer<sycl::float4, 2> output_buf;

public:
  Sobel7Bench(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size; // input size defined by the user
    input.resize(size * size);
    load_bitmap_mirrored("../../share/Brommy.bmp", size, input);
    output.resize(size * size);

    input_buf.initialize(args.device_queue, input.data(), s::range<2>(size, size));
    output_buf.initialize(args.device_queue, output.data(), s::range<2>(size, size));
  }

  void run(std::vector<sycl::event>& events) {
    events.push_back(args.device_queue.submit([&](sycl::handler& cgh) {
      auto in = input_buf.get_access<s::access::mode::read>(cgh);
      auto out = output_buf.get_access<s::access::mode::discard_write>(cgh);
      sycl::range<2> ndrange{size, size};

      // Sobel kernel 7x7
      const float kernel[] = {130, 120, 78, 0, -78, -120, -130, 180, 195, 156, 0, -156, -195, -180, 234, 312, 390, 0,
          -390, -312, -234, 260, 390, 780, 0, -780, -390, -260, 234, 312, 390, 0, -390, -312, -234, 180, 195, 156, 0,
          -156, -195, -180, 130, 120, 78, 0, -78, -120, -130};

      cgh.parallel_for<Sobel7BenchKernel>(ndrange, [in, out, kernel, size_ = size](sycl::id<2> gid) {
        int x = gid[0];
        int y = gid[1];
        sycl::float4 Gx = sycl::float4(0, 0, 0, 0);
        sycl::float4 Gy = sycl::float4(0, 0, 0, 0);
        const int radius = 7;

        // constant-size loops in [0,1,2,3,4,5,6]
        for(int x_shift = 0; x_shift < 7; x_shift++) {
          for(int y_shift = 0; y_shift < 7; y_shift++) {
            // sample position
            uint xs = x + x_shift - 3; // [x-3,x-2,x-1,x,x+1,x+2,x+3]
            uint ys = y + y_shift - 3; // [y-3,y-2,y-1,y,y+1,y+2,y+2]
            // for the same pixel, convolution is always 0
            if(x == xs && y == ys)
              continue;
            // boundary check
            if(xs < 0 || xs >= size_ || ys < 0 || ys >= size_)
              continue;

            // sample color
            sycl::float4 sample = in[{xs, ys}];

            // convolution calculation
            int offset_x = x_shift + y_shift * radius;
            int offset_y = y_shift + x_shift * radius;

            float conv_x = kernel[offset_x];
            sycl::float4 conv4_x = sycl::float4(conv_x);
            Gx += conv4_x * sample;

            float conv_y = kernel[offset_y];
            sycl::float4 conv4_y = sycl::float4(conv_y);
            Gy += conv4_y * sample;
          }
        }
        // taking root of sums of squares of Gx and Gy
        sycl::float4 color = hypot(Gx, Gy);
        sycl::float4 minval = sycl::float4(0.0, 0.0, 0.0, 0.0);
        sycl::float4 maxval = sycl::float4(1.0, 1.0, 1.0, 1.0);
        out[gid] = clamp(color, minval, maxval);
      });
    }));
  }

  bool verify(VerificationSetting& ver) {
    // Triggers writeback
    output_buf.reset();
    save_bitmap("sobel7.bmp", size, output);

    const float kernel[] = {130, 120, 78, 0, -78, -120, -130, 180, 195, 156, 0, -156, -195, -180, 234, 312, 390, 0,
        -390, -312, -234, 260, 390, 780, 0, -780, -390, -260, 234, 312, 390, 0, -390, -312, -234, 180, 195, 156, 0,
        -156, -195, -180, 130, 120, 78, 0, -78, -120, -130};

    bool pass = true;
    int radius = 7;
    for(size_t i = ver.begin[0]; i < ver.begin[0] + ver.range[0]; i++) {
      int x = i % size;
      int y = i / size;
      sycl::float4 Gx, Gy;
      for(uint x_shift = 0; x_shift < 7; x_shift++)
        for(uint y_shift = 0; y_shift < 7; y_shift++) {
          uint xs = x + x_shift - 3;
          uint ys = y + y_shift - 3;
          if(x == xs && y == ys)
            continue;
          if(xs < 0 || xs >= size || ys < 0 || ys >= size)
            continue;
          sycl::float4 sample = input[xs + ys * size];
          int offset_x = x_shift + y_shift * radius;
          int offset_y = y_shift + x_shift * radius;
          float conv_x = kernel[offset_x];
          sycl::float4 conv4_x = sycl::float4(conv_x);
          Gx += conv4_x * sample;
          float conv_y = kernel[offset_y];
          sycl::float4 conv4_y = sycl::float4(conv_y);
          Gy += conv4_y * sample;
        }
      sycl::float4 color = hypot(Gx, Gy);
      sycl::float4 minval = sycl::float4(0.0, 0.0, 0.0, 0.0);
      sycl::float4 maxval = sycl::float4(1.0, 1.0, 1.0, 1.0);
      sycl::float4 expected = clamp(color, minval, maxval);
      sycl::float4 dif = fdim(output[i], expected);
      float length = sycl::length(dif);
      if(length > 0.01f) {
        pass = false;
        break;
      }
    }
    return pass;
  }


static std::string getBenchmarkName(BenchmarkArgs& args) {
    return "Sobel7";
  }

}; // SobelBench class


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<Sobel7Bench>();
  return 0;
}
