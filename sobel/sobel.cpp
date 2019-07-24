#include <CL/sycl.hpp>
#include <iostream>

#include "common.h"
#include "bitmap.h"
#include "utils.h"

namespace s = cl::sycl;
namespace c = celerity;
class SobelBenchKernel; // kernel forward declaration

/*
  A Sobel filter with a convolution matrix 3x3.
  Input and output are two-dimensional buffers of floats.     
 */
class SobelBench
{
protected:
  std::vector<s::float4> input;
  std::vector<s::float4> output;

  size_t w, h; // size of the input picture
  size_t size; // user-defined size (input and output will be size x size)
  BenchmarkArgs args;

public:
  SobelBench(const BenchmarkArgs &_args) : args(_args) {}

  void setup()
  {
    size = args.problem_size; // input size defined by the user
    input.resize(size * size);
    load_bitmap_mirrored("../Brommy.bmp", size, input);
    output.resize(size * size);
  }

  void run()
  {
    c::buffer<s::float4, 2> input_buf(input.data(), s::range<2>(size, size));
    c::buffer<s::float4, 2> output_buf(output.data(), s::range<2>(size, size));

    args.device_queue->submit([=](c::handler &cgh) {
      auto in = input_buf.get_access<s::access::mode::read>(cgh, c::access::neighborhood<2>(1, 1));
      auto out = output_buf.get_access<s::access::mode::discard_write>(cgh, c::access::one_to_one<2>());
      cl::sycl::range<2> ndrange{size, size};

      // Sobel kernel 3x3
      const float kernel[] =
          {1, 0, -1,
           2, 0, -2,
           1, 0, -1};

      // NOTE: prevent "this" capture
      int si = size;

      cgh.parallel_for<class SobelBenchKernel>(ndrange, [=](cl::sycl::id<2> gid) {
        int x = gid[0];
        int y = gid[1];
        s::float4 Gx = f4(0);
        s::float4 Gy = f4(0);
        const int radius = 3;

        // constant-size loops in [0,1,2]
        for (int x_shift = 0; x_shift < 3; x_shift++)
        {
          for (int y_shift = 0; y_shift < 3; y_shift++)
          {
            // sample position
            int xs = x + x_shift - 1; // [x-1,x,x+1]
            int ys = y + y_shift - 1; // [y-1,y,y+1]
            // for the same pixel, convolution is always 0
            if (x == xs && y == ys)
              continue;
            // boundary check
            if (xs < 0 || xs >= si || ys < 0 || ys >= si)
              continue;

            // sample color
            s::float4 sample = in[{(size_t)xs, (size_t)ys}];

            // convolution calculation
            int offset_x = x_shift + y_shift * radius;
            int offset_y = y_shift + x_shift * radius;

            float conv_x = kernel[offset_x];
            s::float4 conv4_x = f4(conv_x);
            Gx += conv4_x * sample;

            float conv_y = kernel[offset_y];
            s::float4 conv4_y = f4(conv_y);
            Gy += conv4_y * sample;
          }
        }
        // taking root of sums of squares of Gx and Gy
        s::float4 color = s::hypot(Gx, Gy);
        out[gid] = s::clamp(color, f4(0.0), f4(1.0));
      });
    });

    // synchronize host image (celerity)
    args.device_queue->with_master_access([=](c::handler &cgh) {
      auto out = output_buf.get_access<cl::sycl::access::mode::read>(cgh, s::range<2>(size, size));

      cgh.run([=]() {
        memcpy(output.data(), input.data(), 1);
      });
    });
  }

  bool verify(VerificationSetting &ver)
  {
    save_bitmap("sobel3.bmp", size, output);

    const float kernel[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    bool pass = true;
    int radius = 3;
    for (size_t i = ver.begin[0]; i < ver.begin[0] + ver.range[0]; i++)
    {
      int x = i % size;
      int y = i / size;
      s::float4 Gx, Gy;
      for (uint x_shift = 0; x_shift < 3; x_shift++)
      {
        for (uint y_shift = 0; y_shift < 3; y_shift++)
        {
          uint xs = x + x_shift - 1;
          uint ys = y + y_shift - 1;
          if (x == xs && y == ys)
            continue;
          if (xs < 0 || xs >= size || ys < 0 || ys >= size)
            continue;
          s::float4 sample = input[xs + ys * size];
          int offset_x = x_shift + y_shift * radius;
          int offset_y = y_shift + x_shift * radius;
          float conv_x = kernel[offset_x];
          s::float4 conv4_x = f4(conv_x);
          Gx += conv4_x * sample;
          float conv_y = kernel[offset_y];
          s::float4 conv4_y = f4(conv_y);
          Gy += conv4_y * sample;
        }
      }
      s::float4 color = s::hypot(Gx, Gy);
      s::float4 expected = s::clamp(color, f4(0.0), f4(1.0));
      s::float4 dif = s::fdim(output[i], expected);
      float length = cl::sycl::length(dif);
      if (length > 0.01f)
      {
        pass = false;
        break;
      }
    }
    return pass;
  }

  static std::string getBenchmarkName()
  {
    return "Sobel3";
  }

}; // SobelBench class

int main(int argc, char **argv)
{
  BenchmarkApp app(argc, argv);
  app.run<SobelBench>();
  return 0;
}
