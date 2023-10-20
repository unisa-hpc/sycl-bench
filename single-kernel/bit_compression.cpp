#include <iostream>
#include <sycl/sycl.hpp>

#include "common.h"

namespace s = sycl;


class BitCompressionKernel; // kernel forward declaration


class BitCompression {
protected:
  size_t w, h; // size of the input picture
  size_t size; // user-defined size (input and output will be size x size)
  size_t local_size;
  std::vector<s::uint4> input;

  BenchmarkArgs args;

  PrefetchedBuffer<uint, 1> buf_bits;
  PrefetchedBuffer<s::uint4, 1> buf_input;
  PrefetchedBuffer<uint, 1> buf_output;

public:
  BitCompression(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size;     // input size defined by the user
    local_size = args.local_size; // set local work_group size
                                  // declare some variables for intializing data
    uint* num_bits = (uint*)malloc(sizeof(uint) * size);

    input.resize(size);
    uint* output = (uint*)malloc(sizeof(uint) * size);
    // init data
    for(uint i = 0; i < size; ++i) {
      input[i] = {15, 15, 15, 15};
      num_bits[i] = (int)pow(2, ((i % 3) + 1));
      output[i] = 0;
    }

    // init buffer
    buf_input.initialize(args.device_queue, input.data(), s::range<1>(size));
    buf_bits.initialize(args.device_queue, num_bits, s::range<1>(size));
    buf_output.initialize(args.device_queue, output, s::range<1>(size));
  }

  void run(std::vector<s::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
      auto input_acc = buf_input.get_access<s::access::mode::read>(cgh);
      auto num_bits_acc = buf_bits.get_access<s::access::mode::read>(cgh);
      auto output_acc = buf_output.get_access<s::access::mode::write>(cgh);


      s::range<1> ndrange{size};

      cgh.parallel_for<class BitCompressionKernel>(
          ndrange, [input_acc, num_bits_acc, output_acc, size_ = size](s::id<1> id) {
            int gid = id[0];
            if(gid >= size_)
              return;

            s::uint4 in = input_acc[gid];
            int bits = num_bits_acc[gid];
            uint tmp = 0;
            if(bits == 2) {
              tmp |= (in.x() << (32 - bits)) & 3221225472u;
              tmp |= (in.y() << (28 - bits)) & 805306368u;
              tmp |= (in.z() << (24 - bits)) & 201326592u;
              tmp |= (in.w() << (20 - bits)) & 50331648u;
            } else if(bits == 4) {
              tmp |= (in.x() << (32 - bits)) & 4026531840u;
              tmp |= (in.y() << (28 - bits)) & 251658240u;
              tmp |= (in.z() << (24 - bits)) & 15728640u;
              tmp |= (in.w() << (20 - bits)) & 983040u;
            } else if(bits == 8) {
              tmp |= (in.x() << (32 - bits)) & 4278190080u;
              tmp |= (in.y() << (28 - bits)) & 16711680u;
              tmp |= (in.z() << (24 - bits)) & 65280u;
              tmp |= (in.w() << (20 - bits)) & 255u;
            }
            output_acc[gid] = tmp;
          });
    }));
  }

  bool verify(VerificationSetting& ver) { return true; }


  static std::string getBenchmarkName() { return "Bit Compression"; }

}; // BitCompression class


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<BitCompression>();
  return 0;
}
