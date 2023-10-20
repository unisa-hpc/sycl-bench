#include <iostream>
#include <sycl/sycl.hpp>

#include "common.h"

#define MT_RNG_COUNT 4096
#define MT_MM 9
#define MT_NN 19
#define MT_WMASK 0xFFFFFFFFU
#define MT_UMASK 0xFFFFFFFEU
#define MT_LMASK 0x1U
#define MT_SHIFT0 12
#define MT_SHIFTB 7
#define MT_SHIFTC 15
#define MT_SHIFT1 18
#define PI 3.14159265358979f

namespace s = sycl;


class MerseTwisterKernel; // kernel forward declaration


class MerseTwister {
protected:
  size_t w, h; // size of the input picture
  size_t size; // user-defined size (input and output will be size x size)
  size_t local_size;
  BenchmarkArgs args;


  std::vector<uint> ma;
  std::vector<uint> b;
  std::vector<uint> c;
  std::vector<uint> seed;
  std::vector<s::float4> result;


  PrefetchedBuffer<uint, 1> buf_ma;
  PrefetchedBuffer<uint, 1> buf_b;
  PrefetchedBuffer<uint, 1> buf_c;
  PrefetchedBuffer<uint, 1> buf_seed;
  PrefetchedBuffer<s::float4, 1> buf_result;


public:
  MerseTwister(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size;     // input size defined by the user
    local_size = args.local_size; // set local work_group size
    ma.resize(size);
    b.resize(size);
    c.resize(size);
    seed.resize(size);
    result.resize(size);

    for(uint i = 0; i < size; ++i) {
      ma[i] = i;
      b[i] = i;
      c[i] = i;
      seed[i] = i;
    }

    // init buffer
    buf_ma.initialize(args.device_queue, ma.data(), s::range<1>(size));
    buf_b.initialize(args.device_queue, b.data(), s::range<1>(size));
    buf_c.initialize(args.device_queue, c.data(), s::range<1>(size));
    buf_seed.initialize(args.device_queue, seed.data(), s::range<1>(size));
    buf_result.initialize(args.device_queue, result.data(), s::range<1>(size));
  }

  void run(std::vector<s::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
      auto ma_acc = buf_ma.get_access<s::access::mode::read>(cgh);
      auto b_acc = buf_b.get_access<s::access::mode::read>(cgh);
      auto c_acc = buf_c.get_access<s::access::mode::read>(cgh);
      auto seed_acc = buf_seed.get_access<s::access::mode::read>(cgh);
      auto result_acc = buf_result.get_access<s::access::mode::write>(cgh);

      s::range<1> ndrange{size};

      cgh.parallel_for<class MerseTwisterKernel>(
          ndrange, [this, ma_acc, b_acc, c_acc, seed_acc, result_acc, length = size](s::id<1> id) {
            int gid = id[0];

            if(gid >= length)
              return;

            int iState, iState1, iStateM;
            unsigned int mti, mti1, mtiM, x;
            unsigned int matrix_a, mask_b, mask_c;

            unsigned int mt[MT_NN]; // FIXME

            matrix_a = ma_acc[gid];
            mask_b = b_acc[gid];
            mask_c = c_acc[gid];

            mt[0] = seed_acc[gid];
            for(iState = 1; iState < MT_NN; iState++)
              mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;

            iState = 0;
            mti1 = mt[0];

            float tmp[5];
            for(int i = 0; i < 4; ++i) {
              iState1 = iState + 1;
              iStateM = iState + MT_MM;
              if(iState1 >= MT_NN)
                iState1 -= MT_NN;
              if(iStateM >= MT_NN)
                iStateM -= MT_NN;
              mti = mti1;
              mti1 = mt[iState1];
              mtiM = mt[iStateM];

              x = (mti & MT_UMASK) | (mti1 & MT_LMASK);
              x = mtiM ^ (x >> 1) ^ ((x & 1) ? matrix_a : 0);

              mt[iState] = x;
              iState = iState1;

              // Tempering transformation
              x ^= (x >> MT_SHIFT0);
              x ^= (x << MT_SHIFTB) & mask_b;
              x ^= (x << MT_SHIFTC) & mask_c;
              x ^= (x >> MT_SHIFT1);

              tmp[i] = ((float)x + 1.0f) / 4294967296.0f;
            }

            s::float4 val;
            val.s0() = tmp[0];
            val.s1() = tmp[1];
            val.s2() = tmp[2];
            val.s3() = tmp[3];

            result_acc[gid] = val;
          });
    }));
  }

  bool verify(VerificationSetting& ver) { return true; }


  static std::string getBenchmarkName() { return "Merse Twister"; }

}; // MerseTwister class


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<MerseTwister>();
  return 0;
}
