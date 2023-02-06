#include <iostream>
#include <sycl.hpp>

#include "common.h"

namespace s = sycl;


class SinewaveKernel; // kernel forward declaration


class Sinewave {
protected:
  size_t size; // user-defined size (input and output will be size x size)
  size_t local_size;
  BenchmarkArgs args;

  std::vector<s::float4> output;


  PrefetchedBuffer<s::float4, 1> buf_output;


public:
  Sinewave(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size;     // input size defined by the user
    local_size = args.local_size; // set local work_group size
    output.resize(size);

    buf_output.initialize(args.device_queue, output.data(), s::range<1>(size));
  }

  void run(std::vector<s::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
      auto output_acc = buf_output.get_access<s::access::mode::write>(cgh);


      s::range<1> ndrange{size};

      cgh.parallel_for<class SinewaveKernel>(ndrange, [this, output_acc, num_elements = size](s::id<1> id) {
        int gid = id[0];
        if(gid >= num_elements)
          return;

        float time = 5.0f;
        float u = gid * 2.0f - 1.0f;
        float v = gid * 3.0f - 2.0f;
        float w = gid * 4.0f - 3.0f;
        float z = gid * 5.0f - 4.0f;
        float freq = 4.0f;
        for(int i = 0; i < 50; ++i) {
          u = s::sin(u * freq + time) * s::cos(v * freq + time) * 0.5f +
              s::sin(w * freq + time) * s::cos(z * freq + time) * 0.5f;
          v = s::sin(u * freq + time) * s::cos(v * freq + time) * 0.5f +
              s::sin(w * freq + time) * s::cos(z * freq + time) * 0.5f;
          w = s::sin(u * freq + time) * s::cos(v * freq + time) * 0.5f +
              s::sin(w * freq + time) * s::cos(z * freq + time) * 0.5f;
          z = s::sin(u * freq + time) * s::cos(v * freq + time) * 0.5f +
              s::sin(w * freq + time) * s::cos(z * freq + time) * 0.5f;
        }

        s::float4 out{u, v, w, z};
        output_acc[gid] = out;
      });
    }));
  }

  bool verify(VerificationSetting& ver) { return true; }


  static std::string getBenchmarkName() { return "Sinewave"; }

}; // Sinewave class


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<Sinewave>();
  return 0;
}
