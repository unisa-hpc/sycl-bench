#include <iostream>
#include <sycl.hpp>

#include "common.h"

namespace s = sycl;


class FtleKernel; // kernel forward declaration


class Ftle {
protected:
  size_t size; // user-defined size (input and output will be size x size)
  size_t local_size;
  int numTimesteps;
  s::float2 origin;
  s::float2 cellSize;
  float startTime;
  float advectionTime;
  int width;
  std::vector<s::float2> flowMap;
  std::vector<s::float2> output;


  BenchmarkArgs args;


  PrefetchedBuffer<s::float2, 1> buf_output;
  PrefetchedBuffer<s::float2, 1> buf_flowMap;


  s::float4 float4mul(s::float4 a, s::float4 b) {
    s::float4 c;
    c.x() = a.x() * b.x() + a.y() * b.y();
    c.y() = a.x() * b.y() + a.y() * b.w();
    c.z() = a.z() * b.x() + a.w() * b.z();
    c.w() = a.z() * b.y() + a.w() * b.w();
    return c;
  }

  s::float4 float4trp(s::float4 a) {
    s::float4 b = a;
    float x;
    x = b.y();
    b.y() = b.z();
    b.z() = x;
    return b;
  }

  s::float4 float4symm(s::float4 a) {
    s::float4 b = float4trp(a);
    b = b + a;
    b = b * 0.5f;
    return b;
  }

  s::float2 float4invariants(s::float4 m) {
    s::float2 pqr;
    pqr.x() = m.x() * m.w() - m.y() * m.z();
    pqr.y() = -(m.x() + m.w());
    return pqr;
  }


  s::float2 float2squareroots(s::float2 a) {
    float discrim, root;
    s::float2 b;
    discrim = a.y() * a.y() - 4 * a.x();

    if(discrim >= 0) {
      root = s::sqrt(discrim);
      b.x() = (-a.y() - root) / 2.0f;
      b.y() = (-a.y() + root) / 2.0f;
    } else {
      root = s::sqrt(-discrim);
      b.x() = -a.x() / 2.0f;
      b.y() = root / 2.0f;
    }
    return b;
  }


  s::float2 float4eigenvalues(s::float4 m) {
    s::float2 pqr;
    pqr = float4invariants(m);
    return (float2squareroots(pqr));
  }


  static inline float random01_float() { return (float)rand() / ((float)RAND_MAX); }

  void fillrandom_float(float* arrayPtr, int width, int height, float rangeMin, float rangeMax) {
    if(!arrayPtr) {
      fprintf(stderr, "Cannot fill array: NULL pointer.\n");
      return;
    }
    double range = (double)(rangeMax - rangeMin);
    for(int i = 0; i < height; i++)
      for(int j = 0; j < width; j++) {
        int index = i * width + j;
        arrayPtr[index] = rangeMin + (float)(range * random01_float());
      }
  }


public:
  Ftle(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size;     // input size defined by the user
    local_size = args.local_size; // set local work_group size


    int numTimesteps = 4;
    origin = {0.f, 0.f};
    cellSize = {0.1f, 0.1f};
    startTime = 1.0f;
    advectionTime = 0.5f;

    int width = (int)floor(sqrt(size));

    size = width * width;

    flowMap.resize(size);
    output.resize(size);
    fillrandom_float((float*)flowMap.data(), size, 1, -1.0f, 1.0f); // filled in case we only run the 2nd kernel

    // init buffer
    buf_output.initialize(args.device_queue, output.data(), s::range<1>(size));
    buf_flowMap.initialize(args.device_queue, flowMap.data(), s::range<1>(size));
  }

  void run(std::vector<s::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
      auto flowMap_acc = buf_flowMap.get_access<s::access::mode::read>(cgh);
      auto output_acc = buf_output.get_access<s::access::mode::write>(cgh);

      // size_t szLocalWorkSize = local_size;
      // float multiplier = size/(float)szLocalWorkSize;
      // if(multiplier > (int)multiplier)
      // 	multiplier += 1;
      // size_t szGlobalWorkSize = (int)multiplier * szLocalWorkSize;

      s::range<1> ndrange{size};

      cgh.parallel_for<class FtleKernel>(
          ndrange, [this, flowMap_acc, width_ = width, dataCellSize = cellSize, advectionTime_ = advectionTime,
                       output_acc, num_elements = size](s::id<1> id) {
            int gid = id[0];
            if(gid >= num_elements)
              return;
            int tx = gid % width_;
            int ty = gid / width_;

            if(tx >= 1 && tx < (width_ - 1) && ty >= 1 && ty < num_elements / width_ - 1) {
              s::float2 left = flowMap_acc[gid - 1];
              s::float2 right = flowMap_acc[gid + 1];
              s::float2 top = flowMap_acc[gid - width_];
              s::float2 bottom = flowMap_acc[gid + width_];

              s::float2 delta2;
              delta2.x() = 2.0f * dataCellSize.x();
              delta2.y() = 2.0f * dataCellSize.y();

              s::float4 jacobi;
              jacobi.x() = (right.x() - left.x()) / delta2.x();
              jacobi.y() = (bottom.x() - top.x()) / delta2.y();
              jacobi.z() = (right.y() - left.y()) / delta2.x();
              jacobi.w() = (bottom.y() - top.y()) / delta2.y();

              s::float4 jacobiT, cauchy, cauchySymm;
              jacobiT = float4trp(jacobi);
              cauchy = float4mul(jacobiT, jacobi);
              cauchySymm = float4symm(cauchy);

              s::float2 eigenvalues;
              eigenvalues = float4eigenvalues(cauchySymm);
              eigenvalues = eigenvalues + float4eigenvalues(cauchySymm);
              eigenvalues = eigenvalues + float4eigenvalues(cauchySymm);
              eigenvalues = eigenvalues + float4eigenvalues(cauchySymm);
              float maxEigenvalue = s::max(eigenvalues.x(), eigenvalues.y());

              output_acc[gid] = 1.0 / s::fabs(advectionTime) * s::log(s::sqrt(maxEigenvalue));
            }
          });
    }));
  }

  bool verify(VerificationSetting& ver) { return true; }


  static std::string getBenchmarkName() { return "Ftle"; }

}; //


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<Ftle>();
  return 0;
}
