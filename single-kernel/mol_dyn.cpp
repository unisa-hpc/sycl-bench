#include "common.h"
#include <iostream>

//using namespace cl::sycl;
namespace s = cl::sycl;
class MolecularDynamicsKernel;

class MolecularDynamicsBench
{
protected:
  std::vector<s::float4> input;
  std::vector<s::float4> output;
  std::vector<int> neighbour;
  int neighCount;
  int cutsq;
  int lj1;
  float lj2;
  int inum;
  BenchmarkArgs args;

  PrefetchedBuffer<s::float4, 1> input_buf;
  PrefetchedBuffer<int, 1> neighbour_buf;
  PrefetchedBuffer<s::float4, 1> output_buf;

public:
  MolecularDynamicsBench(const BenchmarkArgs &_args) : args(_args) {}

  void setup() {
    // host memory allocation and initialization
    neighCount = 15;
    cutsq = 50;
    lj1 = 20;
    lj2 = 0.003f;
    inum = 0;

    input.resize(args.problem_size * sizeof(s::float4));
    neighbour.resize(args.problem_size);
    output.resize(args.problem_size * sizeof(s::float4));

    for(size_t i = 0; i < args.problem_size; i++) {
      input[i] = s::float4{
          (float)i, (float)i, (float)i, (float)i}; // Same value for all 4 elements. Could be changed if needed
    }
    for(size_t i = 0; i < args.problem_size; i++) {
      neighbour[i] = i + 1;
    }

    input_buf.initialize(args.device_queue, input.data(), s::range<1>(args.problem_size * sizeof(s::float4)));
    neighbour_buf.initialize(args.device_queue, neighbour.data(), s::range<1>(args.problem_size));
    output_buf.initialize(args.device_queue, output.data(), s::range<1>(args.problem_size * sizeof(s::float4)));
  }

  void run(std::vector<cl::sycl::event>& events) {
    
    events.push_back(args.device_queue.submit(
        [&](cl::sycl::handler& cgh) {
      auto in = input_buf.get_access<s::access::mode::read>(cgh);
      auto neigh = neighbour_buf.get_access<s::access::mode::read>(cgh);
      auto out = output_buf.get_access<s::access::mode::discard_write>(cgh);

      cl::sycl::range<1> ndrange (args.problem_size);

      cgh.parallel_for<class MolecularDynamicsKernel>(ndrange,
        [=, problem_size = args.problem_size, neighCount_ = neighCount,
         inum_ = inum, cutsq_ = cutsq, lj1_ = lj1, lj2_ = lj2]
        (cl::sycl::id<1> idx)
        {
            size_t gid= idx[0];

            if (gid < problem_size) {
                s::float4 ipos = in[gid];
                s::float4 f = {0.0f, 0.0f, 0.0f, 0.0f};
                int j = 0;
                while (j < neighCount_) {
                    int jidx = neigh[j*inum_ + gid];
                    s::float4 jpos = in[jidx];

                    // Calculate distance
                    float delx = ipos.x() - jpos.x();
                    float dely = ipos.y() - jpos.y();
                    float delz = ipos.z() - jpos.z();
                    float r2inv = delx*delx + dely*dely + delz*delz;

                    // If distance is less than cutoff, calculate force
                    if (r2inv < cutsq_) {
                        r2inv = 10.0f/r2inv;
                        float r6inv = r2inv * r2inv * r2inv;
                        float forceC = r2inv*r6inv*(lj1_*r6inv - lj2_);

                        f.x() += delx * forceC;
                        f.y() += dely * forceC;
                        f.z() += delz * forceC;
                    }
                    j++;
                }
                out[gid] = f;
            }
        });
    }));
  }

  bool verify(VerificationSetting &ver) {
    auto output_acc = output_buf.get_access<s::access::mode::read>();

    bool pass = true;
    unsigned equal = 1;
    const float tolerance = 0.00001;
    for(unsigned int i = 0; i < args.problem_size; ++i) {
        s::float4 ipos = input[i];
        s::float4 f = {0.0f, 0.0f, 0.0f, 0.0f};
        int j = 0;
        while (j < neighCount) {
            int jidx = neighbour[j*inum + i];
            s::float4 jpos = input[jidx];

            // Calculate distance
            float delx = ipos.x() - jpos.x();
            float dely = ipos.y() - jpos.y();
            float delz = ipos.z() - jpos.z();
            float r2inv = delx*delx + dely*dely + delz*delz;

            // If distance is less than cutoff, calculate force
            if (r2inv < cutsq) {
                r2inv = 10.0f/r2inv;
                float r6inv = r2inv * r2inv * r2inv;
                float forceC = r2inv*r6inv*(lj1*r6inv - lj2);

                f.x() += delx * forceC;
                f.y() += dely * forceC;
                f.z() += delz * forceC;
            }
            j++;
        }

        if(fabs(f.x() - output_acc[i].x()) > tolerance || fabs(f.y() - output_acc[i].y()) > tolerance ||
            fabs(f.z() - output_acc[i].z()) > tolerance) {
          pass = false;
          break;
        }
    }
    return pass;
  }
  
  static std::string getBenchmarkName() {
    return "MolecularDynamics";
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);
  app.run<MolecularDynamicsBench>();  
  return 0;
}
