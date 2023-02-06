#include <iostream>
#include <sycl.hpp>

#include "common.h"

namespace s = sycl;


class FlowMapKernel; // kernel forward declaration


class FlowMap {
protected:
  size_t size;
  size_t local_size;
  int numTimesteps;
  s::float2 origin;
  s::float2 cellSize;
  float startTime;
  float advectionTime;
  int width;
  std::vector<s::float2> data;
  std::vector<float> timesteps;
  std::vector<s::float2> flowMap;

  BenchmarkArgs args;


  PrefetchedBuffer<s::float2, 1> buf_data;
  PrefetchedBuffer<float, 1> buf_timesteps;
  PrefetchedBuffer<s::float2, 1> buf_flowMap;


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


  s::float2 blend(float factor, s::float2 value1, s::float2 value2) {
    float factor2 = 1.0f - factor;
    s::float2 tmp;
    tmp.x() = factor2 * value1.x() + factor * value2.x();
    tmp.y() = factor2 * value1.y() + factor * value2.y();
    return tmp;
  }


public:
  FlowMap(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    size = args.problem_size;     // input size defined by the user
    local_size = args.local_size; // set local work_group size

    numTimesteps = 4;
    origin = {0.f, 0.f};
    cellSize = {0.1f, 0.1f};
    startTime = 1.0f;
    advectionTime = 0.5f;

    width = (int)floor(sqrt(size));
    size = width * width;


    // prepare inputs
    data.resize(size * numTimesteps);
    timesteps.resize(numTimesteps);
    flowMap.resize(size);

    fillrandom_float((float*)data.data(), size * 2, 1, -1.0f, 1.0f);
    for(int i = 0; i < numTimesteps; i++) timesteps[i] = 0.05f * i;

    fillrandom_float((float*)flowMap.data(), size, 1, -1.0f, 1.0f); // filled in case we only run the 2nd kernel


    // init buffer
    buf_data.initialize(args.device_queue, data.data(), s::range<1>(size * numTimesteps));
    buf_timesteps.initialize(args.device_queue, timesteps.data(), s::range<1>(numTimesteps));
    buf_flowMap.initialize(args.device_queue, flowMap.data(), s::range<1>(size));
  }

  void run(std::vector<s::event>& events) {
    events.push_back(args.device_queue.submit([&](s::handler& cgh) {
      auto data_acc = buf_data.get_access<s::access::mode::read>(cgh);
      auto timesteps_acc = buf_timesteps.get_access<s::access::mode::read>(cgh);
      auto output_acc = buf_flowMap.get_access<s::access::mode::write>(cgh);

      s::range<1> ndrange{size};

      cgh.parallel_for<class FlowMapKernel>(
          ndrange, [this, data_acc, width_ = width, dataOrigin = origin, dataCellSize = cellSize, timesteps_acc,
                       numTimesteps_ = numTimesteps, startTime_ = startTime, advectionTime_ = advectionTime, output_acc,
                       num_elements = size](s::id<1> id) {
            int gid = id[0];
            if(gid >= num_elements)
              return;
            int tx = gid % width_;
            int ty = gid / width_;

            const unsigned int numSteps = 1000;
            float timestep = advectionTime_ / numSteps;

            s::float2 pos;
            pos.x() = dataOrigin.x() + tx * dataCellSize.x();
            pos.y() = dataOrigin.y() + ty * dataCellSize.y();

            for(unsigned int step = 0; step < numSteps; ++step) {
              float currentTime = startTime_ + step * timestep;

              // previous time index
              int prevIndex = -1;

              for(unsigned int previous = 0; previous < numTimesteps_; ++previous) {
                if(timesteps_acc[previous] <= currentTime)
                  prevIndex = previous;
              }

              if(prevIndex < 0 || prevIndex > numTimesteps_ - 2) {
                output_acc[gid].x() = 0;
                output_acc[gid].y() = 0;
                return;
              }

              // next time index
              int nextIndex = prevIndex + 1;

              s::float2 interpolatedPrev;
              s::float2 interpolatedNext;

              s::float2 posW;
              posW.x() = (pos.x() - dataOrigin.x()) / dataCellSize.x();
              posW.y() = (pos.y() - dataOrigin.y()) / dataCellSize.y();

              // posXi,posYi is integral coordinate of "upper left corner"
              float posX = s::floor(posW.x());
              float posY = s::floor(posW.y());

              posX = s::clamp(posX, 0.0f, (float)(width_ - 2));
              posY = s::clamp(posY, 0.0f, (float)(width_ - 2));

              // get local coordinates
              s::float2 lpos;
              lpos.x() = s::clamp((float)(posW.x() - posX), 0.0f, 1.0f);
              lpos.y() = s::clamp((float)(posW.y() - posY), 0.0f, 1.0f);

              int posXi = (int)posX;
              int posYi = (int)posY;

              unsigned int timeSlice1 = width_ * width_ * prevIndex;
              unsigned int timeSlice2 = width_ * width_ * nextIndex;

              s::float2 a;
              a.x() = 0;
              a.y() = 1;
              s::float2 b;
              b.x() = 3;
              b.y() = 6;

              s::float2 vecMid1 = blend(lpos.y(),
                  blend(lpos.x(), data_acc[(posXi + 1) + posYi * width_ + timeSlice1],
                      data_acc[(posXi + 1) + posYi * width_ + timeSlice1]),
                  blend(lpos.x(), data_acc[posXi + (posYi + 1) * width_ + timeSlice1],
                      data_acc[(posXi + 1) + (posYi + 1) * width_ + timeSlice1]));

              s::float2 vecMid2 = blend(lpos.y(),
                  blend(lpos.x(), data_acc[posXi + posYi * width_ + timeSlice2],
                      data_acc[(posXi + 1) + posYi * width_ + timeSlice2]),
                  blend(lpos.x(), data_acc[posXi + (posYi + 1) * width_ + timeSlice2],
                      data_acc[(posXi + 1) + (posYi + 1) * width_ + timeSlice2]));


              interpolatedPrev = vecMid1;
              interpolatedNext = vecMid2;

              float localTime =
                  (currentTime - timesteps_acc[prevIndex]) / (timesteps_acc[nextIndex] - timesteps_acc[prevIndex]);

              s::float2 interpolated = blend(localTime, interpolatedPrev, interpolatedNext);

              pos.x() += interpolated.x() * timestep;
              pos.y() += interpolated.y() * timestep;
            }
            output_acc[gid] = pos;
          });
    }));
  }

  bool verify(VerificationSetting& ver) { return true; }


  static std::string getBenchmarkName() { return "Flow map"; }

}; // FlowMap class


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<FlowMap>();
  return 0;
}
