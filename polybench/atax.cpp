#include <string>
#include <vector>

#include <cstdlib>

#include <sycl/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

#ifndef M_PI
#define M_PI 3.14159
#endif

using DATA_TYPE = float;

class Atax1;
class Atax2;

void init_array(DATA_TYPE* x, DATA_TYPE* A, size_t size) {
  const auto NX = size;
  const auto NY = size;

  for(size_t i = 0; i < NX; i++) {
    x[i] = i * M_PI;
    for(size_t j = 0; j < NY; j++) {
      A[i * NY + j] = ((DATA_TYPE)i * (j)) / NX;
    }
  }
}

void atax_cpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp, size_t size) {
  const auto NX = size;
  const auto NY = size;

  for(size_t i = 0; i < NX; i++) {
    for(size_t j = 0; j < NY; j++) {
      tmp[i] += A[i * NY + j] * x[j];
    }

    for(size_t j = 0; j < NY; j++) {
      y[j] += A[i * NY + j] * tmp[i];
    }
  }
}

class Polybench_Atax {
public:
  Polybench_Atax(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

  void setup() {
    A.resize(size * size);
    x.resize(size);
    y.resize(size);
    tmp.resize(size);

    init_array(x.data(), A.data(), size);

    A_buffer.initialize(args.device_queue, A.data(), sycl::range<2>{size, size});
    x_buffer.initialize(args.device_queue, x.data(), sycl::range<1>{size});
    y_buffer.initialize(args.device_queue, y.data(), sycl::range<1>{size});
    tmp_buffer.initialize(args.device_queue, tmp.data(), sycl::range<1>{size});
  }

  void run(std::vector<sycl::event>& events) {
    using namespace sycl;

    events.push_back(args.device_queue.submit([&](handler& cgh) {
      auto A = A_buffer.get_access<access::mode::read>(cgh);
      auto x = x_buffer.get_access<access::mode::read>(cgh);
      auto tmp = tmp_buffer.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<Atax1>(tmp_buffer.get_range(), [=, size_ = size](item<1> item) {
        const auto i = item[0];

        for(size_t j = 0; j < size_; j++) {
          tmp[item] += A[{i, j}] * x[j];
        }
      });
    }));

    events.push_back(args.device_queue.submit([&](handler& cgh) {
      auto A = A_buffer.get_access<access::mode::read>(cgh);
      auto y = y_buffer.get_access<access::mode::read_write>(cgh);
      auto tmp = tmp_buffer.get_access<access::mode::read>(cgh);

      cgh.parallel_for<Atax2>(y_buffer.get_range(), [=, size_ = size](item<1> item) {
        const auto j = item[0];

        for(size_t i = 0; i < size_; i++) {
          y[item] += A[{i, j}] * tmp[i];
        }
      });
    }));
  }

  bool verify(VerificationSetting&) {
    constexpr auto ERROR_THRESHOLD = 0.05;

    init_array(x.data(), A.data(), size);

    std::vector<DATA_TYPE> y_cpu(size);
    std::vector<DATA_TYPE> tmp_cpu(size);

    atax_cpu(A.data(), x.data(), y_cpu.data(), tmp_cpu.data(), size);

    auto y_acc = y_buffer.get_host_access();

    for(size_t i = 0; i < size; i++) {
      const auto diff = percentDiff(y_cpu[i], y_acc[i]);
      if(diff > ERROR_THRESHOLD)
        return false;
    }

    return true;
  }

  static std::string getBenchmarkName() { return "Polybench_Atax"; }

private:
  BenchmarkArgs args;

  const size_t size;
  std::vector<DATA_TYPE> A;
  std::vector<DATA_TYPE> x;
  std::vector<DATA_TYPE> y;
  std::vector<DATA_TYPE> tmp;

  PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
  PrefetchedBuffer<DATA_TYPE, 1> x_buffer;
  PrefetchedBuffer<DATA_TYPE, 1> y_buffer;
  PrefetchedBuffer<DATA_TYPE, 1> tmp_buffer;
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<Polybench_Atax>();
  return 0;
}
