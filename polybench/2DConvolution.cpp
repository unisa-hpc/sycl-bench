#include <string>
#include <vector>

#include <cstdlib>

#include <sycl/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class conv2D;

void init(DATA_TYPE* A, size_t size) {
  const auto NI = size;
  const auto NJ = size;

  for(size_t i = 0; i < NI; ++i) {
    for(size_t j = 0; j < NJ; ++j) {
      A[i * NJ + j] = (float)rand() / (float)RAND_MAX;
    }
  }
}

void conv2D(DATA_TYPE* A, DATA_TYPE* B, size_t size) {
  const auto NI = size;
  const auto NJ = size;

  const DATA_TYPE c11 = +0.2, c21 = +0.5, c31 = -0.8;
  const DATA_TYPE c12 = -0.3, c22 = +0.6, c32 = -0.9;
  const DATA_TYPE c13 = +0.4, c23 = +0.7, c33 = +0.10;

  for(size_t i = 1; i < NI - 1; ++i) {
    for(size_t j = 1; j < NJ - 1; ++j) {
      B[i * NJ + j] =
          c11 * A[(i - 1) * NJ + (j - 1)] + c12 * A[(i + 0) * NJ + (j - 1)] + c13 * A[(i + 1) * NJ + (j - 1)] +
          c21 * A[(i - 1) * NJ + (j + 0)] + c22 * A[(i + 0) * NJ + (j + 0)] + c23 * A[(i + 1) * NJ + (j + 0)] +
          c31 * A[(i - 1) * NJ + (j + 1)] + c32 * A[(i + 0) * NJ + (j + 1)] + c33 * A[(i + 1) * NJ + (j + 1)];
    }
  }
}

class Polybench_2DConvolution {
public:
  Polybench_2DConvolution(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

  void setup() {
    A.resize(size * size);
    B.resize(size * size);

    init(A.data(), size);

    A_buffer.initialize(args.device_queue, A.data(), sycl::range<2>(size, size));
    B_buffer.initialize(args.device_queue, B.data(), sycl::range<2>(size, size));
  }

  void run(std::vector<sycl::event>& events) {
    using namespace sycl;

    events.push_back(args.device_queue.submit([&](handler& cgh) {
      auto A = A_buffer.get_access<access::mode::read>(cgh);
      auto B = B_buffer.get_access<access::mode::discard_write>(cgh);

      cgh.parallel_for<class conv2D>(B_buffer.get_range(), [=, size_ = size](item<2> item) {
        const auto i = item[0];
        const auto j = item[1];

        const DATA_TYPE c11 = +0.2, c21 = +0.5, c31 = -0.8;
        const DATA_TYPE c12 = -0.3, c22 = +0.6, c32 = -0.9;
        const DATA_TYPE c13 = +0.4, c23 = +0.7, c33 = +0.10;

        if((i > 0) && (j > 0) && (i < size_ - 1) && (j < size_ - 1)) {
          B[item] = c11 * A[{(i - 1), (j - 1)}] + c12 * A[{(i + 0), (j - 1)}] + c13 * A[{(i + 1), (j - 1)}] +
                    c21 * A[{(i - 1), (j + 0)}] + c22 * A[{(i + 0), (j + 0)}] + c23 * A[{(i + 1), (j + 0)}] +
                    c31 * A[{(i - 1), (j + 1)}] + c32 * A[{(i + 0), (j + 1)}] + c33 * A[{(i + 1), (j + 1)}];
        }
      });
    }));
  }

  bool verify(VerificationSetting&) {
    constexpr auto ERROR_THRESHOLD = 0.05;

    auto B_acc = B_buffer.get_host_access();

    std::vector<DATA_TYPE> B_cpu(size * size);
    conv2D(A.data(), B_cpu.data(), size);

    for(size_t i = 0; i < size; i++) {
      for(size_t j = 0; j < size; j++) {
        if((i > 0) && (j > 0) && (i < size - 1) && (j < size - 1)) {
          const auto diff = percentDiff(B_cpu[i * size + j], B_acc.get_pointer()[i * size + j]);
          if(diff > ERROR_THRESHOLD)
            return false;
        }
      }
    }

    return true;
  }

  static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_2DConvolution"; }

private:
  BenchmarkArgs args;

  const size_t size;
  std::vector<DATA_TYPE> A;
  std::vector<DATA_TYPE> B;

  PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
  PrefetchedBuffer<DATA_TYPE, 2> B_buffer;
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<Polybench_2DConvolution>();
  return 0;
}
