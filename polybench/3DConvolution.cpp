#include <string>
#include <vector>

#include <cstdlib>

#include <sycl/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class conv3D;

void init(DATA_TYPE* A, size_t size) {
  const auto NI = size;
  const auto NJ = size;
  const auto NK = size;

  for(size_t i = 0; i < NI; ++i) {
    for(size_t j = 0; j < NJ; ++j) {
      for(size_t k = 0; k < NK; ++k) {
        A[i * (NK * NJ) + j * NK + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
      }
    }
  }
}

void conv3D(DATA_TYPE* A, DATA_TYPE* B, size_t size) {
  const auto NI = size;
  const auto NJ = size;
  const auto NK = size;

  const DATA_TYPE c11 = +2, c21 = +5, c31 = -8;
  const DATA_TYPE c12 = -3, c22 = +6, c32 = -9;
  const DATA_TYPE c13 = +4, c23 = +7, c33 = +10;

  for(size_t i = 1; i < NI - 1; ++i) {
    for(size_t j = 1; j < NJ - 1; ++j) {
      for(size_t k = 1; k < NK - 1; ++k) {
        B[i * (NK * NJ) + j * NK + k] = c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
                                        c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
                                        c21 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
                                        c23 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
                                        c31 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
                                        c33 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] +
                                        c12 * A[(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] +
                                        c22 * A[(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)] +
                                        c32 * A[(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] +
                                        c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
                                        c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] +
                                        c21 * A[(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
                                        c23 * A[(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] +
                                        c31 * A[(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)] +
                                        c33 * A[(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
      }
    }
  }
}

class Polybench_3DConvolution {
public:
  Polybench_3DConvolution(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

  void setup() {
    A.resize(size * size * size);
    B.resize(size * size * size);

    init(A.data(), size);

    A_buffer.initialize(args.device_queue, A.data(), sycl::range<3>(size, size, size));
    B_buffer.initialize(args.device_queue, B.data(), sycl::range<3>(size, size, size));
  }

  void run(std::vector<sycl::event>& events) {
    using namespace sycl;

    events.push_back(args.device_queue.submit([&](handler& cgh) {
      auto A = A_buffer.get_access<access::mode::read>(cgh);
      auto B = B_buffer.get_access<access::mode::discard_write>(cgh);

      cgh.parallel_for<class conv3D>(B_buffer.get_range(), [=, size_ = size](item<3> item) {
        const auto i = item[0];
        const auto j = item[1];
        const auto k = item[2];

        const DATA_TYPE c11 = +2, c21 = +5, c31 = -8;
        const DATA_TYPE c12 = -3, c22 = +6, c32 = -9;
        const DATA_TYPE c13 = +4, c23 = +7, c33 = +10;

        if((i > 0) && (j > 0) && (k > 0) && (i < (size_ - 1)) && (j < (size_ - 1)) && (k < (size_ - 1))) {
          B[item] = c11 * A[{(i - 1), (j - 1), (k - 1)}] + c13 * A[{(i + 1), (j - 1), (k - 1)}] +
                    c21 * A[{(i - 1), (j - 1), (k - 1)}] + c23 * A[{(i + 1), (j - 1), (k - 1)}] +
                    c31 * A[{(i - 1), (j - 1), (k - 1)}] + c33 * A[{(i + 1), (j - 1), (k - 1)}] +
                    c12 * A[{(i + 0), (j - 1), (k + 0)}] + c22 * A[{(i + 0), (j + 0), (k + 0)}] +
                    c32 * A[{(i + 0), (j + 1), (k + 0)}] + c11 * A[{(i - 1), (j - 1), (k + 1)}] +
                    c13 * A[{(i + 1), (j - 1), (k + 1)}] + c21 * A[{(i - 1), (j + 0), (k + 1)}] +
                    c23 * A[{(i + 1), (j + 0), (k + 1)}] + c31 * A[{(i - 1), (j + 1), (k + 1)}] +
                    c33 * A[{(i + 1), (j + 1), (k + 1)}];
        }
      });
    }));
  }


  bool verify(VerificationSetting&) {
    constexpr auto ERROR_THRESHOLD = 0.05;

    std::vector<DATA_TYPE> B_cpu(size * size * size);
    conv3D(A.data(), B_cpu.data(), size);

    auto B_acc = B_buffer.get_host_access();

    for(size_t i = 0; i < size; i++) {
      for(size_t j = 0; j < size; j++) {
        for(size_t k = 0; k < size; k++) {
          if((i > 0) && (j > 0) && (k > 0) && (i < (size - 1)) && (j < (size - 1)) && (k < (size - 1))) {
            const auto diff = percentDiff(
                B_cpu[i * (size * size) + j * size + k], B_acc.get_pointer()[i * (size * size) + j * size + k]);
            if(diff > ERROR_THRESHOLD)
              return false;
          }
        }
      }
    }

    return true;
  }

  static std::string getBenchmarkName(BenchmarkArgs& args) { return "Polybench_3DConvolution"; }

private:
  BenchmarkArgs args;

  const size_t size;
  std::vector<DATA_TYPE> A;
  std::vector<DATA_TYPE> B;

  PrefetchedBuffer<DATA_TYPE, 3> A_buffer;
  PrefetchedBuffer<DATA_TYPE, 3> B_buffer;
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<Polybench_3DConvolution>();
  return 0;
}
