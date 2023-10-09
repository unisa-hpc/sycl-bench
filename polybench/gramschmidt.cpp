#include <string>
#include <vector>

#include <cmath>
#include <cstdlib>

#include <sycl/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

using DATA_TYPE = float;

class Gramschmidt1;
class Gramschmidt2;
class Gramschmidt3;

void init_array(DATA_TYPE* A, size_t size) {
	const auto M = size;
	const auto N = size;

  for(size_t i = 0; i < M; i++) {
    for(size_t j = 0; j < N; j++) {
      A[i * N + j] = ((DATA_TYPE)(i + 1) * (j + 1)) / (M + 1);
    }
  }
}

void gramschmidt(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q, size_t size) {
	const auto M = size;
	const auto N = size;

  for(size_t k = 0; k < N; k++) {
    DATA_TYPE nrm = 0;
    for(size_t i = 0; i < M; i++) {
      nrm += A[i * N + k] * A[i * N + k];
    }

    R[k * N + k] = sqrt(nrm);
    for(size_t i = 0; i < M; i++) {
      Q[i * N + k] = A[i * N + k] / R[k * N + k];
    }

    for(size_t j = k + 1; j < N; j++) {
      R[k * N + j] = 0;
      for(size_t i = 0; i < M; i++) {
        R[k * N + j] += Q[i * N + k] * A[i * N + j];
      }
      for(size_t i = 0; i < M; i++) {
        A[i * N + j] = A[i * N + j] - Q[i * N + k] * R[k * N + j];
      }
    }
  }
}

class Polybench_Gramschmidt {
public:
  Polybench_Gramschmidt(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

  void setup() {
    A.resize(size * size);
    R.resize(size * size);
    Q.resize(size * size);

    init_array(A.data(), size);

    A_buffer.initialize(args.device_queue, A.data(), sycl::range<2>(size, size));
    R_buffer.initialize(args.device_queue, R.data(), sycl::range<2>(size, size));
    Q_buffer.initialize(args.device_queue, Q.data(), sycl::range<2>(size, size));
  }

  void run(std::vector<sycl::event>& events) {
    using namespace sycl;

    for(size_t k = 0; k < size; k++) {
      events.push_back(args.device_queue.submit([&](handler& cgh) {
        auto A = A_buffer.get_access<access::mode::read>(cgh);
        auto R = R_buffer.get_access<access::mode::write>(cgh);

        cgh.parallel_for<Gramschmidt1>(range<2>(1, 1), [=, M_ = size](item<2> item) {
          DATA_TYPE nrm = 0;
          for(size_t i = 0; i < M_; i++) {
            nrm += A[{i, k}] * A[{i, k}];
          }
          R[{k, k}] = sycl::sqrt(nrm);
        });
      }));

      events.push_back(args.device_queue.submit([&](handler& cgh) {
        auto A = A_buffer.get_access<access::mode::read>(cgh);
        auto R = R_buffer.get_access<access::mode::read>(cgh);
        auto Q = Q_buffer.get_access<access::mode::write>(cgh);

        cgh.parallel_for<Gramschmidt2>(range<2>(size, 1), [=](item<2> gid) {
          const id<2> offset(0, k);
          Q[gid + offset] = A[gid + offset] / R[{k, k}];
        });
      }));

      events.push_back(args.device_queue.submit([&](handler& cgh) {
        auto A = A_buffer.get_access<access::mode::read_write>(cgh);
        auto R = R_buffer.get_access<access::mode::write>(cgh);
        auto Q = Q_buffer.get_access<access::mode::read>(cgh);

        cgh.parallel_for<Gramschmidt3>(range<2>(size, 1), [=, M_ = size, N_ = size](item<2> item) {
          const auto j = item[0];

          if(j <= k || j >= N_)
            return;

          R[item] = 0;
          for(size_t i = 0; i < M_; i++) {
            R[item] += Q[{i, k}] * A[{i, j}];
          }

          for(size_t i = 0; i < M_; i++) {
            A[{i, j}] -= Q[{i, k}] * R[item];
          }
        });
      }));
    }
  }

  bool verify(VerificationSetting&) {
    constexpr auto ERROR_THRESHOLD = 0.05;

    std::vector<DATA_TYPE> A_cpu(size * size);
    std::vector<DATA_TYPE> R_cpu(size * size);
    std::vector<DATA_TYPE> Q_cpu(size * size);

    // Trigger writeback
    A_buffer.reset();

    init_array(A_cpu.data(), size);

    gramschmidt(A_cpu.data(), R_cpu.data(), Q_cpu.data(), size);

    for(size_t i = 0; i < size; i++) {
      for(size_t j = 0; j < size; j++) {
        const auto diff = percentDiff(A_cpu[i * size + j], A[i * size + j]);
        if(diff > ERROR_THRESHOLD)
          return false;
      }
    }

    return true;
  }

  static std::string getBenchmarkName() { return "Polybench_Gramschmidt"; }

private:
  BenchmarkArgs args;

  const size_t size;
  std::vector<DATA_TYPE> A;
  std::vector<DATA_TYPE> R;
  std::vector<DATA_TYPE> Q;

  PrefetchedBuffer<DATA_TYPE, 2> A_buffer;
  PrefetchedBuffer<DATA_TYPE, 2> R_buffer;
  PrefetchedBuffer<DATA_TYPE, 2> Q_buffer;
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<Polybench_Gramschmidt>();
  return 0;
}
