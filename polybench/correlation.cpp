#include <string>
#include <vector>

#include <cmath>
#include <cstdlib>

#include <sycl/sycl.hpp>

#include "common.h"
#include "polybenchUtilFuncts.h"

#define FLOAT_N 3214212.01
#define EPS 0.005

#define sqrt_of_array_cell(x, j) sqrt(x[j])

using DATA_TYPE = float;

class CorrelationMean;
class CorrelationStd;
class CorrelationReduce;
class CorrelationCorr;
class Correlation5;

void init_arrays(DATA_TYPE* data, size_t size) {
  const auto M = size;
  const auto N = size;

  for(size_t i = 0; i <= M; i++) {
    for(size_t j = 0; j <= N; j++) {
      data[i * N + j] = ((DATA_TYPE)i * j) / (M + 1);
    }
  }
}

void correlation(DATA_TYPE* data, DATA_TYPE* mean, DATA_TYPE* stddev, DATA_TYPE* symmat, size_t size) {
  const auto M = size;
  const auto N = size;

  // Determine mean of column vectors of input data matrix
  for(size_t j = 1; j <= M; j++) {
    mean[j] = 0.0;

    for(size_t i = 1; i <= N; i++) {
      mean[j] += data[i * (M + 1) + j];
    }

    mean[j] /= (DATA_TYPE)FLOAT_N;
  }

  // Determine standard deviations of column vectors of data matrix.
  for(size_t j = 1; j <= M; j++) {
    stddev[j] = 0.0;

    for(size_t i = 1; i <= N; i++) {
      stddev[j] += (data[i * (M + 1) + j] - mean[j]) * (data[i * (M + 1) + j] - mean[j]);
    }

    stddev[j] /= FLOAT_N;
    stddev[j] = sqrt_of_array_cell(stddev, j);
    stddev[j] = stddev[j] <= EPS ? 1.0 : stddev[j];
  }

  // Center and reduce the column vectors.
  for(size_t i = 1; i <= N; i++) {
    for(size_t j = 1; j <= M; j++) {
      data[i * (M + 1) + j] -= mean[j];
      data[i * (M + 1) + j] /= sqrt(FLOAT_N);
      data[i * (M + 1) + j] /= stddev[j];
    }
  }

  // Calculate the m * m correlation matrix.
  for(size_t j1 = 1; j1 <= M - 1; j1++) {
    symmat[j1 * (M + 1) + j1] = 1.0;

    for(size_t j2 = j1 + 1; j2 <= M; j2++) {
      symmat[j1 * (M + 1) + j2] = 0.0;

      for(size_t i = 1; i <= N; i++) {
        symmat[j1 * (M + 1) + j2] += (data[i * (M + 1) + j1] * data[i * (M + 1) + j2]);
      }

      symmat[j2 * (M + 1) + j1] = symmat[j1 * (M + 1) + j2];
    }
  }

  symmat[M * (M + 1) + M] = 1.0;
}

class Polybench_Correlation {
public:
  Polybench_Correlation(const BenchmarkArgs& args) : args(args), size(args.problem_size) {}

  void setup() {
    data.resize((size + 1) * (size + 1));
    mean.resize(size + 1);
    stddev.resize(size + 1);
    symmat.resize((size + 1) * (size + 1));

    init_arrays(data.data(), size);

    data_buffer.initialize(args.device_queue, data.data(), sycl::range<2>(size + 1, size + 1));
    mean_buffer.initialize(args.device_queue, mean.data(), sycl::range<1>(size + 1));
    stddev_buffer.initialize(args.device_queue, stddev.data(), sycl::range<1>(size + 1));
    symmat_buffer.initialize(args.device_queue, symmat.data(), sycl::range<2>(size + 1, size + 1));
  }

  void run(std::vector<sycl::event>& events) {
    using namespace sycl;

    events.push_back(args.device_queue.submit([&](handler& cgh) {
      auto data = data_buffer.get_access<access::mode::read>(cgh);
      auto mean = mean_buffer.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<CorrelationMean>(range<1>(size), [=, N_ = size](id<1> gid) {
        const id<1> offset(1);
        const auto j = gid[0] + offset[0];

        for(size_t i = 1; i <= N_; i++) {
          mean[gid + offset] += data[{i, j}];
        }
        mean[gid + offset] /= ((DATA_TYPE)FLOAT_N);
      });
    }));

    events.push_back(args.device_queue.submit([&](handler& cgh) {
      auto data = data_buffer.get_access<access::mode::read>(cgh);
      auto mean = mean_buffer.get_access<access::mode::read>(cgh);
      auto stddev = stddev_buffer.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<CorrelationStd>(range<1>(size), [=, N_ = size](id<1> gid) {
        const id<1> offset(1);
        const auto adj_id = gid + offset;
        const auto j = gid[0] + offset[0];

        for(size_t i = 1; i <= N_; i++) {
          stddev[adj_id] += (data[{i, j}] - mean[adj_id]) * (data[{i, j}] - mean[adj_id]);
        }

        stddev[adj_id] /= FLOAT_N;
        stddev[adj_id] = sycl::sqrt(stddev[adj_id]);
        stddev[adj_id] = stddev[adj_id] <= EPS ? 1.0 : stddev[adj_id];
      });
    }));

    events.push_back(args.device_queue.submit([&](handler& cgh) {
      auto data = data_buffer.get_access<access::mode::read_write>(cgh);
      auto mean = mean_buffer.get_access<access::mode::read>(cgh);
      auto stddev = stddev_buffer.get_access<access::mode::read>(cgh);

      cgh.parallel_for<CorrelationReduce>(range<2>(size, size), [=](id<2> gid) {
        const id<2> offset(1, 1);
        const auto adj_id = gid + offset;
        const auto j = gid[1] + offset[1];

        data[adj_id] -= mean[j];
        data[adj_id] /= sycl::sqrt(FLOAT_N);
        data[adj_id] /= stddev[j];
      });
    }));

    events.push_back(args.device_queue.submit([&](handler& cgh) {
      auto data = data_buffer.get_access<access::mode::read>(cgh);
      auto symmat = symmat_buffer.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<CorrelationCorr>(range<1>(size), [=, M_ = size, N_ = size](id<1> gid) {
        // if(item[0] >= M_ - 1) return;
        const id<1> offset(1);
        const auto j1 = gid[0] + offset[0];

        symmat[{j1, j1}] = 1.0;

        for(size_t j2 = j1 + 1; j2 <= M_; j2++) {
          symmat[{j1, j2}] = 0.0;

          for(size_t i = 1; i <= N_; i++) {
            symmat[{j1, j2}] += data[{i, j1}] * data[{i, j2}];
          }

          symmat[{j2, j1}] = symmat[{j1, j2}];
        }
      });
    }));

    events.push_back(args.device_queue.submit([&](handler& cgh) {
      auto symmat = symmat_buffer.get_access<access::mode::discard_write>(cgh);
      cgh.parallel_for<Correlation5>(range<2>(1, 1), [=, M_ = size](id<2> gid) {
        const id<2> offset(M_, M_);
        symmat[gid + offset] = 1.0;
      });
    }));
  }

  bool verify(VerificationSetting&) {
    constexpr auto ERROR_THRESHOLD = 0.05;

    std::vector<DATA_TYPE> data_cpu((size + 1) * (size + 1));
    std::vector<DATA_TYPE> mean_cpu(size + 1);
    std::vector<DATA_TYPE> stddev_cpu(size + 1);
    std::vector<DATA_TYPE> symmat_cpu((size + 1) * (size + 1));

    // Trigger writeback
    symmat_buffer.reset();

    init_arrays(data_cpu.data(), size);
    correlation(data_cpu.data(), mean_cpu.data(), stddev_cpu.data(), symmat_cpu.data(), size);

    for(size_t i = 1; i < size + 1; i++) {
      for(size_t j = 1; j < size + 1; j++) {
        const auto diff = percentDiff(symmat_cpu[i * (size + 1) + j], symmat[i * (size + 1) + j]);
        if(diff > ERROR_THRESHOLD)
          return false;
      }
    }

    return true;
  }

  static std::string getBenchmarkName() { return "Polybench_Correlation"; }

private:
  BenchmarkArgs args;

  const size_t size;
  std::vector<DATA_TYPE> data;
  std::vector<DATA_TYPE> mean;
  std::vector<DATA_TYPE> stddev;
  std::vector<DATA_TYPE> symmat;

  PrefetchedBuffer<DATA_TYPE, 2> data_buffer;
  PrefetchedBuffer<DATA_TYPE, 1> mean_buffer;
  PrefetchedBuffer<DATA_TYPE, 1> stddev_buffer;
  PrefetchedBuffer<DATA_TYPE, 2> symmat_buffer;
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  app.run<Polybench_Correlation>();
  return 0;
}
