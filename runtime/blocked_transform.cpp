


#include "common.h"

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>


using complex = sycl::vec<float, 2>;

inline complex mandelbrot_iteration(complex z, complex c) {
  complex result = c;

  result.x() += z.x() * z.x() - z.y() * z.y();
  result.y() += 2 * z.x() * z.y();

  return result;
}

template <int Num_iterations>
complex mandelbrot_sequence(complex z0, complex c) {
  complex z = z0;
  for(int i = 0; i < Num_iterations; ++i) {
    z = mandelbrot_iteration(z, c);
  }
  return z;
}

template <int Num_iterations>
class MandelbrotKernel;

/// Performs a blocked transform operation using the mandelbrot sequence
/// as kernels. The number of iterations of the sequence (and hence the runtime
/// of the kernel can be adjusted using \c Num_iterations ).
/// This benchmark processes the data in chunks that are assigned to independent
/// kernels, therefore this benchmark tests
/// * Overlapping of compute and data transfers
/// * concurrent kernel execution
/// * if the implementation of ranged accessors creates independent accesses if
/// accessed ranges are non-overlapping. In order for the benchmark to stress
/// these aspects, \c Num_iterations should be tuned such that the kernel
/// runtime is similar to the data transfer time of one block.
template <int Num_iterations>
class BlockedTransform {
private:
  std::vector<complex> data;
  BenchmarkArgs args;
  std::size_t block_size;

public:
  BlockedTransform(const BenchmarkArgs& _args, std::size_t _block_size) : args(_args), block_size{_block_size} {
    assert(block_size > 0);
  }

  void setup() { init_data(data); }

  void run() {
    sycl::buffer<complex, 1> buff{data.data(), sycl::range<1>{data.size()}};

    sycl::id<1> begin{0};
    sycl::range<1> current_batch_size{block_size};
    for(; begin[0] < data.size(); begin[0] += this->block_size) {
      current_batch_size[0] = std::min(this->block_size, data.size() - begin[0]);

      args.device_queue.submit([&](sycl::handler& cgh) {
        auto acc = buff.get_access<sycl::access::mode::read_write>(cgh, current_batch_size, begin);

        cgh.parallel_for<MandelbrotKernel<Num_iterations>>(current_batch_size, [=](sycl::id<1> idx) {
          const complex z0{0.0f, 0.0f};
          acc[idx + begin] = mandelbrot_sequence<Num_iterations>(z0, acc[idx + begin]);
        });
      });
    }
  }

  bool verify(VerificationSetting& ver) {
    std::vector<complex> v;
    init_data(v);

    const double tol = 1.e-5;

    for(std::size_t i = 0; i < v.size(); ++i) {
      v[i] = mandelbrot_sequence<Num_iterations>(complex{0.0f, 0.0f}, v[i]);

      if(std::abs(v[i].x() - data[i].x()) > tol)
        return false;
      if(std::abs(v[i].y() - data[i].y()) > tol)
        return false;
    }

    return true;
  }

  std::string getBenchmarkName() {
    std::stringstream name;
    name << "Runtime_BlockedTransform_iter_";
    name << Num_iterations << "_blocksize_";
    name << block_size;
    return name.str();
  }

private:
  void init_data(std::vector<complex>& initial_data) {
    initial_data.clear();
    initial_data.resize(args.problem_size);

    for(std::size_t i = 0; i < initial_data.size(); ++i) {
      initial_data[i].x() = 0.8 * std::cos(i / args.problem_size);
      initial_data[i].y() = 0.8 * std::sin(i / args.problem_size);
    }
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  for(std::size_t block_size = app.getArgs().local_size; block_size < app.getArgs().problem_size; block_size *= 2) {
    app.run<BlockedTransform<64>>(block_size);
    app.run<BlockedTransform<128>>(block_size);
    app.run<BlockedTransform<256>>(block_size);
    app.run<BlockedTransform<512>>(block_size);
  }

  return 0;
}
