#include <cstdio>
#include <vector>

#include <common.h>

// Performs chained matrix multiply of the form (AB)(CD)
// Uses two intermediate buffers and one for the result

template <typename T>
class MatmulChain;

template <typename T>
void multiply(sycl::queue& queue, sycl::buffer<T, 2>& mat_a, sycl::buffer<T, 2>& mat_b, sycl::buffer<T, 2>& mat_c,
    const size_t mat_size) {
  queue.submit([&](sycl::handler& cgh) {
    auto a = mat_a.template get_access<sycl::access::mode::read>(cgh);
    auto b = mat_b.template get_access<sycl::access::mode::read>(cgh);
    auto c = mat_c.template get_access<sycl::access::mode::discard_write>(cgh);

    cgh.parallel_for<class MatmulChain<T>>(sycl::range<2>(mat_size, mat_size), [=](sycl::item<2> item) {
      auto sum = 0;
      for(size_t k = 0; k < mat_size; ++k) {
        const auto a_ik = a[{item[0], k}];
        const auto b_kj = b[{k, item[1]}];
        sum += a_ik * b_kj;
      }
      c[item] = sum;
    });
  });
}


template <typename T>
class MatmulChain {
protected:
  std::vector<T> mat_a;
  std::vector<T> mat_b;
  std::vector<T> mat_c;
  std::vector<T> mat_d;
  std::vector<T> mat_res;
  BenchmarkArgs args;
  int mat_size;

  PrefetchedBuffer<T, 2> mat_a_buf;
  PrefetchedBuffer<T, 2> mat_b_buf;
  PrefetchedBuffer<T, 2> mat_c_buf;
  PrefetchedBuffer<T, 2> mat_d_buf;
  PrefetchedBuffer<T, 2> mat_res_buf;
  PrefetchedBuffer<T, 2> mat_p_buf;
  PrefetchedBuffer<T, 2> mat_q_buf;

public:
  MatmulChain(const BenchmarkArgs& _args) : args(_args) { mat_size = args.problem_size; }

  void setup() {
    mat_a = std::vector<T>(mat_size * mat_size);
    mat_b = std::vector<T>(mat_size * mat_size);
    mat_c = std::vector<T>(mat_size * mat_size);
    mat_d = std::vector<T>(mat_size * mat_size);
    mat_res = std::vector<T>(mat_size * mat_size);

    // Initialize matrices to the identity
    for(size_t i = 0; i < mat_size; ++i) {
      for(size_t j = 0; j < mat_size; ++j) {
        mat_a[i * mat_size + j] = i == j;
        mat_b[i * mat_size + j] = i == j;
        mat_c[i * mat_size + j] = i == j;
        mat_d[i * mat_size + j] = i == j;
      }
    }

    mat_a_buf.initialize(args.device_queue, mat_a.data(), sycl::range<2>(mat_size, mat_size));
    mat_b_buf.initialize(args.device_queue, mat_b.data(), sycl::range<2>(mat_size, mat_size));
    mat_c_buf.initialize(args.device_queue, mat_c.data(), sycl::range<2>(mat_size, mat_size));
    mat_d_buf.initialize(args.device_queue, mat_d.data(), sycl::range<2>(mat_size, mat_size));
    mat_res_buf.initialize(args.device_queue, mat_res.data(), sycl::range<2>(mat_size, mat_size));
    mat_p_buf.initialize(args.device_queue, sycl::range<2>(mat_size, mat_size));
    mat_q_buf.initialize(args.device_queue, sycl::range<2>(mat_size, mat_size));
  }

  void run() {
    multiply(args.device_queue, mat_a_buf.get(), mat_b_buf.get(), mat_p_buf.get(), mat_size);
    multiply(args.device_queue, mat_c_buf.get(), mat_d_buf.get(), mat_q_buf.get(), mat_size);
    multiply(args.device_queue, mat_p_buf.get(), mat_q_buf.get(), mat_res_buf.get(), mat_size);
  }

  static std::string getBenchmarkName(BenchmarkArgs& args) { return "MatmulChain"; }

  bool verify(VerificationSetting& ver) {
    // Triggers writeback
    auto* mat_res = mat_res_buf.get_host_access().get_pointer();
    bool verification_passed = true;

    for(size_t i = 0; i < mat_size; ++i) {
      for(size_t j = 0; j < mat_size; ++j) {
        const T kernel_value = mat_res[i * mat_size + j];
        const T host_value = i == j;
        if(kernel_value != host_value) {
          fprintf(stderr, "VERIFICATION FAILED for element %ld,%ld: %f != %f\n", i, j, kernel_value, host_value);
          verification_passed = false;
          break;
        }
      }
      if(!verification_passed) {
        break;
      }
    }
    return verification_passed;
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  // float
  app.run<MatmulChain<float>>();
}
