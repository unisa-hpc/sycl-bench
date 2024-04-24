// Specialization constant benchmark
// - runs a generic 9 point stencil of which only 5 points are used in practice
// - weights are provided either
//   * fully dynamically (AccessVariants::dynamic_value),
//   * as specialization constants (AccessVariants::spec_const_value), or
//   * statically at compile time (AccessVariants::constexpr_value)
// Example run: ./spec_constant_convolution --device=gpu --no-verification --size=8192 --output=out.csv

#include "common.h"
#include <iostream>

enum class AccessVariants {
  dynamic_value,
  spec_const_value,
  constexpr_value,
};

namespace s = sycl;
template <typename T, AccessVariants AccessVariant, int InnerLoops>
class ConvKernel;

// T is the data type operated on
// AccessVariant determines if coefficients are accessed dynamically, use specialization constants, or are static
// InnerLoops allows tuning the arithmetic intensity of the kernel
template <typename T, AccessVariants AccessVariant, int InnerLoops>
class SpecConstConvBench {
  int problem_size = 1;

  using coeff_t = std::array<std::array<T, 3>, 3>;

  // internal function to generate some coefficients for the specialization constant
  coeff_t getCoefficients() {
    // trick the compiler a bit - problem size is always < 0, but the compiler doesn't know that
    T val(problem_size < 0 ? problem_size : 2);
    T val0(problem_size < 0 ? problem_size : 0);
    return {{{val0, val, val0}, {val, val, val}, {val0, val, val0}}};
  }

  T getDivider() {
    // analogous to above
    return problem_size < 0 ? T(problem_size) : T(5);
  }

  T getInitValue() {
    // analogous to above
    return problem_size < 0 ? T(problem_size) : T(1);
  }

#ifdef HIPSYCL_EXT_SPECIALIZED
  // ACPP implements sycl::specialized instead of spec constants
  sycl::specialized<coeff_t> coeff_spec;
  sycl::specialized<T> div_spec;
#else
  // ids for the specialization constants
  static constexpr s::specialization_id<coeff_t> coeff_id;
  static constexpr s::specialization_id<T> div_id;
#endif

  BenchmarkArgs args;

  PrefetchedBuffer<T, 2> in_buf;
  PrefetchedBuffer<T, 2> out_buf;

  std::vector<T> in_vec;

public:
  SpecConstConvBench(const BenchmarkArgs& _args) : args(_args) {}

  void setup() {
    problem_size = (int)args.problem_size;

    in_vec.resize(problem_size * problem_size);
    std::fill(in_vec.begin(), in_vec.end(), getInitValue());

    in_buf.initialize(args.device_queue, in_vec.data(), s::range<2>(problem_size, problem_size));
    out_buf.initialize(args.device_queue, in_vec.data(), s::range<2>(problem_size, problem_size));
  }


  void run(std::vector<sycl::event>& events) {
    events.push_back(args.device_queue.submit([&](sycl::handler& cgh) {
      auto in = in_buf.template get_access<s::access::mode::read>(cgh);
      auto out = out_buf.template get_access<s::access::mode::write>(cgh);

      // set the specialization constants
      coeff_t dynamic_coeff;
      T dynamic_div;
      if constexpr(AccessVariant == AccessVariants::dynamic_value) {
        dynamic_coeff = getCoefficients();
        dynamic_div = getDivider();
      } else if constexpr(AccessVariant == AccessVariants::spec_const_value) {
#ifndef HIPSYCL_EXT_SPECIALIZED
        cgh.set_specialization_constant<coeff_id>(getCoefficients());
        cgh.set_specialization_constant<div_id>(getDivider());
#else
        coeff_spec = getCoefficients();
        div_spec = getDivider();
#endif
      }

      cgh.parallel_for<class ConvKernel<T, AccessVariant, InnerLoops>>(in.get_range(),
#ifdef HIPSYCL_EXT_SPECIALIZED
          [=, coeff_spec_copy = coeff_spec, div_spec_copy = div_spec](
              s::item<2> item_id) // Copy to avoid this ptr access in lambda
#else
      [=](s::item<2> item_id, s::kernel_handler h)
#endif
          {
            T acc = 0;
            coeff_t coeff;
            T div;
            if constexpr(AccessVariant == AccessVariants::dynamic_value) {
              coeff = dynamic_coeff;
              div = dynamic_div;
            } else if constexpr(AccessVariant == AccessVariants::spec_const_value) {
#ifndef HIPSYCL_EXT_SPECIALIZED
              coeff = h.get_specialization_constant<coeff_id>();
              div = h.get_specialization_constant<div_id>();
#else
              coeff = coeff_spec_copy;
              div = div_spec_copy;
#endif
            } else if constexpr(AccessVariant == AccessVariants::constexpr_value) {
              coeff = {{{0, 2, 0}, {2, 2, 2}, {0, 2, 0}}};
              div = 5;
            }
            for(int k = 0; k < InnerLoops; ++k) {
              for(int i = -1; i <= 1; i++) {
                if(item_id[0] + i < 0 || item_id[0] + i >= in.get_range()[0])
                  continue;
                for(int j = -1; j <= 1; j++) {
                  if(item_id[1] + j < 0 || item_id[1] + j >= out.get_range()[1])
                    continue;
                  acc += coeff[i + 1][j + 1] * in[item_id[0] + i][item_id[1] + j];
                }
              }
            }
            out[item_id] = acc / (div * T(InnerLoops));
          });
    }));
  }

  bool verify(VerificationSetting& ver) {
    auto out_acc = out_buf.get_host_access();

    bool pass = true;

    auto c = getCoefficients();
    auto d = getDivider();
    auto v = getInitValue();
    T expected_val = 0;
    for(int i = 0; i < InnerLoops; ++i) {
      expected_val += v * c[0][0] + v * c[0][1] + v * c[0][2]   //
                      + v * c[1][0] + v * c[1][1] + v * c[1][2] //
                      + v * c[2][0] + v * c[2][1] + v * c[2][2];
    }
    expected_val /= d * InnerLoops;

    for(size_t x = 1; x < args.problem_size - 1 && pass; ++x) {
      for(size_t y = 1; y < args.problem_size - 1 && pass; ++y) {
        if(out_acc[x][y] != expected_val) {
          std::cout << "Fail at = " << x << " / " << y << "\nExpected = " << expected_val << "Actual =" << out_acc[x][y]
                    << std::endl;
          pass = false;
          break;
        }
      }
    }

    return pass;
  }

  static std::string getBenchmarkName(BenchmarkArgs& args) {
    std::stringstream name;
    name << "SpecConstantConvolution_";
    name << ReadableTypename<T>::name;
    if constexpr(AccessVariant == AccessVariants::dynamic_value) {
      name << "_DynamicValue";
    } else if constexpr(AccessVariant == AccessVariants::spec_const_value) {
      name << "_SpecConstValue";
    } else if constexpr(AccessVariant == AccessVariants::constexpr_value) {
      name << "_ConstExprValue";
    }
    name << "_IL" << InnerLoops;
    return name.str();
  }
};


template <typename T, AccessVariants AccessVariant>
void runLoopCounts(BenchmarkApp& app) {
  app.run<SpecConstConvBench<T, AccessVariant, 1>>();
  app.run<SpecConstConvBench<T, AccessVariant, 16>>();
  app.run<SpecConstConvBench<T, AccessVariant, 64>>();
}

template <typename T>
void runAccessVariants(BenchmarkApp& app) {
  runLoopCounts<T, AccessVariants::dynamic_value>(app);
  runLoopCounts<T, AccessVariants::spec_const_value>(app);
  runLoopCounts<T, AccessVariants::constexpr_value>(app);
}

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  runAccessVariants<int>(app);
  runAccessVariants<long long>(app);
  runAccessVariants<float>(app);
  if constexpr(SYCL_BENCH_HAS_FP64_SUPPORT) {
    runAccessVariants<double>(app);
  }
  return 0;
}
