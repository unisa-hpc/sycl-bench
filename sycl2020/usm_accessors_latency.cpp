#include "common.h"

#include <iostream>

namespace s = cl::sycl;

// TODO: As kernel parameter
#define NUM_KERNELS 50000


/**
 * @brief Measure Accessors latency compared to USM
        The benchmark submits multiple small kernels which stress
        SYCL dependency tracking.
 *
 * @tparam DATA_TYPE
 */


template <typename DATA_TYPE>
class AccessorLatency {
protected:
  PrefetchedBuffer<DATA_TYPE, 1> buff_A;
  PrefetchedBuffer<DATA_TYPE, 1> buff_B;
  PrefetchedBuffer<DATA_TYPE, 1> buff_C;

  BenchmarkArgs args;

public:
  AccessorLatency(const BenchmarkArgs& args) : args(args) {}

  // TODO: Problem size?
  void setup() {
    buff_A.initialize(args.device_queue, {args.problem_size});
    buff_B.initialize(args.device_queue, {args.problem_size});
    buff_C.initialize(args.device_queue, {args.problem_size});
  }

  void run() {
    auto& queue = args.device_queue;
    for(int i = 0; i < NUM_KERNELS; i++) {
      queue.submit([&](s::handler& cgh) {
        auto acc_A = buff_A.template get_access<s::access::mode::read>(cgh, buff_A.get_range());
        auto acc_B = buff_A.template get_access<s::access::mode::read>(cgh, buff_A.get_range());
        auto acc_C = buff_A.template get_access<s::access::mode::write>(cgh, buff_A.get_range());

        cgh.parallel_for(
            s::range<1>{args.problem_size}, [=](s::item<1> item) { acc_C[item] = acc_A[item] + acc_B[item]; });
      });
      // swap buffers to
      std::swap(buff_A, buff_B);
      std::swap(buff_A, buff_C);
    }
  }

  bool verify(VerificationSetting& settings) {
    // TODO
    return true;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "SYCL2020_Accessors_Latency_";
    name << ReadableTypename<DATA_TYPE>::name << "_";
    name << NUM_KERNELS;
	return name.str();
  }
};

template <typename DATA_TYPE>
class USMLatency {
protected:
  USMBuffer<DATA_TYPE> buff_A;
  USMBuffer<DATA_TYPE> buff_B;
  USMBuffer<DATA_TYPE> buff_C;

  BenchmarkArgs args;

public:
  USMLatency(const BenchmarkArgs& args) : args(args) {}

  // TODO: Problem size?
  void setup() {
    buff_A.initialize(args.device_queue, args.problem_size);
    buff_B.initialize(args.device_queue, args.problem_size);
    buff_C.initialize(args.device_queue, args.problem_size);
  }

  void run() {
    auto& queue = args.device_queue;
    cl::sycl::event event;
    for(int i = 0; i < NUM_KERNELS; i++) {
      event = queue.submit([&](s::handler& cgh) {
        cgh.depends_on(event);
        cgh.parallel_for(s::range<1>{args.problem_size},
               [acc_A = buff_A.get(), acc_B = buff_B.get(), acc_C = buff_C.get()](s::item<1> item) {
                 const auto id = item.get_linear_id();
                 acc_C[id] = acc_A[id] + acc_B[id];
               });
      });
      // swap buffers to
      std::swap(buff_A, buff_B);
      std::swap(buff_A, buff_C);
    }
  }

  bool verify(VerificationSetting& settings) {
    // TODO
    return true;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "SYCL2020_USM_Latency_";
    name << ReadableTypename<DATA_TYPE>::name << "_";
    name << NUM_KERNELS;
	return name.str();
  }
};


int main(int argc, char** argv){
	BenchmarkApp app(argc, argv);
	
	app.run<AccessorLatency<float>>();
	app.run<USMLatency<float>>();
	if (app.deviceSupportsFP64()){
		app.run<AccessorLatency<double>>();
		app.run<USMLatency<double>>();
	}
}
