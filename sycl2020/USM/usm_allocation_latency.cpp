#include "common.h"
#include "polybenchUtilFuncts.h"
#include "usm_utils.hpp"

/**
Measure USM allocation time for different types of USM memory.
*/
template <typename DATA_TYPE, sycl::usm::alloc usm_type>
class USMAllocationLatency {
protected:
  BenchmarkArgs args;
  DATA_TYPE* buffer;

public:
  USMAllocationLatency(const BenchmarkArgs& _args) : args(_args), buffer(nullptr) {}

  ~USMAllocationLatency() {
    if(buffer != nullptr) {
      sycl::free(buffer, args.device_queue);
    }
  }

  void setup() {}

  void run(std::vector<sycl::event>& events) {
    sycl::queue& queue = args.device_queue;
    buffer = static_cast<DATA_TYPE*>(sycl::malloc(args.problem_size * sizeof(DATA_TYPE), queue, usm_type));
  }

  bool verify(VerificationSetting& settings) {
    sycl::queue& queue = args.device_queue;
    queue.fill(buffer, DATA_TYPE{1}, args.problem_size).wait();
    DATA_TYPE* host_ptr = buffer;
    if constexpr(usm_type == sycl::usm::alloc::device) {
      host_ptr = static_cast<DATA_TYPE*>(
          sycl::malloc(args.problem_size * sizeof(DATA_TYPE), args.device_queue, sycl::usm::alloc::host));
      queue.copy(buffer, host_ptr, args.problem_size).wait();
    }

    bool pass = true;
    for(int i = 0; i < args.problem_size; i++) {
      if(host_ptr[i] != DATA_TYPE{1}) {
        pass = false;
      }
    }
    if constexpr(usm_type == sycl::usm::alloc::device) {
      sycl::free(host_ptr, queue);
    }
    return pass;
  }


  static std::string getBenchmarkName(BenchmarkArgs& args) {
    std::stringstream name;
    name << "USM_Allocation_latency_";
    name << ReadableTypename<DATA_TYPE>::name << "_";
    name << usm_to_string(usm_type);
    return name.str();
  }
};


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<USMAllocationLatency<float, sycl::usm::alloc::device>>();
  app.run<USMAllocationLatency<float, sycl::usm::alloc::host>>();
  app.run<USMAllocationLatency<float, sycl::usm::alloc::shared>>();

  return 0;
}