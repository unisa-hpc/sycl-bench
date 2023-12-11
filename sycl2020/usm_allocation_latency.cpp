#include "common.h"
#include "polybenchUtilFuncts.h"
#include "usm_common.hpp"

template <typename DATA_TYPE, sycl::usm::alloc usm_type>
class USMAllocationLatency {
protected:
  BenchmarkArgs args;
  DATA_TYPE* buffer;
  int runs = 0;

public:
  USMAllocationLatency(const BenchmarkArgs& _args) : args(_args), buffer(nullptr) {}

  ~USMAllocationLatency() {
	if (runs != 0)
    	sycl::free(buffer, args.device_queue);
  }

  void setup() {}

  void run(std::vector<sycl::event>& events) {
    runs++;
    sycl::queue& queue = args.device_queue;
    buffer = (DATA_TYPE*)sycl::malloc(args.problem_size * sizeof(DATA_TYPE), queue, usm_type);
  }


  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "USM_Allocation_latency_";
    name << ReadableTypename<DATA_TYPE>::name << "_";
    name << usm_to_string(usm_type);
	return name.str();
  }
};


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<USMAllocationLatency<double, sycl::usm::alloc::device>>();
  app.run<USMAllocationLatency<double, sycl::usm::alloc::host>>();
  app.run<USMAllocationLatency<double, sycl::usm::alloc::shared>>();

  return 0;
}