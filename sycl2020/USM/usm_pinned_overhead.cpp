#include "common.h"

static constexpr int HOST_DEVICE = 0;
static constexpr int DEVICE_HOST = 1;
static constexpr int TEST_VAL = 1;
static constexpr std::size_t d_num_copies = 1;

/**
Measure the overhead of copying data from host to device and vice versa using pinned and non-pinned memory.
Takes a --num-copies parameter to specify how many copies to perform.
*/
template <typename DATA_TYPE, bool use_pinned_memory, int direction, bool include_init>
class USMPienndOverhead {
protected:
  BenchmarkArgs args;
  DATA_TYPE* buffer;
  DATA_TYPE* host_memory;
  size_t num_copies;

private:
  void init() {
    sycl::queue& queue = args.device_queue;
    if constexpr(use_pinned_memory) {
      host_memory = (DATA_TYPE*)sycl::malloc_host(args.problem_size * sizeof(DATA_TYPE), queue);
    } else {
      host_memory = (DATA_TYPE*)malloc(args.problem_size * sizeof(DATA_TYPE));
    }
  }

public:
  USMPienndOverhead(const BenchmarkArgs& _args, size_t num_copies)
      : args(_args), buffer(nullptr), host_memory(nullptr), num_copies(num_copies) {}

  ~USMPienndOverhead() {
    if(buffer == nullptr || host_memory == nullptr) {
      return;
    }
    if constexpr(use_pinned_memory) {
      sycl::free(host_memory, args.device_queue);
    } else {
      free(host_memory);
    }
    sycl::free(buffer, args.device_queue);
  }

  void setup() {
    sycl::queue& queue = args.device_queue;
    if constexpr(!include_init) {
      init();
    }
    buffer = (DATA_TYPE*) sycl::malloc_device(args.problem_size * sizeof(DATA_TYPE), queue);
  }

  void run(std::vector<sycl::event>& events) {
    sycl::queue& queue = args.device_queue;
    if constexpr(include_init) {
      init();
    }

    for(size_t i = 0; i < num_copies; i++) {
      if constexpr(direction == HOST_DEVICE){
        events.push_back(queue.copy(host_memory, buffer, args.problem_size));
      }
      else{
        events.push_back(queue.copy(buffer, host_memory, args.problem_size));
      }
    }
  }

  static ThroughputMetric getThroughputMetric(const BenchmarkArgs& args) {
    const size_t num_copies = args.cli.getOrDefault("--num-copies", d_num_copies);
    const double copiedGiB = args.problem_size * sizeof(DATA_TYPE) * num_copies / 1024.0 / 1024.0 / 1024.0;
    return {copiedGiB, "GiB"};
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    const size_t num_copies= args.cli.getOrDefault("--num-copies", d_num_copies);
    
    name << "USM_Pinned_Overhead_";
    name << ReadableTypename<DATA_TYPE>::name << "_";
    name << (direction == HOST_DEVICE ? "HostDevice" : "DeviceHost") << "_";
    name << (use_pinned_memory ? "Pinned" : "NonPinned") << "_";
    name << (include_init ? "Init" : "NoInit") << "_"; 
    name << num_copies;
    
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  const size_t num_copies= app.getArgs().cli.getOrDefault("--num-copies", d_num_copies);

  app.run<USMPienndOverhead<float, false, HOST_DEVICE, true>>(num_copies);
  app.run<USMPienndOverhead<float, true, HOST_DEVICE, true>>(num_copies);
  app.run<USMPienndOverhead<float, false, DEVICE_HOST, true>>(num_copies);
  app.run<USMPienndOverhead<float, true, DEVICE_HOST, true>>(num_copies);


  return 0;
}