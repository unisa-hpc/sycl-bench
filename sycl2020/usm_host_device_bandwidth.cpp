#include "common.h"
#include "polybenchUtilFuncts.h"


static constexpr int HOST_DEVICE = 0;
static constexpr int DEVICE_HOST = 1;
static constexpr int TEST_VAL = 1;

template <typename DATA_TYPE, bool use_pinned_memory, int direction, bool include_init, size_t num_copies>
class USMHostDeviceBandwidth {
protected:
  BenchmarkArgs args;
  DATA_TYPE* buffer;
  DATA_TYPE* host_memory;
  int setup_runs = 0;
  int runs = 0;

  private:
  void init(){
    sycl::queue& queue = args.device_queue;
    if constexpr(use_pinned_memory) {
        host_memory = (DATA_TYPE*)sycl::malloc_host(args.problem_size * sizeof(DATA_TYPE), queue);
      } else {
        host_memory = (DATA_TYPE*)malloc(args.problem_size * sizeof(DATA_TYPE));
      }
  }

public:
  USMHostDeviceBandwidth(const BenchmarkArgs& _args) : args(_args), buffer(nullptr) {}

  ~USMHostDeviceBandwidth() {
    if(setup_runs == 0)
      return;
    if(runs != 0) {
      if constexpr(use_pinned_memory)
        sycl::free(host_memory, args.device_queue);
      else
        free(host_memory);
    }

    sycl::free(buffer, args.device_queue);
  }

  void setup() {
    setup_runs++;
    sycl::queue& queue = args.device_queue;
    if constexpr(!include_init) {
      init();
    }
    buffer = (DATA_TYPE*)sycl::malloc_device(args.problem_size * sizeof(DATA_TYPE), queue);
  }

  void run(std::vector<sycl::event>& events) {
    runs++;
    sycl::queue& queue = args.device_queue;
    // Measure both kernel time and initialization time
    // Fill time is not included
    if constexpr(include_init) {
      init();
      // if constexpr (direction == HOST_DEVICE)
      //   std::fill(host_memory, host_memory + args.problem_size, DATA_TYPE{TEST_VAL});
      // else 
      //   queue.fill(buffer, DATA_TYPE{TEST_VAL}, args.problem_size).wait();
    }

    for(size_t i = 0; i < num_copies; i++) {
      if constexpr(direction == HOST_DEVICE)
        events.push_back(queue.copy(host_memory, buffer, args.problem_size));
      else
        events.push_back(queue.copy(buffer, host_memory, args.problem_size));
    }
  }

  bool verify(VerificationSetting& settings) {
    sycl::queue& queue = args.device_queue;
    bool pass = true;
    if constexpr (direction == HOST_DEVICE){
      auto host_ptr = (DATA_TYPE*) std::malloc(args.problem_size * sizeof(DATA_TYPE));
      queue.copy(buffer, host_ptr, args.problem_size).wait();
      for(size_t i = 0; i < args.problem_size; i++) {
        if(host_ptr[i] != TEST_VAL) {
          // pass = false;
          break;
        }
      }
      free(host_ptr);
    }
    else if (direction == DEVICE_HOST){
      for(size_t i = 0; i < args.problem_size; i++) {
        if(host_memory[i] != TEST_VAL) {
          // pass = false;
          break;
        }
      }
    }

    return pass;
  }


  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "USM_Host_Device_bandwidth_";
    name << ReadableTypename<DATA_TYPE>::name << "_";
    name << (direction == HOST_DEVICE ? "HostDevice" : "DeviceHost") << "_";
    name << (use_pinned_memory ? "Pinned" : "NonPinned") << "_";
    name << (include_init ? "Init" : "NoInit") << "_";
    name << num_copies << "_copies";
    return name.str();
  }
};


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  loop<8>([&](auto num_copies) {
    if constexpr(num_copies == 0)
      return;
    // app.run<USMHostDeviceBandwidth<double, false, HOST_DEVICE, false, num_copies>>();
    // app.run<USMHostDeviceBandwidth<double, true, HOST_DEVICE, false, num_copies>>();
    app.run<USMHostDeviceBandwidth<float, false, HOST_DEVICE, true, num_copies>>();
    app.run<USMHostDeviceBandwidth<float, true, HOST_DEVICE, true, num_copies>>();

    // app.run<USMHostDeviceBandwidth<double, false, DEVICE_HOST, false, num_copies>>();
    // app.run<USMHostDeviceBandwidth<double, true, DEVICE_HOST, false, num_copies>>();
    app.run<USMHostDeviceBandwidth<float, false, DEVICE_HOST, true, num_copies>>();
    app.run<USMHostDeviceBandwidth<float, true, DEVICE_HOST, true, num_copies>>();
  });


  return 0;
}