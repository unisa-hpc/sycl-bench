#include "common.h"
#include "polybenchUtilFuncts.h"

#define KERNEL_LAUNCHES_DEFAULT 50000
static constexpr auto offset = 4;


std::string usm_to_string(sycl::usm::alloc usm_type) {
  if(usm_type == sycl::usm::alloc::device)
    return "device";
  else if(usm_type == sycl::usm::alloc::host)
    return "host";
  else if(usm_type == sycl::usm::alloc::shared)
    return "shared";
  else
    return "unknown";
}


template <typename DATA_TYPE, sycl::usm::alloc usm_type, bool strided>
class USMHostDeviceBenchmark {
protected:
  BenchmarkArgs args;
  size_t kernel_launches;
  USMBuffer<DATA_TYPE, 1, usm_type> buff1;


public:
  USMHostDeviceBenchmark(const BenchmarkArgs& _args, size_t kernel_launches)
      : args(_args), buff1(args.device_queue), kernel_launches(kernel_launches) {}

  void setup() {
    // buff1.initialize(args.problem_size);
    // args.device_queue.fill(buff1.get(), 0, buff1.size());
  }


  void run(std::vector<sycl::event>& events) {
    if constexpr(strided) {
      run_strided(events);
    } else {
      run_normal(events);
    }
  }

  void run_strided(std::vector<sycl::event>& events) {
    sycl::queue& queue = args.device_queue;
    // Init
    buff1.initialize(args.problem_size);
    {
      auto host_ptr = buff1.update_and_get_host_ptr();
      std::fill(host_ptr, host_ptr + buff1.size(), 0);
    }

    for(size_t i = 0; i < kernel_launches; i++) {
      auto device_copy_event = buff1.update_device();
      auto kernel_event = queue.submit([&](sycl::handler& cgh) {
        auto* acc_1 = buff1.get();
        cgh.depends_on(device_copy_event);
        cgh.parallel_for(sycl::nd_range<1>{{args.problem_size / offset}, {args.local_size}}, [=](sycl::nd_item<1> item) {
          acc_1[item.get_global_linear_id() * offset] = static_cast<DATA_TYPE>(item.get_global_linear_id());
        });
      });
      events.push_back(kernel_event);
      auto [host_ptr, copy_event] = buff1.update_and_get_host_ptr(kernel_event);
      copy_event.wait(); // Need this wait 'cause we can't use host tasks and synchronization with the device is needed
      for(size_t i = 0; i < buff1.size() / offset; i++) {
        host_ptr[i * offset] -= DATA_TYPE{1};
      }
    }
  }

  void run_normal(std::vector<sycl::event>& events) {
    sycl::queue& queue = args.device_queue;
    // Init
    buff1.initialize(args.problem_size);
    {
      auto host_ptr = buff1.update_and_get_host_ptr();
      std::fill(host_ptr, host_ptr + buff1.size(), 0);
    }

    for(size_t i = 0; i < kernel_launches; i++) {
      auto device_copy_event = buff1.update_device();
      auto kernel_event = queue.submit([&](sycl::handler& cgh) {
        auto* acc_1 = buff1.get();
        cgh.depends_on(device_copy_event);
        cgh.parallel_for(sycl::nd_range<1>{{args.problem_size}, {args.local_size}}, [=](sycl::nd_item<1> item) {
          acc_1[item.get_global_linear_id()] = static_cast<DATA_TYPE>(item.get_global_linear_id());
        });
      });
      events.push_back(kernel_event);
      // Host op
      // TODO: Host tasks?
      // TODO: Prefetch?
      // TODO: Strided copy?
      auto [host_ptr, copy_event] = buff1.update_and_get_host_ptr(kernel_event);
      copy_event.wait(); // Need this wait 'cause we can't use host tasks and synchronization with the device is needed
      // Host op
      for(size_t i = 0; i < buff1.size(); i++) {
        host_ptr[i] -= DATA_TYPE{1};
      }
    }
  }

  bool verify(VerificationSetting& settings) {
    auto host_ptr = buff1.get_host_ptr();
    constexpr auto ERROR_THRESHOLD = 0.05;
    for(int i = 0; i < buff1.size() / (strided ? offset : 1); i++) {
      int index = strided ? i * offset : i;
      const auto diff = percentDiff(host_ptr[index], i - 1);
      if(diff > ERROR_THRESHOLD) {
        std::cout << i << " -- " << host_ptr[index] << " : " << static_cast<DATA_TYPE>(i - 1) << "\n";
        return false;
      }
    }
    return true;
  }

  static std::string getBenchmarkName() {
    std::stringstream name;
    name << "USM_Host_Device_";
    name << ReadableTypename<DATA_TYPE>::name << "_";
    name << usm_to_string(usm_type) << "_";
    if constexpr(strided)
      name << "strided_";
    return name.str();
  }
};


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  const size_t kernel_launches_num = app.getArgs().cli.getOrDefault("--num-launches", KERNEL_LAUNCHES_DEFAULT);

  app.run<USMHostDeviceBenchmark<int, sycl::usm::alloc::device, false>>(kernel_launches_num);
  app.run<USMHostDeviceBenchmark<int, sycl::usm::alloc::host, false>>(kernel_launches_num);
  app.run<USMHostDeviceBenchmark<int, sycl::usm::alloc::shared, false>>(kernel_launches_num);

  app.run<USMHostDeviceBenchmark<float, sycl::usm::alloc::device, false>>(kernel_launches_num);
  app.run<USMHostDeviceBenchmark<float, sycl::usm::alloc::host, false>>(kernel_launches_num);
  app.run<USMHostDeviceBenchmark<float, sycl::usm::alloc::shared, false>>(kernel_launches_num);

  if constexpr(SYCL_BENCH_ENABLE_FP64_BENCHMARKS) {
    if(app.deviceSupportsFP64()) {
      app.run<USMHostDeviceBenchmark<double, sycl::usm::alloc::device, false>>(kernel_launches_num);
      app.run<USMHostDeviceBenchmark<double, sycl::usm::alloc::host, false>>(kernel_launches_num);
      app.run<USMHostDeviceBenchmark<double, sycl::usm::alloc::shared, false>>(kernel_launches_num);
    }
  }

  
  app.run<USMHostDeviceBenchmark<int, sycl::usm::alloc::device, true>>(kernel_launches_num);
  app.run<USMHostDeviceBenchmark<int, sycl::usm::alloc::host,   true>>(kernel_launches_num);
  app.run<USMHostDeviceBenchmark<int, sycl::usm::alloc::shared, true>>(kernel_launches_num);
  app.run<USMHostDeviceBenchmark<float, sycl::usm::alloc::device, true>>(kernel_launches_num);
  app.run<USMHostDeviceBenchmark<float, sycl::usm::alloc::host,   true>>(kernel_launches_num);
  app.run<USMHostDeviceBenchmark<float, sycl::usm::alloc::shared, true>>(kernel_launches_num);

  if constexpr(SYCL_BENCH_ENABLE_FP64_BENCHMARKS) {
    if(app.deviceSupportsFP64()) {
      app.run<USMHostDeviceBenchmark<double, sycl::usm::alloc::device, true>>(kernel_launches_num);
      app.run<USMHostDeviceBenchmark<double, sycl::usm::alloc::host,   true>>(kernel_launches_num);
      app.run<USMHostDeviceBenchmark<double, sycl::usm::alloc::shared, true>>(kernel_launches_num);
    }
  }


}