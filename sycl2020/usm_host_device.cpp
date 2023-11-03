#include "common.h"
#include "polybenchUtilFuncts.h"

#define KERNEL_LAUNCHES_DEFAULT 50000

std::string usm_to_string(sycl::usm::alloc usm_type) {
  if(usm_type == sycl::usm::alloc::device)
    return "device";
  else if(usm_type == sycl::usm::alloc::host)
    return "host";
  else if(usm_type == sycl::usm::alloc::shared)
    return "shared";
  else
    throw std::invalid_argument("Unknown USM type");
}

template <typename DATA_TYPE, sycl::usm::alloc usm_type>
class USMHostDeviceBenchmark {
protected:
  BenchmarkArgs args;
  size_t kernel_launches;
  USMBuffer<DATA_TYPE, 1, usm_type> buff1;

public:
  USMHostDeviceBenchmark(const BenchmarkArgs& _args, size_t kernel_launches)
      : args(_args), buff1(args.device_queue), kernel_launches(kernel_launches) {}

  void setup() {
    buff1.initialize(args.problem_size);
    args.device_queue.fill(buff1.get(), 0, buff1.size());
  }

  void run(std::vector<sycl::event>& events) {
    sycl::queue& queue = args.device_queue;
    sycl::event first_event; // Empty event
    for(size_t i = 0; i < kernel_launches; i++) {
      auto kernel_event = queue.submit([&](sycl::handler& cgh) {
        auto* acc_1 = buff1.get();
        cgh.depends_on(first_event);
        cgh.parallel_for(sycl::nd_range<1>{{args.problem_size}, {args.local_size}}, [=](sycl::nd_item<1> item) {
          acc_1[item.get_global_linear_id()] = static_cast<DATA_TYPE>(item.get_global_linear_id());
        });
      });
      events.push_back(kernel_event);
      // queue.wait();
      // Host op
      // TODO: Host tasks?
      // TODO: Prefetch?
      // TODO: Strided copy?
      auto [host_ptr, copy_event] = buff1.update_and_get_host_ptr(kernel_event);
      copy_event.wait(); // Need this wait 'cause we can't use host tasks and synchronization with the device is needed
      for(size_t i = 0; i < buff1.size(); i++) {
        host_ptr[i] -= DATA_TYPE{1};
      }

      first_event = buff1.update_device(copy_event);
    }
  }


  bool verify(VerificationSetting& settings) {
    auto host_ptr = buff1.update_and_get_host_ptr();
    constexpr auto ERROR_THRESHOLD = 0.05;
    for(int i = 0; i < buff1.size(); i++) {
      const auto diff = percentDiff(host_ptr[i], i - 1);
      if(diff > ERROR_THRESHOLD) {
        std::cout << i << " -- " << host_ptr[i] << " : " << static_cast<DATA_TYPE>(i - 1) << "\n";
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
    return name.str();
  }

  // static ThroughputMetric getThroughputMetric(const BenchmarkArgs& args) {
  //   // const double copiedGiB =
  //   //     getBufferSize<Dims, false>(args.problem_size).size() * sizeof(DataT) / 1024.0 / 1024.0 / 1024.0;
  //   // return {copiedGiB, "GiB"};
  //   // TODO
  // }
};


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  const size_t kernel_launches_num = app.getArgs().cli.getOrDefault("--num-launches", KERNEL_LAUNCHES_DEFAULT);

  app.run<USMHostDeviceBenchmark<int, sycl::usm::alloc::device>>(kernel_launches_num);
  app.run<USMHostDeviceBenchmark<int, sycl::usm::alloc::host>>(kernel_launches_num);
  app.run<USMHostDeviceBenchmark<int, sycl::usm::alloc::shared>>(kernel_launches_num);

  app.run<USMHostDeviceBenchmark<float, sycl::usm::alloc::device>>(kernel_launches_num);
  app.run<USMHostDeviceBenchmark<float, sycl::usm::alloc::host>>(kernel_launches_num);
  app.run<USMHostDeviceBenchmark<float, sycl::usm::alloc::shared>>(kernel_launches_num);

  if constexpr(SYCL_BENCH_ENABLE_FP64_BENCHMARKS) {
    if(app.deviceSupportsFP64()) {
      app.run<USMHostDeviceBenchmark<double, sycl::usm::alloc::device>>(kernel_launches_num);
      app.run<USMHostDeviceBenchmark<double, sycl::usm::alloc::host>>(kernel_launches_num);
      app.run<USMHostDeviceBenchmark<double, sycl::usm::alloc::shared>>(kernel_launches_num);
    }
  }
}