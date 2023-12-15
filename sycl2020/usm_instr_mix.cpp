#include "common.h"
#include "polybenchUtilFuncts.h"
#include "usm_common.hpp"

#define KERNEL_LAUNCHES_DEFAULT 50000
static constexpr auto offset = 4;


template <std::size_t exp>
struct pow_2 {
  static constexpr auto value = 2 * pow_2<exp - 1>::value;
};

template <>
struct pow_2<0> {
  static constexpr auto value = 1;
};

static constexpr std::array<float, 11> ratios = {1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6};

// static constexpr std::array<float, 8> ratios = {1, 2, 4, 8, 16, 32, 64, 128};


template <typename DATA_TYPE, sycl::usm::alloc usm_type, bool include_init, size_t device_op_ratio>
class USMHostDeviceBenchmark {
protected:
  BenchmarkArgs args;
  size_t kernel_launches;
  USMBuffer<DATA_TYPE, 1, usm_type> buff1;
  DATA_TYPE* tmp;

public:
  USMHostDeviceBenchmark(const BenchmarkArgs& _args, size_t kernel_launches)
      : args(_args), kernel_launches(kernel_launches) {}

  // ~USMHostDeviceBenchmark() {
  //   sycl::free(tmp, args.device_queue);
  // }

  void setup() {
    if constexpr (!include_init) {
      buff1.initialize(args.device_queue, args.problem_size);
      args.device_queue.fill(buff1.get_host_ptr(), DATA_TYPE{0}, buff1.size());
    }
  }

  void run(std::vector<sycl::event>& events) {
    sycl::queue& queue = args.device_queue;
    // Init
    if constexpr(include_init) {
      buff1.initialize(args.device_queue, args.problem_size);
      // args.device_queue.fill(buff1.get_host_ptr(), DATA_TYPE{0}, buff1.size());
    }

    for(size_t i = 0; i < kernel_launches; i++) {
      auto device_copy_event = buff1.update_device();
      // Prefetch if using shared memory, should increase performance
      if constexpr(usm_type == sycl::usm::alloc::shared) {
        queue.prefetch(buff1.get(), buff1.size() * sizeof(DATA_TYPE));
      }
      auto kernel_event = queue.submit([&](sycl::handler& cgh) {
        auto* acc_1 = buff1.get();
        cgh.depends_on(device_copy_event);
        cgh.parallel_for(sycl::nd_range<1>{{args.problem_size}, {args.local_size}}, [=](sycl::nd_item<1> item) {
          constexpr auto device_op = ratios[device_op_ratio];
          const auto id = item.get_global_id(0);
          const auto num_ops = item.get_global_range(0) * device_op;
          for(size_t i = id, j = 0; i < num_ops; i += item.get_global_range(0), j++) {
            acc_1[(id + j) % item.get_global_range(0)] += DATA_TYPE{1};
          }
        });
      });
      events.push_back(kernel_event);
      auto [host_ptr, copy_event] = buff1.update_and_get_host_ptr(kernel_event);
      copy_event.wait(); // Need this wait 'cause we can't use host tasks and synchronization with the device is needed
      // Host op
      for(size_t i = 0; i < buff1.size(); i++) {
        host_ptr[i] -= DATA_TYPE{1};
      }
    }
  }

  // bool verify(VerificationSetting& settings) {
  //   // auto host_ptr = buff1.get_host_ptr();
  //   // constexpr auto ERROR_THRESHOLD = 0.05;
  //   // for(int i = 0; i < buff1.size() / (strided ? offset : 1); i++) {
  //   //   int index = strided ? i * offset : i;
  //   //   const auto diff = percentDiff(host_ptr[index], i - 1);
  //   //   if(diff > ERROR_THRESHOLD) {
  //   //     std::cout << i << " -- " << host_ptr[index] << " : " << static_cast<DATA_TYPE>(i - 1) << "\n";
  //   //     return false;
  //   //   }
  //   // }
  //   // return true;
  //   return false;
  // }

  static std::string getBenchmarkName() {
    constexpr auto device_op = ratios[device_op_ratio];
    std::stringstream name;
    name << "USM_Host_Device_";
    name << ReadableTypename<DATA_TYPE>::name << "_";
    name << usm_to_string(usm_type) << "_";
    name << "1:" << (std::size_t(device_op) == device_op ? std::size_t(device_op) : device_op)
         << "mix_"; // avoid .0 if it's an integer
    if constexpr(include_init)
      name << "with_init_";
    else
      name << "no_init_";
    return name.str();
  }
};


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  const size_t kernel_launches_num = app.getArgs().cli.getOrDefault("--num-launches", KERNEL_LAUNCHES_DEFAULT);

  if constexpr(SYCL_BENCH_ENABLE_FP64_BENCHMARKS) {
    if(app.deviceSupportsFP64()) {
      loop<ratios.size()>([&](auto idx) {
        app.run<USMHostDeviceBenchmark<float, sycl::usm::alloc::device, true, idx>>(kernel_launches_num);
        app.run<USMHostDeviceBenchmark<float, sycl::usm::alloc::host, true, idx>>(kernel_launches_num);
        app.run<USMHostDeviceBenchmark<float, sycl::usm::alloc::shared, true, idx>>(kernel_launches_num);
      });

      loop<ratios.size()>([&](auto idx) {
        app.run<USMHostDeviceBenchmark<float, sycl::usm::alloc::device, false, idx>>(kernel_launches_num);
        app.run<USMHostDeviceBenchmark<float, sycl::usm::alloc::host, false, idx>>(kernel_launches_num);
        app.run<USMHostDeviceBenchmark<float, sycl::usm::alloc::shared, false, idx>>(kernel_launches_num);
      });
    }
  }
}