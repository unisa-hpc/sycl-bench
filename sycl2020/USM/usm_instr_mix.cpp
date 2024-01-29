#include "common.h"
#include "polybenchUtilFuncts.h"
#include "usm_utils.hpp"

static constexpr std::size_t d_kernel_launch = 100;
static constexpr std::size_t offset = 4;
static constexpr float d_instr_mix = 1;

/**
This benchmark measure the performance of USM allocations with different Host-Device instruction mixes
It copies some data on the device, then performs some operations on the device, copies the data back to the host and
performs some operations on the host. This is done in a loop.

It takes a float parameter --instr-mix that specifies the percentage of operations that are performed on the device.

A parameter --num-launches specifies the number of times the operation loop is executed

The benchmark is run with 4 different configurations:
  - USM allocations with device memory
  - USM allocations with host memory
  - USM allocations with shared memory
  - USM allocations with shared memory and prefetching  

The benchmark uses 2 additional configurations:
  - USM allocations with initialization
  - USM allocations without initialization
This helps to measure the overhead of the initialization operation
*/
template <typename DATA_TYPE, sycl::usm::alloc usm_type, bool include_init, bool use_prefetch>
class USMInstructionMix {
protected:
  BenchmarkArgs args;
  size_t kernel_launches;
  USMBuffer<DATA_TYPE, 1, usm_type> buff1;
  float instr_mix;

public:
  USMInstructionMix(const BenchmarkArgs& _args, size_t kernel_launches, float instr_mix)
      : args(_args), kernel_launches(kernel_launches), instr_mix(instr_mix) {}


  void setup() {
    if constexpr(!include_init) {
      buff1.initialize(args.device_queue, args.problem_size);
    }
  }

  void run(std::vector<sycl::event>& events) {
    sycl::queue& queue = args.device_queue;

    if constexpr(include_init) {
      buff1.initialize(args.device_queue, args.problem_size);
    }

    for(size_t i = 0; i < kernel_launches; i++) {
      auto device_copy_event = buff1.update_device();
      // Prefetch if using shared memory, should increase performance
      if constexpr(usm_type == sycl::usm::alloc::shared && use_prefetch) {
        device_copy_event = queue.prefetch(buff1.get(), buff1.size() * sizeof(DATA_TYPE), device_copy_event);
      }
      auto kernel_event = queue.submit([&](sycl::handler& cgh) {
        auto* acc_1 = buff1.get();
        cgh.depends_on(device_copy_event);

        cgh.parallel_for(sycl::nd_range<1>{{args.problem_size}, {args.local_size}},
            [=, _instr_mix = instr_mix](sycl::nd_item<1> item) {
              const auto id = item.get_global_id(0);
              const auto num_ops = item.get_global_range(0) * _instr_mix;
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

  bool verify(VerificationSetting& ver) {
    buff1.update_host();
    bool pass = false;
    for(size_t i = 0; i < buff1.size(); i++) {
      if(buff1.get_host_ptr()[i] != DATA_TYPE{0}) {
        pass = true;
      }
    }
    return pass;
  }

  static std::string getBenchmarkName() {
    const float device_op = args.cli.getOrDefault("--instr-mix", d_instr_mix);

    std::stringstream name;
    name << "USM_Instr_Mix_";
    name << ReadableTypename<DATA_TYPE>::name << "_";
    name << usm_to_string(usm_type) << "_";
    name << "1:" << (std::size_t(device_op) == device_op ? std::size_t(device_op) : device_op)
         << "mix_"; // avoid .0 if it's an integer
    name << (include_init ? "with_init_" : "no_init_");
    name << (use_prefetch ? "with_prefetch" : "no_prefetch");
    return name.str();
  }
};

int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);
  const size_t kernel_launches_num = app.getArgs().cli.getOrDefault("--num-launches", d_kernel_launch);
  const float instr_mix = app.getArgs().cli.getOrDefault("--instr-mix", d_instr_mix);

  app.run<USMInstructionMix<float, sycl::usm::alloc::device, true, false>>(kernel_launches_num, instr_mix);
  app.run<USMInstructionMix<float, sycl::usm::alloc::host, true, false>>(kernel_launches_num, instr_mix);
  app.run<USMInstructionMix<float, sycl::usm::alloc::shared, true, false>>(kernel_launches_num, instr_mix);
  app.run<USMInstructionMix<float, sycl::usm::alloc::shared, true, true>>(kernel_launches_num, instr_mix);

  app.run<USMInstructionMix<float, sycl::usm::alloc::device, false, false>>(kernel_launches_num, instr_mix);
  app.run<USMInstructionMix<float, sycl::usm::alloc::host, false, false>>(kernel_launches_num, instr_mix);
  app.run<USMInstructionMix<float, sycl::usm::alloc::shared, false, false>>(kernel_launches_num, instr_mix);
  app.run<USMInstructionMix<float, sycl::usm::alloc::shared, false, true>>(kernel_launches_num, instr_mix);
}