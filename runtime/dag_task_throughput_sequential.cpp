#include "common.h"

using namespace sycl;

class DagTaskThroughputKernelSingleTask;
class DagTaskThroughputKernelBasicPF;
class DagTaskThroughputKernelNdrangePF;
class DagTaskThroughputKernelHierarchicalPF;

// Measures the time it takes to run <problem-size> trivial single_task and parallel_for kernels
// that depend on each other, and have to be executed in-order (-> Utilization of
// parallel hardware is *not* tested)
// This is influenced by
// * latencies in task submission to the backend, e.g. GPU kernel latencies
// * scheduling latencies caused by the SYCL implementation
// * other overheads
class DagTaskThroughput {
  const int initial_value;
  PrefetchedBuffer<int, 1> dummy_counter;
  BenchmarkArgs args;

public:
  DagTaskThroughput(const BenchmarkArgs& _args) : initial_value{0}, args(_args) {}

  void setup() { dummy_counter.initialize(args.device_queue, &initial_value, sycl::range<1>{1}); }

  void submit_single_task() {
    // Behold! The weirdest, most inefficient summation algorithm ever conceived!
    for(std::size_t i = 0; i < args.problem_size; ++i) {
      args.device_queue.submit([&](sycl::handler& cgh) {
        auto acc = dummy_counter.get_access<sycl::access::mode::read_write>(cgh);

        cgh.single_task<DagTaskThroughputKernelSingleTask>([=]() { acc[0] += 1; });
      }); // submit
    }
  }

  void submit_basic_parallel_for() {
    for(std::size_t i = 0; i < args.problem_size; ++i) {
      args.device_queue.submit([&](sycl::handler& cgh) {
        auto acc = dummy_counter.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<DagTaskThroughputKernelBasicPF>(
            // while we cannot control it, let's hope the SYCL implementation
            // spawns a single work group.
            sycl::range<1>{args.local_size}, [=](sycl::id<1> idx) {
              if(idx[0] == 0)
                acc[0] += 1;
            });
      }); // submit
    }
  }

  void submit_ndrange_parallel_for() {
    for(std::size_t i = 0; i < args.problem_size; ++i) {
      args.device_queue.submit([&](sycl::handler& cgh) {
        auto acc = dummy_counter.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<DagTaskThroughputKernelNdrangePF>(
            sycl::nd_range<1>{sycl::range<1>{args.local_size}, sycl::range<1>{args.local_size}},
            [=](sycl::nd_item<1> idx) {
              if(idx.get_global_id(0) == 0)
                acc[0] += 1;
            });
      }); // submit
    }
  }

  void submit_hierarchical_parallel_for() {
    for(std::size_t i = 0; i < args.problem_size; ++i) {
      args.device_queue.submit([&](sycl::handler& cgh) {
        auto acc = dummy_counter.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for_work_group<DagTaskThroughputKernelHierarchicalPF>(
            sycl::range<1>{1}, sycl::range<1>{args.local_size}, [=](sycl::group<1> grp) {
              grp.parallel_for_work_item([&](sycl::h_item<1> idx) {
                if(idx.get_global_id(0) == 0)
                  acc[0] += 1;
              });
            });
      }); // submit
    }
  }

  bool verify(VerificationSetting& ver) {
    auto host_acc = dummy_counter.get_host_access();

    return host_acc[0] == args.problem_size;
  }
};


class DagTaskThroughputSingleTask : public DagTaskThroughput {
public:
  DagTaskThroughputSingleTask(const BenchmarkArgs& args) : DagTaskThroughput{args} {}

  void run() { submit_single_task(); }

  static std::string getBenchmarkName(BenchmarkArgs& args) {
    return "Runtime_DAGTaskThroughput_SingleTask";
  }
};


class DagTaskThroughputBasicPF : public DagTaskThroughput {
public:
  DagTaskThroughputBasicPF(const BenchmarkArgs& args) : DagTaskThroughput{args} {}

  void run() { submit_basic_parallel_for(); }

  static std::string getBenchmarkName(BenchmarkArgs& args) {
    return "Runtime_DAGTaskThroughput_BasicParallelFor";
  }
};


class DagTaskThroughputNDRangePF : public DagTaskThroughput {
public:
  DagTaskThroughputNDRangePF(const BenchmarkArgs& args) : DagTaskThroughput{args} {}

  void run() { submit_ndrange_parallel_for(); }

  static std::string getBenchmarkName(BenchmarkArgs& args) {
    return "Runtime_DAGTaskThroughput_NDRangeParallelFor";
  }
};


class DagTaskThroughputHierarchicalPF : public DagTaskThroughput {
public:
  DagTaskThroughputHierarchicalPF(const BenchmarkArgs& args) : DagTaskThroughput{args} {}

  void run() { submit_hierarchical_parallel_for(); }

  static std::string getBenchmarkName(BenchmarkArgs& args) {
    return "Runtime_DAGTaskThroughput_HierarchicalParallelFor";
  }
};


int main(int argc, char** argv) {
  BenchmarkApp app(argc, argv);

  app.run<DagTaskThroughputSingleTask>();
  app.run<DagTaskThroughputBasicPF>();
  app.run<DagTaskThroughputHierarchicalPF>();
  // With pure CPU library implementations such as hipSYCL CPU backend
  // or triSYCL, this will be prohibitively slow
  if(app.shouldRunNDRangeKernels())
    app.run<DagTaskThroughputNDRangePF>();

  return 0;
}
