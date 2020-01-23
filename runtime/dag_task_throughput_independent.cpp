#include "common.h"

#include <vector>

using namespace cl;

// Measures the time it takes to run <problem-size> trivial single_task and parallel_for kernels
// that are *independent*. 
// This benchmark can be used to see how well a SYCL implementation
// can utilize hardware concurrency.
class IndependentDagTaskThroughput
{
  std::vector<sycl::buffer<int, 1>> dummy_buffers;
  BenchmarkArgs args;
public:
  IndependentDagTaskThroughput(const BenchmarkArgs &_args) 
  : args(_args)
  {}
  
  void setup() 
  {
    for (std::size_t i = 0; i < args.problem_size; ++i) {
      dummy_buffers.push_back(sycl::buffer<int, 1>{sycl::range<1>{1}});
    }
  }

  void submit_single_task()
  {
    for(std::size_t i = 0; i < args.problem_size; ++i) {

      args.device_queue.submit(
          [&](cl::sycl::handler& cgh) {
        auto acc = dummy_buffers[i].get_access<sycl::access::mode::discard_write>(cgh);
        
        cgh.single_task<class IndependentDagTaskThroughputKernelSingleTask>(
          [=]()
        {
          acc[0] = i;
        });  
      }); // submit
    }
  }

  void submit_basic_parallel_for()
  {
    for(std::size_t i = 0; i < args.problem_size; ++i) {
      args.device_queue.submit(
          [&](cl::sycl::handler& cgh) {
        auto acc = dummy_buffers[i].get_access<sycl::access::mode::discard_write>(cgh);
        
        cgh.parallel_for<class IndependentDagTaskThroughputKernelBasicPF>(
          
          sycl::range<1>{args.local_size},
          [=](sycl::id<1> idx)
        {
          if(idx[0] == 0)
            acc[0] = i;
        });  
      }); // submit
    }
  }

  void submit_ndrange_parallel_for()
  {
    for(std::size_t i = 0; i < args.problem_size; ++i) {
      args.device_queue.submit(
          [&](cl::sycl::handler& cgh) {
        auto acc = dummy_buffers[i].get_access<sycl::access::mode::discard_write>(cgh);
        
        cgh.parallel_for<class DagTaskThroughputKernelNdrangePF>(
          sycl::nd_range<1>{
            sycl::range<1>{args.local_size},
            sycl::range<1>{args.local_size}},
          [=](sycl::nd_item<1> idx)
        {
          if(idx.get_global_id(0) == 0)
            acc[0] = i;
        });  
      }); // submit
    }
  }

  void submit_hierarchical_parallel_for()
  {
    for(std::size_t i = 0; i < args.problem_size; ++i) {
      args.device_queue.submit(
          [&](cl::sycl::handler& cgh) {
        auto acc = dummy_buffers[i].get_access<sycl::access::mode::discard_write>(cgh);
        
        cgh.parallel_for_work_group<class DagTaskThroughputKernelHierarchicalPF>(
          sycl::range<1>{1}, sycl::range<1>{args.local_size},
          [=](sycl::group<1> grp)
        {
          grp.parallel_for_work_item([&](sycl::h_item<1> idx){
            if(idx.get_global_id(0) == 0)
              acc[0] = i;
          });
        });  
      }); // submit
    }
  }

  bool verify(VerificationSetting &ver) { 
    for(std::size_t i = 0; i < dummy_buffers.size(); ++i){
      auto host_acc =
        dummy_buffers[i].get_access<sycl::access::mode::read>();

      if(host_acc[0] != i)
        return false;
    }

    return true;
  }
};

class IndependentDagTaskThroughputSingleTask
    : public IndependentDagTaskThroughput
{
public:
  IndependentDagTaskThroughputSingleTask(const BenchmarkArgs& args)
  : IndependentDagTaskThroughput{args} {}

  void run(){
    submit_single_task();
  }

  static std::string getBenchmarkName() {
    return "Runtime_IndependentDAGTaskThroughput_SingleTask";
  }
};

class IndependentDagTaskThroughputBasicPF
    : public IndependentDagTaskThroughput 
{
public:
  IndependentDagTaskThroughputBasicPF(const BenchmarkArgs& args)
  : IndependentDagTaskThroughput{args} {}

  void run(){
    submit_basic_parallel_for();
  }

  static std::string getBenchmarkName() {
    return "Runtime_IndependentDAGTaskThroughput_BasicParallelFor";
  }
};

class IndependentDagTaskThroughputNDRangePF
    : public IndependentDagTaskThroughput 
{
public:
  IndependentDagTaskThroughputNDRangePF(const BenchmarkArgs& args)
  : IndependentDagTaskThroughput{args} {}

  void run(){
    submit_ndrange_parallel_for();
  }

  static std::string getBenchmarkName() {
    return "Runtime_IndependentDAGTaskThroughput_NDRangeParallelFor";
  }
};

class IndependentDagTaskThroughputHierarchicalPF
    : public IndependentDagTaskThroughput 
{
public:
  IndependentDagTaskThroughputHierarchicalPF(const BenchmarkArgs& args)
  : IndependentDagTaskThroughput{args} {}

  void run(){
    submit_hierarchical_parallel_for();
  }

  static std::string getBenchmarkName() {
    return "Runtime_IndependentDAGTaskThroughput_HierarchicalParallelFor";
  }
};

int main(int argc, char** argv)
{
  BenchmarkApp app(argc, argv);

  app.run<IndependentDagTaskThroughputSingleTask>();
  app.run<IndependentDagTaskThroughputBasicPF>();
  app.run<IndependentDagTaskThroughputHierarchicalPF>();
  // With pure CPU library implementations such as hipSYCL CPU backend
  // or triSYCL, this will be prohibitively slow
  if(app.shouldRunNDRangeKernels())
    app.run<IndependentDagTaskThroughputNDRangePF>();
  
  return 0;
}
