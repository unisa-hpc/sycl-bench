#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler& cgh) {
    q.parallel_for(sycl::nd_range<1>{{1}, {1}}, [=](sycl::nd_item<1> item) {
      // call only the group algorithms used in SYCL-Bench
      sycl::reduce_over_group(item.get_group(), 0, sycl::plus<int>{});
    });
  });
}