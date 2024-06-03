#include <sycl/sycl.hpp>
#include <iostream>


int main() {
  sycl::queue q;
  int* i = sycl::malloc_shared<int>(1, q);
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>{{1}, {1}}, [=](sycl::nd_item<1> item) {
      // call only the group algorithms used in SYCL-Bench
      *i = sycl::reduce_over_group(item.get_group(), 1, sycl::plus<int>{});
    });
  }).wait();

  assert(*i == 1);
  sycl::free(i, q);
}