#include <sycl/sycl.hpp>

#ifndef __ACPP__

static constexpr sycl::specialization_id<int> x;

int main() {
  sycl::queue q;
  int* i = sycl::malloc_shared<int>(1, q);
  q.submit([&](sycl::handler& cgh) { 
    cgh.set_specialization_constant<x>(5); 
    cgh.parallel_for(sycl::range(1), [=](sycl::item<1> item, sycl::kernel_handler h) {
      *i = h.get_specialization_constant<x>();
    });
   }).wait();

  assert(*i == 5);
  sycl::free(i, q);
}

#else

// AdaptiveCpp implements sycl::specialized instead of spec constants

int main() { 
  sycl::queue q;
  sycl::specialized<int> x;
  x = 5; //Requires copy assignment operator
  int* i = sycl::malloc_shared<int>(1, q);
  q.parallel_for(sycl::range(1), [=](sycl::id<1> idx) {
    *i = x;
  }).wait();

  assert(*i == 5);
  sycl::free(i, q);
}

#endif