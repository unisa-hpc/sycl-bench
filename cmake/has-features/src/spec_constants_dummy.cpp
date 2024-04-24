#include <sycl/sycl.hpp>

#ifndef __ACPP__

static constexpr sycl::specialization_id<int> x;

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler& cgh) { cgh.set_specialization_constant<x>(5); });
}

#else

// AdaptiveCpp implements sycl::specialized instead of spec constants

int main() { sycl::specialized<int> x(5); }

#endif