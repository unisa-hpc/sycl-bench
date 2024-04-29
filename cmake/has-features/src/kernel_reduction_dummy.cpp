#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  sycl::buffer<int> x(1);
  q.submit([&](sycl::handler& cgh) {
#ifdef __ACPP__
    auto r = sycl::reduction(x.template get_access<sycl::access_mode::read_write>(cgh), sycl::plus<int>{});
#else
    auto r = sycl::reduction(x, cgh, sycl::plus<int>{});
#endif

    cgh.parallel_for(sycl::range<1>{1}, r, [=](sycl::id<1> idx, auto& op) { op.combine(1); });
  });
}