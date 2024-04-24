#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  sycl::buffer<double> x(1);

  q.submit([&](sycl::handler& cgh) {
    sycl::accessor a(x, cgh, sycl::read_write);
    cgh.parallel_for<class dummy>(sycl::range<1>(1), [=](sycl::id<1> idx) { a[idx] = 0; });
  });
}