#pragma once
#include <CL/sycl.hpp>
#include <memory>

class InitializationDummyKernel;

template <class BufferType>
inline void forceDataTransfer(cl::sycl::queue& q, BufferType b) {
  q.submit([&](cl::sycl::handler& cgh) {
    auto acc = b.template get_access<cl::sycl::access::mode::read>(cgh);
    cgh.single_task<InitializationDummyKernel>([=]() {});
  });
  q.wait_and_throw();
}

template <class T, int Dimensions>
class PrefetchedBuffer {
public:
  void initialize(cl::sycl::queue& q, cl::sycl::range<Dimensions> r) {
    buff = std::make_shared<cl::sycl::buffer<T, Dimensions>>(r);
  }

  void initialize(cl::sycl::queue& q, T* data, cl::sycl::range<Dimensions> r) {
    buff = std::make_shared<cl::sycl::buffer<T, Dimensions>>(data, r);
    forceDataTransfer(q, *buff);
  }

  void initialize(cl::sycl::queue& q, const T* data, cl::sycl::range<Dimensions> r) {
    buff = std::make_shared<cl::sycl::buffer<T, Dimensions>>(data, r);
    forceDataTransfer(q, *buff);
  }


  template <cl::sycl::access::mode mode, cl::sycl::access::target target = cl::sycl::access::target::global_buffer>
  auto get_access(cl::sycl::handler& commandGroupHandler) {
    return buff->template get_access<mode, target>(commandGroupHandler);
  }

  template <cl::sycl::access::mode mode>
  auto get_access() {
    return buff->template get_access<mode>();
  }

  template <cl::sycl::access::mode mode, cl::sycl::access::target target = cl::sycl::access::target::global_buffer>
  auto get_access(cl::sycl::handler& commandGroupHandler, cl::sycl::range<Dimensions> accessRange,
      cl::sycl::id<Dimensions> accessOffset = {}) {
    return buff->template get_access<mode, target>(commandGroupHandler, accessRange, accessOffset);
  }

  template <cl::sycl::access::mode mode>
  auto get_access(cl::sycl::range<Dimensions> accessRange, cl::sycl::id<Dimensions> accessOffset = {}) {
    return buff->template get_access<mode>(accessRange, accessOffset);
  }

  cl::sycl::buffer<T, Dimensions>& get() const { return *buff; }

private:
  // Wrap in a shared_ptr to allow default constructing this class
  std::shared_ptr<cl::sycl::buffer<T, Dimensions>> buff;
};