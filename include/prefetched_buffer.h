#pragma once
#include <memory>
#include <sycl/sycl.hpp>

template <class AccType>
class InitializationDummyKernel {
public:
  InitializationDummyKernel(AccType acc) : acc{acc} {}

  void operator()() const {}

private:
  AccType acc;
};

class InitializationDummyKernel2;

template <class BufferType>
inline void forceDataTransfer(sycl::queue& q, BufferType b) {
  q.submit([&](sycl::handler& cgh) {
    auto acc = b.template get_access<sycl::access::mode::read>(cgh);
    cgh.single_task(InitializationDummyKernel{acc});
  });
  q.wait_and_throw();
}

template <class BufferType>
inline void forceDataAllocation(sycl::queue& q, BufferType b) {
  q.submit([&](sycl::handler& cgh) {
    auto acc = b.template get_access<sycl::access::mode::discard_write>(cgh);
    cgh.single_task(InitializationDummyKernel{acc});
  });
  q.wait_and_throw();
}

template <class T, int Dimensions = 1>
class PrefetchedBuffer {
public:
  void initialize(sycl::queue& q, sycl::range<Dimensions> r) {
    buff = std::make_shared<sycl::buffer<T, Dimensions>>(r);
    forceDataAllocation(q, *buff);
  }

  void initialize(sycl::queue& q, T* data, sycl::range<Dimensions> r) {
    buff = std::make_shared<sycl::buffer<T, Dimensions>>(data, r);
    forceDataTransfer(q, *buff);
  }

  void initialize(sycl::queue& q, const T* data, sycl::range<Dimensions> r) {
    buff = std::make_shared<sycl::buffer<T, Dimensions>>(data, r);
    forceDataTransfer(q, *buff);
  }


  template <sycl::access::mode mode, sycl::access::target target = sycl::access::target::global_buffer>
  auto get_access(sycl::handler& commandGroupHandler) {
    return buff->template get_access<mode, target>(commandGroupHandler);
  }

  template <sycl::access::mode mode>
  auto get_access() {
    return buff->template get_access<mode>();
  }

  template <sycl::access::mode mode, sycl::access::target target = sycl::access::target::global_buffer>
  auto get_access(
      sycl::handler& commandGroupHandler, sycl::range<Dimensions> accessRange, sycl::id<Dimensions> accessOffset = {}) {
    return buff->template get_access<mode, target>(commandGroupHandler, accessRange, accessOffset);
  }

  template <sycl::access::mode mode>
  auto get_access(sycl::range<Dimensions> accessRange, sycl::id<Dimensions> accessOffset = {}) {
    return buff->template get_access<mode>(accessRange, accessOffset);
  }

  sycl::range<Dimensions> get_range() const { return buff->get_range(); }

  sycl::buffer<T, Dimensions>& get() const { return *buff; }

  void reset() { buff = nullptr; }

private:
  // Wrap in a shared_ptr to allow default constructing this class
  std::shared_ptr<sycl::buffer<T, Dimensions>> buff;
};
