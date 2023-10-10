#pragma once
#include <CL/sycl.hpp>
#include <memory>

#include "utils.h"

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
inline void forceDataTransfer(cl::sycl::queue& q, BufferType b) {
  q.submit([&](cl::sycl::handler& cgh) {
    auto acc = b.template get_access<cl::sycl::access::mode::read>(cgh);
    cgh.single_task(InitializationDummyKernel{acc});
  });
  q.wait_and_throw();
}

template <class BufferType>
inline void forceDataAllocation(cl::sycl::queue& q, BufferType b) {
  q.submit([&](cl::sycl::handler& cgh) {
    auto acc = b.template get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.single_task(InitializationDummyKernel{acc});
  });
  q.wait_and_throw();
}

template <class T, int Dimensions = 1>
class PrefetchedBuffer {
public:
  void initialize(cl::sycl::queue& q, cl::sycl::range<Dimensions> r) {
    buff = std::make_shared<cl::sycl::buffer<T, Dimensions>>(r);
    forceDataAllocation(q, *buff);
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

  cl::sycl::range<Dimensions> get_range() const { return buff->get_range(); }

  cl::sycl::buffer<T, Dimensions>& get() const { return *buff; }

  void reset() { buff = nullptr; }

private:
  // Wrap in a shared_ptr to allow default constructing this class
  std::shared_ptr<cl::sycl::buffer<T, Dimensions>> buff;
};

namespace detail {
template <typename T, typename U, size_t val, size_t expected>
struct has_dim_impl {
  static constexpr bool value = val == expected;
};

template <typename T, size_t val, size_t expected>
static constexpr bool has_dim_v = has_dim_impl<T, T, val, expected>::value;

template <typename T, size_t val, size_t expected>
using has_dim_t = std::enable_if_t<has_dim_v<T, val, expected> == true, void>;
} // namespace detail


template <typename T, std::size_t dim, cl::sycl::usm::alloc type = cl::sycl::usm::alloc::device>
class USMBuffer {
protected:
  T* _data;
  cl::sycl::range<dim> _count;
  std::size_t total_size;

public:
  USMBuffer() : _data(nullptr), _count(), total_size(0) {}

  template <typename U = T, typename = detail::has_dim_t<U, dim, 1>>
  void initialize(cl::sycl::queue& q, size_t count) {
    allocate(q, count);
  }

  void initialize(cl::sycl::queue& q, cl::sycl::range<dim> count) { 
      allocate(q, count);
    }

  void initialize(cl::sycl::queue& q, const T* data, size_t count) {
    allocate(q, count);
    copy(q, data, _data, count);
  }

  void initialize(cl::sycl::queue& q, const T* data, cl::sycl::range<dim> count) {
    allocate(q, count);
    copy(q, data, _data, count);
  }

  auto get() const { return _data; }

private:
  template <cl::sycl::usm::alloc alloc_type>
  static T* malloc(cl::sycl::queue& Q, size_t count) {
    if constexpr(alloc_type == cl::sycl::usm::alloc::device)
      return cl::sycl::malloc_device<T>(count, Q);
    if constexpr(alloc_type == cl::sycl::usm::alloc::host)
      return cl::sycl::malloc_host<T>(count, Q);
    else if constexpr(alloc_type == cl::sycl::usm::alloc::shared)
      return cl::sycl::malloc_shared<T>(count, Q);
    else
      throw std::runtime_error("Malloc invoked with unkown allocation type!");
  }

  std::size_t inline getSize(const cl::sycl::range<dim>& count){
    std::size_t total_size = 0;
    loop<dim>([&](std::size_t val) { total_size += count[val]; });
    return total_size;
  }

  template <typename U = T, typename = detail::has_dim_t<U, dim, 1>>
  void allocate(cl::sycl::queue& Q, size_t count) {
    assert(count >= 0 && "Cannot allocate negative num bytes");
    _data = malloc<type>(Q, count);
    this->_count = cl::sycl::range<dim>{count};
    total_size = count;
  }

  void allocate(cl::sycl::queue& Q, const cl::sycl::range<dim>& count) {
    loop<dim>([&](std::size_t idx){
      assert(count[idx] >= 0 && "Cannot allocate negative num bytes");
    });

    const size_t total_size = getSize(count);
    _data = malloc<type>(Q, total_size);
    
    this->_count = count;
    this->total_size = total_size;
  }

  void copy(cl::sycl::queue& Q, const T* src, T* dst, std::size_t count) const {
    // assert(count <= _count[0] && "Cannot copy negative num bytes");
    // assert(_data != nullptr && "Called copy on initialized USM buffer");
    Q.copy(src, dst, count).wait_and_throw();
  }

  void copy(cl::sycl::queue& Q, const T* src, T* dst, cl::sycl::range<dim> count) const {
    loop<dim>([&](std::size_t idx){
      assert(count[idx] >= 0 && "Cannot allocate negative num bytes");
    });

    const size_t total_size = getSize(count);
    Q.copy(src, dst, count).wait_and_throw();
  }


};