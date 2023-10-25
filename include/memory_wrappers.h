#pragma once
#include "common.h"
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
    buff->set_write_back(false);
    forceDataTransfer(q, *buff);
  }

  void initialize(sycl::queue& q, const T* data, sycl::range<Dimensions> r) {
    buff = std::make_shared<sycl::buffer<T, Dimensions>>(data, r);
    buff->set_write_back(false);
    forceDataTransfer(q, *buff);
  }


  template <sycl::access::mode mode, sycl::target target = sycl::target::device>
  auto get_access(sycl::handler& commandGroupHandler) {
    return buff->template get_access<mode, target>(commandGroupHandler);
  }

  template <sycl::access::mode mode>
  auto get_access() {
    return buff->template get_access<mode>();
  }

  template <sycl::access::mode mode, sycl::target target = sycl::target::device>
  auto get_access(
      sycl::handler& commandGroupHandler, sycl::range<Dimensions> accessRange, sycl::id<Dimensions> accessOffset = {}) {
    return buff->template get_access<mode, target>(commandGroupHandler, accessRange, accessOffset);
  }

  template <sycl::access::mode mode>
  auto get_access(sycl::range<Dimensions> accessRange, sycl::id<Dimensions> accessOffset = {}) {
    return buff->template get_access<mode>(accessRange, accessOffset);
  }

  auto get_host_access() { return buff->template get_host_access(); }

  sycl::range<Dimensions> get_range() const { return buff->get_range(); }

  sycl::buffer<T, Dimensions>& get() const { return *buff; }

  void reset() { buff = nullptr; }

private:
  // Wrap in a shared_ptr to allow default constructing this class
  std::shared_ptr<sycl::buffer<T, Dimensions>> buff;
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


template <typename T, std::size_t dim = 1, sycl::usm::alloc type = sycl::usm::alloc::device>
class USMBuffer {
static_assert(dim >= 1 && dim <= 3, "Invalid dim provided");
protected:
  T* _data;
  sycl::range<dim> _count;
  std::size_t total_size;

public:
  USMBuffer() : _data(nullptr), _count(getRange()), total_size(0) {}

  template <typename U = T, typename = detail::has_dim_t<U, dim, 1>>
  void initialize(sycl::queue& q, size_t count) {
    allocate(q, count);
  }

  void initialize(sycl::queue& q, sycl::range<dim> count) { 
      allocate(q, count);
    }

  void initialize(sycl::queue& q, const T* data, size_t count) {
    allocate(q, count);
    copy(q, data, _data, count);
  }

  void initialize(sycl::queue& q, const T* data, sycl::range<dim> count) {
    allocate(q, count);
    copy(q, data, _data, count);
  }

  auto get() const { return _data; }

private:
  template <sycl::usm::alloc alloc_type>
  static T* malloc(sycl::queue& Q, size_t count) {
    if constexpr(alloc_type == sycl::usm::alloc::device)
      return sycl::malloc_device<T>(count, Q);
    if constexpr(alloc_type == sycl::usm::alloc::host)
      return sycl::malloc_host<T>(count, Q);
    else if constexpr(alloc_type == sycl::usm::alloc::shared)
      return sycl::malloc_shared<T>(count, Q);
    else
      throw std::runtime_error("Malloc invoked with unkown allocation type!");
  }

  auto constexpr getRange(){    
    if constexpr (dim == 1){
      return sycl::range<dim>(0);
    }
    if constexpr (dim == 2){
      return sycl::range<dim>(0,0);
    }
    if constexpr (dim == 3){
      return sycl::range<dim>(0,0,0);
    }
  }

  std::size_t inline getSize(const sycl::range<dim>& count){
    std::size_t total_size = 0;
    loop<dim>([&](std::size_t val) { total_size += count[val]; });
    return total_size;
  }

  template <typename U = T, typename = detail::has_dim_t<U, dim, 1>>
  void allocate(sycl::queue& Q, size_t count) {
    assert(count >= 0 && "Cannot allocate negative num bytes");
    _data = malloc<type>(Q, count);
    this->_count = sycl::range<dim>{count};
    total_size = count;
  }

  void allocate(sycl::queue& Q, const sycl::range<dim>& count) {
    loop<dim>([&](std::size_t idx){
      assert(count[idx] >= 0 && "Cannot allocate negative num bytes");
    });

    const size_t total_size = getSize(count);
    _data = malloc<type>(Q, total_size);
    
    this->_count = count;
    this->total_size = total_size;
  }

  void copy(sycl::queue& Q, const T* src, T* dst, std::size_t count) const {
    // assert(count <= _count[0] && "Cannot copy negative num bytes");
    // assert(_data != nullptr && "Called copy on initialized USM buffer");
    Q.copy(src, dst, count).wait_and_throw();
  }

  void copy(sycl::queue& Q, const T* src, T* dst, sycl::range<dim> count) const {
    loop<dim>([&](std::size_t idx){
      assert(count[idx] >= 0 && "Cannot copy negative num bytes");
    });

    const size_t total_size = getSize(count);
    Q.copy(src, dst, count).wait_and_throw();
  }


};