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

template <sycl::usm::alloc type>
struct usm_properties;

using namespace sycl::usm;
template <>
struct usm_properties<alloc::device> {
  static constexpr bool is_device_accessible = true;
  static constexpr bool is_host_accessible = false;
};
template <>
struct usm_properties<alloc::host> {
  static constexpr bool is_device_accessible = true;
  static constexpr bool is_host_accessible = true;
};
template <>
struct usm_properties<alloc::shared> {
  static constexpr bool is_device_accessible = true;
  static constexpr bool is_host_accessible = true;
};


} // namespace detail


template <typename T, std::size_t dim = 1, sycl::usm::alloc type = sycl::usm::alloc::device>
class USMBuffer {
  static_assert(dim >= 1 && dim <= 3, "Invalid dim provided");

protected:
  T* _data;
  T* _host_ptr;
  sycl::range<dim> _count;
  std::size_t total_size;
  sycl::queue queue;

public:
  USMBuffer(const sycl::queue& q) : _data(nullptr), _host_ptr(nullptr), _count(getRange()), total_size(0), queue(q) {}

  ~USMBuffer() {
    if(_data != nullptr) {
      sycl::free(_data, queue);
    }
    if constexpr(!detail::usm_properties<type>::is_host_accessible) {
      if(_host_ptr != nullptr) {
        sycl::free(_host_ptr, queue);
      }
    }
  }

  template <typename U = T, typename = detail::has_dim_t<U, dim, 1>>
  void initialize(size_t count) {
    allocate(count);
  }

  void initialize(sycl::range<dim> count) { allocate(count); }

  void initialize(const T* data, size_t count) {
    allocate(queue, count);
    copy(queue, data, _data, count);
  }

  void initialize(const T* data, sycl::range<dim> count) {
    allocate(count);
    copy(data, _data, count);
  }


  void update_host() {
    if constexpr(!detail::usm_properties<type>::is_host_accessible) {
      if(_host_ptr == nullptr) {
        _host_ptr = static_cast<T*>(sycl::malloc_host(total_size * sizeof(T), queue));
      }
      queue.copy(_data, _host_ptr, total_size);
      queue.wait_and_throw();
    }
  }

  sycl::event update_host(sycl::event event) {
    if constexpr(!detail::usm_properties<type>::is_host_accessible) {
      if(_host_ptr == nullptr) {
        _host_ptr = static_cast<T*>(sycl::malloc_host(total_size * sizeof(T), queue));
      }
      return queue.copy(_data, _host_ptr, total_size, event);
    }
    else return event;
  }

   sycl::event update_device() {
    if constexpr (detail::usm_properties<type>::is_device_accessible && !detail::usm_properties<type>::is_host_accessible){
      assert(_host_ptr != nullptr && "calling update_device when no modification has been made on the host");
      // auto event = queue.copy(_host_ptr, _data, total_size);
      // queue.wait_and_throw();
      return queue.copy(_host_ptr, _data, total_size);
    }
    else return sycl::event{};
  }

  sycl::event update_device(sycl::event event) {
    if constexpr (detail::usm_properties<type>::is_device_accessible && !detail::usm_properties<type>::is_host_accessible){
      assert(_host_ptr != nullptr && "calling update_device when no modification has been made on the host");
      return queue.copy(_host_ptr, _data, total_size, event);
    }
    else return event;
  }

  T* get() const { return _data; }
  
  T* get_host_ptr() const {
    assert(_host_ptr != nullptr && "_host_ptr not initialized. You should first call update_host()");
    return _host_ptr;
  }

  T* update_and_get_host_ptr() {
    update_host();
    return _host_ptr;
  }

  std::tuple<T*, sycl::event> update_and_get_host_ptr(sycl::event event) {
    auto new_event = update_host(event);
    return {_host_ptr, new_event};
  }



  auto size() const { return total_size; }

private:
  template <sycl::usm::alloc alloc_type>
  T* malloc(size_t count) {
    return static_cast<T*>(sycl::malloc(count * sizeof(T), queue, alloc_type));
  }

  auto constexpr getRange() {
    if constexpr(dim == 1) {
      return sycl::range<dim>(0);
    }
    if constexpr(dim == 2) {
      return sycl::range<dim>(0, 0);
    }
    if constexpr(dim == 3) {
      return sycl::range<dim>(0, 0, 0);
    }
  }

  std::size_t inline getSize(const sycl::range<dim>& count) {
    std::size_t total_size = 0;
    loop<dim>([&](std::size_t val) { total_size += count[val]; });
    return total_size;
  }

  template <typename U = T, typename = detail::has_dim_t<U, dim, 1>>
  void allocate(size_t count) {
    assert(count >= 0 && "Cannot allocate negative num bytes");
    _data = malloc<type>(count);
    this->_count = sycl::range<dim>{count};
    total_size = count;

    // All the USM allocations, apart from usm::device, are accessible from the host
    if constexpr(!detail::usm_properties<type>::is_host_accessible) {
      _host_ptr = nullptr;
    } else {
      _host_ptr = _data;
    }
  }

  void allocate(const sycl::range<dim>& count) {
    loop<dim>([&](std::size_t idx) { assert(count[idx] >= 0 && "Cannot allocate negative num bytes"); });

    const size_t total_size = getSize(count);
    _data = malloc<type>(queue, total_size);

    this->_count = count;
    this->total_size = total_size;
    // All the USM allocations, apart from usm::device, are accessible from the host
    if constexpr(type == sycl::usm::alloc::device) {
      _host_ptr = nullptr;
    } else {
      _host_ptr = _data;
    }
  }

  void copy(const T* src, T* dst, std::size_t count) const {
    // assert(count <= _count[0] && "Cannot copy negative num bytes");
    // assert(_data != nullptr && "Called copy on initialized USM buffer");
    queue.copy(src, dst, count).wait_and_throw();
  }

  void copy(const T* src, T* dst, sycl::range<dim> count) const {
    loop<dim>([&](std::size_t idx) { assert(count[idx] >= 0 && "Cannot copy negative num bytes"); });

    const size_t total_size = getSize(count);
    queue.copy(src, dst, count).wait_and_throw();
  }
};