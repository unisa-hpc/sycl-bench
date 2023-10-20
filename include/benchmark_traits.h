#pragma once

#include <utility>

#include <sycl/sycl.hpp>

namespace detail {

template <typename T, typename = void>
struct SupportsQueueProfiling {
  static constexpr bool value = false;
};

template <typename T>
struct SupportsQueueProfiling<T,
    std::void_t<decltype(std::declval<T>().run(std::declval<std::vector<sycl::event>&>()))>> {
  static constexpr bool value = true;
};

#define MAKE_HAS_METHOD_TRAIT(T, method, name)                                                                         \
  template <typename _T>                                                                                               \
  static constexpr std::false_type _has_##method(...);                                                                 \
  template <typename _T>                                                                                               \
  static constexpr std::true_type _has_##method(_T, decltype(&_T::method) = nullptr);                                  \
  static constexpr bool name = std::is_same_v<decltype(_has_##method<T>(std::declval<T>())), std::true_type>;

template <typename T>
struct BenchmarkTraits {
  MAKE_HAS_METHOD_TRAIT(T, verify, hasVerify)
  MAKE_HAS_METHOD_TRAIT(T, getThroughputMetric, hasGetThroughputMetric)

  static constexpr bool supportsQueueProfiling = SupportsQueueProfiling<T>::value;
};

} // namespace detail
