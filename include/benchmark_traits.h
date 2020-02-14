#pragma once

namespace detail {

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
};

} // namespace detail
