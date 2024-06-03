#include <array>
#include <type_traits.h>

template <std::size_t... Idx, typename F>
void loop_impl(std::integer_sequence<std::size_t, Idx...>, F&& f) {
  (f(std::integral_constant<std::size_t, Idx>{}), ...);
}

template <std::size_t count, typename F>
void loop(F&& f) {
  loop_impl(std::make_index_sequence<count>{}, std::forward<F>(f));
}