#ifndef BOOST_STRINGIFY_ARGF_HPP_INCLUDED
#define BOOST_STRINGIFY_ARGF_HPP_INCLUDED

#include <string>

namespace boost
{
namespace stringify
{
  template <typename T>
  inline auto argf(T value) noexcept -> decltype(basic_argf<char>(value))
  {
    return basic_argf<char>(value); 
  }

  template <typename T>
  inline auto wargf(T value) noexcept -> decltype(basic_argf<wchar_t>(value))
  {
    return basic_argf<wchar_t>(value); 
  }

  template <typename T>
  inline auto argf16(T value) noexcept -> decltype(basic_argf<char16_t>(value))
  {
    return basic_argf<char16_t>(value); 
  }

  template <typename T>
  inline auto argf32(T value) noexcept -> decltype(basic_argf<char32_t>(value))
  {
    return basic_argf<char32_t>(value); 
  }
} // namespace stringify
} // namespace boost

#endif

















