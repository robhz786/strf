#ifndef BOOST_ROSE_ARGF_HPP_INCLUDED
#define BOOST_ROSE_ARGF_HPP_INCLUDED

#include <string>

namespace boost
{
namespace rose
{
  template <typename T>
  inline
  decltype(basic_argf<char>(*(const T*)(0)))
  argf(T value)
  {
    return basic_argf<char>(value); 
  }

  template <typename T>
  inline
  decltype(basic_argf<wchar_t>(*(const T*)(0)))
  wargf(T value)
  {
    return basic_argf<wchar_t>(value); 
  }

  template <typename T>
  inline
  decltype(basic_argf<char16_t>(*(const T*)(0)))
  argf16(T value)
  {
    return basic_argf<char16_t>(value); 
  }

  template <typename T>
  inline
  decltype(basic_argf<char32_t>(*(const T*)(0)))
  argf32(T value)
  {
    return basic_argf<char32_t>(value); 
  }
} // namespace rose
} // namespace boost

#endif

















