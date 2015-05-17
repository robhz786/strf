#ifndef BOOST_ROSE_ARGF_HPP_INCLUDED
#define BOOST_ROSE_ARGF_HPP_INCLUDED

#include <string>

namespace boost
{
namespace rose
{
  template <typename T, typename traits = std::char_traits<char> >
  inline
  decltype(basic_argf<char, traits>(*(const T*)(0)))
  argf(T value)
  {
    return basic_argf<char, traits>(value); 
  }

  template <typename T, typename traits = std::char_traits<wchar_t> >
  inline
  decltype(basic_argf<wchar_t, traits>(*(const T*)(0)))
  wargf(T value)
  {
    return basic_argf<wchar_t, traits>(value); 
  }

  template <typename T, typename traits = std::char_traits<char16_t> >
  inline
  decltype(basic_argf<char16_t, traits>(*(const T*)(0)))
  argf16(T value)
  {
    return basic_argf<char16_t, traits>(value); 
  }

  template <typename T, typename traits = std::char_traits<char32_t> >
  inline
  decltype(basic_argf<char32_t, traits>(*(const T*)(0)))
  argf32(T value)
  {
    return basic_argf<char32_t, traits>(value); 
  }
} // namespace rose
} // namespace boost

#endif

















