#ifndef BOOST_ROSE_INTPUT_TYPES_HPP_INCLUDED
#define BOOST_ROSE_INTPUT_TYPES_HPP_INCLUDED

#include <boost/rose/detail/int_writer.hpp>
#include <boost/rose/detail/char_writer.hpp>
#include <boost/rose/detail/string_writer.hpp>


namespace boost
{
namespace rose
{
  //
  // integers
  //

  template <typename charT>
  inline                                    
  detail::int_writer<int, charT> basic_argf(int i) noexcept
  {                                               
    return i;                                     
  }

  template <typename charT>
  inline
  detail::int_writer<long, charT> basic_argf(long i) noexcept
  {
    return i;
  }

  template <typename charT>
  inline
  detail::int_writer<long long, charT> basic_argf(long long i) noexcept
  {
    return i;
  }

  template <typename charT>
  inline
  detail::int_writer<unsigned int, charT> basic_argf(unsigned int i) noexcept
  {
    return i;
  }

  template <typename charT>
  inline
  detail::int_writer<unsigned long, charT> basic_argf(unsigned long i) noexcept
  {
    return i;
  }

  template <typename charT>
  inline
  detail::int_writer<unsigned long long, charT>
  basic_argf(unsigned long long i) noexcept
  {
    return i;
  }

  //
  // charT and char32_t
  //

  namespace detail
  {
    template <int charT_size, typename charT>
    struct char32_to_utf_traits
    {
    };

    template <typename charT>
    struct char32_to_utf_traits<1, charT>
    {
      typedef char32_to_utf8 writer_type;
    };

    template <typename charT>
    struct char32_to_utf_traits<2, charT>
    {
      typedef char32_to_utf16<charT> writer_type;
    };

    template <typename charT>
    struct char32_to_utf_traits<4, charT>
    {
      typedef detail::char_writer<charT> writer_type;
    };
  };//namespace detail

  template <typename charT>
  inline
  typename detail::char32_to_utf_traits<sizeof(charT), charT>::writer_type
  basic_argf(char32_t c) noexcept
  {
    return c;
  }

  template <typename charT>
  inline typename std::enable_if<
    (sizeof(charT) == 1),
    detail::char_writer<char> >
  ::type
  basic_argf(char c) noexcept
  {
    return c;
  }

  template <typename charT>
  inline typename std::enable_if<
    (sizeof(charT) == 2),
    detail::char_writer<char16_t> >
  ::type
  basic_argf(char16_t c) noexcept
  {
    return c;
  }

  template <typename charT>
  inline typename std::enable_if<
    (sizeof(charT) == sizeof(wchar_t)),
    detail::char_writer<wchar_t> >
  ::type
  basic_argf(wchar_t c) noexcept
  {
    return c;
  }


  //
  // string as (const charT*)
  // 

  template <typename charT>
  inline
  detail::char_ptr_writer<charT> basic_argf(const charT* str) noexcept
  {
    return str;
  }


  //
  // std::basic_string
  // 

  template <typename charT, typename traits>
  inline
  detail::std_string_writer<charT, traits>
  basic_argf(const std::basic_string<charT, traits>& str) noexcept
  {
    return str;
  }

}//namespace rose
}//namespace boost


#endif
