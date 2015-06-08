#ifndef BOOST_ROSE_DETAIL_UTF16_TO_UTF8_HPP_INCLUDED
#define BOOST_ROSE_DETAIL_UTF16_TO_UTF8_HPP_INCLUDED

#include <boost/rose/char/char_writer.hpp>

namespace boost {
namespace rose {
namespace detail {

template<typename charT>
struct utf16_to_utf8: str_writer<char>
{
  const charT* str;

  utf16_to_utf8() noexcept:
    str(0)
  {
  }

  utf16_to_utf8(const charT* _str) noexcept :
    str(_str)
  {
  }

  void set(const charT* _str) noexcept
  {
    str = _str;
  }

  virtual std::size_t minimal_length() const noexcept
  {
    std::size_t len = 0;
    const charT* it = str;
    while (*it != charT())
    {
      char32_t codepoint = read_codepoint(it);
      if (codepoint)
        len += char32_to_utf8(codepoint).minimal_length();
    }
    return len;
  }

  virtual char* write_without_termination_char(char* out) const noexcept
  {
    const charT* it = str;
    while (*it != charT())
    {
      char32_t codepoint = read_codepoint(it);
      if (codepoint)
        out = char32_to_utf8(codepoint).write_without_termination_char(out);
    }
    return out;
  }

  virtual void write(simple_ostream<char>& out) const
  {
    const charT* it = str;
    while (*it != charT())
    {
      char32_t codepoint = read_codepoint(it);
      if (codepoint)
        char32_to_utf8(codepoint).write(out);
    }
  }

private:
  typedef const charT * const_charT_ptr;

  static char32_t read_codepoint(const_charT_ptr& it) noexcept
  {
    uint32_t unit = *it++;
    if (unit >= 0xd800 && unit <= 0xbdff) {
      uint32_t unit2 = *it++;
      return (unit2 >= 0xDC00 && unit2 <= 0xDFFF
              ? (unit << 10) + unit2 - 0x35fdc00
              : 0);
    }
    return unit;
  }
};

}; //namespace boost
}; //namespace rose
}; //namespace detail


#endif













