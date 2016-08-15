#ifndef BOOST_STRINGIFY_DETAIL_CHAR_WRITER_HPP_INCLUDED
#define BOOST_STRINGIFY_DETAIL_CHAR_WRITER_HPP_INCLUDED

#include <boost/stringify/str_writer.hpp>
#include <type_traits>

namespace boost
{
namespace stringify
{
namespace detail
{
  template <typename charT>
  class char_writer: public str_writer<charT>
  {
    charT character;

  public:
    char_writer() noexcept
      :character()
    {
    }

    char_writer(charT _character) noexcept
      :character(_character)
    {
    }

    void set(charT _character) noexcept
    {
      character = _character;
    }

    virtual std::size_t minimal_length() const noexcept
    {
      return 1;
    }

    virtual charT* write_without_termination_char(charT* out) const noexcept
    {
      *out = character;
      return out + 1;
    }

    virtual void write(simple_ostream<charT>& out) const
    {
      out.put(character);
    }
  };


  class char32_to_utf8: public str_writer<char>
  {
  public:

    char32_to_utf8() noexcept
    :codepoint(0XFFFFFFFF)
    {
    }

    char32_to_utf8(char32_t _codepoint) noexcept
    :codepoint(_codepoint)
    {
    }

    void set(char32_t _codepoint) noexcept
    {
      codepoint = _codepoint;
    }

    virtual std::size_t minimal_length() const noexcept
    {
      return (codepoint <     0x80 ? 1 :
              codepoint <    0x800 ? 2 :
              codepoint <  0x10000 ? 3 :
              codepoint < 0x110000 ? 4 :
              /* invalid codepoit */ 0);
    }

    virtual char* write_without_termination_char(char* out) const noexcept
    {
      return (codepoint <     0x80 ? write_utf8_range1(out) :
              codepoint <    0x800 ? write_utf8_range2(out) :
              codepoint <  0x10000 ? write_utf8_range3(out) :
              codepoint < 0x110000 ? write_utf8_range4(out) :
              /* invalid codepoit */ out);
    }

    virtual void write(simple_ostream<char>& out) const
    {
      char buff[4];
      write_without_termination_char(buff);
      out.write(buff, 4);
    }

  private:

    char32_t codepoint;

    char* write_utf8_range1(char* out) const noexcept
    {
      *out =  static_cast<char>(codepoint);
      return ++out;
    }

    char* write_utf8_range2(char* out) const noexcept
    {
      *  out = static_cast<char>(0xC0 | ((codepoint & 0x7C0) >> 6));
      *++out = static_cast<char>(0x80 |  (codepoint &  0x3F));
      return ++out;
    }

    char* write_utf8_range3(char* out) const noexcept
    {
      *  out = static_cast<char>(0xE0 | ((codepoint & 0xF000) >> 12));
      *++out = static_cast<char>(0x80 | ((codepoint &  0xFC0) >> 6));
      *++out = static_cast<char>(0x80 |  (codepoint &   0x3F));
      return ++out;
    }

    char* write_utf8_range4(char* out) const noexcept
    {
      *  out = static_cast<char>(0xF0 | ((codepoint & 0x1C0000) >> 18));
      *++out = static_cast<char>(0x80 | ((codepoint &  0x3F000) >> 12));
      *++out = static_cast<char>(0x80 | ((codepoint &    0xFC0) >> 6));
      *++out = static_cast<char>(0x80 |  (codepoint &     0x3F));
      return ++out;
    }
  };

  template <typename charT>
  class char32_to_utf16: public str_writer<charT>
  {

  public:

    char32_to_utf16() noexcept
    :codepoint(0xFFFFFFFF)
    {
    }

    char32_to_utf16(char32_t _codepoint) noexcept
    :codepoint(_codepoint)
    {
    }

    void set(char32_t _codepoint) noexcept
    {
      codepoint = _codepoint;
    }

    virtual std::size_t minimal_length() const noexcept
    {
      return (single_char_range() ? 1 :
              two_chars_range()   ? 2 : 0);
    }

    virtual charT* write_without_termination_char(charT* out) const noexcept
    {
      if (single_char_range())
      {
        *out++ = static_cast<charT>(codepoint);
      }
      else if (two_chars_range())
      {
        char32_t sub_codepoint = codepoint - 0x10000;
        char32_t high_surrogate = 0xD800 + ((sub_codepoint & 0xFFC00) >> 10);
        char32_t low_surrogate  = 0xDC00 +  (sub_codepoint &  0x3FF);
        *out++ = static_cast<charT>(high_surrogate);
        *out++ = static_cast<charT>(low_surrogate);
      }
      return out;
    }

    virtual void write(simple_ostream<charT>& out) const
    {
      if (single_char_range())
      {
        out.put(static_cast<charT>(codepoint));
      }
      else if (two_chars_range())
      {
        char32_t sub_codepoint = codepoint - 0x10000;
        char32_t high_surrogate = 0xD800 + ((sub_codepoint & 0xFFC00) >> 10);
        char32_t low_surrogate  = 0xDC00 +  (sub_codepoint &  0x3FF);
        out.put(static_cast<charT>(high_surrogate));
        out.put(static_cast<charT>(low_surrogate));
      }
    }

  private:

    char32_t codepoint;

    bool single_char_range() const
    {
      return codepoint < 0xd800 || (0xdfff < codepoint && codepoint <  0x10000);
    }

    bool two_chars_range() const
    {
      return 0xffff < codepoint && codepoint < 0x110000;
    }
  };

} // namespace detail
} // namespace stringify
} // namespace boost

#endif



