#ifndef BOOST_ROSE_CHAR_CHAR_WRITER_HPP_INCLUDED
#define BOOST_ROSE_CHAR_CHAR_WRITER_HPP_INCLUDED

#include <boost/rose/str_writer.hpp>

namespace boost
{
namespace rose
{
  template <class charT, class traits>
  struct char_writer: public str_writer<charT>
  {
    charT character;

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

    virtual charT* write_without_termination_char(charT* output) const noexcept
    {
      traits::assign(*output, character);
      return output + 1;
    }
  };

  template <typename charT, typename traits>
  inline char_writer<charT, traits> basic_argf(charT c) noexcept
  {
    return c;
  }

} // namespace rose
} // namespace boost

#endif



