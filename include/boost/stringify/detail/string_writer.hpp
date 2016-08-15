#ifndef BOOST_STRINGIFY_DETAIL_STRING_WRITER_HPP_INCLUDED
#define BOOST_STRINGIFY_DETAIL_STRING_WRITER_HPP_INCLUDED

#include <string>
#include <limits>
#include <boost/stringify/str_writer.hpp>

#ifndef BOOST_PREVENT_MACRO_SUBSTITUTION
#define BOOST_PREVENT_MACRO_SUBSTITUTION
#endif

namespace boost
{
namespace stringify
{
namespace detail
{  
  template <class charT, class traits>
  struct std_string_writer: public str_writer<charT>
  {
    const std::basic_string<charT, traits>* str;    

    std_string_writer() noexcept
      :str(0)
    {
    }

    std_string_writer(const std::basic_string<charT, traits>& _str) noexcept
      :str(&_str)
    {
    }

    void set(const std::basic_string<charT, traits>& _str) noexcept
    {
      str = &_str;
    }

    virtual std::size_t minimal_length() const noexcept
    {
      return str ? str->length() : 0;
    }

    virtual charT* write_without_termination_char(charT* out) const noexcept
    {
      if( ! str)
        return out;

      return std::copy(str->begin(), str->end(), out);
    }

    virtual void write(simple_ostream<charT>& out) const
    {
      if(str)
        out.write(str->c_str(), str->length());
    }
  };


  template<typename charT>
  struct char_ptr_writer: str_writer<charT>
  {
    const charT* str;    

    char_ptr_writer() noexcept:
      str(0),
      len(0)
    {
    }

    char_ptr_writer(const charT* _str) noexcept :
      str(_str),
      len(std::numeric_limits<std::size_t>::max BOOST_PREVENT_MACRO_SUBSTITUTION ())
    {
    }

    void set(const charT* _str) noexcept
    {
      str = _str;
      len = std::numeric_limits<std::size_t>::max BOOST_PREVENT_MACRO_SUBSTITUTION ();
    }

    virtual std::size_t minimal_length() const noexcept
    {
      return get_length();
    }

    virtual charT* write_without_termination_char(charT* out) const noexcept
    {
      return std::copy(str, str + get_length(), out);
    }

    virtual void write(simple_ostream<charT>& out) const
    {
      if(str)
        out.write(str, get_length());
    }


  private:
    mutable std::size_t len;

    std::size_t get_length() const noexcept
    {
      if (len == std::numeric_limits<std::size_t>::max BOOST_PREVENT_MACRO_SUBSTITUTION ())
      {
        try
        { len = std::char_traits<charT>::length(str); }
        catch(...)
        { len = 0; }
      }
      return len;
    }

  }; 


} // namespace detail
} // namespace stringify
} // namespace boost


#endif
