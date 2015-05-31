#ifndef BOOST_ROSE_STRING_STRING_WRITER_HPP_INCLUDED
#define BOOST_ROSE_STRING_STRING_WRITER_HPP_INCLUDED

#include <string>
#include <boost/rose/str_writer.hpp>

namespace boost
{
namespace rose
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

    virtual charT* write_without_termination_char(charT* output) const noexcept
    {
      if( ! str)
        return output;

      return std::copy(str->begin(), str->end(), output);
    }
  };

  template <typename charT, typename traits>
  inline
  std_string_writer<charT, traits>
  basic_argf(const std::basic_string<charT, traits>& str) noexcept
  {
    return str;
  }

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
      len(-1)
    {
    }

    void set(const charT* _str) noexcept
    {
      str = _str;
      len = -1;
    }

    virtual std::size_t minimal_length() const noexcept
    {
      return get_length();
    }

    virtual charT* write_without_termination_char(charT* output) const noexcept
    {
      return std::copy(str, str + get_length(), output);
    }

  private:
    mutable std::size_t len;

    std::size_t get_length() const noexcept
    {
      if (len == -1)
      {
        try
        {
          len = std::char_traits<charT>::length(str);
        }
        catch(...)
        {
          len = 0;
        }
      }
      return len;
    }

  }; 

  template <typename charT>
  inline
  char_ptr_writer<charT> basic_argf(const charT* str) noexcept
  {
    return str;
  }


} // namespace rose
} // namespace boost


#endif
