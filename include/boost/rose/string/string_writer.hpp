#ifndef BOOST_LISTF_STRING_WRITER_HPP_INCLUDED
#define BOOST_LISTF_STRING_WRITER_HPP_INCLUDED

#include <string>
#include <algorithm>
#include <boost/rose/str_writer.hpp>

namespace boost
{
namespace rose
{
  template <class charT, class traits>
  struct std_string_writer: public str_writer<charT>
  {
    const std::basic_string<charT, traits>* str;    

    std_string_writer()
      :str(0)
    {
    }

    std_string_writer(const std::basic_string<charT, traits>& _str)
      :str(&_str)
    {
    }

    void set(const std::basic_string<charT, traits>& _str)
    {
      str = &_str;
    }

    virtual std::size_t minimal_length() const
    {
      return str ? str->length() : 0;
    }

    virtual charT* write_without_termination_char(charT* output) const
    {
      if( ! str)
        return output;
      std::size_t length = str->length();
      traits::copy(output, &*str->begin(), length);
      return output + length;
    }
  };

  template <typename charT, typename traits>
  inline
  std_string_writer<charT, traits>
  basic_argf(const std::basic_string<charT, traits>& str)
  {
    return str;
  }

  template<typename charT,typename traits>
  struct char_ptr_writer: str_writer<charT>
  {
    const charT* str;    

    char_ptr_writer()
      :str(0)
    {
    }

    char_ptr_writer(const charT* _str)
      :str(_str)
    {
    }

    void set(const charT* _str)
    {
      str = _str;
    }

    virtual std::size_t minimal_length() const
    {
      return str ? traits::length(str) : 0;
    }

    virtual charT* write_without_termination_char(charT* output) const
    {
      if( ! str)
        return output;
      std::size_t length = traits::length(str);
      traits::copy(output, str, length);
      return output + length;
    }
  }; 

  template <typename charT, typename traits>
  inline
  char_ptr_writer<charT, traits> basic_argf(const charT* str)
  {
    return str;
  }


} // namespace rose
} // namespace boost


#endif
