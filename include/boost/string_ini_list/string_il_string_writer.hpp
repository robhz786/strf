#ifndef BOOST_DETAIL_STRING_IL_STRING_WRITER_HPP_INCLUDED
#define BOOST_DETAIL_STRING_IL_STRING_WRITER_HPP_INCLUDED

#include <string>
#include <algorithm>
#include <boost/string_ini_list/string_il_writer.hpp>

namespace boost
{
  //--------------------------------------------------
  // std::string 

  template<>
  template<typename charT,
           typename traits>
  class string_il_writer<std::basic_string<charT, traits>, charT, traits>:
     public string_il_writer_base<charT>
  {
  public:
    const std::basic_string<charT, traits>& str;    

    string_il_writer(const std::basic_string<charT, traits>& _str)
      :str(_str)
    {
    }

    virtual std::size_t minimal_length() const
    {
      return str.length();
    }

    virtual charT* write_without_termination_char(charT* output) const
    {
      std::size_t length = str.length();
      traits::copy(output, &*str.begin(), length);
      return output + length;
    }
  };

  //--------------------------------------------------
  // const charT* 

  template<>
  template<typename charT,
           typename traits>
  class string_il_writer<const charT*, charT, traits>:
    public string_il_writer_base<charT>
  {
  public:
    const charT* str;    

    string_il_writer(const charT* _str)
      :str(_str)
    {
    }

    virtual std::size_t minimal_length() const
    {
      return traits::length(str);
    }

    virtual charT* write_without_termination_char(charT* output) const
    {
      std::size_t length = traits::length(str);
      traits::copy(output, str, length);
      return output + length;
    }
  }; 

}; // namespace boost


#endif
