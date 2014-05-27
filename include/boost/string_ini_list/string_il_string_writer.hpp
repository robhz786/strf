#ifndef BOOST_DETAIL_STRING_IL_STRING_WRITER_HPP_INCLUDED
#define BOOST_DETAIL_STRING_IL_STRING_WRITER_HPP_INCLUDED

#include <string>
#include <algorithm>
#include <boost/string_ini_list/string_il_writer.hpp>

namespace boost
{

  //--------------------------------------------------
  // std::string 
  struct string_ini_lists_std_string_arg_traits
  {
    template <class charT, class traits>
    struct writer: public string_il_writer_base<charT>
    {
      const std::basic_string<charT, traits>& str;    

      writer(const std::basic_string<charT, traits>& _str)
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
  };
  
  template <class charT, class traits>
  inline
  string_ini_lists_std_string_arg_traits 
  string_ini_list_argument_traits(const std::basic_string<charT, traits>&)
  {
    return string_ini_lists_std_string_arg_traits();
  }

  //--------------------------------------------------
  // const charT* 

  struct string_ini_lists_char_ptr_arg_traits
  {
    template<typename charT,typename traits>
    struct writer: string_il_writer_base<charT>
    {
      const charT* str;    

      writer(const charT* _str)
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
  };

  template <class charT>
  inline
  string_ini_lists_char_ptr_arg_traits
  string_ini_list_argument_traits(const charT*)
  {
    return string_ini_lists_char_ptr_arg_traits();
  }

}; // namespace boost


#endif
