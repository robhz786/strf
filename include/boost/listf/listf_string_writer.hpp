#ifndef BOOST_LISTF_STRING_WRITER_HPP_INCLUDED
#define BOOST_LISTF_STRING_WRITER_HPP_INCLUDED

#include <string>
#include <algorithm>
#include <boost/listf/listf_writer_base.hpp>

namespace boost
{

  //--------------------------------------------------
  // std::string 
  struct listf_std_string_arg_traits
  {
    template <class charT, class traits>
    struct writer: public listf_writer_base<charT>
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
  listf_std_string_arg_traits 
  listf_argument_traits(const std::basic_string<charT, traits>&)
  {
    return listf_std_string_arg_traits();
  }

  //--------------------------------------------------
  // const charT* 

  struct listf_char_ptr_arg_traits
  {
    template<typename charT,typename traits>
    struct writer: listf_writer_base<charT>
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
  listf_char_ptr_arg_traits
  listf_argument_traits(const charT*)
  {
    return listf_char_ptr_arg_traits();
  }

}; // namespace boost


#endif
