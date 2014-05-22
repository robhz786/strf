#ifndef BOOST_STRING_INI_LIST_HPP_INCLUDED
#define BOOST_STRING_INI_LIST_HPP_INCLUDED

#include <initializer_list>
#include <boost/string_ini_list/detail/basic_string_il_element.hpp>

namespace boost 
{

  // using template <typename charT, typename traits>
  // basic_string_il = std::initializer_list<detail::basic_string_il_element<charT, traits> >;
  // typedef std::initializer_list<detail::basic_string_il_element<char> >      string_il;
  // typedef std::initializer_list<detail::basic_string_il_element<wchar_t> >   wstring_il;
  // typedef std::initializer_list<detail::basic_string_il_element<char16_t> >  string16_il;
  // typedef std::initializer_list<detail::basic_string_il_element<char32_t> >  string32_il;


  template <typename charT, typename traits=std::char_traits<charT> >
  class basic_string_il
  {
    typedef 
      std::initializer_list<detail::basic_string_il_element<charT, traits> >
      initializer_list_type;

    const initializer_list_type inilist;

  public:

    basic_string_il(const initializer_list_type& _inilist):
      inilist(_inilist)
    {
    }

    std::size_t minimal_length() const
    {
      std::size_t sum=0;
      for(auto it = inilist.begin(); it != inilist.end(); ++it)
      {
        sum += it->minimal_length();
      }
      return sum;
    }


    charT* write_without_termination_char(charT* output) const
    {
      for(auto it = inilist.begin(); it != inilist.end(); ++it)
      {
        output = it->write_without_termination_char(output);
      }
      return output;
    }

    charT* write(charT* output) const
    {
      output = write_without_termination_char(output);
      *output = charT();
      return output;
    }

    template<class Allocator>
    void append_to(std::basic_string<charT, traits, Allocator>& str) const
    {
      std::size_t initial_length = str.length();
      str.append(minimal_length(), charT());
      charT* begin_append = & str[initial_length];
      charT* end_append   = write_without_termination_char(begin_append);
      str.resize(initial_length + (end_append - begin_append));
    }

  };

  typedef basic_string_il<char>      string_il;
  typedef basic_string_il<wchar_t>   wstring_il;
  typedef basic_string_il<char16_t>  string16_il;
  typedef basic_string_il<char32_t>  string32_il;


  template<typename charT, 
           typename traits=std::char_traits<char> >
  charT*
  write(charT* output, 
        const basic_string_il<charT, traits>& inilist)
  {
    return inilist.write(output);
  }



  template<typename charT,
           typename traits,
           typename Allocator>
  std::basic_string<charT, traits, Allocator>& 
  operator+=(std::basic_string<charT, traits, Allocator>& str,
             const basic_string_il<charT, traits>& inilist)
  {
    std::size_t initial_length = str.length();
    str.append(inilist.minimal_length(), charT());
    charT* begin_append = & str[initial_length];
    charT* end_append   = inilist.write_without_termination_char(begin_append);
    str.resize(initial_length + (end_append - begin_append));
    return str;
  }

};

#endif
