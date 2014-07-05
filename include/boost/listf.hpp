#ifndef BOOST_LISTF_HPP_INCLUDED
#define BOOST_LISTF_HPP_INCLUDED

#include <initializer_list>
#include <boost/listf/detail/basic_listf_element.hpp>

namespace boost 
{
  template <typename charT, typename traits=std::char_traits<charT> >
  class basic_listf
  {
    typedef 
      std::initializer_list<detail::basic_listf_element<charT, traits> >
      initializer_list_type;

    const initializer_list_type inilist;

  public:

    basic_listf(const initializer_list_type& _inilist):
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
  };

  typedef basic_listf<char>      listf;
  typedef basic_listf<wchar_t>   wlistf;
  typedef basic_listf<char16_t>  listf16;
  typedef basic_listf<char32_t>  listf32;


  template<typename charT, typename traits>
  charT* operator<<(charT* output, const basic_listf<charT, traits>& lsf)
  {
    return lsf.write(output);
  }


  template<typename charT, typename traits, typename Allocator>
  std::basic_string<charT, traits, Allocator>& 
  operator<<(std::basic_string<charT, traits, Allocator>& str,
             const basic_listf<charT, traits>& lsf)
  {
    std::size_t initial_length = str.length();
    str.append(lsf.minimal_length(), charT());
    charT* begin_append = & str[initial_length];
    charT* end_append   = lsf.write_without_termination_char(begin_append);
    str.resize(initial_length + (end_append - begin_append));
    return str;
  }

};


// BOOST_STRING_LITERAL: I dont know where to put this macro so I left it here.
// example of how it can be used:
// std::basic_string<someCharT> someString = BOOST_STRING_LITERAL(someCharT, "foobar");

#define BOOST_STRING_LITERAL(charT, str)                                \
  (::boost::is_same<charT, char32_t>::value ? reinterpret_cast<const charT*>(U ## str) : \
   ::boost::is_same<charT, char16_t>::value ? reinterpret_cast<const charT*>(u ## str) : \
   ::boost::is_same<charT, wchar_t>::value  ? reinterpret_cast<const charT*>(L ## str) : \
   /*else: */                                 reinterpret_cast<const charT*>(str)) 


#endif
