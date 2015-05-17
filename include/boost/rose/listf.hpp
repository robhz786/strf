#ifndef BOOST_LISTF_HPP_INCLUDED
#define BOOST_LISTF_HPP_INCLUDED

#include <initializer_list>
#include <boost/rose/str_writer.hpp>
#include <boost/rose/argf.hpp>

namespace boost
{
namespace rose
{
  template <typename charT, typename traits=std::char_traits<charT> >
  class basic_listf: public str_writer<charT>
  {

    struct str_writer_ref
    {
      str_writer_ref(const str_writer<charT>& w): writer(w) {}

      template <class T>
      using str_writer_of = decltype(basic_argf<charT, traits>(*(const T*)(0)));

      template <typename T>
      str_writer_ref(
        const T& value,
        str_writer_of<T> && wt = str_writer_of<T>()):
        writer(wt)
      {
        wt.set(value);
      }

      const str_writer<charT>& writer;
    };


    typedef 
      std::initializer_list<str_writer_ref >
      initializer_list_type;

    const initializer_list_type inilist;

  public:

    basic_listf(const initializer_list_type& _inilist):
      inilist(_inilist)
    {
    }

    virtual std::size_t minimal_length() const
    {
      std::size_t sum=0;
      for(auto it = inilist.begin(); it != inilist.end(); ++it)
      {
        sum += it->writer.minimal_length();
      }
      return sum;
    }


    virtual charT* write_without_termination_char(charT* output) const
    {
      for(auto it = inilist.begin(); it != inilist.end(); ++it)
      {
        output = it->writer.write_without_termination_char(output);
      }
      return output;
    }
  };

  typedef basic_listf<char>      listf;
  typedef basic_listf<wchar_t>   wlistf;
  typedef basic_listf<char16_t>  listf16;
  typedef basic_listf<char32_t>  listf32;
} // namespace rose
} // namespace boost


// BOOST_STRING_LITERAL: I dont know where to put this macro so I left it here.
// example of how it can be used:
// std::basic_string<someCharT> someString = BOOST_STRING_LITERAL(someCharT, "foobar");

#define BOOST_STRING_LITERAL(charT, str)                                \
  (::boost::is_same<charT, char32_t>::value ? reinterpret_cast<const charT*>(U ## str) : \
   ::boost::is_same<charT, char16_t>::value ? reinterpret_cast<const charT*>(u ## str) : \
   ::boost::is_same<charT, wchar_t>::value  ? reinterpret_cast<const charT*>(L ## str) : \
   /*else: */                                 reinterpret_cast<const charT*>(str)) 


#endif
