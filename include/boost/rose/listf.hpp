#ifndef BOOST_ROSE_LISTF_HPP_INCLUDED
#define BOOST_ROSE_LISTF_HPP_INCLUDED

#include <initializer_list>
#include <boost/rose/str_writer.hpp>
#include <boost/rose/argf.hpp>

namespace boost
{
namespace rose
{
  template <typename charT>
  class basic_listf: public str_writer<charT>
  {

    struct str_writer_ref
    {
      str_writer_ref(const str_writer<charT>& w) noexcept: writer(w) {}

      template <class T>
      using str_writer_of = decltype(basic_argf<charT>(*(const T*)(0)));

      template <typename T>
      str_writer_ref(
        const T& value,
        str_writer_of<T> && wt = str_writer_of<T>())
      noexcept :
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

    basic_listf(const initializer_list_type& _inilist) noexcept:
      inilist(_inilist)
    {
    }

    virtual std::size_t minimal_length() const noexcept
    {
      std::size_t sum=0;
      for(auto it = inilist.begin(); it != inilist.end(); ++it)
      {
        sum += it->writer.minimal_length();
      }
      return sum;
    }


    virtual charT* write_without_termination_char(charT* out) const noexcept
    {
      for(auto it = inilist.begin(); it != inilist.end(); ++it)
      {
        out = it->writer.write_without_termination_char(out);
      }
      return out;
    }

    virtual void write(simple_ostream<charT>& out) const
    {
      for(auto it = inilist.begin(); it != inilist.end() && out.good(); ++it)
      {
        it->writer.write(out);
      }
    }

  };

  typedef basic_listf<char>      listf;
  typedef basic_listf<wchar_t>   wlistf;
  typedef basic_listf<char16_t>  listf16;
  typedef basic_listf<char32_t>  listf32;
} // namespace rose
} // namespace boost


#endif
