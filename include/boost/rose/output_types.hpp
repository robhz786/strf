#ifndef BOOST_ROSE_OUTPUT_TYPES_HPP_INCLUDED
#define BOOST_ROSE_OUTPUT_TYPES_HPP_INCLUDED

#include <boost/rose/str_writer.hpp>
#include <string>
#include <ostream>


namespace boost
{
namespace rose 
{
  //
  // charT*
  //

  template<typename charT>
  charT* operator<<(charT* out, const str_writer<charT>& lsf) noexcept
  {
    return lsf.write(out);
  }

  //
  // std::basic_string
  //

  template<typename charT, typename traits, typename Allocator>
  std::basic_string<charT, traits, Allocator>& 
  operator<<(std::basic_string<charT, traits, Allocator>& str,
             const str_writer<charT>& writer)
  {
    std::size_t initial_length = str.length();
    str.append(writer.minimal_length(), charT());
    charT* append_begin = & str[initial_length];

    //write
    charT* append_end = writer.write_without_termination_char(append_begin);

    //apply char_traits if necessary
    std::size_t append_length = append_end - append_begin;
    BOOST_ASSERT(append_length <= writer.minimal_length());
    if( ! std::is_same<traits, std::char_traits<charT> >::value)
      traits::move(append_begin, append_begin, append_length);

    //set correct size ( current size might be greater )
    str.resize(initial_length + append_length);

    return str;
  }

  //
  // std::ostream
  //

  namespace detail
  {
    template <typename charT, typename traits>
    class std_ostream_adapter: public simple_ostream<charT>
    {
    public:
      std_ostream_adapter(std::basic_ostream<charT, traits>& _out):
        out(_out)
      {
      }

      virtual bool good() noexcept
      {
        return out.good();
      }

      virtual void put(charT c) noexcept
      {
        out.put(c);
      }

      virtual void write(const charT* str, std::size_t len) noexcept
      {
        out.write(str, len);
      }

    private:
      std::basic_ostream<charT, traits>& out;
    };
  } //namespace detail

  template<typename charT, typename traits>
  std::basic_ostream<charT, traits>& operator << (
    std::basic_ostream<charT, traits>& out,
    const str_writer<charT>& input
  )
  {
    detail::std_ostream_adapter<charT, traits> adapted_out(out);
    adapted_out << input;
    return out;
  }

} //namespace rose
} //namespace boost

#endif
