#ifndef BOOST_ROSE_STR_WRITER_HPP_INCLUDED
#define BOOST_ROSE_STR_WRITER_HPP_INCLUDED

#include <boost/assert.hpp>
#include <cstddef>
#include <ostream>

namespace boost
{
namespace rose 
{
  template <typename charT>
  class simple_ostream;

  template <typename charT>
  class str_writer
  {
  public:
    virtual ~str_writer()
    {}

    /**
       return position after last written character
     */
    virtual charT* write_without_termination_char(charT* out) const noexcept =0;

    charT* write(charT* out) const noexcept
    {
      charT* end = write_without_termination_char(out);
      *end = charT();
      return ++end;
    }

    /**
       return the amount of character that needs to be allocated,
       not counting the termination character. The implementer is
       not required to calculate the exact value if this is 
       difficult. But it must be greater or equal. And should not
       be much greater.
     */
    virtual std::size_t minimal_length() const noexcept = 0;

    std::size_t minimal_size() const noexcept
    {
      return 1 + minimal_length();
    }

    virtual void write(simple_ostream<charT>& out) const = 0;
  };


  template <typename charT>
  class simple_ostream
  {
  public:
    virtual ~simple_ostream()
    {}

    virtual bool good() = 0;

    virtual void put(charT) = 0;

    virtual void write(const charT*, std::size_t) = 0;

    simple_ostream& operator<<(const str_writer<charT>& input)
    {
      input.write(*this);
      return *this;
    }
  };

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

}//namespace detail


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


  template<typename charT>
  charT* operator<<(charT* out, const str_writer<charT>& lsf) noexcept
  {
    return lsf.write(out);
  }


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

} //namespace rose
} //namespace boost


#endif






