#ifndef BOOST_ROSE_STR_WRITER_HPP_INCLUDED
#define BOOST_ROSE_STR_WRITER_HPP_INCLUDED

#include <cstddef>

namespace boost
{
namespace rose 
{

  template <typename charT>
  class str_writer
  {
  public:
    virtual ~str_writer()
    {
    }

    /**
       return position after last written character
     */
    virtual charT* write_without_termination_char(charT* output) const =0;

    charT* write(charT* output) const
    {
      charT* end = write_without_termination_char(output);
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
    virtual std::size_t minimal_length() const = 0;

    std::size_t minimal_size() const
    {
      return 1 + minimal_length();
    }
  };


  template<typename charT>
  charT* operator<<(charT* output, const str_writer<charT>& lsf)
  {
    return lsf.write(output);
  }


  template<typename charT, typename traits, typename Allocator>
  std::basic_string<charT, traits, Allocator>& 
  operator<<(std::basic_string<charT, traits, Allocator>& str,
             const str_writer<charT>& writer)
  {
    std::size_t initial_length = str.length();
    str.append(writer.minimal_length(), charT());
    charT* begin_append = & str[initial_length];
    charT* end_append   = writer.write_without_termination_char(begin_append);
    str.resize(initial_length + (end_append - begin_append));
    return str;
  }

} //namespace rose
} //namespace boost


#endif
