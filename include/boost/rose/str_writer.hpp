#ifndef BOOST_ROSE_STR_WRITER_HPP_INCLUDED
#define BOOST_ROSE_STR_WRITER_HPP_INCLUDED

#include <boost/assert.hpp>
#include <cstddef>

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

} //namespace rose
} //namespace boost


#endif






