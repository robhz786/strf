#ifndef BOOST_STRINGIFY_INPUT_BASE_HPP_INCLUDED
#define BOOST_STRINGIFY_INPUT_BASE_HPP_INCLUDED

#include <boost/assert.hpp>
#include <cstddef>

namespace boost
{
namespace stringify 
{
template <typename charT>
class simple_ostream;

template <typename charT, typename Formating>
class input_base
{
public:
      virtual ~input_base()
      {}

      /**
         return position after last written character
      */
      virtual charT* write_without_termination_char
          ( charT* out
          , const Formating& 
          ) const noexcept =0;

      charT* write
          ( charT* out
          , const Formating& fmt
          ) const noexcept
      {
          charT* end = write_without_termination_char(out, fmt);
          *end = charT();
          return end;
      }

      /**
         return the amount of character that needs to be allocated,
         not counting the termination character. The implementer is
         not required to calculate the exact value if this is 
         difficult. But it must be greater or equal. And should not
         be much greater.
      */
      virtual std::size_t minimal_length(const Formating&) const noexcept = 0;

      std::size_t minimal_size(const Formating& fmt) const noexcept
      {
          return 1 + minimal_length(fmt);
      }

      virtual void write
          ( simple_ostream<charT>& out
          , const Formating& fmt
          ) const = 0;
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
};

} //namespace stringify
} //namespace boost


#endif






