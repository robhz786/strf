#ifndef BOOST_ROSE_INT_INT_WRITER_HPP_INCLUDED
#define BOOST_ROSE_INT_INT_WRITER_HPP_INCLUDED

#include <boost/rose/detail/characters_catalog.hpp>
#include <boost/rose/str_writer.hpp>
#include <boost/rose/detail/uint_traits.hpp>

namespace boost
{
namespace rose
{
  template <typename intT, typename charT>
  struct int_writer: public str_writer<charT>
  {
  private:
    typedef typename std::make_unsigned<intT>::type  unsigned_intT;
    typedef detail::uint_traits<unsigned_intT> uint_traits;

  public:

    int_writer() noexcept:
      value(0),
      abs_value(0)
    {
    }

    int_writer(intT _value) noexcept
    {
      set(_value);
    }

    void set(intT _value) noexcept
    {
      value = (_value);
      abs_value = (value > 0
                   ? static_cast<unsigned_intT>(value)
                   : static_cast<unsigned_intT>(-(value+1)) +1 );
    }


    virtual std::size_t minimal_length() const noexcept
    {
      return (uint_traits::number_of_digits(static_cast<unsigned_intT>(value))
              + (value < 0 ? 1 : 0));
    }

    virtual charT* write_without_termination_char(charT* out) const noexcept
    {
      if (value < 0)
        *out++ = detail::the_sign_minus<charT>();

      out += uint_traits::number_of_digits(abs_value);
      unsigned_intT it_value = abs_value;
      charT* end = out;
      do
      {
        *--out = correspondig_character_of_digit(it_value % 10);
      }
      while(it_value /= 10);

      return end;
    }

    virtual void write(simple_ostream<charT>& out) const
    {
      if (value < 0)
        out.put(detail::the_sign_minus<charT>());

      unsigned_intT div = uint_traits::greatest_power_of_10_less_than(abs_value);
      do
      {
        out.put(correspondig_character_of_digit((abs_value / div) % 10));
      }
      while(div /= 10);
    }


private:
    intT value;
    unsigned_intT abs_value;

    charT correspondig_character_of_digit(unsigned int digit) const noexcept
    {
      return detail::the_digit_zero<charT>() + digit;
    }
  };

  template <typename charT>
  inline                                    
  int_writer<int, charT> basic_argf(int i) noexcept
  {                                               
    return i;                                     
  }

  template <typename charT>
  inline
  int_writer<long, charT> basic_argf(long i) noexcept
  {
    return i;
  }

  template <typename charT>
  inline
  int_writer<long long, charT> basic_argf(long long i) noexcept
  {
    return i;
  }

  template <typename charT>
  inline
  int_writer<unsigned int, charT> basic_argf(unsigned int i) noexcept
  {
    return i;
  }

  template <typename charT>
  inline
  int_writer<unsigned long, charT> basic_argf(unsigned long i) noexcept
  {
    return i;
  }

  template <typename charT>
  inline
  int_writer<unsigned long long, charT>
  basic_argf(unsigned long long i) noexcept
  {
    return i;
  }

}//namespace rose
}//namespace boost


#endif
