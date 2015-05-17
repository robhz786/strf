#ifndef BOOST_ROSE_INT_INT_WRITER_HPP_INCLUDED
#define BOOST_ROSE_INT_INT_WRITER_HPP_INCLUDED

#include <boost/rose/detail/characters_catalog.hpp>
#include <boost/rose/str_writer.hpp>
#include <type_traits>

namespace boost
{
namespace rose
{
  template <typename intT, typename charT, typename traits>
  struct int_writer: public str_writer<charT>
  {
    typedef typename std::make_unsigned<intT>::type  unsigned_intT;

    intT value;

    int_writer() noexcept:
      value(0)
    {
    }

    int_writer(intT _value) noexcept:
      value(_value)
    {
    }

    void set(intT _value) noexcept
    {
      value = (_value);
    }


    virtual std::size_t minimal_length() const noexcept
    {
      if(value < 0)
        return number_of_digits() + 1;
      return number_of_digits();
    }

    virtual charT* write_without_termination_char(charT* output) const noexcept
    {
      if(value < 0){
        traits::assign(*output, detail::the_sign_minus<charT>());
        output++;
      }
      output += number_of_digits();
      charT* end = output;
        

      unsigned_intT abs_value = (value >= 0 
                                 ? static_cast<unsigned_intT>(value)
                                 : static_cast<unsigned_intT>(-(value + 1)) + 1);
      do
      {
        traits::assign(*--output,
                       correspondig_character_of_digit(abs_value % 10));
      }
      while(abs_value /= 10);

      return end;
    }

    std::size_t number_of_digits() const noexcept
    {
      int num = 1;
      intT val = value;
      while(val /= 10)
        ++num;
      return num;
    }

    charT correspondig_character_of_digit(int digit) const noexcept
    {
      return detail::the_digit_zero<charT>() + digit;
    }
  };

  template <typename charT, typename traits>
  inline                                    
  int_writer<int, charT, traits> basic_argf(int i) noexcept
  {                                               
    return i;                                     
  }

  template <typename charT, typename traits>
  inline
  int_writer<long, charT, traits> basic_argf(long i) noexcept
  {
    return i;
  }

  template <typename charT, typename traits>
  inline
  int_writer<long long, charT, traits> basic_argf(long long i) noexcept
  {
    return i;
  }

  template <typename charT, typename traits>
  inline
  int_writer<unsigned int, charT, traits> basic_argf(unsigned int i) noexcept
  {
    return i;
  }

  template <typename charT, typename traits>
  inline
  int_writer<unsigned long, charT, traits> basic_argf(unsigned long i) noexcept
  {
    return i;
  }

  template <typename charT, typename traits>
  inline
  int_writer<unsigned long long, charT, traits> basic_argf(unsigned long long i) noexcept
  {
    return i;
  }

}//namespace rose
}//namespace boost


#endif
