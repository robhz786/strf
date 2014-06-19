#ifndef BOOST_STRING_IL_INT_WRITER_HPP_INCLUDED
#define BOOST_STRING_IL_INT_WRITER_HPP_INCLUDED

#include <boost/string_ini_list/detail/characters_catalog.hpp>
#include <type_traits>

namespace boost
{
  template <typename intT>
  struct listf_int_arg_traits
  {
    typedef typename std::make_unsigned<intT>::type  unsigned_intT;

    template <typename charT, typename traits>
    struct writer: public listf_writer_base<charT>
    {
      intT value;
      writer(intT _value):
        value(_value)
      {
      }

      virtual std::size_t minimal_length() const
      {
        if(value < 0)
          return number_of_digits() + 1;
        return number_of_digits();
      }

      virtual charT* write_without_termination_char(charT* output) const
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

      std::size_t number_of_digits() const
      {
        int num = 1;
        intT val = value;
        while(val /= 10)
          ++num;
        return num;
      }

      charT correspondig_character_of_digit(int digit) const
      {
        return detail::the_digit_zero<charT>() + digit;
      }

    };
  };


  inline
  listf_int_arg_traits<int>
  listf_argument_traits(int)
  {
    return listf_int_arg_traits<int>();
  }

  inline
  listf_int_arg_traits<unsigned int>
  listf_argument_traits(unsigned int)
  {
    return listf_int_arg_traits<unsigned int>();
  }

  inline
  listf_int_arg_traits<long>
  listf_argument_traits(long)
  {
    return listf_int_arg_traits<long>();
  }

  inline
  listf_int_arg_traits<unsigned long>
  listf_argument_traits(unsigned long)
  {
    return listf_int_arg_traits<unsigned long>();
  }

  inline
  listf_int_arg_traits<long long>
  listf_argument_traits(long long)
  {
    return listf_int_arg_traits<long long>();
  }

  inline
  listf_int_arg_traits<unsigned long long>
  listf_argument_traits(unsigned long long)
  {
    return listf_int_arg_traits<unsigned long long>();
  }

}//namespace boost


#endif
