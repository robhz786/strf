#ifndef BOOST_STRINGIFY_DETAIL_UINT_TRAITS_HPP_INCLUDED
#define BOOST_STRINGIFY_DETAIL_UINT_TRAITS_HPP_INCLUDED

#include <cstdint>
#include <type_traits>

namespace boost{
namespace stringify{
namespace detail{

  template<int INT_SIZE>
  struct uint_size_traits
  {
  private:
    typedef uint_size_traits<INT_SIZE + 1> delegated;

  public:
    typedef  typename delegated::fast_type fast_type;

    static fast_type greatest_power_of_10_less_than(fast_type d) noexcept
    {
      return delegated::greatest_power_of_10_less_than(d);
    }
    static int number_of_digits(fast_type d) noexcept
    {
      return delegated::number_of_digits(d);
    }
  };

  template<>
  struct uint_size_traits<1>
  {
    typedef uint_fast8_t fast_type;

    static uint_fast8_t greatest_power_of_10_less_than(uint_fast8_t value) noexcept
    {
      return (value > 99 ? 100 :
              value > 9  ? 10  : 1);
    }
    static int number_of_digits(fast_type value) noexcept
    {
      return (value > 99 ? 3 :
              value > 9  ? 2  : 1);
    }
    static uint_fast8_t greatest_power_of_10_less_than_hex(uint_fast8_t value) noexcept
    {
      return value > 0x0f ? 2 : 1;
    }
  };

  template<>
  struct uint_size_traits<2>
  {
    typedef uint_fast16_t fast_type;

    static fast_type greatest_power_of_10_less_than(fast_type value) noexcept
    {
      fast_type result = 1;

      if (value > 9999)
        return 10000;
      if (value > 99) {
        value /= 100;
        result *= 100;
      }
      if(value > 9)
        result *= 10;
      return result;
    }

    static int number_of_digits(fast_type value) noexcept
    {
      int num_digits = 1;

      if (value > 9999) {
        return 5;
      }
      if( value > 99 ) {
        value /= 100;
        num_digits += 2 ;
      }
      if (value > 9) {
        value /= 10;
        ++num_digits;
      }
      return num_digits;
    }
  };

  template<>
  struct uint_size_traits<4>
  {
    typedef uint_fast32_t fast_type;

    static fast_type greatest_power_of_10_less_than(fast_type value) noexcept
    {
      fast_type result = 1l;

      if (value > 99999999l) {
        value /= 100000000l;
        result *= 100000000l;
        goto value_less_than_100;
      }
      //now value must be < 100000000
      if (value > 9999l) {
        value /= 10000l;
        result *= 10000l;
      }
      //now value must be < 10000
      if (value > 99l) {
        value /= 100l;
        result *= 100l;
      }
      value_less_than_100:
      //now value must be < 100
      if (value > 9l) {
        value /= 10l;
        result *= 10l;
      }

      return result;
    }

    static int number_of_digits(fast_type value) noexcept
    {
      int num_digits = 1;

      if (value > 99999999l) {
        value /= 100000000l;
        num_digits += 8;
        goto value_less_than_100;
      }
      if (value > 9999l) {
        value /= 10000l;
        num_digits += 4;
      }
      if( value > 99l ) {
        value /= 100l;
        num_digits += 2 ;
      }
      value_less_than_100:
      if (value > 9l) {
        value /= 10l;
        ++num_digits;
      }

      return num_digits;
    }

  };

  template<>
  struct uint_size_traits<8>
  {
    typedef uint_fast64_t fast_type;

    static fast_type greatest_power_of_10_less_than(fast_type value) noexcept
    {
      fast_type result = 1LL;

      if (value > 9999999999999999LL) {
        value /= 10000000000000000LL;
        result *= 10000000000000000LL;
        goto value_less_than_10000;
      }
      if (value > 99999999LL) {
        value /= 100000000LL;
        result *= 100000000LL;
      }
      value_less_than_10000:
      if (value > 9999LL) {
        value /= 10000LL;
        result *= 10000LL;
      }
      if (value > 99LL) {
        value /= 100LL;
        result *= 100LL;
      }
      if (value > 9LL) {
        result *= 10LL;
      }
      return result;
    }


    static int number_of_digits(fast_type value) noexcept
    {
      int num_digits = 1;
      
      if (value > 9999999999999999LL) {
        value /= 10000000000000000LL;
        num_digits += 16;
        goto value_less_than_10000;
      }
      if (value > 99999999LL) {
        value /= 100000000LL;
        num_digits += 8;
      }
      value_less_than_10000:
      if (value > 9999LL) {
        value /= 10000LL;
        num_digits += 4;
      }
      if(value > 99LL) {
        value /= 100LL;
        num_digits += 2;
      }
      if(value > 9LL)
        ++num_digits;

      return num_digits;
    }

  };


  template <typename uintT>
  struct uint_traits
  {
  private:
    typedef uint_size_traits<sizeof(uintT) > delegated;

  public:
    static_assert(std::is_unsigned<uintT>::value, "must be unsigned"); //todo: use concepts instead

    typedef typename delegated::fast_type fast_type;      

    static fast_type greatest_power_of_10_less_than(fast_type v) noexcept
    {
      return delegated::greatest_power_of_10_less_than(v);
    }

    static int number_of_digits(fast_type v) noexcept
    {
      return delegated::number_of_digits(v);
    }
  };


} //namespace boost
} //namespace stringify
} //namespace detail

      
#endif //#ifndef BOOST_STRINGIFY_DETAIL_INT_HPP_INCLUDED












