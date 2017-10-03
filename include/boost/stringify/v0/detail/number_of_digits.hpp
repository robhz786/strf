#ifndef BOOST_STRINGIFY_V0_DETAIL_NUMBER_OF_DIGITS_HPP
#define BOOST_STRINGIFY_V0_DETAIL_NUMBER_OF_DIGITS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <int Base, int IntSize>
struct digits_counter;

template<>
struct digits_counter<8, 2>
{
    static unsigned count_digits(uint_fast16_t value)
    {
        unsigned num_digits = 1;
        if(value > 07777)
        {
            value >>= 12;
            num_digits += 4;
        }
        if(value > 077)
        {
            value >>= 6;
            num_digits += 2;
        }
        if(value > 07)
        {
            ++num_digits;
        }
        return num_digits;
    }
};

template<>
struct digits_counter<8, 4>
{
    static unsigned count_digits(uint_fast32_t value)
    {
        unsigned num_digits = 1;
        if(value > 077777777l)
        {
            value >>= 24;
            num_digits += 8;
        }
        if(value > 07777l)
        {
            value >>= 12;
            num_digits += 4;
        }
        if(value > 077l)
        {
            value >>= 6;
            num_digits += 2;
        }
        if(value > 07l)
        {
            ++num_digits;
        }
        return num_digits;
    }
};

template<>
struct digits_counter<8, 8>
{
    static unsigned count_digits(uint_fast64_t value)
    {
        unsigned num_digits = 1;
        if(value > 07777777777777777LL)
        {
            value >>= 48;
            num_digits += 16;
        }
        if(value > 077777777LL)
        {
            value >>= 24;
            num_digits += 8;
        }
        if(value > 07777LL)
        {
            value >>= 12;
            num_digits += 4;
        }
        if(value > 077LL)
        {
            value >>= 6;
            num_digits += 2;
        }
        if(value > 07LL)
        {
            ++num_digits;
        }
        return num_digits;
    }
};


template<>
struct digits_counter<10, 2>
{
    static unsigned count_digits(uint_fast16_t value)
    {
        unsigned num_digits = 1;
        if (value > 9999) {
            return 5;
        }
        if( value > 99 ) {
            value /= 100;
            num_digits += 2 ;
        }
        if (value > 9) {
            ++num_digits;
        }
        return num_digits;
    }
};


template<>
struct digits_counter<10, 4>
{
    static unsigned count_digits(uint_fast32_t value)
    {
        unsigned num_digits = 1l;
 
        if (value > 99999999l)
        {
            value /= 100000000l;
            num_digits += 8;
            goto value_less_than_100;
        }
        if (value > 9999l)
        {
            value /= 10000l;
            num_digits += 4;
        }
        if( value > 99l )
        {
            value /= 100l;
            num_digits += 2 ;
        }
        value_less_than_100:
        if (value > 9l)
        {
             ++num_digits;
        }
     
        return num_digits;
    }
};


template<>
struct digits_counter<10, 8>
{
    static unsigned count_digits(uint_fast64_t value)
    {
        unsigned num_digits = 1LL;
              
        if (value > 9999999999999999LL)
        {
            value /= 10000000000000000LL;
            num_digits += 16;
            //  goto value_less_than_10000;
        }
        if (value > 99999999LL)
        {
            value /= 100000000LL;
            num_digits += 8;
        }
        //value_less_than_10000:
        if (value > 9999LL)
        {
            value /= 10000LL;
            num_digits += 4;
        }
        if(value > 99LL)
        {
            value /= 100LL;
            num_digits += 2;
        }
        if(value > 9LL)
        {
            ++num_digits;
        }
        return num_digits;
    }
};


template<>
struct digits_counter<16, 2>
{
    static unsigned count_digits(uint_fast16_t value)
    {
        unsigned num_digits = 1;
        if( value > 0xff ) {
            value >>= 8;
            num_digits += 2 ;
        }
        if (value > 0xf) {
            ++num_digits;
        }
        return num_digits;
    }
};


template<>
struct digits_counter<16, 4>
{
    static unsigned count_digits(uint_fast32_t value)
    {
        unsigned num_digits = 1;
        if( value > 0xffffl ) {
            value >>= 16;
            num_digits += 4 ;
        }
        if( value > 0xffl ) {
            value >>= 8;
            num_digits += 2 ;
        }
        if (value > 0xfl) {
            ++num_digits;
        }
        return num_digits;
    }
};


template<>
struct digits_counter<16, 8>
{
    static unsigned count_digits(uint_fast64_t value)
    {
        unsigned num_digits = 1;
        if( value > 0xffffffffLL ) {
            value >>= 32;
            num_digits += 8 ;
        }
        if( value > 0xffffLL ) {
            value >>= 16;
            num_digits += 4 ;
        }
        if( value > 0xffLL ) {
            value >>= 8;
            num_digits += 2 ;
        }
        if (value > 0xfLL) {
            ++num_digits;
        }
        return num_digits;
    }
};


template <unsigned Base, typename intT>
unsigned number_of_digits(intT value)
{
    static_assert(std::is_unsigned<intT>::value, "");
    return boost::stringify::v0::detail::digits_counter<Base, sizeof(intT)>
        ::count_digits(value);
}


} // namespace detail
BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_NUMBER_OF_DIGITS_HPP

