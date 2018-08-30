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


template
    < typename IntT
    , typename unsigned_IntT = typename std::make_unsigned<IntT>::type
    >
typename std::enable_if<std::is_signed<IntT>::value, unsigned_IntT>::type
unsigned_abs(IntT value)
{
    return ( value > 0
           ? static_cast<unsigned_IntT>(value)
           : 1 + static_cast<unsigned_IntT>(-(value + 1)));
}

template<typename IntT>
typename std::enable_if<std::is_unsigned<IntT>::value, IntT>::type
unsigned_abs(IntT value)
{
    return value;
}

template <unsigned Base, typename intT>
unsigned count_digits(intT value)
{
    return stringify::v0::detail::digits_counter<Base, sizeof(intT)>
        ::count_digits(stringify::v0::detail::unsigned_abs(value));
}

template <typename intT>
unsigned count_digits(intT value, int base)
{
    if (base == 10) return count_digits<10>(value);
    if (base == 16) return count_digits<16>(value);
    BOOST_ASSERT(base == 8);
    return count_digits<8>(value);
}

inline char to_xdigit(unsigned digit, bool uppercase)
{
    if (digit < 10)
    {
        return '0' + digit;
    }
    const char char_a = uppercase ? 'A' : 'a';
    return  char_a + digit - 10;
}

inline char to_digit(unsigned digit)
{
    return '0' + digit;
}

inline const char* chars_00_to_99()
{
    static const char array[] =
        "00010203040506070809"
        "10111213141516171819"
        "20212223242526272829"
        "30313233343536373839"
        "40414243444546474849"
        "50515253545556575859"
        "60616263646566676869"
        "70717273747576777879"
        "80818283848586878889"
        "90919293949596979899";

    return array;
}

template <typename IntT, typename CharT>
CharT* write_int_dec_txtdigits_backwards(IntT value, CharT* it) noexcept
{
    auto uvalue = stringify::v0::detail::unsigned_abs(value);
    const char* arr = stringify::v0::detail::chars_00_to_99();
    while(uvalue > 99)
    {
        auto index = (uvalue % 100) << 1;
        it[-2] = arr[index];
        it[-1] = arr[index + 1];
        it -= 2;
        uvalue /= 100;
    }
    if (uvalue < 10)
    {
        *--it = to_digit(uvalue);
        return it;
    }
    else
    {
        auto index = uvalue << 1;
        it[-2] = arr[index];
        it[-1] = arr[index + 1];
        return it - 2;
    }
}

template <typename IntT, typename CharT>
CharT* write_int_hex_txtdigits_backwards(IntT value, bool uppercase, CharT* it) noexcept
{
    do
    {
        *--it = stringify::v0::detail::to_xdigit(value & 0xF, uppercase);
        value >>= 4;
    }
    while(value != 0);
    return it;
}

template <typename IntT, typename CharT>
CharT* write_int_oct_txtdigits_backwards(IntT value, CharT* it) noexcept
{
    do
    {
        *--it = stringify::v0::detail::to_digit(value & 7);
        value >>= 3;
    }
    while(value != 0);
    return it;
}

template <typename IntT, typename CharT>
CharT* write_int_txtdigits_backwards(IntT value, int base, bool uppercase, CharT* it) noexcept
{
    if (base == 10)
    {
        return write_int_dec_txtdigits_backwards(value, it);
    }
    if (base == 16)
    {
        return write_int_hex_txtdigits_backwards(value, uppercase, it);
    }
    BOOST_ASSERT(base == 8);
    return write_int_oct_txtdigits_backwards(value, it);
}

template <typename IntT, typename CharT>
class intdigits_writer: public stringify::v0::piecemeal_writer<CharT>
{
public:

    intdigits_writer
        ( const char* dig_it
        , const unsigned char* grp
        , const unsigned char* grp_it
        , const stringify::v0::encoder<CharT>& encoder
        , char32_t sepchar
        , unsigned sepchar_size )
        : m_dig_it{dig_it}
        , m_grp{grp}
        , m_grp_it{grp_it}
        , m_encoder{encoder}
        , m_sepchar{sepchar}
        , m_sepchar_size{sepchar_size}
    {
    }

    CharT* write(CharT* begin, CharT* end) override;

private:

    CharT* write_sep(CharT* begin, CharT* end)
    {
        auto it = m_encoder.encode(m_sepchar, begin, end, true);
        BOOST_ASSERT(it != nullptr);
        BOOST_ASSERT(it != end + 1);
        return it;
    }

    CharT* write_grp(unsigned grp_size, CharT* it);

    const char* m_dig_it;
    const unsigned char* const m_grp;
    const unsigned char* m_grp_it;
    const stringify::v0::encoder<CharT>& m_encoder;
    char32_t m_sepchar;
    unsigned m_sepchar_size;
    bool m_first_grp = true;
};


template <typename IntT, typename CharT>
CharT* intdigits_writer<IntT, CharT>::write(CharT* begin, CharT* end)
{
    auto it = begin;
    if(m_first_grp)
    {
        auto grp_size = *m_grp_it;
        if (begin + grp_size > end)
        {
            return begin;
        }
        it = write_grp(grp_size, it);
        -- m_grp_it;
        m_first_grp = false;
    }
    BOOST_ASSERT(m_grp_it >= m_grp);
    do
    {
        auto grp_size = *m_grp_it;
        if (it + grp_size + m_sepchar_size > end)
        {
            return it;
        }
        it = write_sep(it, end);
        it = write_grp(grp_size, it);
        -- m_grp_it;
    }
    while(m_grp_it >= m_grp);

    this->report_success();
    return it;
}

template <typename IntT, typename CharT>
CharT* intdigits_writer<IntT, CharT>::write_grp(unsigned grp_size, CharT* it)
{
    BOOST_ASSERT(grp_size != 0);
    do
    {
        *it++ = *m_dig_it++;
    }
    while(--grp_size > 0);
    return it;
}

// template <typename IntT, typename CharT>
// CharT* intdigits_writer<IntT, CharT>::write_first_sep(CharT* begin, CharT* end)
// {
//     auto r = this->encode_char( m_encoder, m_err_sig, m_sepchar
//                               , begin, end, m_allow_surr );
//     BOOST_ASSERT(r.it != end + 1);
//     if (r.it == nullptr)
//     {
//         return nullptr;
//     }
//     m_sepchar = r.ch;
//     m_sepchar_size = r.it - begin;
//     return r.it;
// }

// template <typename IntT, typename CharT>
// CharT* intdigits_writer<IntT, CharT>::write_sep(CharT* begin, CharT* end)
// {
//     if(m_sepchar_size > 0)
//     {
//         auto r = this->encode_char( m_encoder, m_err_sig, m_sepchar
//                                   , begin, end, m_allow_surr );
//         BOOST_ASSERT(r.it != nullptr);
//         BOOST_ASSERT(r.it <= end);
//         return r.it;
//     }
//     return begin;
// }


} // namespace detail
BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_NUMBER_OF_DIGITS_HPP

