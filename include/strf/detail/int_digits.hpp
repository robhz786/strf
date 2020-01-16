#ifndef STRF_DETAIL_NUMBER_OF_DIGITS_HPP
#define STRF_DETAIL_NUMBER_OF_DIGITS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/lettercase.hpp>
#ifdef __cpp_lib_bitops
#include <bit>
#endif

namespace strf {

namespace detail {

inline STRF_HD unsigned long long pow10(unsigned n) noexcept
{
    static const unsigned long long p10[] =
        { 1, 10, 100, 1000, 10000, 100000, 1000000
        , 10000000ull
        , 100000000ull
        , 1000000000ull
        , 10000000000ull
        , 100000000000ull
        , 1000000000000ull
        , 10000000000000ull
        , 100000000000000ull
        , 1000000000000000ull
        , 10000000000000000ull
        , 100000000000000000ull
        , 1000000000000000000ull
        , 10000000000000000000ull };
    STRF_ASSERT(n <= 19);

    return p10[n];
};

template <typename IntT, unsigned Base> struct max_num_digits_impl;
template <typename IntT> struct max_num_digits_impl<IntT, 10>
{
    static constexpr unsigned value = (240824 * sizeof(IntT) + 99999) / 100000;
};
template <typename IntT> struct max_num_digits_impl<IntT, 16>
{
    static constexpr unsigned value = sizeof(IntT) * 2;
};
template <typename IntT> struct max_num_digits_impl<IntT, 8>
{
    static constexpr unsigned value = (sizeof(IntT) * 8 + 2) / 3;
};
template <typename IntT> struct max_num_digits_impl<IntT, 2>
{
    static constexpr unsigned value = sizeof(IntT) * 8;
};

template<class IntT, unsigned Base>
constexpr unsigned max_num_digits =
    strf::detail::max_num_digits_impl<IntT, Base>::value;


template
    < typename IntT
    , typename unsigned_IntT = typename std::make_unsigned<IntT>::type >
inline STRF_HD typename std::enable_if<std::is_signed<IntT>::value, unsigned_IntT>::type
unsigned_abs(IntT value) noexcept
{
    return ( value > 0
           ? static_cast<unsigned_IntT>(value)
           : 1 + static_cast<unsigned_IntT>(-(value + 1)));
}

template<typename IntT>
inline STRF_HD typename std::enable_if<std::is_unsigned<IntT>::value, IntT>::type
unsigned_abs(IntT value) noexcept
{
    return value;
}

template <int Base, int IntSize>
struct digits_counter;

#if defined(__cpp_lib_bitops)

template<>
struct digits_counter<2, 4>
{
    static inline STRF_HD unsigned count_digits(uint_fast32_t value) noexcept
    {
        return sizeof(value) * 8 - std::countl_zero(value);
    }
};
template<>
struct digits_counter<2, 8>
{
    static inline STRF_HD unsigned count_digits(uint_fast64_t value) noexcept
    {
        return sizeof(value) * 8 - std::countl_zero(value);
    }
};
template<>
struct digits_counter<8, 4>
{
    static STRF_HD unsigned count_digits(uint_fast32_t value) noexcept
    {
        int bin_digits = sizeof(value) * 8 - std::countl_zero(value);
        return (bin_digits + 2) / 3;
    }
};
template<>
struct digits_counter<8, 8>
{
    static STRF_HD unsigned count_digits(uint_fast64_t value) noexcept
    {
        int bin_digits = sizeof(value) * 8 - std::countl_zero(value);
        return (bin_digits + 2) / 3;
    }
};
template<>
struct digits_counter<16, 4>
{
    static STRF_HD unsigned count_digits(uint_fast32_t value) noexcept
    {
        int bin_digits = sizeof(value) * 8 - std::countl_zero(value);
        return (bin_digits + 3) >> 2;
    }
};
template<>
struct digits_counter<16, 8>
{
    static STRF_HD unsigned count_digits(uint_fast64_t value) noexcept
    {
        int bin_digits = sizeof(value) * 8 - std::countl_zero(value);
        return (bin_digits + 3) >> 2;
    }
};

#else // defined(__cpp_lib_bitops)

template<>
struct digits_counter<2, 4>
{
    static STRF_HD unsigned count_digits(uint_fast32_t value) noexcept
    {
        unsigned num_digits = 1;
        if( value > 0xfffful ) {
            value >>= 16;
            num_digits += 16 ;
        }
        if( value > 0xfful ) {
            value >>= 8;
            num_digits += 8 ;
        }
        if (value > 0xful) {
            value >>= 4;
            num_digits += 4 ;
        }
        if (value > 3) {
            value >>= 2;
            num_digits += 2 ;
        }
        if (value > 1) {
            return num_digits + 1;
        }
        return num_digits;
    }
};

template<>
struct digits_counter<2, 8>
{
    static STRF_HD unsigned count_digits(uint_fast64_t value) noexcept
    {

        unsigned num_digits = 1;
        if( value > 0xffffffffull ) {
            value >>= 32;
            num_digits += 32 ;
        }
        if( value > 0xfffful ) {
            value >>= 16;
            num_digits += 16 ;
        }
        if( value > 0xfful ) {
            value >>= 8;
            num_digits += 8 ;
        }
        if (value > 0xful) {
            value >>= 4;
            num_digits += 4 ;
        }
        if (value > 3) {
            value >>= 2;
            num_digits += 2 ;
        }
        if (value > 1) {
            return num_digits + 1;
        }
        return num_digits;
    }
};

template<>
struct digits_counter<8, 4>
{
    static STRF_HD unsigned count_digits(uint_fast32_t value) noexcept
    {
        unsigned num_digits = 1;
        if(value > 077777777ul) {
            value >>= 24;
            num_digits += 8;
        }
        if(value > 07777ul) {
            value >>= 12;
            num_digits += 4;
        }
        if(value > 077ul) {
            value >>= 6;
            num_digits += 2;
        }
        if(value > 07ul) {
            ++num_digits;
        }
        return num_digits;
    }
};

template<>
struct digits_counter<8, 8>
{
    static STRF_HD unsigned count_digits(uint_fast64_t value) noexcept
    {
        unsigned num_digits = 1;
        if(value > 07777777777777777uLL) {
            value >>= 48;
            num_digits += 16;
        }
        if(value > 077777777uLL) {
            value >>= 24;
            num_digits += 8;
        }
        if(value > 07777uLL) {
            value >>= 12;
            num_digits += 4;
        }
        if(value > 077uLL) {
            value >>= 6;
            num_digits += 2;
        }
        if(value > 07uLL) {
            ++num_digits;
        }
        return num_digits;
    }
};

template<>
struct digits_counter<16, 4>
{
    static STRF_HD unsigned count_digits(uint_fast32_t value) noexcept
    {
        unsigned num_digits = 1;
        if( value > 0xfffful ) {
            value >>= 16;
            num_digits += 4 ;
        }
        if( value > 0xfful ) {
            value >>= 8;
            num_digits += 2 ;
        }
        if (value > 0xful) {
            ++num_digits;
        }
        return num_digits;
    }
};

template<>
struct digits_counter<16, 8>
{
    static STRF_HD unsigned count_digits(uint_fast64_t value) noexcept
    {
        unsigned num_digits = 1;
        if( value > 0xffffffffuLL ) {
            value >>= 32;
            num_digits += 8 ;
        }
        if( value > 0xffffuLL ) {
            value >>= 16;
            num_digits += 4 ;
        }
        if( value > 0xffuLL ) {
            value >>= 8;
            num_digits += 2 ;
        }
        if( value > 0xfuLL ) {
            ++num_digits;
        }
        return num_digits;
    }
};

#endif // defined(__cpp_lib_bitops)

template<>
struct digits_counter<10, 2>
{
    static STRF_HD unsigned count_digits_unsigned(uint_fast16_t value) noexcept
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

    template <typename IntT>
    static STRF_HD unsigned count_digits(IntT value) noexcept
    {
        auto uvalue = strf::detail::unsigned_abs(value);
        return count_digits_unsigned(uvalue);
    }
};

template<>
struct digits_counter<10, 4>
{
    static STRF_HD unsigned count_digits_unsigned(uint_fast32_t value) noexcept
    {
        unsigned num_digits = 1;

        if (value > 99999999ul) {
            value /= 100000000ul;
            num_digits += 8;
            goto value_less_than_100;
        }
        if (value > 9999ul) {
            value /= 10000ul;
            num_digits += 4;
        }
        if( value > 99ul ) {
            value /= 100ul;
            num_digits += 2 ;
        }
        value_less_than_100:
        if (value > 9ul) {
             ++num_digits;
        }

        return num_digits;
    }

    template <typename IntT>
    static STRF_HD unsigned count_digits(IntT value) noexcept
    {
        auto uvalue = strf::detail::unsigned_abs(value);
        return count_digits_unsigned(uvalue);
    }
};


template<>
struct digits_counter<10, 8>
{
    static STRF_HD unsigned count_digits_unsigned(uint_fast64_t value) noexcept
    {
        unsigned num_digits = 1LL;

        if (value > 9999999999999999LL) {
            value /= 10000000000000000LL;
            num_digits += 16;
            //  goto value_less_than_10000;
        }
        if (value > 99999999LL) {
            value /= 100000000LL;
            num_digits += 8;
        }
        //value_less_than_10000:
        if (value > 9999LL) {
            value /= 10000LL;
            num_digits += 4;
        }
        if(value > 99LL) {
            value /= 100LL;
            num_digits += 2;
        }
        if(value > 9LL) {
            ++num_digits;
        }
        return num_digits;
    }

    template <typename IntT>
    static STRF_HD unsigned count_digits(IntT value) noexcept
    {
        auto uvalue = strf::detail::unsigned_abs(value);
        return count_digits_unsigned(uvalue);
    }
};

template <unsigned Base, typename intT>
STRF_HD unsigned count_digits(intT value) noexcept
{
    return strf::detail::digits_counter<Base, sizeof(intT)>
        ::count_digits(value);
}

template <typename intT>
STRF_HD unsigned count_digits(intT value, int base) noexcept
{
    if (base == 10) return count_digits<10>(value);
    if (base == 16) return count_digits<16>(value);
    STRF_ASSERT(base == 8);
    return count_digits<8>(value);
}

inline STRF_HD char to_xdigit(unsigned digit)
{
    if (digit < 10) {
        return static_cast<char>('0' + digit);
    }
    constexpr char offset = 'a' - 10;
    return static_cast<char>(offset + digit);
}

inline STRF_HD const char* chars_00_to_99() noexcept
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

template <int Base>
class intdigits_backwards_writer;

template <>
class intdigits_backwards_writer<10>
{
public:

    template <typename IntT, typename CharT>
    static STRF_HD CharT* write_txtdigits_backwards
        ( IntT value
        , CharT* it
        , strf::lettercase = strf::lowercase ) noexcept
    {
        auto uvalue = strf::detail::unsigned_abs(value);
        const char* arr = strf::detail::chars_00_to_99();
        while(uvalue > 99) {
            auto index = (uvalue % 100) << 1;
            it[-2] = arr[index];
            it[-1] = arr[index + 1];
            it -= 2;
            uvalue /= 100;
        }
        if (uvalue < 10) {
            *--it = static_cast<CharT>('0' + uvalue);
            return it;
        } else {
            auto index = uvalue << 1;
            it[-2] = arr[index];
            it[-1] = arr[index + 1];
            return it - 2;
        }
    }

    template <typename IntT, typename CharT>
    static STRF_HD void write_txtdigits_backwards_little_sep
        ( IntT value
        , CharT* it
        , CharT sep
        , const std::uint8_t* groups
        , strf::lettercase = strf::lowercase ) noexcept
    {
        auto uvalue = strf::detail::unsigned_abs(value);
        const char* arr = strf::detail::chars_00_to_99();
        auto n = *groups;
        while (uvalue > 99) {
            auto index = (uvalue % 100) << 1;
            if (n > 1) {
                it[-2] = arr[index];
                it[-1] = arr[index + 1];
                n -= 2;
                if (n == 0) {
                    it[-3] = sep;
                    n = * ++groups;
                    it -= 3;
                } else {
                    it -= 2;
                }
            } else {
                it[-3] = arr[index];
                it[-2] = sep;
                it[-1] = arr[index + 1];
                n = * ++groups - 1;
                if (n == 0) {
                    it[-4] = sep;
                    it -= 4;
                    n = * ++groups;
                } else {
                    it -= 3;
                }
            }
            uvalue /= 100;
        }
        STRF_ASSERT(n != 0);
        if (uvalue < 10) {
            it[-1] = static_cast<CharT>('0' + uvalue);
        } else {
            auto index = uvalue << 1;
            if (n == 1) {
                it[-3] = arr[index];
                it[-2] = sep;
                it[-1] = arr[index + 1];
            } else {
                it[-2] = arr[index];
                it[-1] = arr[index + 1];
            }
        }
    }
};

template <>
class intdigits_backwards_writer<16>
{
public:

    template <typename IntT, typename CharT>
    static STRF_HD CharT* write_txtdigits_backwards
        ( IntT value
        , CharT* it
        , strf::lettercase lc ) noexcept
    {
        using uIntT = typename std::make_unsigned<IntT>::type;
        uIntT uvalue = value;
        const char offset_digit_a = ('A' | ((lc == strf::lowercase) << 5)) - 10;
        while(uvalue > 0xF) {
            auto digit = uvalue & 0xF;
            *--it = ( digit < 10
                    ? static_cast<CharT>('0' + digit)
                    : static_cast<CharT>(offset_digit_a + digit) );
            uvalue >>= 4;
        }
        auto digit = uvalue & 0xF;
        *--it = ( digit < 10
                ? static_cast<CharT>('0' + digit)
                : static_cast<CharT>(offset_digit_a + digit) );
        return it;
    }

    template <typename IntT, typename CharT>
    static STRF_HD void write_txtdigits_backwards_little_sep
        ( IntT value
        , CharT* it
        , CharT sep
        , const std::uint8_t* groups
        , strf::lettercase lc ) noexcept
    {
        auto uvalue = strf::detail::unsigned_abs(value);
        auto n = *groups;
        const char offset_digit_a = ('A' | ((lc == strf::lowercase) << 5)) - 10;
        while (uvalue > 0xF) {
            unsigned d = uvalue & 0xF;
            --it;
            if (d < 10) {
                *it = static_cast<CharT>('0' + d);
            } else {
                *it = static_cast<CharT>(offset_digit_a + d);
            }
            if (--n == 0) {
                *--it = sep;
                n = *++groups;
            }
            uvalue = uvalue >> 4;
        }
        --it;
        if (uvalue < 10) {
            *it = static_cast<CharT>('0' + uvalue);
        } else {
            *it = static_cast<CharT>(offset_digit_a + uvalue);
        }
    }
};

template <>
class intdigits_backwards_writer<8>
{
public:

    template <typename IntT, typename CharT>
    static STRF_HD CharT* write_txtdigits_backwards
        ( IntT value
        , CharT* it
        , strf::lettercase = strf::lowercase) noexcept
    {
        using uIntT = typename std::make_unsigned<IntT>::type;
        uIntT uvalue = value;
        while (uvalue > 7) {
            *--it = static_cast<CharT>('0' + (uvalue & 7));
            uvalue >>= 3;
        }
        *--it = static_cast<CharT>('0' + uvalue);
        return it;
    }

    template <typename IntT, typename CharT>
    static STRF_HD void write_txtdigits_backwards_little_sep
        ( IntT value
        , CharT* it
        , CharT sep
        , const std::uint8_t* groups
        , strf::lettercase = strf::lowercase ) noexcept
    {
        auto uvalue = strf::detail::unsigned_abs(value);
        auto n = *groups;
        while (uvalue > 0x7) {
            *--it = '0' + (uvalue & 0x7);
            uvalue = uvalue >> 3;
            if (--n == 0) {
                *--it = sep;
                n = *++groups;
            }
        }
        *--it = static_cast<CharT>('0' + uvalue);
    }
};

template <typename IntT, typename CharT>
inline STRF_HD CharT* write_int_dec_txtdigits_backwards(IntT value, CharT* it) noexcept
{
    return intdigits_backwards_writer<10>::write_txtdigits_backwards(value, it);
}

template <typename IntT, typename CharT>
inline STRF_HD CharT* write_int_hex_txtdigits_backwards(IntT value, CharT* it) noexcept
{
    return intdigits_backwards_writer<16>::write_txtdigits_backwards(value, it);
}

template <typename IntT, typename CharT>
inline STRF_HD CharT* write_int_oct_txtdigits_backwards(IntT value, CharT* it) noexcept
{
    return intdigits_backwards_writer<8>::write_txtdigits_backwards(value, it);
}

template <int Base, typename IntT, typename CharT>
inline STRF_HD CharT* write_int_txtdigits_backwards( IntT value
                                                   , CharT* it
                                                   , strf::lettercase lc ) noexcept
{
    using writer = intdigits_backwards_writer<Base>;
    return writer::write_txtdigits_backwards(value, it, lc);
}

template <int Base, typename IntT, typename CharT>
inline STRF_HD void write_int_txtdigits_backwards_little_sep
    ( IntT value
    , CharT* it
    , CharT sep
    , const std::uint8_t* groups
    , strf::lettercase lc ) noexcept
{
    intdigits_backwards_writer<Base>::write_txtdigits_backwards_little_sep
        ( value, it, sep, groups, lc );
}

template <typename CharT>
STRF_HD void write_digits_big_sep
    ( strf::basic_outbuf<CharT>& ob
    , const strf::encoding<CharT> encoding
    , const std::uint8_t* last_grp
    , unsigned char* digits
    , unsigned num_digits
    , char32_t sep
    , std::size_t sep_size )
{
    STRF_ASSERT(sep_size != (std::size_t)-1);
    STRF_ASSERT(sep_size != 1);
    STRF_ASSERT(sep_size == encoding.validate(sep));

    ob.ensure(1);

    auto pos = ob.pos();
    auto end = ob.end();
    auto grp_it = last_grp;
    auto n = *grp_it;

    while(true) {
        *pos = *digits;
        ++pos;
        ++digits;
        if (--num_digits == 0) {
            break;
        }
        --n;
        if (pos == end || (n == 0 && pos + sep_size >= end)) {
            ob.advance_to(pos);
            ob.recycle();
            pos = ob.pos();
            end = ob.end();
        }
        if (n == 0) {
            pos = encoding.encode_char(pos, sep);
            n = *--grp_it;
        }
    }
    ob.advance_to(pos);
}

template <int Base>
class intdigits_writer
{
public:

    template <typename IntT, typename CharT>
    static inline STRF_HD void write
        ( strf::basic_outbuf<CharT>& ob
        , IntT value
        , unsigned digcount
        , strf::lettercase lc )
    {
        static_assert(std::is_unsigned<IntT>::value, "expected unsigned int");

        ob.ensure(digcount);
        auto p = ob.pos() + digcount;
        intdigits_backwards_writer<Base>::write_txtdigits_backwards(value, p, lc);
        ob.advance_to(p);
    }

    template <typename UIntT, typename CharT>
    static STRF_HD void write
          ( strf::basic_outbuf<CharT>& ob
          , const strf::numpunct_base& punct
          , strf::encoding<CharT> enc
          , UIntT value
          , unsigned digcount
          , strf::lettercase lc )
    {
        static_assert(std::is_unsigned<UIntT>::value, "expected unsigned int");

        constexpr auto max_digits = detail::max_num_digits<UIntT, Base>;
        uint8_t groups[max_digits];
        const auto num_groups = punct.groups(digcount, groups);
        if (num_groups == 0) {
            no_punct:
            write(ob, value, digcount, lc);
            return;
        }
        auto sep32 = punct.thousands_sep();
        CharT sep = static_cast<CharT>(sep32);
        if (sep32 >= enc.u32equivalence_end() || sep32 < enc.u32equivalence_begin()) {
            auto sep_size = enc.validate(sep32);
            if (sep_size == (std::size_t)-1) {
                goto no_punct;
            }
            if (sep_size != 1) {
                write_digits_big_sep( ob, enc, groups, value, digcount
                                    , num_groups, sep32, sep_size, lc );
                return;
            }
            enc.encode_char(&sep, sep32);
        }
        std::size_t size = digcount + num_groups - 1;
        ob.ensure(size);
        auto next_p = ob.pos() + size;
        intdigits_backwards_writer<Base>::write_txtdigits_backwards_little_sep
            (value, next_p, sep, groups, lc);
        ob.advance_to(next_p);
    }

private:

    template <typename CharT>
    static STRF_HD void write_digits_big_sep
          ( strf::basic_outbuf<CharT>& ob
          , strf::encoding<CharT> enc
          , const uint8_t* groups
          , unsigned long long value
          , unsigned digcount
          , unsigned num_groups
          , char32_t sep
          , std::size_t sep_size
          , strf::lettercase lc )
    {
        constexpr auto max_digits = detail::max_num_digits<unsigned long long, Base>;
        unsigned char digits_buff[max_digits];

        const auto dig_end = digits_buff + max_digits;
        auto digits = strf::detail::write_int_txtdigits_backwards<Base>
            ( value, dig_end, lc);

        strf::detail::write_digits_big_sep
            ( ob, enc, groups + num_groups - 1, digits, digcount
            , sep, sep_size );
    }
}; // class template intdigits_writer

template <>
class intdigits_writer<2>
{
public:

    template <typename CharT, typename UIntT>
    static STRF_HD void write
        ( strf::basic_outbuf<CharT>& ob
        , UIntT value
        , unsigned digcount
        , strf::lettercase = strf::lowercase )
    {
        static_assert(std::is_unsigned<UIntT>::value, "expected unsigned int");

        if (value <= 1) {
            strf::put(ob, static_cast<CharT>('0' + value));
            return;
        }
        auto it = ob.pos();
        auto end = ob.end();
        UIntT mask = (UIntT)1 << (digcount - 1);
        do {
            if (it == end) {
                ob.advance_to(it);
                ob.recycle();
                it = ob.pos();
                end = ob.end();
            }
            *it = (CharT)'0' + (0 != (value & mask));
            ++it;
            mask = mask >> 1;
        }
        while(mask != 0);

        ob.advance_to(it);
    }

    template <typename UIntT, typename CharT>
    static STRF_HD void write
        ( strf::basic_outbuf<CharT>& ob
        , const strf::numpunct_base& punct
        , strf::encoding<CharT> enc
        , UIntT value
        , unsigned digcount
        , strf::lettercase = strf::lowercase )
    {
        static_assert(std::is_unsigned<UIntT>::value, "expected unsigned int");
        if (value <= 1) {
            strf::put(ob, static_cast<CharT>('0' + value));
            return;
        }

        constexpr auto max_digits = detail::max_num_digits<UIntT, 2>;
        uint8_t groups[max_digits];
        const auto num_groups = punct.groups(digcount, groups);
        if (num_groups == 0) {
            no_punct:
            write(ob, value, digcount);
            return;
        }
        auto sep32 = punct.thousands_sep();
        CharT sep = static_cast<CharT>(sep32);
        if (sep32 >= enc.u32equivalence_end() || sep32 < enc.u32equivalence_begin()) {
            auto sep_size = enc.validate(sep32);
            if (sep_size == (std::size_t)-1) {
                goto no_punct;
            }
            if (sep_size != 1) {
                write_big_sep( ob, enc, groups + num_groups -1, value, digcount
                             , sep32, sep_size );
                return;
            }
            enc.encode_char(&sep, sep32);
        }
        write_little_sep(ob, groups + num_groups -1, value, digcount, sep);
    }

private:

    template <typename UIntT, typename CharT>
    static STRF_HD void write_little_sep
        ( strf::basic_outbuf<CharT>& ob
        , const uint8_t* groups
        , UIntT value
        , unsigned digcount
        , CharT sep )
    {
        auto grp_it = groups;
        auto grp_size = *grp_it;
        ob.ensure(grp_size);
        auto it = ob.pos();
        auto end = ob.end();
        UIntT mask = (UIntT)1 << (digcount - 1);

        while (true) {
            for(;grp_size != 0; --grp_size) {
                *it = (CharT)'0' + (0 != (value & mask));
                mask = mask >> 1;
                ++it;
            }
            if (mask == 0) {
                break;
            }
            grp_size = * --grp_it;
            if (it + grp_size + 1 > end) {
                ob.advance_to(it);
                ob.recycle();
                it = ob.pos();
                end = ob.end();
            }
            *it = sep;
            ++it;
        }
        ob.advance_to(it);
     }

    template <typename UIntT, typename CharT>
    static STRF_HD void write_big_sep
        ( strf::basic_outbuf<CharT>& ob
        , const strf::encoding<CharT> encoding
        , const std::uint8_t* groups
        , UIntT value
        , unsigned digcount
        , char32_t sep
        , std::size_t sep_size )
    {
        auto grp_it = groups;
        auto grp_size = *grp_it;
        auto it = ob.pos();
        auto end = ob.end();
        UIntT mask = (UIntT)1 << (digcount - 1);

        while (true) {
            if (it + grp_size > end) {
                ob.advance_to(it);
                ob.recycle();
                it = ob.pos();
                end = ob.end();
            }
            for(;grp_size != 0; --grp_size) {
                *it = (CharT)'0' + (0 != (value & mask));
                mask = mask >> 1;
                ++it;
            }
            if (mask == 0) {
                break;
            }
            grp_size = * --grp_it;
            if (it + grp_size + sep_size > end) {
                ob.advance_to(it);
                ob.recycle();
                it = ob.pos();
                end = ob.end();
            }
            it = encoding.encode_char(it, sep);
        }
        ob.advance_to(it);
    }

}; // class intdigits_writer<2>

template <int Base, typename CharT, typename UIntT>
inline STRF_HD void write_int
    ( strf::basic_outbuf<CharT>& ob
    , UIntT value
    , unsigned digcount
    , strf::lettercase lc )
{
    intdigits_writer<Base>::write(ob, value, digcount, lc);
}

template <int Base, typename CharT, typename UIntT>
inline STRF_HD void write_int
      ( strf::basic_outbuf<CharT>& ob
      , const strf::numpunct_base& punct
      , strf::encoding<CharT> enc
      , UIntT value
      , unsigned digcount
      , strf::lettercase lc )
{
    intdigits_writer<Base>::write(ob, punct, enc, value, digcount, lc);
}

} // namespace detail

} // namespace strf

#endif  // STRF_DETAIL_NUMBER_OF_DIGITS_HPP

