#ifndef STRF_DETAIL_INT_DIGITS_HPP
#define STRF_DETAIL_INT_DIGITS_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/lettercase.hpp>
#ifdef STRF_USE_STD_BITOPS
#include <bit>
#endif

namespace strf {

namespace detail {

template <typename CharT>
constexpr bool is_digit(CharT ch)
{
    return static_cast<CharT>('0') <= ch && ch <= static_cast<CharT>('9');
}

template <typename CharT>
constexpr bool not_digit(CharT ch)
{
    return ch < static_cast<CharT>('0') || static_cast<CharT>('9') < ch;
}

inline STRF_HD std::uint64_t pow10(int n) noexcept
{
    static const std::uint64_t p10[] =
        { 1, 10, 100, 1000, 10000, 100000, 1000000
        , 10000000ULL
        , 100000000ULL
        , 1000000000ULL
        , 10000000000ULL
        , 100000000000ULL
        , 1000000000000ULL
        , 10000000000000ULL
        , 100000000000000ULL
        , 1000000000000000ULL
        , 10000000000000000ULL
        , 100000000000000000ULL
        , 1000000000000000000ULL
        , 10000000000000000000ULL };
    STRF_ASSERT(0 <= n && n <= 19);

    return p10[n];
}

template <typename IntT, int Base> struct max_num_digits_impl;
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

template<class IntT, int Base>
constexpr STRF_HD unsigned max_num_digits()
{
    return strf::detail::max_num_digits_impl<IntT, Base>::value;
}

#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wconversion"
#endif

template
    < typename IntT
    , typename unsigned_IntT = typename std::make_unsigned<IntT>::type >
constexpr STRF_HD strf::detail::enable_if_t<std::is_signed<IntT>::value, unsigned_IntT>
unsigned_abs(IntT value) noexcept
{
    return ( value >= 0
           ? static_cast<unsigned_IntT>(value)
           : 1 + static_cast<unsigned_IntT>(-(value + 1)));
}

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

template<typename IntT>
constexpr STRF_HD strf::detail::enable_if_t<std::is_unsigned<IntT>::value, IntT>
unsigned_abs(IntT value) noexcept
{
    return value;
}

template < typename ToIntT
         , typename FromIntT
         , strf::detail::enable_if_t<std::is_unsigned<FromIntT>::value, int> = 0>
constexpr STRF_HD ToIntT cast_abs(FromIntT value) noexcept
{
    return value;
}
template < typename ToIntT
         , typename FromIntT
         , strf::detail::enable_if_t
             < (sizeof(ToIntT) > sizeof(FromIntT)) && std::is_signed<FromIntT>::value
             , int > = 0 >
constexpr STRF_HD ToIntT cast_abs(FromIntT value)
{
    using SignedToIntT = strf::detail::make_signed_t<ToIntT>;
    return value >= 0 ? (SignedToIntT)value : -(SignedToIntT)value;
}
template < typename ToIntT
         , typename FromIntT
         , strf::detail::enable_if_t
             < sizeof(ToIntT) == sizeof(FromIntT) && std::is_signed<FromIntT>::value
             , int > = 0 >
constexpr STRF_HD ToIntT cast_abs(FromIntT value)
{
    static_assert( std::is_unsigned<ToIntT>::value
                 , "Expected destination to be unsigned" );
    return value >= 0
        ? static_cast<ToIntT>(value)
        : (1 + static_cast<ToIntT>(-(value + 1)));
}

template <int Base, int IntSize>
struct digits_counter;

#if defined(STRF_HAS_COUNTL_ZERO)

template<int IntSize>
struct digits_counter<2, IntSize>
{
    static_assert(IntSize <= 4, "");
    // NOLINTNEXTLINE(google-runtime-int)
    static inline STRF_HD int count_digits(unsigned long value) noexcept
    {
        return static_cast<int>(sizeof(value)) * 8 - strf::detail::countl_zero_l(value | 1);
    }
};
template<>
struct digits_counter<2, 8>
{
    // NOLINTNEXTLINE(google-runtime-int)
    static inline STRF_HD int count_digits(unsigned long long value) noexcept
    {
        return static_cast<int>(sizeof(value)) * 8 - strf::detail::countl_zero_ll(value | 1);
    }
};
template<int IntSize>
struct digits_counter<8, IntSize>
{
    static_assert(IntSize <= 4, "");
    // NOLINTNEXTLINE(google-runtime-int)
    static STRF_HD int count_digits(unsigned long value) noexcept
    {
        return (static_cast<int>(sizeof(value)) * 8 + 2
                - strf::detail::countl_zero_l(value | 1)) / 3;
}
};
template<>
struct digits_counter<8, 8>
{
    // NOLINTNEXTLINE(google-runtime-int)
    static STRF_HD int count_digits(unsigned long long value) noexcept
    {
        return (static_cast<int>(sizeof(value)) * 8 + 2
                - strf::detail::countl_zero_ll(value | 1)) / 3;
    }
};
template<int IntSize>
struct digits_counter<16, IntSize>
{
    static_assert(IntSize <= 4, "");
    // NOLINTNEXTLINE(google-runtime-int)
    static STRF_HD int count_digits(unsigned long value) noexcept
    {
        return (static_cast<int>(sizeof(value)) * 8 + 3
                - strf::detail::countl_zero_l(value | 1)) >> 2;
    }
};
template<>
struct digits_counter<16, 8>
{
    // NOLINTNEXTLINE(google-runtime-int)
    static STRF_HD int count_digits(unsigned long long value) noexcept
    {
        return (static_cast<int>(sizeof(value)) * 8 + 3
                - strf::detail::countl_zero_ll(value | 1)) >> 2;
    }
};

#else // defined(STRF_HAS_COUNTL_ZERO)

template<>
struct digits_counter<2, 1>
{
    STRF_CONSTEXPR_IN_CXX14 static STRF_HD int count_digits(uint_fast8_t value8) noexcept
    {
        int num_digits = 1;
        unsigned value = value8;
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
struct digits_counter<2, 2>
{
    STRF_CONSTEXPR_IN_CXX14 static STRF_HD int count_digits(uint_fast16_t value) noexcept
    {
        int num_digits = 1;
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
struct digits_counter<2, 4>
{
    STRF_CONSTEXPR_IN_CXX14 static STRF_HD int count_digits(uint_fast32_t value) noexcept
    {
        int num_digits = 1;
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
    STRF_CONSTEXPR_IN_CXX14 static STRF_HD int count_digits(uint_fast64_t value) noexcept
    {

        int num_digits = 1;
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
struct digits_counter<8, 1>
{
    STRF_CONSTEXPR_IN_CXX14 static STRF_HD int count_digits(uint_fast8_t value) noexcept
    {
        if(value > 077ul) {
            return 3;
        }
        if(value > 07ul) {
            return 2;
        }
        return 1;
    }
};

template<>
struct digits_counter<8, 2>
{
    STRF_CONSTEXPR_IN_CXX14 static STRF_HD int count_digits(uint_fast16_t value) noexcept
    {
        int num_digits = 1;
        if(value > 07777u) {
            value >>= 12;
            num_digits += 4;
        }
        if(value > 077u) {
            value >>= 6;
            num_digits += 2;
        }
        if(value > 07u) {
            ++num_digits;
        }
        return num_digits;
    }
};

template<>
struct digits_counter<8, 4>
{
    STRF_CONSTEXPR_IN_CXX14 static STRF_HD int count_digits(uint_fast32_t value) noexcept
    {
        int num_digits = 1;
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
    STRF_CONSTEXPR_IN_CXX14 static STRF_HD int count_digits(uint_fast64_t value) noexcept
    {
        int num_digits = 1;
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
struct digits_counter<16, 1>
{
    constexpr static STRF_HD int count_digits(uint_fast8_t value) noexcept
    {
        return value < 0x10 ? 1 : 2;
    }
};

template<>
struct digits_counter<16, 2>
{
    STRF_CONSTEXPR_IN_CXX14 static STRF_HD int count_digits(uint_fast16_t value) noexcept
    {
        if (value < 0x100ul){
            return value < 0x10ul ? 1 : 2;
        }
        return value < 0x1000ul ? 3 : 4;
    }
};

template<>
struct digits_counter<16, 4>
{
    STRF_CONSTEXPR_IN_CXX14 static STRF_HD int count_digits(uint_fast32_t value) noexcept
    {
        if (value < 0x10000ul) {
            if (value < 0x100ul){
                return value < 0x10ul ? 1 : 2;
            }
            return value < 0x1000ul ? 3 : 4;
        }
        if (value < 0x1000000ul) {
            return value < 0x100000ul ? 5 : 6;
        }
        return value < 0x10000000ul ? 7 : 8;
    }
};

template<>
struct digits_counter<16, 8>
{
    STRF_CONSTEXPR_IN_CXX14 static STRF_HD int count_digits(uint_fast64_t value) noexcept
    {
        if (value < 0x100000000ull) {
            if (value < 0x10000ull) {
                if (value < 0x100ull){
                    return value < 0x10ull ? 1 : 2;
                }
                return value < 0x1000ull ? 3 : 4;
            }
            if (value < 0x1000000ull) {
                return value < 0x100000ull ? 5 : 6;
            }
            return value < 0x10000000ull ? 7 : 8;
        }
        if (value < 0x1000000000000ull) {
            if (value < 0x10000000000ull){
                return value < 0x1000000000ull ? 9 : 10;
            }
            return value < 0x100000000000ull ? 11 : 12;
        }
        if (value < 0x100000000000000ull){
            return value < 0x10000000000000ull ? 13 : 14;
        }
        return value < 0x1000000000000000ull ? 15 : 16;
    }
};

#endif // defined(STRF_HAS_COUNTL_ZERO)

template<>
struct digits_counter<10, 1>
{
    STRF_CONSTEXPR_IN_CXX14 static STRF_HD int count_digits_unsigned
        ( uint_fast8_t value ) noexcept
    {
        if (value <= 99)
            return value <= 9 ? 1 : 2;
        return 3;
    }

    template <typename IntT>
    constexpr static STRF_HD int count_digits(IntT value) noexcept
    {
        return count_digits_unsigned(strf::detail::unsigned_abs(value));
    }
};

template<>
struct digits_counter<10, 2>
{
    STRF_CONSTEXPR_IN_CXX14 static STRF_HD int
    count_digits_unsigned(uint_fast16_t value) noexcept
    {
        if (value <= 99) {
            return value <= 9 ? 1 : 2;
        }
        if (value <= 9999) {
            return value <= 999 ? 3 : 4;
        }
        return 5;
    }

    template <typename IntT>
    constexpr static STRF_HD int count_digits(IntT value) noexcept
    {
        return count_digits_unsigned(strf::detail::unsigned_abs(value));
    }
};

template<>
struct digits_counter<10, 4>
{
    STRF_CONSTEXPR_IN_CXX14 static STRF_HD int count_digits_unsigned(uint_fast32_t value) noexcept
    {
        if (value <= 9999UL) {
            if (value <= 99UL) {
                return value <= 9UL ? 1 : 2;
            }
            return value <= 999UL ? 3 : 4;
        }
        if (value <= 99999999UL) {
            if (value <= 999999UL) {
                return value <= 99999UL ? 5 : 6;
            }
            return value <= 9999999UL ? 7 : 8;
        }
        return value <= 999999999UL ? 9 : 10;
    }

    template <typename IntT>
    constexpr static STRF_HD int count_digits(IntT value) noexcept
    {
        return count_digits_unsigned(strf::detail::unsigned_abs(value));
    }
};

template<>
struct digits_counter<10, 8>
{
    STRF_CONSTEXPR_IN_CXX14 static STRF_HD int count_digits_unsigned
        ( uint_fast64_t value ) noexcept
    {
        if (value <= 99999999ULL) {
            if (value <= 9999) {
                if (value <= 99ULL) {
                    return value <= 9ULL ? 1 : 2;
                }
                return value <= 999ULL ? 3 : 4;
            }
            if (value <= 999999ULL) {
                return value <= 99999ULL ? 5 : 6;
            }
            return value <= 9999999ULL ? 7 : 8;
        }
        if (value <= 9999999999999999ULL) {
            if (value <= 999999999999ULL) {
                if (value <= 9999999999ULL) {
                    return value <= 999999999ULL ? 9 : 10;
                }
                return value <= 99999999999ULL ? 11 : 12;
            }
            if (value <= 99999999999999ULL ) {
                return value <= 9999999999999ULL ? 13 : 14;
            }
            return value <= 999999999999999ULL ? 15 : 16;
        }
        if (value <= 999999999999999999ULL){
            return value <=  99999999999999999ULL ? 17 : 18;
        }
        return value <= 9999999999999999999ULL ? 19 : 20;
    }

    template <typename IntT>
    constexpr static STRF_HD int count_digits(IntT value) noexcept
    {
        return count_digits_unsigned(strf::detail::unsigned_abs(value));
    }
};

template <int Base, typename intT>
STRF_CONSTEXPR_IN_CXX14 STRF_HD int count_digits(intT value) noexcept
{
    static_assert(std::is_unsigned<intT>::value, "");
    return strf::detail::digits_counter<Base, (int)sizeof(intT)>
        ::count_digits(value);
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

    template <typename UIntT, typename CharT>
    static STRF_HD CharT* write_txtdigits_backwards
        ( UIntT uvalue
        , CharT* it
        , strf::lettercase = strf::lowercase ) noexcept
    {
        static_assert(std::is_unsigned<UIntT>::value, "");

        const char* arr = strf::detail::chars_00_to_99();
        while(uvalue > 99) {
            auto index = (uvalue % 100) << 1;
            it[-2] = static_cast<CharT>(arr[index]);
            it[-1] = static_cast<CharT>(arr[index + 1]);
            it -= 2;
            uvalue /= 100;
        }
        if (uvalue < 10) {
            *--it = static_cast<CharT>('0' + uvalue);
            return it;
        }
        auto index = uvalue << 1;
        it[-2] = static_cast<CharT>(arr[index]);
        it[-1] = static_cast<CharT>(arr[index + 1]);
        return it - 2;
    }
    template <typename UIntT, typename CharT>
    static STRF_HD void write_txtdigits_backwards_little_sep
        ( CharT* it
        , UIntT uvalue
        , strf::digits_grouping_iterator git
        , CharT sep
        , strf::lettercase ) noexcept
    {
        STRF_ASSERT(uvalue != 0);

        const char* arr = strf::detail::chars_00_to_99();
        auto digits_before_sep = git.current();

        // if (git.shall_repeat_current() && digits_before_sep > 1) { // common case ( optimization )
        //     auto group_size = digits_before_sep;
        //     while (1) {
        //         if (uvalue < 10) {
        //             if (uvalue) {
        //                 *--it = static_cast<CharT>('0' + uvalue);
        //             }
        //             return;
        //         }
        //         auto dig_index = static_cast<std::uint8_t>((uvalue % 100) << 1);
        //         uvalue /= 100;
        //         if (digits_before_sep >= 2) {
        //             it[-1] = arr[dig_index + 1];
        //             it[-2] = arr[dig_index];
        //             if (uvalue == 0) {
        //                 return;
        //             }
        //             it -= 2;
        //             if (digits_before_sep == 2) {
        //                 *--it = sep;
        //                 digits_before_sep = group_size;
        //             } else {
        //                 digits_before_sep -= 2;
        //             }
        //         } else {
        //             it[-1] = arr[dig_index + 1];
        //             it[-2] = sep;
        //             it[-3] = arr[dig_index];
        //             it -=3;
        //             digits_before_sep = group_size - 1;
        //         }
        //     }
        // }
        while (1) {
            if (uvalue < 10) {
                STRF_ASSERT(uvalue != 0);
                *--it = static_cast<CharT>('0' + uvalue);
                return;
            }
            auto dig_index = static_cast<int>((uvalue % 100) << 1);
            uvalue /= 100;
            if (digits_before_sep >= 2) {
                it[-1] = static_cast<CharT>(arr[dig_index + 1]);
                it[-2] = static_cast<CharT>(arr[dig_index]);
                if (uvalue == 0) {
                    return;
                }
                it -= 2;
                if (digits_before_sep != 2) {
                    digits_before_sep -= 2;
                } else {
                    * --it = sep;
                    if (git.is_final()) {
                        break;
                    }
                    if ( ! git.is_last()) {
                        git.advance();
                    }
                    digits_before_sep = git.current();
                }
            } else {
                it[-1] = static_cast<CharT>(arr[dig_index + 1]);
                it[-2] = sep;
                it[-3] = static_cast<CharT>(arr[dig_index]);
                if (uvalue == 0) {
                    return;
                }
                it -= 3;
                if (git.is_final()) {
                    break;
                }
                if ( ! git.is_last()) {
                    git.advance();
                }
                digits_before_sep = git.current() - 1;
                if (digits_before_sep == 0) {
                    * --it = sep;
                    if (git.is_final()) {
                        break;
                    }
                    if ( ! git.is_last()) {
                        git.advance();
                    }
                    digits_before_sep = git.current();
                }
            }
        }

        STRF_ASSERT(uvalue != 0);
        while (uvalue > 9) {
            auto dig_index = static_cast<int>((uvalue % 100) << 1);
            uvalue /= 100;
            it[-2] = static_cast<CharT>(arr[dig_index]);
            it[-1] = static_cast<CharT>(arr[dig_index + 1]);
            it -= 2;
        }
        if (uvalue) {
            *--it = static_cast<CharT>('0' + uvalue);
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
        STRF_ASSERT(detail::ge_zero(value));
        using uIntT = typename std::make_unsigned<IntT>::type;
        uIntT uvalue = value;
        const int offset_digit_a = (static_cast<int>('A') | ((lc == strf::lowercase) << 5)) - 10;
        while(uvalue > 0xF) {
            int const digit = uvalue & 0xF;
            *--it = ( digit < 10
                    ? static_cast<CharT>('0' + digit)
                    : static_cast<CharT>(offset_digit_a + digit) );
            uvalue >>= 4;
        }
        const auto digit = static_cast<int>(uvalue & 0xF);
        *--it = ( digit < 10
                ? static_cast<CharT>('0' + digit)
                : static_cast<CharT>(offset_digit_a + digit) );
        return it;
    }

    template <typename UIntT, typename CharT>
    static STRF_HD void write_txtdigits_backwards_little_sep
        ( CharT* it
        , UIntT uvalue
        , strf::digits_grouping_iterator git
        , CharT sep
        , strf::lettercase lc ) noexcept
    {
        static_assert(std::is_unsigned<UIntT>::value, "");
        STRF_ASSERT(! git.ended());
        STRF_ASSERT(uvalue > 0xF);

        const int offset_digit_a = (static_cast<int>('A') | ((lc == strf::lowercase) << 5)) - 10;
        auto digits_before_sep = git.current();
        do {
            int const digit = uvalue & 0xF;
            *--it = ( digit < 10
                    ? static_cast<CharT>('0' + digit)
                    : static_cast<CharT>(offset_digit_a + digit) );
            uvalue >>= 4;
            if (digits_before_sep != 1) {
                -- digits_before_sep;
            } else {
                *--it = sep;
                if (git.is_final()) {
                    break;
                }
                if ( ! git.is_last()) {
                    git.advance();
                }
                digits_before_sep = git.current();
            }
        } while(uvalue > 0xF);
        STRF_ASSERT(uvalue);
        do {
            const auto digit = static_cast<int>(uvalue & 0xF);
            *--it = ( digit < 10
                    ? static_cast<CharT>('0' + digit)
                    : static_cast<CharT>(offset_digit_a + digit) );
            uvalue = uvalue >> 4;
        } while(uvalue);
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
    template <typename UIntT, typename CharT>
    static STRF_HD void write_txtdigits_backwards_little_sep
        ( CharT* it
        , UIntT uvalue
        , strf::digits_grouping_iterator git
        , CharT sep
        , strf::lettercase ) noexcept
    {
        static_assert(std::is_unsigned<UIntT>::value, "");
        STRF_ASSERT(uvalue != 0);
        STRF_ASSERT(! git.ended());

        auto digits_before_sep = git.current();
        while (1) {
            STRF_ASSERT(digits_before_sep > 0);
            *--it = static_cast<CharT>('0' + (uvalue & 0x7));
            uvalue = uvalue >> 3;
            if (uvalue == 0) {
                return;
            }
            if (digits_before_sep != 1) {
                -- digits_before_sep;
            } else {
                *--it = sep;
                if (git.is_final()) {
                    break;
                }
                if ( ! git.is_last()) {
                    git.advance();
                }
                digits_before_sep = git.current();
            }
        }
        STRF_ASSERT(uvalue);
        do {
            *--it = static_cast<CharT>('0' + (uvalue & 0x7));
            uvalue = uvalue >> 3;
        } while(uvalue);
    }
};

template <typename IntT, typename CharT>
inline STRF_HD CharT* write_int_dec_txtdigits_backwards(IntT value, CharT* it) noexcept
{
    return intdigits_backwards_writer<10>::write_txtdigits_backwards(value, it);
}

template <typename IntT, typename CharT>
inline STRF_HD CharT* write_int_hex_txtdigits_backwards
    (IntT value, CharT* it, strf::lettercase lc) noexcept
{
    return intdigits_backwards_writer<16>::write_txtdigits_backwards(value, it, lc);
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

// template <int Base, typename IntT, typename CharT>
// inline STRF_HD void write_int_txtdigits_backwards_little_sep
//     ( IntT value
//     , CharT* it
//     , CharT sep
//     , const std::uint8_t* groups
//     , strf::lettercase lc ) noexcept
// {
//     intdigits_backwards_writer<Base>::write_txtdigits_backwards_little_sep
//         ( value, it, sep, groups, lc );
// }

// template <typename CharT>
// STRF_HD void write_digits_big_sep
//     ( strf::destination<CharT>& dst
//     , strf::encode_char_f<CharT> encode_char
//     , const std::uint8_t* last_grp
//     , unsigned char* digits
//     , unsigned num_digits
//     , char32_t sep
//     , std::size_t sep_size )
// {
//     STRF_ASSERT(sep_size != (std::size_t)-1);
//     STRF_ASSERT(sep_size != 1);

//     dst.ensure(1);

//     auto ptr = dst.buffer_ptr();
//     auto end = dst.buffer_end();
//     auto grp_it = last_grp;
//     auto n = *grp_it;

//     while(true) {
//         *ptr = *digits;
//         ++ptr;
//         ++digits;
//         if (--num_digits == 0) {
//             break;
//         }
//         --n;
//         if (ptr == end || (n == 0 && ptr + sep_size >= end)) {
//             dst.advance_to(ptr);
//             dst.flush();
//             ptr = dst.buffer_ptr();
//             end = dst.buffer_end();
//         }
//         if (n == 0) {
//             ptr = encode_char(ptr, sep);
//             n = *--grp_it;
//         }
//     }
//     dst.advance_to(ptr);
// }

template <int Base>
class intdigits_writer
{
public:

    template <typename IntT, typename CharT>
    static inline STRF_HD void write
        ( strf::destination<CharT>& dst
        , IntT value
        , int digcount
        , strf::lettercase lc )
    {
        static_assert(std::is_unsigned<IntT>::value, "expected unsigned int");

        dst.ensure(digcount);
        auto *p = dst.buffer_ptr() + digcount;
        intdigits_backwards_writer<Base>::write_txtdigits_backwards(value, p, lc);
        dst.advance_to(p);
    }

    template <typename UIntT, typename CharT>
    static STRF_HD void write_little_sep
        ( strf::destination<CharT>& dst
        , UIntT uvalue
        , strf::digits_grouping grouping
        , int digcount
        , int seps_count
        , CharT sep
        , strf::lettercase lc )
    {
        static_assert(std::is_unsigned<UIntT>::value, "expected unsigned int");
        auto size = digcount + seps_count;
        dst.ensure(size);
        auto *next_p = dst.buffer_ptr() + size;
        intdigits_backwards_writer<Base>::write_txtdigits_backwards_little_sep
            (next_p, uvalue, grouping.get_iterator(), sep, lc);
        dst.advance_to(next_p);
    }

    template <typename UIntT, typename CharT>
    static STRF_HD void write_big_sep
        ( strf::destination<CharT>& dst
        , strf::encode_char_f<CharT> encode_char
        , UIntT value
        , strf::digits_grouping grouping
        , char32_t sep
        , int sep_size
        , int digcount
        , strf::lettercase lc )
    {
        static_assert(std::is_unsigned<UIntT>::value, "expected unsigned int");
        constexpr auto max_digits = detail::max_num_digits<UIntT, Base>();
        unsigned char digits_buff[max_digits];
        const auto dig_end = digits_buff + max_digits;
        const auto* digits = strf::detail::write_int_txtdigits_backwards<Base>
            ( value, dig_end, lc);

        auto dist = grouping.distribute(digcount);
        dst.ensure(dist.highest_group);
        auto *oit = dst.buffer_ptr();
        auto *end = dst.buffer_end();
        STRF_ASSERT(dist.highest_group >= 0);
        strf::detail::copy_n(digits, detail::cast_unsigned(dist.highest_group), oit);
        oit += dist.highest_group;
        digits += dist.highest_group;

        if (dist.middle_groups_count) {
            const auto middle_groups = dist.low_groups.highest_group();
            STRF_ASSERT(middle_groups >= 0);
            do {
                if (oit + sep_size + middle_groups > end) {
                    dst.advance_to(oit);
                    dst.flush();
                    oit = dst.buffer_ptr();
                    end = dst.buffer_end();
                }
                oit = encode_char(oit, sep);
                strf::detail::copy_n(digits, detail::cast_unsigned(middle_groups), oit);
                oit += middle_groups;
                digits += middle_groups;
            } while (--dist.middle_groups_count);
            dist.low_groups.pop_high();
        }
        while ( ! dist.low_groups.empty()) {
            const auto grp = dist.low_groups.highest_group();
            STRF_ASSERT(grp >= 0);
            if (oit + sep_size + grp > end) {
                dst.advance_to(oit);
                dst.flush();
                oit = dst.buffer_ptr();
                end = dst.buffer_end();
            }
            oit = encode_char(oit, sep);
            strf::detail::copy_n(digits, detail::cast_unsigned(grp), oit);
            oit += grp;
            digits += grp;
            dist.low_groups.pop_high();
        }
        dst.advance_to(oit);
    }

}; // class template intdigits_writer

template <>
class intdigits_writer<2>
{
public:

    template <typename CharT, typename UIntT>
    static STRF_HD void write
        ( strf::destination<CharT>& dst
        , UIntT value
        , int digcount
        , strf::lettercase = strf::lowercase )
    {
        static_assert(std::is_unsigned<UIntT>::value, "expected unsigned int");

        if (value <= 1) {
            strf::put(dst, static_cast<CharT>('0' + value));
            return;
        }
        auto *it = dst.buffer_ptr();
        auto *end = dst.buffer_end();
        UIntT mask = (UIntT)1 << (digcount - 1);
        do {
            if (it == end) {
                dst.advance_to(it);
                dst.flush();
                it = dst.buffer_ptr();
                end = dst.buffer_end();
            }
            *it = (CharT)'0' + (0 != (value & mask));
            ++it;
            mask = mask >> 1;
        }
        while(mask != 0);

        dst.advance_to(it);
    }

    template <typename UIntT, typename CharT>
    static STRF_HD void write_little_sep
        ( strf::destination<CharT>& dst
        , UIntT value
        , strf::digits_grouping grouping
        , int digcount
        , int seps_count
        , CharT sep
        , strf::lettercase = strf::lowercase )
    {
        STRF_ASSERT(value > 1);
        (void)seps_count;
        static_assert(std::is_unsigned<UIntT>::value, "expected unsigned int");

        UIntT mask = (UIntT)1 << (digcount - 1);
        auto dist = grouping.distribute(digcount);
        auto *oit = dst.buffer_ptr();
        auto *end = dst.buffer_end();
        while (dist.highest_group--) {
            *oit++ = (CharT)'0' + (0 != (value & mask));
            mask = mask >> 1;
        }
        if (dist.middle_groups_count) {
            auto middle_groups = dist.low_groups.highest_group();
            do {
                if (oit == end) {
                    dst.advance_to(oit);
                    dst.flush();
                    oit = dst.buffer_ptr();
                    end = dst.buffer_end();
                }
                *oit++ = sep;
                for (auto i = middle_groups; i ; --i) {
                    if (oit == end) {
                        dst.advance_to(oit);
                        dst.flush();
                        oit = dst.buffer_ptr();
                        end = dst.buffer_end();
                    }
                    *oit++ = (CharT)'0' + (0 != (value & mask));
                    mask = mask >> 1;
                }
            } while (--dist.middle_groups_count);
            dist.low_groups.pop_high();
        }
        while ( ! dist.low_groups.empty() ) {
            if (oit == end) {
                dst.advance_to(oit);
                dst.flush();
                oit = dst.buffer_ptr();
                end = dst.buffer_end();
            }
            *oit++ = sep;
            for (auto g = dist.low_groups.highest_group(); g; --g) {
                if (oit == end) {
                    dst.advance_to(oit);
                    dst.flush();
                    oit = dst.buffer_ptr();
                    end = dst.buffer_end();
                }
                *oit++ = (CharT)'0' + (0 != (value & mask));
                mask = mask >> 1;
            }
            dist.low_groups.pop_high();
        }
        dst.advance_to(oit);
    }


    template <typename UIntT, typename CharT>
    static STRF_HD void write_big_sep
        ( strf::destination<CharT>& dst
        , strf::encode_char_f<CharT> encode_char
        , UIntT value
        , strf::digits_grouping grouping
        , char32_t sep
        , int sep_size
        , int digcount
        , strf::lettercase = strf::lowercase )
    {
        STRF_ASSERT(value > 1);
        static_assert(std::is_unsigned<UIntT>::value, "expected int int");

        auto dist = grouping.distribute(digcount);
        UIntT mask = (UIntT)1 << (digcount - 1);
        dst.ensure(dist.highest_group);
        auto *oit = dst.buffer_ptr();
        auto *end = dst.buffer_end();
        while (dist.highest_group--) {
            *oit++ = (CharT)'0' + (0 != (value & mask));
            mask = mask >> 1;
        }
        if (dist.middle_groups_count) {
            auto middle_groups = dist.low_groups.highest_group();
            dist.low_groups.pop_high();
            do {
                if (oit + sep_size > end) {
                    dst.advance_to(oit);
                    dst.flush();
                    oit = dst.buffer_ptr();
                    end = dst.buffer_end();
                }
                oit = encode_char(oit, sep);
                for (auto i = middle_groups; i ; --i) {
                    if (oit == end) {
                        dst.advance_to(oit);
                        dst.flush();
                        oit = dst.buffer_ptr();
                        end = dst.buffer_end();
                    }
                    *oit++ = (CharT)'0' + (0 != (value & mask));
                    mask = mask >> 1;
                }
            } while (--dist.middle_groups_count);
        }
        while ( ! dist.low_groups.empty() ) {
            if (oit + sep_size > end) {
                dst.advance_to(oit);
                dst.flush();
                oit = dst.buffer_ptr();
                end = dst.buffer_end();
            }
            oit = encode_char(oit, sep);
            for (auto g = dist.low_groups.highest_group(); g; --g) {
                if (oit == end) {
                    dst.advance_to(oit);
                    dst.flush();
                    oit = dst.buffer_ptr();
                    end = dst.buffer_end();
                }
                *oit++ = (CharT)'0' + (0 != (value & mask));
                mask = mask >> 1;
            }
            dist.low_groups.pop_high();
        }
        dst.advance_to(oit);
    }

}; // class intdigits_writer<2>

template <int Base, typename CharT, typename UIntT>
inline STRF_HD void write_int
    ( strf::destination<CharT>& dst
    , UIntT value
    , int digcount
    , strf::lettercase lc )
{
    intdigits_writer<Base>::write(dst, value, digcount, lc);
}

template <int Base, typename CharT, typename UIntT>
inline STRF_HD void write_int_little_sep
    ( strf::destination<CharT>& dst
    , UIntT value
    , strf::digits_grouping grouping
    , int digcount
    , int seps_count
    , CharT sep
    , strf::lettercase lc = strf::lowercase )
{
    intdigits_writer<Base>::write_little_sep
        ( dst, value, grouping, digcount, seps_count, sep, lc );
}

template <int Base, typename CharT, typename UIntT>
inline STRF_HD void write_int_big_sep
    ( strf::destination<CharT>& dst
    , strf::encode_char_f<CharT> encode_char
    , UIntT value
    , strf::digits_grouping grouping
    , char32_t sep
    , int sep_size
    , int digcount
    , strf::lettercase lc = strf::lowercase )
{
    intdigits_writer<Base>::write_big_sep
        ( dst, encode_char, value, grouping, sep, sep_size, digcount, lc);
}


} // namespace detail

} // namespace strf

#endif // STRF_DETAIL_INT_DIGITS_HPP

