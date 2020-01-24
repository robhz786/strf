#ifndef STRF_DETAIL_INPUT_TYPES_FLOAT_HPP
#define STRF_DETAIL_INPUT_TYPES_FLOAT_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/printer.hpp>
#include <strf/facets_pack.hpp>
#include <strf/detail/facets/numpunct.hpp>
#include <strf/detail/ryu/double.hpp>
#include <strf/detail/ryu/float.hpp>
#include <algorithm>
#include <cstring>
#include <type_traits>

namespace strf {
namespace detail {

struct double_dec
{
    std::uint64_t m10;
    std::int32_t e10;
    bool negative;
    bool infinity;
    bool nan;
};

struct double_dec_base
{
    std::uint64_t m10;
    std::int32_t e10;
};

#if ! defined(STRF_OMIT_IMPL)

STRF_INLINE STRF_HD double_dec_base trivial_float_dec(
    std::uint32_t ieee_mantissa,
    std::int32_t biased_exponent,
    std::uint32_t k )
{
    constexpr int m_size = 23;

    STRF_ASSERT(-10 <= biased_exponent && biased_exponent <= m_size);
    STRF_ASSERT((std::int32_t)k == (biased_exponent * 179 + 1850) >> 8);
    STRF_ASSERT(0 == (ieee_mantissa & (0x7FFFFF >> k)));

    STRF_ASSERT(k <= m_size);
    STRF_ASSERT(biased_exponent <= (int)k);

    std::int32_t e10 = biased_exponent - k;
    std::uint32_t m = (1ul << k) | (ieee_mantissa >> (m_size - k));
    int p5 = k - biased_exponent;
    STRF_ASSERT(p5 <= 10);
    // when p5 >= 8 , k <= 5; then (m & 0xFF) != 0
    // if (p5 >= 8 && (0 == (m & 0xFF))) {
    //     p5 -= 8;
    //     e10 += 8;
    //     m = m >> 8;
    // }
    if (p5 >= 4 && (0 == (m & 0xF))) {
        p5 -= 4;
        e10 += 4;
        m = m >> 4;
    }
    if (p5 >= 2 && (0 == (m & 0x3))) {
        p5 -= 2;
        e10 += 2;
        m = m >> 2;
    }
    if (p5 != 0 && (0 == (m & 1))) {
        -- p5;
        ++ e10;
        m = m >> 1;
    }
    while ((m % 10) == 0){
        m /= 10;
        ++e10;
    }
    if (p5 >= 8) {
        m *= 390625ul; // (m << 18) + (m << 16) + (m << 15) + (m << 14) + ...
        p5 -= 8;
    }
    if (p5 >= 4) {
        m *= 625;  // = (m << 9) + (m << 6) + (m << 5) + (m << 4) + m
        p5 -= 4;
    }
    if (p5 >= 2) {
        m *= 25; // = (m << 4) + (m << 3) + m
        p5 -= 2;
    }
    if (p5 >= 1) {
        m = (m << 2) + m; // m *= 5
    }
    STRF_ASSERT((m % 10) != 0);
    return {m, e10};
}

STRF_INLINE STRF_HD double_dec_base trivial_double_dec(
    std::uint64_t ieee_mantissa,
    std::int32_t biased_exponent,
    std::uint32_t k )
{
    STRF_ASSERT(-22 <= biased_exponent && biased_exponent <= 52);
    STRF_ASSERT((std::int32_t)k == (biased_exponent * 179 + 4084) >> 8);
    STRF_ASSERT(0 == (ieee_mantissa & (0xFFFFFFFFFFFFFull >> k)));

    STRF_ASSERT(biased_exponent <= (int)k);
    STRF_ASSERT(k <= 52);

    std::int32_t e10 = biased_exponent - k;
    std::uint64_t m = (1ull << k) | (ieee_mantissa >> (52 - k));
    int p5 = k - biased_exponent;
    STRF_ASSERT(p5 <= 22);
    // when p5 >= 16 , k <= 15; then (m & 0xFFFF) != 0
    // if (p5 >= 16 && (0 == (m & 0xFFFF))) {
    //     p5 -= 16;
    //     e10 += 16;
    //     m = m >> 16;
    // }
    if (p5 >= 8 && (0 == (m & 0xFF))) {
        p5 -= 8;
        e10 += 8;
        m = m >> 8;
    }
    if (p5 >= 4 && (0 == (m & 0xF))) {
        p5 -= 4;
        e10 += 4;
        m = m >> 4;
    }
    if (p5 >= 2 && (0 == (m & 0x3))) {
        p5 -= 2;
        e10 += 2;
        m = m >> 2;
    }
    if (p5 != 0 && (0 == (m & 1))) {
        -- p5;
        ++ e10;
        m = m >> 1;
    }
    while ((m % 10) == 0){
        m /= 10;
        ++e10;
    }
    if (p5 >= 16) {
        m *= 152587890625ull;
        p5 -= 16;
    }
    if (p5 >= 8) {
        m *= 390625ull; // (m << 18) + (m << 16) + (m << 15) + (m << 14) + ...
        p5 -= 8;
    }
    if (p5 >= 4) {
        m *= 625;  // = (m << 9) + (m << 6) + (m << 5) + (m << 4) + m
        p5 -= 4;
    }
    if (p5 >= 2) {
        m *= 25; // = (m << 4) + (m << 3) + m
        p5 -= 2;
    }
    if (p5 >= 1) {
        m = (m << 2) + m; // m *= 5
    }
    STRF_ASSERT((m % 10) != 0);
    return {m, e10};
}
STRF_INLINE STRF_HD detail::double_dec decode(float f)
{
    constexpr int bias = 127;
    constexpr int e_size = 8;
    constexpr int m_size = 23;

    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const std::uint32_t exponent
        = static_cast<std::uint32_t>((bits << 1) >> (m_size + 1));
    const bool sign = (bits >> (m_size + e_size));
    const std::uint32_t mantissa = bits & 0x7FFFFF;

    if (exponent == 0 && mantissa == 0) {
        return {0, 0, sign, false, false};
    } else if (bias - 10 <= exponent && exponent <= bias + m_size) {
        const int e = exponent - bias;
        const unsigned k = (179 * e + 1850) >> 8;
        if (0 == (mantissa & (0x7FFFFF >> k))) {
            auto res = trivial_float_dec(mantissa, e, k);
            return {res.m10, res.e10, sign, false, false};
        }
    } else if (exponent == 0xFF) {
        if (mantissa == 0) {
            return {0, 0, sign, true, false};
        } else {
            return {0, 0, sign, false, true};
        }
    }

    auto fdec = detail::ryu::f2d(mantissa, exponent);
    return {fdec.mantissa, fdec.exponent, sign, false, false};
}


STRF_INLINE STRF_HD detail::double_dec decode(double d)
{
    constexpr int bias = 1023;
    constexpr int e_size = 11; // bits in exponent
    constexpr int m_size = 52; // bits in matissa

    std::uint64_t bits;
    std::memcpy(&bits, &d, 8);
    const std::uint32_t exponent
        = static_cast<std::uint32_t>((bits << 1) >> (m_size + 1));
    const bool sign = (bits >> (m_size + e_size));
    const std::uint64_t mantissa = bits & 0xFFFFFFFFFFFFFull;

    if (exponent == 0 && mantissa == 0) {
        return {0, 0, sign, false, false};
    } else if (bias - 22 <= exponent && exponent <= bias + 52) {
        const int e = exponent - bias;
        const unsigned k = (e * 179 + 4084) >> 8;
        if (0 == (mantissa & (0xFFFFFFFFFFFFFull >> k))) {
            auto res = trivial_double_dec(mantissa, e, k);
            return {res.m10, res.e10, sign, false, false};
        }
     } else if (exponent == 0x7FF) {
        if (mantissa == 0) {
            return {0, 0, sign, true, false};
        } else {
            return {0, 0, sign, false, true};
        }
    }
    auto ddec = detail::ryu::d2d(mantissa, exponent);
    return {ddec.mantissa, ddec.exponent, sign, false, false};
}

#else  // ! defined(STRF_OMIT_IMPL)

detail::double_dec decode(double d);
detail::double_dec decode(float f);

#endif // ! defined(STRF_OMIT_IMPL)

} // namespace detail

enum class float_notation{fixed, scientific, general};

struct float_format_data
{
    unsigned precision = (unsigned)-1;
    strf::float_notation notation = float_notation::general;
    bool showpoint = false;
    bool showpos = false;
};

constexpr STRF_HD bool operator==( strf::float_format_data lhs
                                 , strf::float_format_data rhs ) noexcept
{
    return lhs.precision == rhs.precision
        && lhs.notation == rhs.notation
        && lhs.showpoint == rhs.showpoint
        && lhs.showpos == rhs.showpos ;
}

constexpr STRF_HD bool operator!=( strf::float_format_data lhs
                                 , strf::float_format_data rhs ) noexcept
{
    return ! (lhs == rhs);
}

struct float_format;

template <typename T>
class float_format_fn
{
public:
    constexpr float_format_fn() noexcept = default;

    template <typename U>
    constexpr STRF_HD explicit float_format_fn(const float_format_fn<U>& other) noexcept
        : _data(other.get_float_format_data())
    {
    }
    constexpr STRF_HD T&& operator+() && noexcept
    {
        _data.showpos = true;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T&& operator~() && noexcept
    {
        _data.showpoint = true;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T&& p(unsigned _) && noexcept
    {
        _data.precision = _;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T&& sci() && noexcept
    {
        _data.notation = float_notation::scientific;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T&& fixed() && noexcept
    {
        _data.notation = float_notation::fixed;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD T&& gen() && noexcept
    {
        _data.notation = float_notation::general;
        return static_cast<T&&>(*this);
    }
    constexpr strf::float_format_data get_float_format_data() const noexcept
    {
        return _data;
    }

private:

    strf::float_format_data _data;
};

struct float_format
{
    template <typename T>
    using fn = float_format_fn<T>;
};

template<typename FloatT, bool Align = false>
using float_with_format = value_with_format
    < FloatT
    , strf::float_format
    , strf::alignment_format_q<Align> >;

namespace detail {

struct double_printer_data
{
    unsigned m10_digcount;
    unsigned extra_zeros;
    std::uint64_t m10;
    std::int32_t e10;
    bool negative;
    bool infinity;
    bool nan;
    bool showpoint;
    bool showsign;
    bool sci_notation;
};


STRF_HD double_printer_data init_double_printer_data
    ( detail::double_dec d, float_format_data fmt );

inline STRF_HD double_printer_data init_double_printer_data
    ( float f, float_format_data fmt )
{
    return init_double_printer_data(detail::decode(f), fmt);
}

inline STRF_HD double_printer_data init_double_printer_data
    ( double d, float_format_data fmt )
{
    return init_double_printer_data(detail::decode(d), fmt);
}

#if !defined(STRF_OMIT_IMPL)

STRF_INLINE STRF_HD double_printer_data init_double_printer_data
    ( detail::double_dec dd, float_format_data fmt )
{
    double_printer_data data;
    std::memcpy(&data.m10, &dd, sizeof(dd));
    data.showsign = fmt.showpos || data.negative;

    if (data.nan || data.infinity) {
        data.showpoint = false;
        data.sci_notation = false;
        data.m10_digcount = 0;
        data.extra_zeros = 0;
    } else if (fmt.precision == (unsigned)-1) {
        data.m10_digcount = strf::detail::count_digits<10>(data.m10);
        data.extra_zeros = 0;
        switch(fmt.notation) {
            case float_notation::general: {
                data.sci_notation
                    = (data.e10 > 4 + (!fmt.showpoint && data.m10_digcount != 1))
                   || (data.e10 < ( -(int)data.m10_digcount - 2
                                   - (fmt.showpoint || data.m10_digcount != 1) ));
                data.showpoint = fmt.showpoint
                        || (data.sci_notation && data.m10_digcount != 1)
                        || (!data.sci_notation && data.e10 < 0);
                break;
            }
            case float_notation::fixed: {
                data.sci_notation = false;
                data.showpoint = fmt.showpoint || (data.e10 < 0);
                break;
            }
            case float_notation::scientific: {
                data.sci_notation = true;
                data.showpoint = fmt.showpoint || (data.m10_digcount != 1);
                break;
            }
        }
    } else {
        data.m10_digcount = strf::detail::count_digits<10>(data.m10);
        int xz; // number of zeros to be added or ( if negative ) digits to be removed
        switch(fmt.notation) {
            case float_notation::general: {
                int p = fmt.precision + (fmt.precision == 0);
                int sci_notation_exp = data.e10 + (int)data.m10_digcount - 1;
                data.sci_notation = (sci_notation_exp < -4 || sci_notation_exp >= p);
                data.showpoint = fmt.showpoint
                    || (data.sci_notation && data.m10_digcount != 1)
                    || (!data.sci_notation && data.e10 < 0);
                xz = ((unsigned)p < data.m10_digcount || fmt.showpoint)
                   * (p - (int)data.m10_digcount);
                break;
            }
            case float_notation::fixed: {
                const int frac_digits = (data.e10 < 0) * -data.e10;
                xz = (fmt.precision - frac_digits);
                data.sci_notation = false;
                data.showpoint = fmt.showpoint || (fmt.precision != 0);
                break;
            }
            default: {
                STRF_ASSERT(fmt.notation == float_notation::scientific);
                const unsigned frac_digits = data.m10_digcount - 1;
                xz = (fmt.precision - frac_digits);
                data.sci_notation = true;
                data.showpoint = fmt.showpoint || (fmt.precision != 0);
                break;
            }
        }
        if (xz < 0) {
            data.extra_zeros = 0;
            unsigned dp = -xz;
            data.m10_digcount -= dp;
            data.e10 += dp;
            auto p10 = strf::detail::pow10(dp);
            auto remainer = data.m10 % p10;
            data.m10 = data.m10 / p10;
            auto middle = p10 >> 1;
            data.m10 += (remainer > middle || (remainer == middle && (data.m10 & 1) == 1));
            if (fmt.notation == float_notation::general) {
                while (data.m10 % 10 == 0) {
                    data.m10 /= 10;
                    -- data.m10_digcount;
                    ++ data.e10;
                }
                int frac_digits = data.sci_notation * (data.m10_digcount - 1)
                                - !data.sci_notation * (data.e10 < 0) * data.e10;
                data.showpoint = fmt.showpoint || (frac_digits != 0);
            }
         } else {
            data.extra_zeros = xz;
        }
    }
    return data;
}

#endif // !defined(STRF_OMIT_IMPL)

template <int Base, std::size_t CharSize, typename IntT>
inline STRF_HD void write_int_with_leading_zeros
    ( strf::underlying_outbuf<CharSize>& ob
    , IntT value
    , unsigned digcount
    , strf::lettercase lc )
{
    ob.ensure(digcount);
    auto p = ob.pos();
    auto end = p + digcount;
    using writer = detail::intdigits_backwards_writer<Base>;
    auto p2 = writer::write_txtdigits_backwards(value, end, lc);
    if (p != p2) {
        strf::detail::str_fill_n(p, p2 - p, '0');
    }
    ob.advance_to(end);
}

template <std::size_t CharSize>
STRF_HD void print_amplified_integer_small_separator
    ( strf::underlying_outbuf<CharSize>& ob
    , const strf::numpunct_base& punct
    , unsigned long long value
    , unsigned num_digits
    , unsigned num_trailing_zeros
    , strf::underlying_outbuf_char_type<CharSize> separator )
{
    STRF_ASSERT( ! punct.no_group_separation(num_trailing_zeros + num_digits));

    constexpr std::size_t size_after_recycle = strf::min_size_after_recycle
        <strf::underlying_outbuf_char_type<CharSize>>();
    (void) size_after_recycle;

    constexpr auto max_digits = detail::max_num_digits<unsigned long long, 10>;
    char digits_buff[max_digits];
    auto digits = strf::detail::write_int_dec_txtdigits_backwards
        (value, digits_buff + max_digits);

    std::uint8_t groups[std::numeric_limits<double>::max_exponent10 + 1];
    auto num_groups = punct.groups(num_trailing_zeros + num_digits, groups);

    auto grp_it = groups + num_groups - 1;
    unsigned grp_size = *grp_it;
    while (num_digits > grp_size) {
        STRF_ASSERT(grp_size + 1 <= size_after_recycle);
        ob.ensure(grp_size + 1);
        auto it = ob.pos();
        strf::detail::copy_n(digits, grp_size, it);
        it[grp_size] = separator;
        digits += grp_size;
        ob.advance(grp_size + 1);
        num_digits -= grp_size;
        STRF_ASSERT(grp_it != groups);
        grp_size = *--grp_it;
    }
    if (num_digits != 0) {
        STRF_ASSERT(num_digits <= size_after_recycle);
        ob.ensure(num_digits);
        strf::detail::copy_n(digits, num_digits, ob.pos());
        ob.advance(num_digits);
    }
    if (grp_size > num_digits) {
        STRF_ASSERT(num_digits <= size_after_recycle);
        grp_size -= num_digits;
        ob.ensure(grp_size);
        strf::detail::str_fill_n(ob.pos(), grp_size, '0');
        ob.advance(grp_size);
    }
    while (grp_it != groups) {
        grp_size = *--grp_it;
        STRF_ASSERT(grp_size + 1 <= size_after_recycle);
        ob.ensure(grp_size + 1);
        auto it = ob.pos();
        *it = separator;
        strf::detail::str_fill_n(it + 1, grp_size, '0');
        ob.advance(grp_size + 1);
    }
}

template <std::size_t CharSize>
STRF_HD void print_amplified_integer_big_separator
    ( strf::underlying_outbuf<CharSize>& ob
    , strf::encode_char_func<CharSize> encode_char
    , const strf::numpunct_base& punct
    , unsigned long long value
    , unsigned num_digits
    , unsigned num_trailing_zeros
    , unsigned separator_size )
{
    STRF_ASSERT( ! punct.no_group_separation(num_trailing_zeros + num_digits));
    STRF_ASSERT(separator_size > 1);

    constexpr std::size_t size_after_recycle = strf::min_size_after_recycle
        <strf::underlying_outbuf_char_type<CharSize>>();
    (void) size_after_recycle;

    constexpr auto max_digits = detail::max_num_digits<unsigned long long, 10>;
    char digits_buff[max_digits];
    auto digits = strf::detail::write_int_dec_txtdigits_backwards
        (value, digits_buff + max_digits);

    std::uint8_t groups[std::numeric_limits<double>::max_exponent10 + 1];
    auto num_groups = punct.groups(num_trailing_zeros + num_digits, groups);

    auto grp_it = groups + num_groups - 1;
    unsigned grp_size = *grp_it;
    char32_t separator = punct.thousands_sep();
    while (num_digits > grp_size) {
        STRF_ASSERT(grp_size + separator_size <= size_after_recycle);
        ob.ensure(grp_size + separator_size);
        auto it = ob.pos();
        strf::detail::copy_n(digits, grp_size, it);
        digits += grp_size;
        ob.advance_to(encode_char(it + grp_size, separator));
        num_digits -= grp_size;
        STRF_ASSERT(grp_it != groups);
        grp_size = *--grp_it;
    }
    if (num_digits != 0) {
        STRF_ASSERT(num_digits <= size_after_recycle);
        ob.ensure(num_digits);
        strf::detail::copy_n(digits, num_digits, ob.pos());
        ob.advance(num_digits);
    }
    if (grp_size > num_digits) {
        STRF_ASSERT(num_digits <= size_after_recycle);
        grp_size -= num_digits;
        ob.ensure(grp_size);
        strf::detail::str_fill_n(ob.pos(), grp_size, '0');
        ob.advance(grp_size);
    }
    while (grp_it != groups) {
        grp_size = *--grp_it;
        STRF_ASSERT(grp_size + separator_size <= size_after_recycle);
        ob.ensure(grp_size + separator_size);
        auto it = encode_char(ob.pos(), separator);
        strf::detail::str_fill_n(it, grp_size, '0');
        ob.advance_to(it + grp_size);
    }
}

template <std::size_t CharSize>
STRF_HD void print_scientific_notation
    ( strf::underlying_outbuf<CharSize>& ob
    , strf::encode_char_func<CharSize> encode_char
    , unsigned long long digits
    , unsigned num_digits
    , char32_t decimal_point
    , unsigned decimal_point_size
    , int exponent
    , bool print_point
    , unsigned trailing_zeros
    , strf::lettercase lc )
{
    // digits
    using char_type = strf::underlying_outbuf_char_type<CharSize>;

    print_point |= num_digits != 1;
    ob.ensure(num_digits + print_point * decimal_point_size);
    if (num_digits == 1) {
        auto it = ob.pos();
        *it = static_cast<char_type>('0' + digits);
        ++it;
        if (print_point) {
            if (decimal_point_size == 1) {
                *it++ = static_cast<char_type>(decimal_point);
            } else {
                it = encode_char(it, decimal_point);
            }
        }
        ob.advance_to(it);
    } else {
       auto it = ob.pos();
       auto end = it + num_digits + decimal_point_size;
       *it = *write_int_dec_txtdigits_backwards(digits, end);
       ++it;
       if (decimal_point_size == 1) {
           *it++ = static_cast<char_type>(decimal_point);
       } else {
           encode_char(it, decimal_point);
       }
       ob.advance_to(end);
    }

    // extra trailing zeros

    if (trailing_zeros != 0) {
        strf::detail::write_fill(ob, trailing_zeros, char_type('0'));
    }

    // exponent

    unsigned adv = 4;
    char_type* it;
    unsigned e10u = std::abs(exponent);
    STRF_ASSERT(e10u < 1000);

    if (e10u >= 100) {
        ob.ensure(5);
        it = ob.pos();
        it[4] = static_cast<char_type>('0' + e10u % 10);
        e10u /= 10;
        it[3] = static_cast<char_type>('0' + e10u % 10);
        it[2] = static_cast<char_type>('0' + e10u / 10);
        adv = 5;
    } else if (e10u >= 10) {
        ob.ensure(4);
        it = ob.pos();
        it[3] = static_cast<char_type>('0' + e10u % 10);
        it[2] = static_cast<char_type>('0' + e10u / 10);
    } else {
        ob.ensure(4);
        it = ob.pos();
        it[3] = static_cast<char_type>('0' + e10u);
        it[2] = '0';
    }
    it[0] = 'E' | ((lc != strf::uppercase) << 5);
    it[1] = static_cast<char_type>('+' + ((exponent < 0) << 1));
    ob.advance(adv);
}

template <std::size_t CharSize>
STRF_HD void print_nan(strf::underlying_outbuf<CharSize>& ob, strf::lettercase lc)
{
    ob.ensure(3);
    auto p = ob.pos();
    switch (lc) {
        case strf::mixedcase:
            p[0] = 'N';
            p[1] = 'a';
            p[2] = 'N';
            break;
        case strf::uppercase:
            p[0] = 'N';
            p[1] = 'A';
            p[2] = 'N';
            break;
        default:
            p[0] = 'n';
            p[1] = 'a';
            p[2] = 'n';
    }
    ob.advance(3);
}

template <std::size_t CharSize>
STRF_HD void print_inf(strf::underlying_outbuf<CharSize>& ob, strf::lettercase lc)
{
    ob.ensure(3);
    auto p = ob.pos();
    switch (lc) {
        case strf::mixedcase:
            p[0] = 'I';
            p[1] = 'n';
            p[2] = 'f';
            break;
        case strf::uppercase:
            p[0] = 'I';
            p[1] = 'N';
            p[2] = 'F';
            break;
        default:
            p[0] = 'i';
            p[1] = 'n';
            p[2] = 'f';
    }
    ob.advance(3);
}

template <std::size_t CharSize>
STRF_HD void print_inf( strf::underlying_outbuf<CharSize>& ob
                      , strf::lettercase lc
                      , bool negative )
{
    ob.ensure(3 + negative);
    auto p = ob.pos();
    if (negative) {
        *p ++ = '-';
    }
    switch (lc) {
        case strf::mixedcase:
            *p++ = 'I';
            *p++ = 'n';
            *p++ = 'f';
            break;
        case strf::uppercase:
            *p++ = 'I';
            *p++ = 'N';
            *p++ = 'F';
            break;
        default:
            *p++ = 'i';
            *p++ = 'n';
            *p++ = 'f';
    }
    ob.advance_to(p);
}

template <std::size_t CharSize>
class punct_double_printer: public strf::printer<CharSize>
{
public:

    using char_type = strf::underlying_outbuf_char_type<CharSize>;

    template <typename FP, typename Preview, typename FloatT, typename CharT>
    STRF_HD punct_double_printer
        ( const FP& fp
        , Preview& preview
        , strf::float_with_format<FloatT, false> x
        , strf::tag<CharT> )
        : _punct(strf::get_facet<strf::numpunct_c<10>, FloatT>(fp))
        , _enc_err(strf::get_facet<strf::encoding_error_c, FloatT>(fp))
        , _allow_surr(strf::get_facet<strf::surrogate_policy_c, FloatT>(fp))
        , _lettercase(strf::get_facet<strf::lettercase_c, FloatT>(fp))
    {
        const auto fmt = x.get_float_format_data();
        _data = strf::detail::init_double_printer_data(x.value(), fmt);
        decltype(auto) enc = get_facet<strf::encoding_c<CharT>, FloatT>(fp);
        init_(enc, fmt.notation == float_notation::general, fmt.showpoint);
        STRF_IF_CONSTEXPR (Preview::width_required) {
            preview.subtract_width(_content_width());
        }
        STRF_IF_CONSTEXPR (Preview::size_required) {
            preview.add_size(_content_size());
        }
    }

    template <typename FP, typename Preview, typename FloatT, typename CharT>
    STRF_HD punct_double_printer
        ( const FP& fp
        , Preview& preview
        , strf::float_with_format<FloatT, true> x
        , strf::tag<CharT> )
        : _punct(strf::get_facet<strf::numpunct_c<10>, FloatT>(fp))
        , _fillchar(x.fill())
        , _enc_err(strf::get_facet<strf::encoding_error_c, FloatT>(fp))
        , _allow_surr(strf::get_facet<strf::surrogate_policy_c, FloatT>(fp))
        , _lettercase(strf::get_facet<strf::lettercase_c, FloatT>(fp))
    {
        const auto fmt = x.get_float_format_data();
        _data = strf::detail::init_double_printer_data(x.value(), fmt);
        decltype(auto) enc = get_facet<strf::encoding_c<CharT>, FloatT>(fp);
        init_( enc, fmt.notation == float_notation::general, fmt.showpoint);
        init_(preview, x.width(), x.alignment(), enc);
    }


    STRF_HD void print_to(strf::underlying_outbuf<CharSize>&) const override;

private:

    template <typename Encoding>
    STRF_HD void init_
        (const Encoding& enc, bool fmt_general_format, bool fmt_showpoint);

    template <typename Preview, typename Encoding>
    STRF_HD void init_
        ( Preview& preview, std::int16_t w, strf::text_alignment a
        , const Encoding& encoding );

    STRF_HD std::int16_t _content_width() const;
    STRF_HD std::size_t _content_size() const;

    const strf::numpunct_base& _punct;
    strf::encode_char_func<CharSize> _encode_char;
    strf::encode_fill_func<CharSize> _encode_fill;
    char32_t _fillchar = U' ';
    unsigned _left_fillcount = 0;
    unsigned _split_fillcount = 0;
    unsigned _right_fillcount = 0;
    unsigned _sep_count = 0;
    unsigned _sep_size = 0;
    unsigned _decimal_point_size = 0;
    char32_t _decimal_point;
    char_type _little_sep;
    strf::encoding_error _enc_err;
    strf::surrogate_policy _allow_surr = surrogate_policy::strict;
    strf::lettercase _lettercase;
    strf::detail::double_printer_data _data;
};

template <std::size_t CharSize>
template <typename Encoding>
STRF_HD void punct_double_printer<CharSize>::init_
    ( const Encoding& enc, bool general_format, bool fmt_showpoint)
{
    _encode_char = enc.encode_char;
    _encode_fill = enc.encode_fill;
    if (!_data.sci_notation) {
        auto int_dig_count = (int)_data.m10_digcount + _data.e10;
        if (! _punct.no_group_separation(int_dig_count)) {
            auto sep_validation = enc.validate(_punct.thousands_sep());
            if (sep_validation != strf::invalid_char_len) {
                _sep_size = sep_validation;
                _sep_count = _punct.thousands_sep_count(int_dig_count);
                if (general_format) {
                    bool e10neg = _data.e10 < 0;
                    int fixed_width = _data.e10 * !e10neg
                        + (fmt_showpoint || e10neg) + (int)_sep_count;
                    int scientific_width = 5 + (_data.e10 > 99); // assuming dec point
                    if (scientific_width < fixed_width) {
                        _data.sci_notation = true;
                        _data.showpoint |= _data.m10_digcount != 1;
                        _sep_count = 0;
                        _sep_size = 0;
                        goto init_decimal_point;
                    }
                }
                if (_sep_size == 1) {
                    enc.encode_char(&_little_sep, _punct.thousands_sep());
                }
            }
        }
    }
    init_decimal_point:
    if (_data.showpoint) {

        _decimal_point = _punct.decimal_point();
        auto validation = enc.validate(_decimal_point);
        if (validation == 1) {
            _decimal_point_size = 1;
            char_type ch;
            enc.encode_char(&ch, _decimal_point);
            _decimal_point = ch;
        } else if (validation != strf::invalid_char_len) {
            _decimal_point_size = validation;
        } else {
            _decimal_point_size = enc.replacement_char_size();
            _decimal_point = enc.replacement_char();
        }
    }
}

template <std::size_t CharSize>
template <typename Preview, typename Encoding>
STRF_HD void punct_double_printer<CharSize>::init_
    ( Preview& preview, std::int16_t fmt_width, strf::text_alignment a
    , const Encoding& encoding )
{
    (void) encoding;
    auto content_width = _content_width();
    if (content_width >= fmt_width) {
        preview.subtract_width(content_width);
        STRF_IF_CONSTEXPR (Preview::size_required) {
            preview.add_size(_content_size());
        }
    } else {
        auto fillcount = fmt_width - content_width;
        preview.subtract_width(fmt_width);
        STRF_IF_CONSTEXPR (Preview::size_required) {
            std::size_t fillsize = encoding.validate(_fillchar);
            if (fillsize == (size_t)-1) {
                fillsize = encoding.replacement_char_size();
            }
            preview.add_size(_content_size() + fillsize * fillcount);
        }
        switch (a) {
            case strf::text_alignment::right:
                _left_fillcount = fillcount;
                break;
            case strf::text_alignment::left:
                _right_fillcount = fillcount;
                break;
            case strf::text_alignment::split:
                _split_fillcount = fillcount;
                break;
            default:
                STRF_ASSERT(a == strf::text_alignment::center);
                _left_fillcount = fillcount / 2;
                _right_fillcount = fillcount - _left_fillcount;
        }
    }
}

template <std::size_t CharSize>
STRF_HD std::int16_t punct_double_printer<CharSize>::_content_width() const
{
    int decpoint_width = _data.showpoint;
    unsigned w = 0;
    if (_data.infinity || _data.nan) {
        w = 3 + _data.showsign;
    } else if (_data.sci_notation) {
        unsigned e10u = std::abs(_data.e10 + (int)_data.m10_digcount - 1);
        w = _data.m10_digcount + _data.extra_zeros
            + _data.showsign
            + (e10u < 10) + 2
            + detail::count_digits<10>(e10u)
            + decpoint_width;
    } else {
        if (_data.e10 <= -(int)_data.m10_digcount) {
            w = _data.showsign + 1 + decpoint_width
                    - _data.e10 + _data.extra_zeros;
        } else {
            auto idigcount = (int)_data.m10_digcount + _data.e10;
            if (_data.e10 < 0) {
                    w = _data.showsign
                        + (int)_data.m10_digcount
                        + _data.extra_zeros
                        + 1 // decpoint_width
                        + _sep_count;
            } else {
                w = _data.showsign
                    + idigcount
                    + _data.extra_zeros
                    + _data.showpoint
                    + _sep_count;
            }
        }
    }
    return static_cast<std::int16_t>(w);
}

template <std::size_t CharSize>
STRF_HD std::size_t punct_double_printer<CharSize>::_content_size() const
{
    if (_data.infinity || _data.nan) {
        return 3 + _data.showsign;
    }
    if (_data.sci_notation) {
        unsigned e10u = std::abs(_data.e10 + (int)_data.m10_digcount - 1);
        return _data.m10_digcount + _data.extra_zeros
            + _data.showsign
            + (e10u < 10) + 2
            + detail::count_digits<10>(e10u)
            + _decimal_point_size;
    }
    if (_data.e10 <= -(int)_data.m10_digcount) {
        return 1 + _data.showsign + _decimal_point_size
            + (-_data.e10) +_data.extra_zeros;
    }
    return _data.showsign + _sep_count * _sep_size + _decimal_point_size
        + _data.m10_digcount + _data.extra_zeros + (_data.e10 > 0) * _data.e10;
}

template <std::size_t CharSize>
STRF_HD void punct_double_printer<CharSize>::print_to
    (strf::underlying_outbuf<CharSize>& ob) const
{
    if (_left_fillcount != 0) {
        _encode_fill(ob, _left_fillcount, _fillchar, _enc_err, _allow_surr);
    }
    if (_data.showsign) {
        put(ob, static_cast<char_type>('+' + (_data.negative << 1)));
    }
    if (_split_fillcount != 0) {
        _encode_fill(ob, _split_fillcount, _fillchar, _enc_err, _allow_surr);
    }
    if (_data.nan) {
        strf::detail::print_nan(ob, _lettercase);
    } else if (_data.infinity) {
        strf::detail::print_inf(ob, _lettercase);
    } else if (_data.sci_notation) {
        strf::detail::print_scientific_notation
            ( ob, _encode_char, _data.m10, _data.m10_digcount
            , _decimal_point, _decimal_point_size
            , _data.e10 + _data.m10_digcount - 1
            , _data.showpoint, _data.extra_zeros, _lettercase );
    } else if (_data.e10 >= 0) {
        if (_sep_count == 0) {
            strf::detail::write_int<10>( ob, _data.m10, _data.m10_digcount
                                       , strf::lowercase );
            strf::detail::write_fill(ob, _data.e10, (char_type)'0');
        } else if (_sep_size == 1) {
            strf::detail::print_amplified_integer_small_separator
                ( ob, _punct, _data.m10, _data.m10_digcount, _data.e10
                , _little_sep );
        } else {
            strf::detail::print_amplified_integer_big_separator
                ( ob, _encode_char, _punct, _data.m10
                , _data.m10_digcount, _data.e10, _sep_size );
        }
        if (_decimal_point_size == 1) {
            strf::put(ob, static_cast<char_type>(_decimal_point));
        } else if (_decimal_point_size != 0) {
            ob.ensure(_decimal_point_size);
            ob.advance_to(_encode_char(ob.pos(), _decimal_point));
        }
        if (_data.extra_zeros) {
            detail::write_fill(ob, _data.extra_zeros,  (char_type)'0');
        }
    } else {
        STRF_ASSERT(_data.e10 < 0);

        unsigned e10u = - _data.e10;
        if (e10u >= _data.m10_digcount) {
            ob.ensure(1 + _decimal_point_size);
            auto it = ob.pos();
            *it++ = static_cast<char_type>('0');
            if (_decimal_point_size == 1) {
                *it++ = static_cast<char_type>(_decimal_point);
            } else {
                STRF_ASSERT(_decimal_point_size != 0);
                it = _encode_char(it, _decimal_point);
            }
            ob.advance_to(it);

            if (e10u > _data.m10_digcount) {
                strf::detail::write_fill(ob, e10u - _data.m10_digcount, (char_type)'0');
            }
            strf::detail::write_int<10>( ob, _data.m10, _data.m10_digcount
                                       , strf::lowercase);
            if (_data.extra_zeros != 0) {
                strf::detail::write_fill(ob, _data.extra_zeros,  (char_type)'0');
            }
        } else {
            //auto v = std::lldiv(_data.m10, detail::pow10(e10u)); // todo test this
            auto p10 = strf::detail::pow10(e10u);
            auto integral_part = _data.m10 / p10;
            auto fractional_part = _data.m10 % p10;
            auto idigcount = _data.m10_digcount - e10u;

            STRF_ASSERT(idigcount == detail::count_digits<10>(integral_part));

            if (_sep_count == 0) {
                strf::detail::write_int<10>(ob, integral_part, idigcount, strf::lowercase);
            } else if (_sep_size == 1) {
                strf::detail::write_int_little_sep<10>
                    ( ob, _punct, integral_part, idigcount, _little_sep);
            } else {
                strf::detail::write_int_big_sep<10>
                    (  ob, _punct, _encode_char, integral_part
                    , _sep_size, idigcount );
            }

            ob.ensure(_decimal_point_size);
            auto it = ob.pos();
            if (_decimal_point_size == 1) {
                *it++ = static_cast<char_type>(_decimal_point);
            } else {
                STRF_ASSERT(_decimal_point_size != 0);
                it = _encode_char(it, _decimal_point);
            }
            ob.advance_to(it);

            strf::detail::write_int_with_leading_zeros<10>
                (ob, fractional_part, e10u, strf::lowercase);
            if (_data.extra_zeros) {
                detail::write_fill(ob, _data.extra_zeros,  (char_type)'0');
            }
        }
    }
    if (_right_fillcount != 0) {
        _encode_fill(ob, _right_fillcount, _fillchar, _enc_err, _allow_surr);
    }
}

template <std::size_t CharSize>
class double_printer final: public strf::printer<CharSize>
{
public:

    using char_type = strf::underlying_outbuf_char_type<CharSize>;

    template <typename Fpack, typename Preview, typename FloatT, typename CharT>
    STRF_HD double_printer
        ( const Fpack& fp
        , Preview& preview
        , strf::float_with_format<FloatT, false> x
        , strf::tag<CharT> )
        : _data(strf::detail::init_double_printer_data
                (x.value(), x.get_float_format_data()))
        , _lettercase(strf::get_facet<strf::lettercase_c, FloatT>(fp))
    {
        auto content_width = _content_width();
        preview.subtract_width(content_width);
        preview.add_size(content_width);
    }

    template <typename Fpack, typename Preview, typename FloatT, typename CharT>
    STRF_HD double_printer
        ( const Fpack& fp
        , Preview& preview
        , strf::float_with_format<FloatT, true> x
        , strf::tag<CharT>)
        : _data(strf::detail::init_double_printer_data
                (x.value(), x.get_float_format_data()))
        , _fillchar(x.fill())
        , _enc_err(strf::get_facet<strf::encoding_error_c, FloatT>(fp))
        , _allow_surr(strf::get_facet<strf::surrogate_policy_c, FloatT>(fp))
        , _lettercase(strf::get_facet<strf::lettercase_c, FloatT>(fp))
    {
        decltype(auto) enc = strf::get_facet<strf::encoding_c<CharT>, FloatT>(fp);
        _init(preview, x.width(), x.alignment(), enc);
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>&) const override;

private:

    template <typename Preview, typename Encoding>
    STRF_HD void _init( Preview& preview, std::int16_t w, strf::text_alignment a
                      , const Encoding& enc );

    STRF_HD std::int16_t _content_width() const
    {
        return static_cast<std::int16_t>
               ( _data.nan * 3
               + _data.infinity * 3
               + _data.showsign
               + !(_data.infinity | _data.nan)
               * ( _data.extra_zeros
                 + _data.showpoint
                 + _data.m10_digcount
                 + ( _data.sci_notation
                   * ( 4 + ((_data.e10 > 99) || (_data.e10 < -99))) )
                 + ( !_data.sci_notation
                   * ( (0 <= _data.e10)
                     * _data.e10
                     + (_data.e10 <= -(int)_data.m10_digcount)
                       * (-_data.e10 + 1 -(int)_data.m10_digcount) ))));
    }

    strf::detail::double_printer_data _data;
    strf::encode_fill_func<CharSize> _encode_fill;
    char32_t _fillchar = U' ';
    unsigned _left_fillcount = 0;
    unsigned _split_fillcount = 0;
    unsigned _right_fillcount = 0;
    strf::encoding_error _enc_err = encoding_error::replace;
    strf::surrogate_policy _allow_surr = surrogate_policy::strict;
    strf::lettercase _lettercase;
};

template <std::size_t CharSize>
template <typename Preview, typename Encoding>
STRF_HD void double_printer<CharSize>::_init
    ( Preview& preview, std::int16_t w, strf::text_alignment a
    , const Encoding& enc )
{
    _encode_fill = enc.encode_fill;
    auto content_width = _content_width();
    if (content_width >= w) {
        preview.checked_subtract_width(content_width);
        preview.add_size(content_width);
    } else {
        auto fillcount = (w - static_cast<std::int16_t>(content_width));
        preview.subtract_width(w);
        STRF_IF_CONSTEXPR(Preview::size_required) {
            std::size_t fillchar_size = enc.validate(_fillchar);
            if (fillchar_size == (size_t)-1) {
                fillchar_size = enc.replacement_char_size();
            }
            preview.add_size(content_width + fillchar_size * fillcount);
        }
        switch (a) {
            case strf::text_alignment::right:
                _left_fillcount = fillcount;
                break;
            case strf::text_alignment::left:
                _right_fillcount = fillcount;
                break;
            case strf::text_alignment::split:
                _split_fillcount = fillcount;
                break;
            default:
                STRF_ASSERT(a == strf::text_alignment::center);
                _left_fillcount = fillcount / 2;
                _right_fillcount = fillcount - _left_fillcount;
        }
    }
}

template <std::size_t CharSize>
STRF_HD void double_printer<CharSize>::print_to
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (_left_fillcount != 0) {
        _encode_fill(ob, _left_fillcount, _fillchar, _enc_err, _allow_surr);
    }
    if (_data.showsign) {
        put<CharSize>(ob, '+' + (_data.negative << 1));
    }
    if (_split_fillcount != 0) {
        _encode_fill(ob, _split_fillcount, _fillchar, _enc_err, _allow_surr);
    }
    if (_data.nan) {
        strf::detail::print_nan(ob, _lettercase);
    } else if (_data.infinity) {
        strf::detail::print_inf(ob, _lettercase);
    } else if (_data.sci_notation) {
        ob.ensure( _data.m10_digcount
                 + _data.showpoint
                 + 4 + (_data.e10 > 99 || _data.e10 < -99) );
        char_type* it = ob.pos();
        if (_data.m10_digcount == 1) {
            * it = static_cast<char_type>('0' + _data.m10);
            ++it;
            if (_data.showpoint) {
                *it = '.';
                ++it;
            }
            if (_data.extra_zeros > 0) {
                ob.advance_to(it);
                strf::detail::write_fill<CharSize>(ob, _data.extra_zeros, '0');
                it = ob.pos();
            }
        } else {
            auto itz = it + _data.m10_digcount + 1;
            write_int_dec_txtdigits_backwards(_data.m10, itz);
            it[0] = it[1];
            it[1] = '.';
            it = itz;
            if (_data.extra_zeros > 0) {
                ob.advance_to(itz);
                strf::detail::write_fill<CharSize>(ob, _data.extra_zeros, '0');
                it = ob.pos();
            }
        }
        auto e10 = _data.e10 - 1 + (int)_data.m10_digcount;
        it[0] = 'E' | ((_lettercase != strf::uppercase) << 5);
        it[1] = static_cast<char_type>('+' + ((e10 < 0) << 1));
        unsigned e10u = std::abs(e10);
        if (e10u >= 100) {
            it[4] = static_cast<char_type>('0' + e10u % 10);
            e10u /= 10;
            it[3] = static_cast<char_type>('0' + e10u % 10);
            it[2] = static_cast<char_type>('0' + e10u / 10);
            it += 5;
        } else if (e10u >= 10) {
            it[3] = static_cast<char_type>('0' + e10u % 10);
            it[2] = static_cast<char_type>('0' + e10u / 10);
            it += 4;
        } else {
            it[3] = static_cast<char_type>('0' + e10u);
            it[2] = '0';
            it += 4;
        }
        ob.advance_to(it);
    } else {
        ob.ensure( _data.showpoint + _data.m10_digcount
                 + (_data.e10 < -(int)_data.m10_digcount) );
        auto it = ob.pos();
        if (_data.e10 >= 0) {
            it += _data.m10_digcount;
            write_int_dec_txtdigits_backwards(_data.m10, it);
            ob.advance_to(it);
            detail::write_fill(ob, _data.e10, (char_type)'0');
            if (_data.showpoint) {
                ob.ensure(1);
                *ob.pos() = '.';
                ob.advance();
            }
            detail::write_fill(ob, _data.extra_zeros, (char_type)'0');
        } else {
            unsigned e10u = - _data.e10;
            if (e10u >= _data.m10_digcount) {
                it[0] = '0';
                it[1] = '.';
                ob.advance_to(it + 2);
                detail::write_fill(ob, e10u - _data.m10_digcount, (char_type)'0');

                ob.ensure(_data.m10_digcount);
                auto end = ob.pos() + _data.m10_digcount;
                write_int_dec_txtdigits_backwards(_data.m10, end);
                ob.advance_to(end);
                detail::write_fill(ob, _data.extra_zeros, (char_type)'0');
            } else {
                const char* const arr = strf::detail::chars_00_to_99();
                auto m = _data.m10;
                char_type* const end = it + _data.m10_digcount + 1;
                it = end;
                while(e10u >= 2) {
                    auto index = (m % 100) << 1;
                    it[-2] = arr[index];
                    it[-1] = arr[index + 1];
                    it -= 2;
                    m /= 100;
                    e10u -= 2;
                }
                if (e10u != 0) {
                    *--it = static_cast<char_type>('0' + (m % 10));
                    m /= 10;
                }
                * --it = '.';
                while(m > 99) {
                    auto index = (m % 100) << 1;
                    it[-2] = arr[index];
                    it[-1] = arr[index + 1];
                    it -= 2;
                    m /= 100;
                }
                if (m > 9) {
                    auto index = m << 1;
                    it[-2] = arr[index];
                    it[-1] = arr[index + 1];
                } else {
                    *--it = static_cast<char_type>('0' + m);
                }
                ob.advance_to(end);
                detail::write_fill(ob, _data.extra_zeros, (char_type)'0');
            }
        }
    }
    if (_right_fillcount != 0) {
        _encode_fill(ob, _right_fillcount, _fillchar, _enc_err, _allow_surr);
    }
}

template <std::size_t CharSize>
class fast_double_printer: public strf::printer<CharSize>
{
public:

    using char_type = strf::underlying_outbuf_char_type<CharSize>;

    template <typename FPack, typename Preview, typename CharT>
    STRF_HD fast_double_printer
        ( const FPack& fp, Preview& preview, float f, strf::tag<CharT>) noexcept
        : fast_double_printer(f, strf::get_facet<strf::lettercase_c, float>(fp))
    {
        std::size_t s = 0;
        STRF_IF_CONSTEXPR (Preview::width_required || Preview::size_required) {
            s = size();
        }
        preview.checked_subtract_width(s);
        preview.add_size(s);
    }

    template <typename FPack, typename Preview, typename CharT>
    STRF_HD fast_double_printer
        ( const FPack& fp, Preview& preview, double d, strf::tag<CharT>) noexcept
        : fast_double_printer(d, strf::get_facet<strf::lettercase_c, double>(fp))
    {
        std::size_t s = 0;
        STRF_IF_CONSTEXPR (Preview::width_required || Preview::size_required) {
            s = size();
        }
        preview.checked_subtract_width(s);
        preview.add_size(s);
    }

    STRF_HD fast_double_printer(float f, strf::lettercase lc) noexcept
        : _value(decode(f))
        , _m10_digcount(strf::detail::count_digits<10>(_value.m10))
        , _lettercase(lc)

    {
        STRF_ASSERT(!_value.nan || !_value.infinity);
        _sci_notation = (_value.e10 > 4 + (_m10_digcount != 1))
            || (_value.e10 < -(int)_m10_digcount - 2 - (_m10_digcount != 1));
    }

    STRF_HD fast_double_printer(double d, strf::lettercase lc) noexcept
        : _value(decode(d))
        , _m10_digcount(strf::detail::count_digits<10>(_value.m10))
        , _lettercase(lc)

    {
        STRF_ASSERT(!_value.nan || !_value.infinity);
        _sci_notation = (_value.e10 > 4 + (_m10_digcount != 1))
            || (_value.e10 < -(int)_m10_digcount - 2 - (_m10_digcount != 1));
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>&) const override;

    STRF_HD std::size_t size() const;

private:

    const detail::double_dec _value;
    bool _sci_notation ;
    const unsigned _m10_digcount;
    strf::lettercase _lettercase;
};

template <std::size_t CharSize>
STRF_HD std::size_t fast_double_printer<CharSize>::size() const
{
    return ( _value.nan * 3
           + (_value.infinity * 3)
           + (_value.negative && !_value.nan)
           + !(_value.infinity | _value.nan)
           * ( ( _sci_notation
               * ( 4 // e+xx
                 + (_m10_digcount != 1) // decimal point
                 + _m10_digcount
                 + ((_value.e10 > 99) || (_value.e10 < -99))) )
             + ( !_sci_notation
               * ( (int)_m10_digcount
                 + (_value.e10 > 0) * _value.e10 // trailing zeros
                 + (_value.e10 <= -(int)_m10_digcount) * (2 -_value.e10 - (int)_m10_digcount) // leading zeros and point
                 + (-(int)_m10_digcount < _value.e10 && _value.e10 < 0) ))));
}

template <std::size_t CharSize>
STRF_HD void fast_double_printer<CharSize>::print_to
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (_value.nan) {
        strf::detail::print_nan(ob, _lettercase);
    } else if (_value.infinity) {
        strf::detail::print_inf(ob, _lettercase, _value.negative);
    } else if (_sci_notation) {
        ob.ensure( _value.negative + _m10_digcount + (_m10_digcount != 1) + 4
                 + (_value.e10 > 99 || _value.e10 < -99) );
        char_type* it = ob.pos();
        if (_value.negative) {
            * it = '-';
            ++it;
        }
        if (_m10_digcount == 1) {
            * it = static_cast<char_type>('0' + _value.m10);
            ++ it;
        } else {
            auto next = it + _m10_digcount + 1;
            write_int_dec_txtdigits_backwards(_value.m10, next);
            it[0] = it[1];
            it[1] = '.';
            it = next;
        }
        auto e10 = _value.e10 - 1 + (int)_m10_digcount;
        it[0] = 'E' | ((_lettercase != strf::uppercase) << 5);
        it[1] = static_cast<char_type>('+' + ((e10 < 0) << 1));
        unsigned e10u = std::abs(e10);
        if (e10u >= 100) {
            it[4] = static_cast<char_type>('0' + e10u % 10);
            e10u /= 10;
            it[3] = static_cast<char_type>('0' + e10u % 10);
            it[2] = static_cast<char_type>('0' + e10u / 10);
            it += 5;
        } else if (e10u >= 10) {
            it[3] = static_cast<char_type>('0' + e10u % 10);
            it[2] = static_cast<char_type>('0' + e10u / 10);
            it += 4;
        } else {
            it[3] = static_cast<char_type>('0' + e10u);
            it[2] = '0';
            it += 4;
        }
        ob.advance_to(it);
    } else {
        ob.ensure( _value.negative
                 + _m10_digcount * (_value.e10 > - (int)_m10_digcount)
                 + (_value.e10 < - (int)_m10_digcount)
                 + (_value.e10 < 0) );
        auto it = ob.pos();
        if (_value.negative) {
            *it = '-';
            ++it;
        }
        if (_value.e10 >= 0) {
            it += _m10_digcount;
            write_int_dec_txtdigits_backwards(_value.m10, it);
            ob.advance_to(it);
            if (_value.e10 != 0) {
                detail::write_fill(ob, _value.e10, (char_type)'0');
            }
        } else {
            unsigned e10u = - _value.e10;
            if (e10u >= _m10_digcount) {
                it[0] = '0';
                it[1] = '.';
                ob.advance_to(it + 2);
                detail::write_fill(ob, e10u - _m10_digcount, (char_type)'0');

                ob.ensure(_m10_digcount);
                auto end = ob.pos() + _m10_digcount;
                write_int_dec_txtdigits_backwards(_value.m10, end);
                ob.advance_to(end);
            } else {
                const char* const arr = strf::detail::chars_00_to_99();
                auto m = _value.m10;
                it += _m10_digcount + 1;
                char_type* const end = it;
                while(e10u >= 2) {
                    auto index = (m % 100) << 1;
                    it[-2] = arr[index];
                    it[-1] = arr[index + 1];
                    it -= 2;
                    m /= 100;
                    e10u -= 2;
                }
                if (e10u != 0) {
                    *--it = static_cast<char_type>('0' + (m % 10));
                    m /= 10;
                }
                * --it = '.';
                while(m > 99) {
                    auto index = (m % 100) << 1;
                    it[-2] = arr[index];
                    it[-1] = arr[index + 1];
                    it -= 2;
                    m /= 100;
                }
                if (m > 9) {
                    auto index = m << 1;
                    it[-2] = arr[index];
                    it[-1] = arr[index + 1];
                } else {
                    *--it = static_cast<char_type>('0' + m);
                }
                ob.advance_to(end);
            }
        }
    }
}


template <std::size_t CharSize>
class fast_punct_double_printer: public strf::printer<CharSize>
{
public:

    using char_type = strf::underlying_outbuf_char_type<CharSize>;

    template <typename FPack, typename Preview, typename FloatT, typename CharT>
    STRF_HD fast_punct_double_printer
        ( const FPack& fp, Preview& preview, FloatT d, strf::tag<CharT> )
        : _punct(strf::get_facet<strf::numpunct_c<10>, FloatT>(fp))
        , _value(decode(d))
        , _m10_digcount(strf::detail::count_digits<10>(_value.m10))
        , _sep_count(0)
        , _lettercase(strf::get_facet<strf::lettercase_c, FloatT>(fp))
    {
        init_(strf::get_facet<strf::encoding_c<CharT>, FloatT>(fp));
        STRF_IF_CONSTEXPR (Preview::width_required) {
            preview.subtract_width(width());
        }
        STRF_IF_CONSTEXPR (Preview::size_required) {
            preview.add_size(size());
        }
    }

    STRF_HD strf::width_t width() const;

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>&) const override;

    STRF_HD std::size_t size() const;

private:

    template <typename Encoding>
    STRF_HD void init_(const Encoding& encoding);

    const strf::numpunct_base& _punct;
    strf::encode_char_func<CharSize> _encode_char;
    const detail::double_dec _value;
    const unsigned _m10_digcount;
    unsigned _sep_count = 0;
    unsigned _sep_size = 0;
    unsigned _decimal_point_size = 0;
    char32_t _decimal_point = '.';
    char_type _little_sep;
    strf::lettercase _lettercase;
    bool _sci_notation ;

};

template <std::size_t CharSize>
template <typename Encoding>
STRF_HD void fast_punct_double_printer<CharSize>::init_(const Encoding& enc)
{
    _encode_char = enc.encode_char;
    bool showpoint;
    if (_value.e10 > -(int)_m10_digcount) {
        bool e10neg = _value.e10 < 0;
        int fixed_width = _value.e10 * !e10neg  + e10neg + (int)_sep_count;
        int scientific_width = 4 + (_value.e10 > 99) + (_m10_digcount != 1);
        if (scientific_width < fixed_width) {
            _sci_notation = true;
            showpoint = _m10_digcount != 1;
        } else {
            auto int_dig_count = (int)_m10_digcount + _value.e10;
            if (! _punct.no_group_separation(int_dig_count)) {
                auto sep_validation = enc.validate(_punct.thousands_sep());
                if (sep_validation != strf::invalid_char_len) {
                    _sep_count = _punct.thousands_sep_count(int_dig_count);
                    if (scientific_width < fixed_width + (int)_sep_count) {
                        _sep_count = 0;
                        _sci_notation = true;
                        showpoint = _m10_digcount != 1;
                        goto init_decimal_point;
                    }
                    _sep_size = sep_validation;
                    if (_sep_size == 1) {
                        _encode_char(&_little_sep, _punct.thousands_sep());
                    }
                }
            }
            showpoint = _value.e10 < 0;
            _sci_notation = false;
        }
    } else {
        _sep_count = 0;
        int tmp = _m10_digcount + 2 + (_value.e10 < -99)
            + (_m10_digcount != 1);
        _sci_notation = -_value.e10 > tmp;
        showpoint = _m10_digcount != 1 || !_sci_notation;
    }
    init_decimal_point:
    if (showpoint) {
        _decimal_point = _punct.decimal_point();
        auto validation = enc.validate(_decimal_point);
        if (validation == 1) {
            _decimal_point_size = 1;
            char_type ch;
            enc.encode_char(&ch, _decimal_point);
            _decimal_point = ch;
        } else if (validation != strf::invalid_char_len) {
            _decimal_point_size = validation;
        } else {
            _decimal_point_size = enc.replacement_char_size();
            _decimal_point = enc.replacement_char();
        }
    }
}


template <std::size_t CharSize>
STRF_HD std::size_t fast_punct_double_printer<CharSize>::size() const
{
    if (_value.infinity || _value.nan) {
        return 3 + (_value.negative && _value.infinity);
    }
    if (_sci_notation) {
        unsigned e10u = std::abs(_value.e10 + (int)_m10_digcount - 1);
        return _m10_digcount
            + _value.negative
            + (e10u < 10) + 2
            + detail::count_digits<10>(e10u)
            + _decimal_point_size;
    }
    if (_value.e10 <= -(int)_m10_digcount) {
        return 1 + _decimal_point_size + (-_value.e10);
    }
    return _sep_count * _sep_size + _decimal_point_size + _m10_digcount
        + _value.negative + (_value.e10 > 0) * _value.e10;
}

template <std::size_t CharSize>
STRF_HD strf::width_t fast_punct_double_printer<CharSize>::width() const
{
    if (_value.infinity || _value.nan) {
        return static_cast<std::int16_t>
            (3 + (_value.negative && _value.infinity));
    }
    constexpr unsigned decpoint_width = 1;
    if (_sci_notation) {
        unsigned e10u = std::abs(_value.e10 + (int)_m10_digcount - 1);
        auto w = _m10_digcount
            + _value.negative
            + (e10u < 10) + 2
            + detail::count_digits<10>(e10u)
            + decpoint_width * (_m10_digcount != 1);
        return static_cast<std::int16_t>(w);
    }
    if (_value.e10 <= -(int)_m10_digcount) {
        return static_cast<std::int16_t>
            (_value.negative + 1 - _value.e10 +  decpoint_width);
    }
    auto sep_w = _sep_size * _sep_count;
    auto idigcount = (int)_m10_digcount + _value.e10;
    if (_value.e10 < 0) {
        auto w = _value.negative + _m10_digcount + decpoint_width + sep_w;
        return static_cast<std::int16_t>(w);
    }
    return static_cast<std::int16_t>(_value.negative + idigcount + sep_w);
}

template <std::size_t CharSize>
STRF_HD void fast_punct_double_printer<CharSize>::print_to
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (_value.negative) {
        put(ob, static_cast<char_type>('-'));
    }
    if (_value.nan) {
        strf::detail::print_nan(ob, _lettercase);
    } else if (_value.infinity) {
        strf::detail::print_inf(ob, _lettercase);
    } else if (_sci_notation) {
        strf::detail::print_scientific_notation
            ( ob, _encode_char, _value.m10, _m10_digcount
            , _decimal_point, _decimal_point_size
            , _value.e10 + _m10_digcount - 1
            , false, 0, _lettercase );
    } else {
        if (_value.e10 >= 0) {
            if (_sep_count == 0) {
                strf::detail::write_int<10>( ob, _value.m10, _m10_digcount
                                           , strf::lowercase);
                strf::detail::write_fill(ob, _value.e10, (char_type)'0');
            } else if (_sep_size == 1) {
                strf::detail::print_amplified_integer_small_separator
                    ( ob, _punct, _value.m10, _m10_digcount, _value.e10
                    , _little_sep );
            } else {
                strf::detail::print_amplified_integer_big_separator
                    ( ob, _encode_char, _punct, _value.m10
                    , _m10_digcount, _value.e10, _sep_size );
            }
        } else {
            unsigned e10u = - _value.e10;
            if (e10u >= _m10_digcount) {
                ob.ensure(1 + _decimal_point_size);
                auto it = ob.pos();
                *it = static_cast<char_type>('0');
                if (_decimal_point_size == 1) {
                    it[1] = static_cast<char_type>(_decimal_point);
                    ob.advance_to(it + 2);
                } else {
                    ob.advance_to(_encode_char(it + 1, _decimal_point));
                }
                if (e10u > _m10_digcount) {
                    strf::detail::write_fill(ob, e10u - _m10_digcount, (char_type)'0');
                }
                strf::detail::write_int<10>( ob, _value.m10, _m10_digcount
                                           , strf::lowercase );
            } else {
                //auto v = std::lldiv(_value.m10, detail::pow10(e10u)); // todo test this
                auto p10 = strf::detail::pow10(e10u);
                auto integral_part = _value.m10 / p10;
                auto fractional_part = _value.m10 % p10;
                auto idigcount = _m10_digcount - e10u;
                STRF_ASSERT(idigcount == detail::count_digits<10>(integral_part));
                if (_sep_count == 0) {
                    strf::detail::write_int<10>( ob, integral_part, idigcount
                                               , strf::lowercase );
                } else if (_sep_size == 1) {
                    strf::detail::write_int_little_sep<10>
                        ( ob, _punct, integral_part, idigcount
                        , _little_sep, strf::lowercase);
                } else {
                    strf::detail::write_int_big_sep<10>
                        ( ob, _punct, _encode_char, integral_part
                        , _sep_size, idigcount, strf::lowercase );
                }
                ob.ensure(_decimal_point_size);
                if (_decimal_point_size == 1) {
                    *ob.pos() = static_cast<char_type>(_decimal_point);
                } else {
                    _encode_char(ob.pos(), _decimal_point);
                }
                ob.advance(_decimal_point_size);

                strf::detail::write_int_with_leading_zeros<10>
                    (ob, fractional_part, e10u, strf::lowercase);
            }
        }
    }
}


#if defined(STRF_SEPARATE_COMPILATION)

STRF_EXPLICIT_TEMPLATE class punct_double_printer<1>;
STRF_EXPLICIT_TEMPLATE class punct_double_printer<2>;
STRF_EXPLICIT_TEMPLATE class punct_double_printer<4>;

STRF_EXPLICIT_TEMPLATE class double_printer<1>;
STRF_EXPLICIT_TEMPLATE class double_printer<2>;
STRF_EXPLICIT_TEMPLATE class double_printer<4>;

STRF_EXPLICIT_TEMPLATE class fast_double_printer<1>;
STRF_EXPLICIT_TEMPLATE class fast_double_printer<2>;
STRF_EXPLICIT_TEMPLATE class fast_double_printer<4>;

STRF_EXPLICIT_TEMPLATE class fast_punct_double_printer<1>;
STRF_EXPLICIT_TEMPLATE class fast_punct_double_printer<2>;
STRF_EXPLICIT_TEMPLATE class fast_punct_double_printer<4>;

#endif // defined(STRF_SEPARATE_COMPILATION)

} // namespace detail

inline STRF_HD auto make_fmt(strf::rank<1>, float x)
{
    return strf::float_with_format<float, false>{x};
}

inline STRF_HD auto make_fmt(strf::rank<1>, double x)
{
    return strf::float_with_format<double, false>{x};
}

inline STRF_HD void make_fmt(strf::rank<1>, long double x) = delete;

template <typename CharT, typename FPack, typename Preview>
inline STRF_HD typename std::conditional
    < strf::detail::has_punct<CharT, FPack, float, 10>
    , strf::detail::fast_punct_double_printer<sizeof(CharT)>
    , strf::detail::fast_double_printer<sizeof(CharT)> >::type
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, float d)
{
    return {fp, preview, d, strf::tag<CharT>()};
}

template <typename CharT, typename FPack, typename Preview>
inline STRF_HD typename std::conditional
    < strf::detail::has_punct<CharT, FPack, double, 10>
    , strf::detail::fast_punct_double_printer<sizeof(CharT)>
    , strf::detail::fast_double_printer<sizeof(CharT)> >::type
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, double d)
{
    return {fp, preview, d, strf::tag<CharT>()};
}

template <typename CharT, typename FPack, typename Preview>
STRF_HD void make_printer(strf::rank<1>, const FPack& fp, Preview& preview, long double d) = delete;

template <typename CharT, typename FPack, typename Preview, bool Align>
inline STRF_HD typename std::conditional
    < strf::detail::has_punct<CharT, FPack, float, 10>
    , strf::detail::punct_double_printer<sizeof(CharT)>
    , strf::detail::double_printer<sizeof(CharT)> >::type
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , strf::float_with_format<float, Align> x )
{
    return {fp, preview, x, strf::tag<CharT>()};
}

template <typename CharT, typename FPack, typename Preview, bool Align>
inline STRF_HD typename std::conditional
    < strf::detail::has_punct<CharT, FPack, double, 10>
    , strf::detail::punct_double_printer<sizeof(CharT)>
    , strf::detail::double_printer<sizeof(CharT)> >::type
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , strf::float_with_format<double, Align> x )
{
    return {fp, preview, x, strf::tag<CharT>()};
}

} // namespace strf

#endif  // STRF_DETAIL_INPUT_TYPES_FLOAT_HPP

