#ifndef BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_FLOAT_HPP
#define BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_FLOAT_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/printer.hpp>
#include <boost/stringify/v0/facets_pack.hpp>
#include <boost/stringify/v0/detail/facets/numpunct.hpp>
#include <boost/stringify/v0/detail/ryu/double.hpp>
#include <boost/stringify/v0/detail/ryu/float.hpp>
#include <boost/assert.hpp>
#include <algorithm>
#include <cstring>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
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

#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_INLINE double_dec_base trivial_float_dec(
    std::uint32_t ieee_mantissa,
    std::int32_t biased_exponent,
    std::uint32_t k )
{
    constexpr int m_size = 23;

    BOOST_ASSERT(-10 <= biased_exponent && biased_exponent <= m_size);
    BOOST_ASSERT((std::int32_t)k == (biased_exponent * 179 + 1850) >> 8);
    BOOST_ASSERT(0 == (ieee_mantissa & (0x7FFFFF >> k)));

    BOOST_ASSERT(k <= m_size);
    BOOST_ASSERT(biased_exponent <= (int)k);

    std::int32_t e10 = biased_exponent - k;
    std::uint32_t m = (1ul << k) | (ieee_mantissa >> (m_size - k));
    int p5 = k - biased_exponent;
    BOOST_ASSERT(p5 <= 10);
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
    BOOST_ASSERT((m % 10) != 0);
    return {m, e10};
}

BOOST_STRINGIFY_INLINE double_dec_base trivial_double_dec(
    std::uint64_t ieee_mantissa,
    std::int32_t biased_exponent,
    std::uint32_t k )
{
    BOOST_ASSERT(-22 <= biased_exponent && biased_exponent <= 52);
    BOOST_ASSERT((std::int32_t)k == (biased_exponent * 179 + 4084) >> 8);
    BOOST_ASSERT(0 == (ieee_mantissa & (0xFFFFFFFFFFFFFull >> k)));

    BOOST_ASSERT(biased_exponent <= (int)k);
    BOOST_ASSERT(k <= 52);

    std::int32_t e10 = biased_exponent - k;
    std::uint64_t m = (1ull << k) | (ieee_mantissa >> (52 - k));
    int p5 = k - biased_exponent;
    BOOST_ASSERT(p5 <= 22);
    if (p5 >= 16 && (0 == (m & 0xFFFF))) {
        p5 -= 16;
        e10 += 16;
        m = m >> 16;
    }
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
    BOOST_ASSERT((m % 10) != 0);
    return {m, e10};
}
BOOST_STRINGIFY_INLINE detail::double_dec decode(float f)
{
    constexpr int bias = 127;
    constexpr int e_size = 8;
    constexpr int m_size = 23;

    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const std::uint32_t exponent
        = static_cast<std::uint32_t>((bits << 1) >> (m_size + 1));
    const bool sign = (bits >> (m_size + e_size));
    const std::uint64_t mantissa = bits & 0x7FFFFF;

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


BOOST_STRINGIFY_INLINE detail::double_dec decode(double d)
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

#else  // ! defined(BOOST_STRINGIFY_OMIT_IMPL)

detail::double_dec decode(double d);
detail::double_dec decode(float f);

#endif // ! defined(BOOST_STRINGIFY_OMIT_IMPL)

} // namespace detail

enum class float_notation{fixed, scientific, general};

struct decimal_float_format_data
{
    unsigned precision = (unsigned)-1;
    stringify::v0::float_notation notation = float_notation::general;
    bool showpoint = false;
    bool showpos = false;
};

struct decimal_float_format
{
    template <typename T>
    class fn: public decimal_float_format_data
    {
        using derived_type = stringify::v0::fmt_derived<alignment_format, T>;
        // using as_hex = stringify::v0::fmt_replace
        //     <T, decimal_float_format, stringify::v0::hex_float_format>;

    public:

        constexpr fn() = default;
        constexpr fn(const fn&) = default;
        constexpr fn(fn&&) = default;
        constexpr fn(const decimal_float_format_data& data)
            : decimal_float_format_data(data)
        {
        }
        constexpr derived_type&& operator+() &&
        {
            this->showpos = true;
            return static_cast<derived_type&&>(*this);
        }
        constexpr derived_type&& operator~() &&
        {
            this->showpoint = true;
            return static_cast<derived_type&&>(*this);
        }
        constexpr derived_type&& p(unsigned _) &&
        {
            this->precision = _;
            return static_cast<derived_type&&>(*this);
        }
        constexpr derived_type&& sci() &&
        {
            this->notation = float_notation::scientific;
            return static_cast<derived_type&&>(*this);
        }
        constexpr derived_type&& fixed() &&
        {
            this->notation = float_notation::fixed;
            return static_cast<derived_type&&>(*this);
        }
        constexpr derived_type&& dec() &&
        {
            return static_cast<derived_type&&>(*this);
        }
        // constexpr as_hex hex() const &
        // {
        //     return as_hex{static_cast<const derived_type&>(*this)};
        // }
    };
};

// struct hex_float_format
// {
//     template <typename T>
//     class fn
//     {
//         using derived_type = stringify::v0::fmt_derived<alignment_format, T>;
//         using as_dec = stringify::v0::fmt_replace
//             <T, decimal_float_format, stringify::v0::hex_float_format>;

//     public:

//         constexpr fn() = default;
//         constexpr fn(const fn&) = default;
//         constexpr fn(fn&&) = default;

//         template <typename U>
//         constexpr fn(const fn<U>& cp)
//             : _precision(cp._precision)
//             , _showpoint(cp._showpoint)
//             , _showpos(cp._showpos)
//         {
//         }

//         template <typename U>
//         explicit constexpr fn(const decimal_float_format::fn<U>& cp)
//             : _precision(cp._precision)
//             , _showpoint(cp._showpoint)
//             , _showpos(cp._showpos)
//         {
//         }

//         constexpr derived_type&& operator+() &&
//         {
//             _showpos = true;
//             return static_cast<derived_type&&>(*this);
//         }
//         constexpr derived_type&& operator~() &&
//         {
//             _showpoint = true;
//             return static_cast<derived_type&&>(*this);
//         }
//         constexpr derived_type&& p(unsigned _) &&
//         {
//             _precision = _;
//             return static_cast<derived_type&&>(*this);
//         }

//     private:

//         unsigned _precision = static_cast<unsigned>(-1);
//         bool _showpoint = false;
//         bool _showpos = false;
//     };
// };

inline auto make_fmt(stringify::v0::tag, float x)
{
    return stringify::v0::value_with_format
        < float
        , stringify::v0::decimal_float_format
        , stringify::v0::empty_alignment_format >{x};
}

inline auto make_fmt(stringify::v0::tag, double x)
{
    return stringify::v0::value_with_format
        < double
        , stringify::v0::decimal_float_format
        , stringify::v0::empty_alignment_format >{x};
}

namespace detail {

struct double_printer_data: detail::double_dec
{
    constexpr double_printer_data(const double_printer_data&) = default;

    template <typename FloatT>
    double_printer_data
        ( FloatT f
        , decimal_float_format_data fmt
        , const stringify::v0::numpunct_base* punct = nullptr )
        :  double_printer_data(detail::decode(f), fmt, punct)
    {
    }
    double_printer_data
        ( detail::double_dec d
        , decimal_float_format_data fmt
        , const stringify::v0::numpunct_base* punct = nullptr );

    bool showpoint;
    bool showsign;
    bool sci_notation;
    unsigned m10_digcount;
    unsigned extra_zeros;
};

#if !defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_INLINE double_printer_data::double_printer_data
    ( detail::double_dec d
    , decimal_float_format_data fmt
    , const stringify::v0::numpunct_base* punct )
    : stringify::v0::detail::double_dec(d)
    , showsign(fmt.showpos || negative)
{
    if (nan || infinity)
    {
        showpoint = false;
        sci_notation = false;
        m10_digcount = 0;
        extra_zeros = 0;
    }
    else if (fmt.precision == (unsigned)-1)
    {
        m10_digcount = stringify::v0::detail::count_digits<10>(m10);
        extra_zeros = 0;
        switch(fmt.notation)
        {
            case float_notation::general:
            {
                if (punct == nullptr)
                {
                    sci_notation = (e10 > 4 + (!fmt.showpoint && m10_digcount > 1))
                        || (e10 < -(int)m10_digcount - 2 - (fmt.showpoint || m10_digcount > 1));
                }
                else if (e10 > - (int)m10_digcount)
                {
                    auto sep_count = punct->thousands_sep_count(m10_digcount + e10);
                    bool e10neg = e10 < 0;
                    int fw = e10 * !e10neg + (fmt.showpoint || e10neg) + (int)sep_count;
                    int sw = 4 + (e10 > 99) + (m10_digcount > 1 || fmt.showpoint);
                    sci_notation = sw < fw;
                }
                else
                {
                    int tmp = m10_digcount + 2 + (e10 < -99)
                        + (m10_digcount > 1 || fmt.showpoint);
                    sci_notation = -e10 > tmp;
                }
                showpoint = fmt.showpoint
                        || (sci_notation && m10_digcount > 1)
                        || (!sci_notation && e10 < 0);
                break;
            }
            case float_notation::fixed:
            {
                sci_notation = false;
                showpoint = fmt.showpoint || (e10 < 0);
                break;
            }
            case float_notation::scientific:
            {
                sci_notation = true;
                showpoint = fmt.showpoint || (m10_digcount > 1);
                break;
            }
        }
    }
    else
    {
        m10_digcount = stringify::v0::detail::count_digits<10>(m10);
        int xz; // number of zeros to be added or ( if negative ) digits to be removed
        switch(fmt.notation)
        {
            case float_notation::general:
            {
                int p = fmt.precision + (fmt.precision == 0);
                int sci_notation_exp = e10 + (int)m10_digcount - 1;
                sci_notation = (sci_notation_exp < -4 || sci_notation_exp >= p);
                showpoint = fmt.showpoint || (sci_notation && m10_digcount != 1)
                                          || (!sci_notation && e10 < 0);
                xz = ((unsigned)p < m10_digcount || fmt.showpoint)
                   * (p - (int)m10_digcount);
                break;
            }
            case float_notation::fixed:
            {
                const int frac_digits = (e10 < 0) * -e10;
                xz = (fmt.precision - frac_digits);
                sci_notation = false;
                showpoint = fmt.showpoint || (fmt.precision != 0);
                break;
            }
            default:
            {
                BOOST_ASSERT(fmt.notation == float_notation::scientific);
                const unsigned frac_digits = m10_digcount - 1;
                xz = (fmt.precision - frac_digits);
                sci_notation = true;
                showpoint = fmt.showpoint || (fmt.precision != 0);
                break;
            }
        }
        if (xz < 0)
        {
            extra_zeros = 0;
            unsigned dp = -xz;
            m10_digcount -= dp;
            e10 += dp;
            auto p10 = stringify::v0::detail::pow10(dp);
            auto remainer = m10 % p10;
            m10 = m10 / p10;
            auto middle = p10 >> 1;
            m10 += (remainer > middle || (remainer == middle && (m10 & 1) == 1));
            if (fmt.notation == float_notation::general && ! fmt.showpoint)
            {
                while (m10 % 10 == 0)
                {
                    m10 /= 10;
                    -- m10_digcount;
                    ++ e10;
                }
                int frac_digits = sci_notation * (m10_digcount - 1)
                                - !sci_notation * (e10 < 0) * e10;
                showpoint = fmt.showpoint || (frac_digits != 0);
            }
         }
        else
        {
            extra_zeros = xz;
        }
    }
}

#endif // !defined(BOOST_STRINGIFY_OMIT_IMPL)

template <typename CharT>
void _print_amplified_integer_small_separator
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , const std::uint8_t* groups
    , unsigned num_groups
    , CharT separator
    , const char* digits
    , unsigned num_digits )
{
    (void)enc;
    BOOST_ASSERT(num_groups != 0);
    auto grp_it = groups + num_groups - 1;
    unsigned grp_size = *grp_it;
    while (num_digits > grp_size)
    {
        BOOST_ASSERT(grp_size + 1 <= boost::min_size_after_recycle<CharT>());
        ob.ensure(grp_size + 1);
        auto it = ob.pos();
        auto digits_2 = digits + grp_size;
        std::copy(digits, digits_2, it);
        it[grp_size] = separator;
        digits = digits_2;
        ob.advance(grp_size + 1);
        num_digits -= grp_size;
        BOOST_ASSERT(grp_it != groups);
        grp_size = *--grp_it;
    }
    if (num_digits != 0)
    {
        BOOST_ASSERT(num_digits <= boost::min_size_after_recycle<CharT>());
        ob.ensure(num_digits);
        std::copy(digits, digits + num_digits, ob.pos());
        ob.advance(num_digits);
    }
    if (grp_size > num_digits)
    {
        BOOST_ASSERT(num_digits <= boost::min_size_after_recycle<CharT>());
        grp_size -= num_digits;
        ob.ensure(grp_size);
        std::char_traits<CharT>::assign(ob.pos(), grp_size, '0');
        ob.advance(grp_size);
    }
    while (grp_it != groups)
    {
        grp_size = *--grp_it;
        BOOST_ASSERT(grp_size + 1 <= boost::min_size_after_recycle<CharT>());
        ob.ensure(grp_size + 1);
        auto it = ob.pos();
        *it = separator;
        std::char_traits<CharT>::assign(it + 1, grp_size, '0');
        ob.advance(grp_size + 1);
    }
}

template <typename CharT>
void _print_amplified_integer_big_separator
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , const std::uint8_t* groups
    , unsigned num_groups
    , char32_t separator
    , std::size_t separator_size
    , const char* digits
    , unsigned num_digits )
{
    BOOST_ASSERT(num_groups != 0);
    auto grp_it = groups + num_groups - 1;
    unsigned grp_size = *grp_it;
    while (num_digits > grp_size)
    {
        BOOST_ASSERT(grp_size + separator_size <= boost::min_size_after_recycle<CharT>());
        ob.ensure(grp_size + separator_size);
        auto it = ob.pos();
        auto digits_2 = digits + grp_size;
        std::copy(digits, digits_2, it);
        digits = digits_2;
        ob.advance_to(enc.encode_char(it + grp_size, separator));
        num_digits -= grp_size;
        BOOST_ASSERT(grp_it != groups);
        grp_size = *--grp_it;
    }
    if (num_digits != 0)
    {
        BOOST_ASSERT(num_digits <= boost::min_size_after_recycle<CharT>());
        ob.ensure(num_digits);
        std::copy(digits, digits + num_digits, ob.pos());
        ob.advance(num_digits);
    }
    if (grp_size > num_digits)
    {
        BOOST_ASSERT(num_digits <= boost::min_size_after_recycle<CharT>());
        grp_size -= num_digits;
        ob.ensure(grp_size);
        std::char_traits<CharT>::assign(ob.pos(), grp_size, '0');
        ob.advance(grp_size);
    }
    while (grp_it != groups)
    {
        grp_size = *--grp_it;
        BOOST_ASSERT(grp_size + separator_size <= boost::min_size_after_recycle<CharT>());
        ob.ensure(grp_size + separator_size);
        auto it = enc.encode_char(ob.pos(), separator);
        std::char_traits<CharT>::assign(it, grp_size, '0');
        ob.advance_to(it + separator_size);
    }
}

template <int Base, typename CharT>
void print_amplified_integer( boost::basic_outbuf<CharT>& ob
                            , const stringify::v0::numpunct_base& punct
                            , stringify::v0::encoding<CharT> enc
                            , unsigned long long value
                            , unsigned num_digits
                            , unsigned num_trailing_zeros )
{
    constexpr auto max_digits = detail::max_num_digits<unsigned long long, Base>;
    char digits_buff[max_digits];
    auto digits = stringify::v0::detail::write_int_txtdigits_backwards<Base>
        (value, digits_buff + max_digits);
    BOOST_ASSERT(num_digits == ((digits_buff + max_digits) - digits));

    std::uint8_t groups[std::numeric_limits<double>::max_exponent10 + 1];
    auto num_groups = punct.groups(num_trailing_zeros + num_digits, groups);
    auto sep32 = punct.thousands_sep();
    if (sep32 >= enc.u32equivalence_end() || sep32 < enc.u32equivalence_begin())
    {
        auto sep_size = enc.validate(sep32);
        if (sep_size == (std::size_t)-1)
        {
            stringify::v0::detail::write_int<10>(ob, value, num_digits);
            stringify::v0::detail::write_fill(ob, num_trailing_zeros, (CharT)'0');
            return;
        }
        if (sep_size != 1)
        {
            stringify::v0::detail::_print_amplified_integer_big_separator<CharT>
                ( ob, enc, groups, num_groups, sep32, sep_size
                , digits, num_digits );
            return;
        }
    }
    CharT sep = static_cast<CharT>(sep32);
    stringify::v0::detail::_print_amplified_integer_small_separator<CharT>
        ( ob, enc, groups, num_groups, sep, digits, num_digits );
}

template <typename CharT>
void print_scientific_notation
    ( boost::basic_outbuf<CharT>& ob
    , stringify::v0::encoding<CharT> enc
    , unsigned long long digits
    , unsigned num_digits
    , char32_t decimal_point
    , int exponent
    , bool print_point
    , unsigned trailing_zeros )
{
    BOOST_ASSERT(num_digits == detail::count_digits<10>(digits));

    CharT small_decimal_point = static_cast<CharT>(decimal_point);
    std::size_t psize = 1;
    print_point = print_point || num_digits > 1|| trailing_zeros != 0;
    if ( print_point
      && ( decimal_point >= enc.u32equivalence_end()
        || decimal_point < enc.u32equivalence_begin() ) )
    {
        psize = enc.validate(decimal_point);
        if (psize == (std::size_t)-1)
        {
            psize = enc.replacement_char_size();
        }
        else if (psize == 1)
        {
            enc.encode_char(&small_decimal_point, decimal_point);
        }
    }
    if (num_digits == 1)
    {
        ob.ensure(num_digits + print_point * psize);
        auto it = ob.pos();
        *it = '0' + digits;
        if (print_point)
        {
            if (psize == 1)
            {
                it[1] = small_decimal_point;
                ob.advance(2);
            }
            else
            {
                ob.advance_to(enc.encode_char(it + 1, decimal_point));
            }
        }
        else
        {
            ob.advance(1);
        }
    }
    else
    {
        ob.ensure(num_digits + psize);
        auto it = ob.pos() + num_digits + psize;
        const char* arr = stringify::v0::detail::chars_00_to_99();

        while(digits > 99)
        {
            auto index = (digits % 100) << 1;
            it[-2] = arr[index];
            it[-1] = arr[index + 1];
            it -= 2;
            digits /= 100;
        }
        CharT highest_digit;
        if (digits < 10)
        {
            highest_digit = static_cast<CharT>('0' + digits);
        }
        else
        {
            auto index = digits << 1;
            highest_digit = arr[index];
            * --it = arr[index + 1];
        }
        if (psize == 1)
        {
            * --it = small_decimal_point;
        }
        else
        {
            it -= psize;
            enc.encode_char(it, decimal_point);
        }
        * --it = highest_digit;
        BOOST_ASSERT(it == ob.pos());
        ob.advance(num_digits + psize);
    }
    if (trailing_zeros != 0)
    {
        stringify::v0::detail::write_fill(ob, trailing_zeros, CharT('0'));
    }

    unsigned adv = 4;
    CharT* it;
    unsigned e10u = std::abs(exponent);
    BOOST_ASSERT(e10u < 1000);

    if (e10u >= 100)
    {
        ob.ensure(5);
        it = ob.pos();
        it[4] = static_cast<CharT>('0' + e10u % 10);
        e10u /= 10;
        it[3] = static_cast<CharT>('0' + e10u % 10);
        it[2] = static_cast<CharT>('0' + e10u / 10);
        adv = 5;
    }
    else if (e10u >= 10)
    {
        ob.ensure(4);
        it = ob.pos();
        it[3] = static_cast<CharT>('0' + e10u % 10);
        it[2] = static_cast<CharT>('0' + e10u / 10);
    }
    else
    {
        ob.ensure(4);
        it = ob.pos();
        it[3] = static_cast<CharT>('0' + e10u);
        it[2] = '0';
    }
    it[0] = 'e';
    it[1] = static_cast<CharT>('+' + ((exponent < 0) << 1));
    ob.advance(adv);
}

template <typename CharT>
class punct_double_printer: public stringify::v0::printer<CharT>
{
public:

    template <typename FP, typename FloatT>
    punct_double_printer
        ( const FP& fp
        , stringify::v0::value_with_format
            < FloatT
            , stringify::v0::decimal_float_format
            , stringify::v0::empty_alignment_format > x )
         : _data{x.value(), x}
         , _punct(get_facet<stringify::v0::numpunct_c<10>, FloatT>(fp))
         , _encoding(get_facet<stringify::v0::encoding_c<CharT>, FloatT>(fp))
         , _enc_err(get_facet<stringify::v0::encoding_error_c, FloatT>(fp))
    {
    }

    template <typename FP, typename FloatT>
    punct_double_printer
        ( const FP& fp
        , stringify::v0::value_with_format
            < FloatT
            , stringify::v0::decimal_float_format
            , stringify::v0::alignment_format > x )
        : _data{x.value(), x}
        , _punct(get_facet<stringify::v0::numpunct_c<10>, FloatT>(fp))
        , _encoding(get_facet<stringify::v0::encoding_c<CharT>, FloatT>(fp))
        , _fillchar(x.fill())
        , _enc_err(fp.template get_facet<stringify::v0::encoding_error_c, FloatT>())
        , _allow_surr(fp.template get_facet<stringify::v0::surrogate_policy_c, FloatT>())
    {
        void init_fill(int w, stringify::v0::alignment a);
    }

    int width(int) const override;

    std::size_t necessary_size() const override;

    void write(boost::basic_outbuf<CharT>&) const override;

private:

    void init_fill(int w, stringify::v0::alignment a);

    stringify::v0::detail::double_printer_data _data;
    const stringify::v0::numpunct_base& _punct;
    const stringify::v0::encoding<CharT> _encoding;
    char32_t _fillchar = U' ';
    unsigned _left_fillcount = 0;
    unsigned _internal_fillcount = 0;
    unsigned _right_fillcount = 0;
    stringify::v0::encoding_error _enc_err;
    stringify::v0::surrogate_policy _allow_surr = surrogate_policy::strict;
};

template <typename CharT>
void punct_double_printer<CharT>::init_fill(int w, stringify::v0::alignment a)
{
    auto fillcount = w - this->width(w);
    if (fillcount > 0)
    {
        switch (a)
        {
            case stringify::v0::alignment::right:
                _left_fillcount = fillcount;
                break;
            case stringify::v0::alignment::left:
                _right_fillcount = fillcount;
                break;
            case stringify::v0::alignment::internal:
                _internal_fillcount = fillcount;
                break;
            default:
                BOOST_ASSERT(a == stringify::v0::alignment::center);
                _left_fillcount = fillcount / 2;
                _right_fillcount = fillcount - _left_fillcount;
        }
    }
}

template <typename CharT>
int punct_double_printer<CharT>::width(int) const
{
    auto fillcount = _left_fillcount + _internal_fillcount + _right_fillcount;

    if (_data.infinity || _data.nan)
    {
        return 3 + _data.showsign + fillcount;
    }
    int decpoint_width = _data.showpoint;
    if (_data.sci_notation)
    {
        unsigned e10u = std::abs(_data.e10 + (int)_data.m10_digcount - 1);
        return fillcount + _data.m10_digcount + _data.extra_zeros
            + _data.showsign
            + (e10u < 10) + 2
            + detail::count_digits<10>(e10u)
            + decpoint_width;
    }
    if (_data.e10 < 0)
    {
        if (_data.e10 <= -(int)_data.m10_digcount)
        {
            return fillcount + _data.showsign + 1 + decpoint_width
                - _data.e10 + _data.extra_zeros;
        }
        else
        {
            auto idigcount = (int)_data.m10_digcount + _data.e10;
            return fillcount + _data.showsign
                + (int)_data.m10_digcount
                + _data.extra_zeros
                + 1 // decpoint_width
                + _punct.thousands_sep_count(idigcount);
        }
    }
    auto idigcount = _data.m10_digcount + _data.e10;
    return fillcount + _data.showsign
        + idigcount
        + _data.extra_zeros
        + _data.showpoint
        + _punct.thousands_sep_count(idigcount);
}

template <typename CharT>
std::size_t punct_double_printer<CharT>::necessary_size() const
{
    auto fillcount = _left_fillcount + _internal_fillcount + _right_fillcount;
    std::size_t fillsize = 0;
    if (fillcount != 0)
    {
        fillsize = _encoding.validate(_fillchar);
        if (fillsize == (size_t)-1)
        {
            fillsize = _encoding.replacement_char_size();
        }
        fillsize *= fillcount;
    }
    if (_data.infinity || _data.nan)
    {
        return 3 + _data.showsign + fillsize;
    }
    std::size_t point_size = 0;
    if (_data.showpoint)
    {
        point_size = _encoding.validate(_punct.decimal_point());
        if (point_size == (size_t)-1)
        {
            point_size = _encoding.replacement_char_size();
        }
    }
    if (_data.sci_notation)
    {
        unsigned e10u = std::abs(_data.e10 + (int)_data.m10_digcount - 1);
        return fillsize + _data.m10_digcount + _data.extra_zeros
            + _data.showsign
            + (e10u < 10) + 2
            + detail::count_digits<10>(e10u)
            + point_size;
    }
    if (_data.e10 <= -(int)_data.m10_digcount)
    {
        return fillsize + 1 + point_size + (-_data.e10) +_data.extra_zeros;
    }

    std::size_t seps_size = 0;
    auto idigcount = (int)_data.m10_digcount + _data.e10;
    BOOST_ASSERT(idigcount > 0);

    if (idigcount > 1 && ! _punct.no_group_separation(idigcount))
    {
        auto s = _encoding.validate(_punct.thousands_sep());
        if (s != (std::size_t)-1)
        {
            seps_size = s * _punct.thousands_sep_count(idigcount);
        }
    }
    return fillsize + _data.showsign + seps_size + point_size
        + _data.m10_digcount + _data.extra_zeros + (_data.e10 > 0) * _data.e10;
}

template <typename CharT>
void punct_double_printer<CharT>::write(boost::basic_outbuf<CharT>& ob) const
{
    if (_data.showsign)
    {
        put(ob, static_cast<CharT>('+' + (_data.negative << 1)));
    }
    if (_data.nan)
    {
        ob.ensure(3);
        ob.pos()[0] = 'n';
        ob.pos()[1] = 'a';
        ob.pos()[2] = 'n';
        ob.advance(3);
    }
    else if (_data.infinity)
    {
        ob.ensure(3);
        ob.pos()[0] = 'i';
        ob.pos()[1] = 'n';
        ob.pos()[2] = 'f';
        ob.advance(3);
    }
    else if (_data.sci_notation)
    {
        stringify::v0::detail::print_scientific_notation
            ( ob, _encoding, _data.m10, _data.m10_digcount
            , _punct.decimal_point()
            , _data.e10 + _data.m10_digcount - 1
            , _data.showpoint
            , _data.extra_zeros );
    }
    else if (_data.e10 >= 0)
    {
        if (_punct.no_group_separation(_data.m10_digcount + _data.e10))
        {
            stringify::v0::detail::write_int<10>(ob, _data.m10, _data.m10_digcount);
            stringify::v0::detail::write_fill(ob, _data.e10, (CharT)'0');
        }
        else
        {
            stringify::v0::detail::print_amplified_integer<10>
                ( ob, _punct, _encoding, _data.m10
                , _data.m10_digcount, _data.e10 );
        }
        if (_data.showpoint)
        {
            _encoding.encode_char( ob, _punct.decimal_point()
                                 , stringify::v0::encoding_error::replace);
        }
        if (_data.extra_zeros)
        {
            detail::write_fill(ob, _data.extra_zeros,  (CharT)'0');
        }
    }
    else
    {
        BOOST_ASSERT(_data.e10 < 0);

        unsigned e10u = - _data.e10;
        if (e10u >= _data.m10_digcount)
        {
            put(ob, static_cast<CharT>('0'));
            _encoding.encode_char( ob, _punct.decimal_point()
                                 , stringify::v0::encoding_error::replace );

            if (e10u > _data.m10_digcount)
            {
                stringify::v0::detail::write_fill(ob, e10u - _data.m10_digcount, (CharT)'0');
            }
            stringify::v0::detail::write_int<10>(ob, _data.m10, _data.m10_digcount);
            if (_data.extra_zeros != 0)
            {
                stringify::v0::detail::write_fill(ob, _data.extra_zeros,  (CharT)'0');
            }
        }
        else
        {
            //auto v = std::lldiv(_data.m10, detail::pow10(e10u)); // todo test this
            auto p10 = stringify::v0::detail::pow10(e10u);
            auto integral_part = _data.m10 / p10;
            auto fractional_part = _data.m10 % p10;
            auto idigcount = _data.m10_digcount - e10u;

            BOOST_ASSERT(idigcount == detail::count_digits<10>(integral_part));

            if (_punct.no_group_separation(idigcount))
            {
                stringify::v0::detail::write_int<10>(ob, integral_part, idigcount);
            }
            else
            {
                stringify::v0::detail::write_int<10>( ob, _punct, _encoding
                                                    , integral_part, idigcount );
            }
            _encoding.encode_char( ob, _punct.decimal_point()
                                 , stringify::v0::encoding_error::replace );
            stringify::v0::detail::write_int_with_leading_zeros<10>
                (ob, fractional_part, e10u);
            if (_data.extra_zeros)
            {
                detail::write_fill(ob, _data.extra_zeros,  (CharT)'0');
            }
        }
    }
}

template <typename CharT>
class double_printer final: public stringify::v0::printer<CharT>
{
public:

    template <typename Fpack, typename FloatT>
    double_printer
        ( const Fpack&
        , stringify::v0::value_with_format
            < FloatT
            , stringify::v0::decimal_float_format
            , stringify::v0::empty_alignment_format > x )
            : _data(x.value(), x)
    {
    }

    template <typename Fpack, typename FloatT>
    double_printer
        ( const Fpack& fp
        , stringify::v0::value_with_format
            < FloatT
            , stringify::v0::decimal_float_format
            , stringify::v0::alignment_format > x )
        : _data(x.value(), x)
        , _encoding(fp.template get_facet<stringify::v0::encoding_c<CharT>, FloatT>())
        , _fillchar(x.fill())
        , _enc_err(fp.template get_facet<stringify::v0::encoding_error_c, FloatT>())
        , _allow_surr(fp.template get_facet<stringify::v0::surrogate_policy_c, FloatT>())
    {
        init_fill(x.width(), x.alignment());
    }

    int width(int) const override;

    void write(boost::basic_outbuf<CharT>&) const override;

    std::size_t necessary_size() const override;

private:

    void init_fill(int w, stringify::v0::alignment a);

    stringify::v0::detail::double_printer_data _data;
    stringify::v0::encoding<CharT> _encoding
        = stringify::v0::encoding_c<CharT>::get_default();
    char32_t _fillchar = U' ';
    unsigned _left_fillcount = 0;
    unsigned _internal_fillcount = 0;
    unsigned _right_fillcount = 0;
    stringify::v0::encoding_error _enc_err = encoding_error::ignore;
    stringify::v0::surrogate_policy _allow_surr = surrogate_policy::strict;
};

template <typename CharT>
void double_printer<CharT>::init_fill(int w, stringify::v0::alignment a)
{
    auto fillcount = w - this->width(w);
    if (fillcount > 0)
    {
        switch (a)
        {
            case stringify::v0::alignment::right:
                _left_fillcount = fillcount;
                break;
            case stringify::v0::alignment::left:
                _right_fillcount = fillcount;
                break;
            case stringify::v0::alignment::internal:
                _internal_fillcount = fillcount;
                break;
            default:
                BOOST_ASSERT(a == stringify::v0::alignment::center);
                _left_fillcount = fillcount / 2;
                _right_fillcount = fillcount - _left_fillcount;
        }
    }
}

template <typename CharT>
std::size_t double_printer<CharT>::necessary_size() const
{
    auto fillcount = _left_fillcount + _internal_fillcount + _right_fillcount;
    std::size_t fillsize = 0;
    if (fillcount != 0)
    {
        fillsize = _encoding.validate(_fillchar);
        if (fillsize == (size_t)-1)
        {
            fillsize = _encoding.replacement_char_size();
        }
        fillsize *= fillcount;
    }
    return ( fillsize
           + _data.nan * 3
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

template <typename CharT>
int double_printer<CharT>::width(int) const
{
    return ( _left_fillcount + _internal_fillcount + _right_fillcount
           + _data.nan * 3
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

template <typename CharT>
void double_printer<CharT>::write
    ( boost::basic_outbuf<CharT>& ob ) const
{
    if (_left_fillcount != 0)
    {
        _encoding.encode_fill( ob, _left_fillcount, _fillchar
                             , _enc_err, _allow_surr);
    }
    if (_data.showsign)
    {
        put<CharT>(ob, '+' + (_data.negative << 1));
    }
    if (_internal_fillcount != 0)
    {
        _encoding.encode_fill( ob, _internal_fillcount, _fillchar
                             , _enc_err, _allow_surr);
    }
    if (_data.nan)
    {
        ob.ensure(3);
        ob.pos()[0] = 'n';
        ob.pos()[1] = 'a';
        ob.pos()[2] = 'n';
        ob.advance(3);
    }
    else if (_data.infinity)
    {
        ob.ensure(3);
        ob.pos()[0] = 'i';
        ob.pos()[1] = 'n';
        ob.pos()[2] = 'f';
        ob.advance(3);
    }
    else if (_data.sci_notation)
    {
        ob.ensure( _data.m10_digcount
                 + _data.showpoint
                 + 4 + (_data.e10 > 99 || _data.e10 < -99) );
        CharT* it = ob.pos();
        if (_data.m10_digcount == 1)
        {
            * it = static_cast<CharT>('0' + _data.m10);
            ++it;
            if (_data.showpoint)
            {
                *it = '.';
                ++it;
            }
            if (_data.extra_zeros > 0)
            {
                ob.advance_to(it);
                stringify::v0::detail::write_fill<CharT>(ob, _data.extra_zeros, '0');
                it = ob.pos();
            }
        }
        else
        {
            auto itz = it + _data.m10_digcount + 1;
            write_int_dec_txtdigits_backwards(_data.m10, itz);
            it[0] = it[1];
            it[1] = '.';
            it = itz;
            if (_data.extra_zeros > 0)
            {
                ob.advance_to(itz);
                stringify::v0::detail::write_fill<CharT>(ob, _data.extra_zeros, '0');
                it = ob.pos();
            }
        }
        auto e10 = _data.e10 - 1 + (int)_data.m10_digcount;
        it[0] = 'e';
        it[1] = static_cast<CharT>('+' + ((e10 < 0) << 1));
        unsigned e10u = std::abs(e10);
        if (e10u >= 100)
        {
            it[4] = static_cast<CharT>('0' + e10u % 10);
            e10u /= 10;
            it[3] = static_cast<CharT>('0' + e10u % 10);
            it[2] = static_cast<CharT>('0' + e10u / 10);
            it += 5;
        }
        else if (e10u >= 10)
        {
            it[3] = static_cast<CharT>('0' + e10u % 10);
            it[2] = static_cast<CharT>('0' + e10u / 10);
            it += 4;
        }
        else
        {
            it[3] = static_cast<CharT>('0' + e10u);
            it[2] = '0';
            it += 4;
        }
        ob.advance_to(it);
    }
    else
    {
        ob.ensure( _data.showpoint + _data.m10_digcount
                 + (_data.e10 < -(int)_data.m10_digcount) );
        auto it = ob.pos();
        if (_data.e10 >= 0)
        {
            it += _data.m10_digcount;
            write_int_dec_txtdigits_backwards(_data.m10, it);
            ob.advance_to(it);
            detail::write_fill(ob, _data.e10, (CharT)'0');
            if (_data.showpoint)
            {
                ob.ensure(1);
                *ob.pos() = '.';
                ob.advance();
            }
            detail::write_fill(ob, _data.extra_zeros, (CharT)'0');
        }
        else
        {
            unsigned e10u = - _data.e10;
            if (e10u >= _data.m10_digcount)
            {
                it[0] = '0';
                it[1] = '.';
                ob.advance_to(it + 2);
                detail::write_fill(ob, e10u - _data.m10_digcount, (CharT)'0');

                ob.ensure(_data.m10_digcount);
                auto end = ob.pos() + _data.m10_digcount;
                write_int_dec_txtdigits_backwards(_data.m10, end);
                ob.advance_to(end);
                detail::write_fill(ob, _data.extra_zeros, (CharT)'0');
            }
            else
            {
                const char* const arr = stringify::v0::detail::chars_00_to_99();
                auto m = _data.m10;
                CharT* const end = it + _data.m10_digcount + 1;
                it = end;
                while(e10u >= 2)
                {
                    auto index = (m % 100) << 1;
                    it[-2] = arr[index];
                    it[-1] = arr[index + 1];
                    it -= 2;
                    m /= 100;
                    e10u -= 2;
                }
                if (e10u != 0)
                {
                    *--it = static_cast<CharT>('0' + (m % 10));
                    m /= 10;
                }
                * --it = '.';
                while(m > 99)
                {
                    auto index = (m % 100) << 1;
                    it[-2] = arr[index];
                    it[-1] = arr[index + 1];
                    it -= 2;
                    m /= 100;
                }
                if (m > 9)
                {
                    auto index = m << 1;
                    it[-2] = arr[index];
                    it[-1] = arr[index + 1];
                }
                else
                {
                    *--it = static_cast<CharT>('0' + m);
                }
                ob.advance_to(end);
                detail::write_fill(ob, _data.extra_zeros, (CharT)'0');
            }
        }
    }
    if (_right_fillcount != 0)
    {
        _encoding.encode_fill( ob, _right_fillcount, _fillchar
                             , _enc_err, _allow_surr );
    }
}

template <typename CharT>
class fast_double_printer: public stringify::v0::printer<CharT>
{
public:

    template <typename FPack>
    fast_double_printer(const FPack, float f)
        : fast_double_printer(f)
    {
    }

    template <typename FPack>
    fast_double_printer(const FPack, double d)
        : fast_double_printer(d)
    {
    }

    explicit fast_double_printer(float f)
        : _value(decode(f))
        , _m10_digcount(stringify::v0::detail::count_digits<10>(_value.m10))

    {
        BOOST_ASSERT(!_value.nan || !_value.infinity);
        _sci_notation = (_value.e10 > 4 + (_m10_digcount > 1))
            || (_value.e10 < -(int)_m10_digcount - 2 - (_m10_digcount > 1));
    }

    explicit fast_double_printer(double d)
        : _value(decode(d))
        , _m10_digcount(stringify::v0::detail::count_digits<10>(_value.m10))

    {
        BOOST_ASSERT(!_value.nan || !_value.infinity);
        _sci_notation = (_value.e10 > 4 + (_m10_digcount > 1))
            || (_value.e10 < -(int)_m10_digcount - 2 - (_m10_digcount > 1));
    }

    int width(int) const override;

    void write(boost::basic_outbuf<CharT>&) const override;

    std::size_t necessary_size() const override;

private:

    unsigned _size_sci() const
    {
        return _value.negative + _m10_digcount + (_m10_digcount != 1) + 4
            + (_value.e10 > 99 || _value.e10 < -99);

    }

    const detail::double_dec _value;
    bool _sci_notation ;
    const unsigned _m10_digcount;
};

template <typename CharT>
std::size_t fast_double_printer<CharT>::necessary_size() const
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
                 + (_value.e10 > 0) * _value.e10
                 + (_value.e10 <= -(int)_m10_digcount) * (2 -_value.e10 - (int)_m10_digcount)
                 + (-(int)_m10_digcount < _value.e10 && _value.e10 < 0) ))));
}

template <typename CharT>
int fast_double_printer<CharT>::width(int) const
{
    return ( _value.nan * 3
           + (_value.infinity * 3)
           + (_value.negative && !_value.nan)
           + !(_value.infinity | _value.nan)
           * ( ( _sci_notation
               * ( 4
                 + (_m10_digcount != 1)
                 + _m10_digcount
                 + ((_value.e10 > 99) || (_value.e10 < -99))) )
             + ( !_sci_notation
               * ( (int)_m10_digcount
                 + (_value.e10 > 0) * _value.e10
                 + (_value.e10 <= -(int)_m10_digcount) * (2 -_value.e10 - (int)_m10_digcount)
                 + (-(int)_m10_digcount < _value.e10 && _value.e10 < 0)
                 + (_value.e10 == -(int)_m10_digcount) ))));
}

template <typename CharT>
void fast_double_printer<CharT>::write
    ( boost::basic_outbuf<CharT>& ob ) const
{
    if (_value.nan)
    {
        ob.ensure(3);
        ob.pos()[0] = 'n';
        ob.pos()[1] = 'a';
        ob.pos()[2] = 'n';
        ob.advance(3);
    }
    else if (_value.infinity)
    {
        if (_value.negative)
        {
            ob.ensure(4);
            ob.pos()[0] = '-';
            ob.pos()[1] = 'i';
            ob.pos()[2] = 'n';
            ob.pos()[3] = 'f';
            ob.advance(4);
        }
        else
        {
            ob.ensure(3);
            ob.pos()[0] = 'i';
            ob.pos()[1] = 'n';
            ob.pos()[2] = 'f';
            ob.advance(3);
        }
    }
    else if (_sci_notation)
    {
        ob.ensure( _value.negative + _m10_digcount + (_m10_digcount != 1) + 4
                 + (_value.e10 > 99 || _value.e10 < -99) );
        CharT* it = ob.pos();

        if (_value.negative)
        {
            * it = '-';
            ++it;
        }
        if (_m10_digcount == 1)
        {
            * it = static_cast<CharT>('0' + _value.m10);
            ++ it;
        }
        else
        {
            auto next = it + _m10_digcount + 1;
            write_int_dec_txtdigits_backwards(_value.m10, next);
            it[0] = it[1];
            it[1] = '.';
            it = next;
        }
        auto e10 = _value.e10 - 1 + (int)_m10_digcount;
        it[0] = 'e';
        it[1] = static_cast<CharT>('+' + ((e10 < 0) << 1));
        unsigned e10u = std::abs(e10);
        if (e10u >= 100)
        {
            it[4] = static_cast<CharT>('0' + e10u % 10);
            e10u /= 10;
            it[3] = static_cast<CharT>('0' + e10u % 10);
            it[2] = static_cast<CharT>('0' + e10u / 10);
            it += 5;
        }
        else if (e10u >= 10)
        {
            it[3] = static_cast<CharT>('0' + e10u % 10);
            it[2] = static_cast<CharT>('0' + e10u / 10);
            it += 4;
        }
        else
        {
            it[3] = static_cast<CharT>('0' + e10u);
            it[2] = '0';
            it += 4;
        }
        ob.advance_to(it);
    }
    else
    {
        ob.ensure( _value.negative
                 + _m10_digcount * (_value.e10 > - (int)_m10_digcount)
                 + (_value.e10 < - (int)_m10_digcount)
                 + (_value.e10 < 0) );
        auto it = ob.pos();
        if (_value.negative)
        {
            *it = '-';
            ++it;
        }
        if (_value.e10 >= 0)
        {
            it += _m10_digcount;
            write_int_dec_txtdigits_backwards(_value.m10, it);
            ob.advance_to(it);
            if (_value.e10 != 0)
            {
                detail::write_fill(ob, _value.e10, (CharT)'0');
            }
        }
        else
        {
            unsigned e10u = - _value.e10;
            if (e10u >= _m10_digcount)
            {
                it[0] = '0';
                it[1] = '.';
                ob.advance_to(it + 2);
                detail::write_fill(ob, e10u - _m10_digcount, (CharT)'0');

                ob.ensure(_m10_digcount);
                auto end = ob.pos() + _m10_digcount;
                write_int_dec_txtdigits_backwards(_value.m10, end);
                ob.advance_to(end);
            }
            else
            {
                const char* const arr = stringify::v0::detail::chars_00_to_99();
                auto m = _value.m10;
                it += _m10_digcount + 1;
                CharT* const end = it;
                while(e10u >= 2)
                {
                    auto index = (m % 100) << 1;
                    it[-2] = arr[index];
                    it[-1] = arr[index + 1];
                    it -= 2;
                    m /= 100;
                    e10u -= 2;
                }
                if (e10u != 0)
                {
                    *--it = static_cast<CharT>('0' + (m % 10));
                    m /= 10;
                }
                * --it = '.';
                while(m > 99)
                {
                    auto index = (m % 100) << 1;
                    it[-2] = arr[index];
                    it[-1] = arr[index + 1];
                    it -= 2;
                    m /= 100;
                }
                if (m > 9)
                {
                    auto index = m << 1;
                    it[-2] = arr[index];
                    it[-1] = arr[index + 1];
                }
                else
                {
                    *--it = static_cast<CharT>('0' + m);
                }
                ob.advance_to(end);
            }
        }
    }
}


template <typename CharT>
class fast_punct_double_printer: public stringify::v0::printer<CharT>
{
public:

    template <typename FPack, typename FloatT>
    fast_punct_double_printer(const FPack& fp, FloatT d)
        : _punct(get_facet<stringify::v0::numpunct_c<10>, FloatT>(fp))
        , _encoding(get_facet<stringify::v0::encoding_c<CharT>, FloatT>(fp))
        , _value(decode(d))
        , _m10_digcount(stringify::v0::detail::count_digits<10>(_value.m10))
    {
        constexpr bool showpoint = false;
        if (_value.e10 > -(int)_m10_digcount)
        {
            _sep_count = _punct.thousands_sep_count((int)_m10_digcount + _value.e10);
            bool e10neg = _value.e10 < 0;
            int fw = _value.e10 * !e10neg  + (showpoint || e10neg) + (int)_sep_count;
            int sw = 4 + (_value.e10 > 99) + (_m10_digcount > 1 || showpoint);
            _sci_notation = sw < fw;
        }
        else
        {
            _sep_count = 0;
            int tmp = _m10_digcount + 2 + (_value.e10 < -99)
                    + (_m10_digcount > 1 || showpoint);
            _sci_notation = -_value.e10 > tmp;
        }
    }

    int width(int) const override;

    void write(boost::basic_outbuf<CharT>&) const override;

    std::size_t necessary_size() const override;

private:

    unsigned _size_sci() const
    {
        return _value.negative + _m10_digcount + (_m10_digcount != 1) + 4
            + (_value.e10 > 99 || _value.e10 < -99);

    }

    const stringify::v0::numpunct_base& _punct;
    stringify::v0::encoding<CharT> _encoding;
    const detail::double_dec _value;
    const unsigned _m10_digcount;
    unsigned _sep_count;
    bool _sci_notation ;

};

template <typename CharT>
std::size_t fast_punct_double_printer<CharT>::necessary_size() const
{
    if (_value.infinity || _value.nan)
    {
        return 3 + (_value.negative && _value.infinity);
    }
    std::size_t point_size = 0;
    if (_sci_notation || _value.e10 < 0)
    {
        point_size = _encoding.validate(_punct.decimal_point());
        if (point_size == (size_t)-1)
        {
            point_size = _encoding.replacement_char_size();
        }
    }
    if (_sci_notation)
    {
        unsigned e10u = std::abs(_value.e10 + (int)_m10_digcount - 1);
        return _m10_digcount
            + _value.negative
            + (e10u < 10) + 2
            + detail::count_digits<10>(e10u)
            + point_size * (_m10_digcount > 1);
    }
    if (_value.e10 <= -(int)_m10_digcount)
    {
        return 1 + point_size + (-_value.e10);
    }
    std::size_t seps_size = 0;
    auto idigcount = (int)_m10_digcount + _value.e10;
    if (idigcount > 1 && ! _punct.no_group_separation(idigcount))
    {
        auto s = _encoding.validate(_punct.thousands_sep());
        if (s != (std::size_t)-1)
        {
            seps_size = s * _sep_count;
        }
    }
    return seps_size + point_size + _m10_digcount  + _value.negative
        + (_value.e10 > 0) * _value.e10;
}

template <typename CharT>
int fast_punct_double_printer<CharT>::width(int) const
{
    if (_value.infinity || _value.nan)
    {
        return 3 + (_value.negative && _value.infinity);
    }
    constexpr unsigned decpoint_width = 1;
    constexpr unsigned sep_width = 1;
    if (_sci_notation)
    {
        unsigned e10u = std::abs(_value.e10 + (int)_m10_digcount - 1);
        return _m10_digcount
            + _value.negative
            + (e10u < 10) + 2
            + detail::count_digits<10>(e10u)
            + decpoint_width * (_m10_digcount > 1);
    }
    if (_value.e10 < 0)
    {
        if (_value.e10 <= -(int)_m10_digcount)
        {
            return _value.negative + 1 - _value.e10 +  decpoint_width;
        }
        else
        {
            auto idigcount = (int)_m10_digcount + _value.e10;
            return _value.negative
                + (int)_m10_digcount
                + decpoint_width
                + _punct.thousands_sep_count(idigcount) * sep_width;
        }
    }
    auto idigcount = _m10_digcount + _value.e10;
    return _value.negative + idigcount
        + _punct.thousands_sep_count(idigcount);
}

template <typename CharT>
void fast_punct_double_printer<CharT>::write
    ( boost::basic_outbuf<CharT>& ob ) const
{
    if (_value.negative)
    {
        put(ob, static_cast<CharT>('-'));
    }
    if (_value.nan)
    {
        ob.ensure(3);
        ob.pos()[0] = 'n';
        ob.pos()[1] = 'a';
        ob.pos()[2] = 'n';
        ob.advance(3);
    }
    else if (_value.infinity)
    {
        ob.ensure(3);
        ob.pos()[0] = 'i';
        ob.pos()[1] = 'n';
        ob.pos()[2] = 'f';
        ob.advance(3);
    }
    else if (_sci_notation)
    {
        stringify::v0::detail::print_scientific_notation
            ( ob, _encoding, _value.m10, _m10_digcount
            , _punct.decimal_point()
            , _value.e10 + _m10_digcount - 1
            , false, 0 );
    }
    else
    {
        if (_value.e10 >= 0)
        {
            if (_punct.no_group_separation(_m10_digcount + _value.e10))
            {
                stringify::v0::detail::write_int<10>(ob, _value.m10, _m10_digcount);
                stringify::v0::detail::write_fill(ob, _value.e10, (CharT)'0');
            }
            else
            {
                stringify::v0::detail::print_amplified_integer<10>
                    ( ob, _punct, _encoding, _value.m10
                    , _m10_digcount, _value.e10 );
            }
        }
        else
        {
            unsigned e10u = - _value.e10;
            if (e10u >= _m10_digcount)
            {
                put(ob, static_cast<CharT>('0'));
                _encoding.encode_char( ob, _punct.decimal_point()
                                     , stringify::v0::encoding_error::replace );
                if (e10u > _m10_digcount)
                {
                    stringify::v0::detail::write_fill(ob, e10u - _m10_digcount, (CharT)'0');
                }
                stringify::v0::detail::write_int<10>(ob, _value.m10, _m10_digcount);
            }
            else
            {
                //auto v = std::lldiv(_value.m10, detail::pow10(e10u)); // todo test this
                auto p10 = stringify::v0::detail::pow10(e10u);
                auto integral_part = _value.m10 / p10;
                auto fractional_part = _value.m10 % p10;
                auto idigcount = _m10_digcount - e10u;
                BOOST_ASSERT(idigcount == detail::count_digits<10>(integral_part));

                if (_punct.no_group_separation(_m10_digcount - e10u))
                {
                    stringify::v0::detail::write_int<10>(ob, integral_part, idigcount);
                }
                else
                {
                    stringify::v0::detail::write_int<10>( ob, _punct, _encoding
                                                        , integral_part, idigcount );
                }
                _encoding.encode_char( ob, _punct.decimal_point()
                                     , stringify::v0::encoding_error::replace );
                stringify::v0::detail::write_int_with_leading_zeros<10>
                    (ob, fractional_part, e10u);
            }
        }
    }
}

#if defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class punct_double_printer<char8_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class double_printer<char8_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_double_printer<char8_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_punct_double_printer<char8_t>;
#endif

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class punct_double_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class punct_double_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class punct_double_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class punct_double_printer<wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class double_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class double_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class double_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class double_printer<wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_double_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_double_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_double_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_double_printer<wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_punct_double_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_punct_double_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_punct_double_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_punct_double_printer<wchar_t>;

#endif // defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

} // namespace detail

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_punct<CharT, FPack, float, 10>
    , stringify::v0::detail::fast_punct_double_printer<CharT>
    , stringify::v0::detail::fast_double_printer<CharT> >::type
make_printer(const FPack& fp, float d)
{
    return {fp, d};
}

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_punct<CharT, FPack, double, 10>
    , stringify::v0::detail::fast_punct_double_printer<CharT>
    , stringify::v0::detail::fast_double_printer<CharT> >::type
make_printer(const FPack& fp, double d)
{
    return {fp, d};
}

template <typename CharT, typename FPack, typename FloatT>
inline typename std::conditional
    < stringify::v0::detail::has_punct<CharT, FPack, FloatT, 10>
    , stringify::v0::detail::punct_double_printer<CharT>
    , stringify::v0::detail::double_printer<CharT> >::type
make_printer
    ( const FPack& fp
    , const stringify::v0::value_with_format
            < FloatT
            , stringify::v0::decimal_float_format
            , stringify::v0::empty_alignment_format >& x )
{
    return {fp, x};
}

template <typename CharT, typename FPack, typename FloatT>
inline typename std::conditional
    < stringify::v0::detail::has_punct<CharT, FPack, FloatT, 10>
    , stringify::v0::detail::punct_double_printer<CharT>
    , stringify::v0::detail::double_printer<CharT> >::type
make_printer
    ( const FPack& fp
    , const stringify::v0::value_with_format
            < FloatT
            , stringify::v0::decimal_float_format
            , stringify::v0::alignment_format >& x )
{
    return {fp, x};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_FLOAT_HPP

