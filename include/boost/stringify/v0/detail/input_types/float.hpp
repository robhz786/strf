#ifndef BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_FLOAT_HPP
#define BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_FLOAT_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/printer.hpp>
#include <boost/stringify/v0/facets_pack.hpp>
#include <boost/stringify/v0/detail/facets/numchars.hpp>
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

    BOOST_ASSERT(-8 <= biased_exponent && biased_exponent <= m_size);
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

    double_printer_data(float d)
        : double_printer_data(detail::decode(d))
    {
    }
    double_printer_data(float d, decimal_float_format_data fmt)
        :  double_printer_data(detail::decode(d), fmt)
    {
    }
    double_printer_data(double d)
        : double_printer_data(detail::decode(d))
    {
    }
    double_printer_data(double d, decimal_float_format_data fmt)
        :  double_printer_data(detail::decode(d), fmt)
    {
    }
    double_printer_data(detail::double_dec d);
    double_printer_data(detail::double_dec d, decimal_float_format_data fmt);

    bool showpoint;
    bool showsign;
    bool sci_notation;
    unsigned m10_digcount;
    unsigned extra_zeros;
};

#if !defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_INLINE double_printer_data::double_printer_data(detail::double_dec d)
    : detail::double_dec(d)
    , showsign(negative)
    , extra_zeros(0)
{
    BOOST_ASSERT(!nan || !infinity);
    if (nan || infinity)
    {
        showpoint = false;
        sci_notation = false;
        m10_digcount = 0;
    }
    else
    {
        m10_digcount = stringify::v0::detail::count_digits<10>(m10);
        sci_notation = ( e10 < - static_cast<int>(m10_digcount) - 3
                      || e10 > static_cast<int>(4 + (m10_digcount > 1)) );
        showpoint = (sci_notation && m10_digcount > 1)
                 || (!sci_notation && e10 < 0);
    }
}

BOOST_STRINGIFY_INLINE double_printer_data::double_printer_data
    ( detail::double_dec d, decimal_float_format_data fmt)
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
                sci_notation = (e10 > 4 + (!fmt.showpoint && m10_digcount > 1))
                    || (e10 < -(int)m10_digcount - 2 - (fmt.showpoint || m10_digcount > 1));
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

                // unsigned p = has_p * fmt.precision
                //     + (fmt.precision == 0)
                //     + (!has_p) * m10_digcount;
                // extra_zeros = 0;
                // sci_notation = ((int)m10_digcount + e10 < -4)
                //     || (e10 - (int)m10_digcount >= p);
                // showpoint = fmt.showpoint
                //     || (sci_notation && p > 1)
                //     || ( !sci_notation
                //       && (e10 < 0 || (p > ((unsigned)e10 + m10_digcount))) );
                break;
            }
            case float_notation::fixed:
            {
                const int frac_digits = (e10 < 0) * -e10;
                //bool has_p = (fmt.precision != (unsigned)-1);
                xz = /*has_p * */ (fmt.precision - frac_digits);
                sci_notation = false;
                showpoint = fmt.showpoint || (fmt.precision != 0);
                // showpoint = fmt.showpoint
                //     || (has_p * fmt.precision != 0)
                //     || (!has_p * frac_digits != 0)
                break;
            }
            default:
            {
                BOOST_ASSERT(fmt.notation == float_notation::scientific);
                const unsigned frac_digits = m10_digcount - 1;
                //bool has_p = (fmt.precision != (unsigned)-1);
                xz = /*has_p * */ (fmt.precision - frac_digits);
                sci_notation = true;
                showpoint = fmt.showpoint || (fmt.precision != 0);
                    // || (has_p * fmt.precision != 0)
                    // || (!has_p * frac_digits != 0);
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
class i18n_double_printer: public stringify::v0::printer<CharT>
{
public:

    template <typename FP, typename FloatT>
    i18n_double_printer
        ( const FP& fp
        , stringify::v0::value_with_format
            < FloatT
            , stringify::v0::decimal_float_format
            , stringify::v0::empty_alignment_format > x );

    int width(int) const override;

    std::size_t necessary_size() const override;

    void write(stringify::v0::output_buffer<CharT>&) const override;

private:

    stringify::v0::detail::double_printer_data _data;
    stringify::v0::encoding_error _enc_err;
    const stringify::v0::encoding<CharT> _encoding;
    const stringify::v0::numchars<CharT>& _chars;
    const stringify::v0::numpunct_base& _punct;
};


template <typename CharT>
template <typename FP, typename FloatT>
i18n_double_printer<CharT>::i18n_double_printer
    ( const FP& fp
    , stringify::v0::value_with_format
        < FloatT
        , stringify::v0::decimal_float_format
        , stringify::v0::empty_alignment_format > x )
    : _data{x.value(), x}
    , _enc_err(get_facet<stringify::v0::encoding_error_c, FloatT>(fp))
    , _encoding(get_facet<stringify::v0::encoding_c<CharT>, FloatT>(fp))
    , _chars(get_facet<stringify::v0::numchars_c<CharT, 10>, FloatT>(fp))
    , _punct(get_facet<stringify::v0::numpunct_c<10>, FloatT>(fp))
{

}

template <typename CharT>
int i18n_double_printer<CharT>::width(int) const
{
    if (_data.infinity || _data.nan)
    {
        return 3 + (_data.showsign && _data.infinity);
    }
    int decpoint_width = _data.showpoint;
    if (_data.sci_notation)
    {
        return _chars.scientific_notation_printwidth
            ( _data.m10_digcount + _data.extra_zeros
            , _data.e10 + (int)_data.m10_digcount - 1
            , _data.showsign )
            + decpoint_width;
    }
    if (_data.e10 < 0)
    {
        bool idig = _data.e10 > -(int)_data.m10_digcount;
        auto idigcount = !idig + idig * ((int)_data.m10_digcount + _data.e10);
        auto fdigcount = _data.extra_zeros + (-_data.e10);
        auto iwidth = _chars.integer_printwidth(idigcount, _data.showsign, false);
        auto fwidth = _chars.fractional_digits_printwidth(fdigcount);
        unsigned seps_width = _punct.thousands_sep_count((idigcount > 0) * idigcount);
        return iwidth + fwidth + decpoint_width + seps_width;
    }
    auto idigcount = _data.m10_digcount + _data.e10;
    auto w = _chars.integer_printwidth( idigcount, _data.showsign, false );
    if (_data.extra_zeros != 0)
    {
        w += _chars.fractional_digits_printwidth(_data.extra_zeros);
    }
    unsigned seps_width = _punct.thousands_sep_count(idigcount);
    return w + seps_width + decpoint_width;
}

template <typename CharT>
std::size_t i18n_double_printer<CharT>::necessary_size() const
{
    if (_data.infinity || _data.nan)
    {
        return 3 + (_data.showsign && _data.infinity);
    }
    if (_data.sci_notation)
    {
        return _chars.scientific_notation_printsize
            ( _encoding
            , _data.m10_digcount + _data.extra_zeros
            , _punct.decimal_point()
            , _data.e10 + (int)_data.m10_digcount - 1
            , _data.showsign, _data.showpoint);
    }
    std::size_t seps_size = 0;
    auto idigcount = (int)_data.m10_digcount + _data.e10;
    if (idigcount > 1 && ! _punct.no_group_separation(idigcount))
    {
        auto s = _encoding.validate(_punct.thousands_sep());
        if (s != (std::size_t)-1)
        {
            seps_size = s * _punct.thousands_sep_count(idigcount);
        }
    }
    if (_data.e10 < 0)
    {
        bool idig = _data.e10 > -(int)_data.m10_digcount;
        auto idigcount = !idig + idig * ((int)_data.m10_digcount + _data.e10);
        auto isize = _chars.integer_printsize( _encoding
                                             , idigcount
                                             , _data.negative
                                             , false );
        auto fsize = _chars.fractional_digits_printsize( _encoding
                                                       , _punct.decimal_point()
                                                       , _data.extra_zeros - _data.e10 );
        return isize + fsize + seps_size;
    }
    auto s = _chars.integer_printsize
        (_encoding, idigcount, _data.showsign, false);
    if (_data.showpoint)
    {
        s += _chars.fractional_digits_printsize( _encoding
                                               , _punct.decimal_point()
                                               , _data.extra_zeros );
    }
    return s + seps_size;
}

template <typename CharT>
void i18n_double_printer<CharT>::write(stringify::v0::output_buffer<CharT>& ob) const
{
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
        if (_data.showsign)
        {
            ob.ensure(4);
            ob.pos()[0] = static_cast<CharT>('+' + (_data.negative << 1));
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
    else if (_data.sci_notation)
    {
        if (_data.showsign)
        {
            _chars.print_sign(ob, _encoding, _data.negative);
        }
        _chars.print_scientific_notation
            ( ob, _encoding, _data.m10, _data.m10_digcount
            , _punct.decimal_point()
            , _data.e10 + _data.m10_digcount - 1
            , _data.showpoint
            , _data.extra_zeros );
    }
    else
    {
        if (_data.showsign)
        {
            _chars.print_sign(ob, _encoding, _data.negative);
        }
        if (_data.e10 >= 0)
        {
            if (_punct.no_group_separation(_data.m10_digcount + _data.e10))
            {
                _chars.print_amplified_integer( ob, _encoding, _data.m10
                                              , _data.m10_digcount, _data.e10 );
            }
            else
            {
                std::uint8_t grps_pool[std::numeric_limits<double>::max_exponent10 + 1];
                char buff[std::numeric_limits<double>::max_exponent10 + 1];
                auto digits = detail::write_int_dec_txtdigits_backwards
                    ( _data.m10, buff + sizeof(buff) );
                _chars.print_amplified_integer
                    ( ob, _encoding, _punct, grps_pool, digits
                    , _data.m10_digcount, _data.e10 );
            }
            if (_data.showpoint)
            {
                _chars.print_fractional_digits
                    ( ob, _encoding, 0, 0, _punct.decimal_point(), _data.extra_zeros);
            }
        }
        else
        {
            unsigned e10u = - _data.e10;
            if (e10u >= _data.m10_digcount)
            {
                _chars.print_single_digit(ob, _encoding, 0);
                _chars.print_fractional_digits
                    ( ob, _encoding, _data.m10, e10u
                    , _punct.decimal_point()
                    , _data.extra_zeros);
            }
            else
            {
                //auto v = std::lldiv(_data.m10, detail::pow10(e10u)); // todo test this
                auto p10 = stringify::v0::detail::pow10(e10u);
                auto integral_part = _data.m10 / p10;
                auto fractional_part = _data.m10 % p10;
                auto idigcount = _data.m10_digcount - e10u;
                std::uint8_t grps_pool[std::numeric_limits<double>::max_exponent10 + 1];
                BOOST_ASSERT(idigcount == detail::count_digits<10>(integral_part));

                if (_punct.no_group_separation(_data.m10))
                {
                    _chars.print_integer( ob, _encoding
                                        , integral_part, idigcount );
                }
                else
                {
                    _chars.print_integer( ob, _encoding, _punct, grps_pool
                                        , integral_part, idigcount );
                }
                _chars.print_fractional_digits
                    ( ob, _encoding, fractional_part, e10u
                    , _punct.decimal_point(), _data.extra_zeros );
            }
        }
    }
}

template <typename CharT>
class double_printer: public stringify::v0::printer<CharT>
{
public:

    template <typename Fp>
    double_printer
        ( const Fp&
        , stringify::v0::value_with_format
            < double
            , stringify::v0::decimal_float_format
            , stringify::v0::empty_alignment_format > x )
            : _data(x.value(), x)
    {
    }

    int width(int) const override;

    void write(stringify::v0::output_buffer<CharT>&) const override;

    std::size_t necessary_size() const override;

private:

    stringify::v0::detail::double_printer_data _data;
};

template <typename CharT>
std::size_t double_printer<CharT>::necessary_size() const
{
    return ( _data.nan * 3
           + _data.infinity * 3
           + (_data.showsign && !_data.nan)
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
    return ( _data.nan * 3
           + _data.infinity * 3
           + (_data.showsign && !_data.nan)
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
    ( stringify::v0::output_buffer<CharT>& ob ) const
{
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
        if (_data.showsign)
        {
            ob.ensure(4);
            ob.pos()[0] = static_cast<CharT>('+' + (_data.negative << 1));
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
    else if (_data.sci_notation)
    {
        ob.ensure( _data.showsign
                 + _data.m10_digcount
                 + _data.showpoint
                 + 4 + (_data.e10 > 99 || _data.e10 < -99) );
        CharT* it = ob.pos();
        if (_data.showsign)
        {
            * it = static_cast<CharT>('+' + (_data.negative << 1));
            ++it;
        }
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
        ob.ensure( _data.showsign + _data.showpoint + _data.m10_digcount
                 + (_data.e10 < -(int)_data.m10_digcount) );
        auto it = ob.pos();
        if (_data.showsign)
        {
            *it = static_cast<CharT>('+' + ((_data.negative) << 1));
            ++it;
        }
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
}


template <typename CharT>
class fast_double_printer: public stringify::v0::printer<CharT>
{
public:

    template <typename FPack>
    fast_double_printer(const FPack, double d)
        : fast_double_printer(d)
    {
    }

    explicit fast_double_printer(double d)
        : _value(decode(d))
        , _m10_digcount(stringify::v0::detail::count_digits<10>(_value.m10))

    {
        BOOST_ASSERT(!_value.nan || !_value.infinity);
        _sci_notation = ( _value.e10 < - static_cast<int>(_m10_digcount) - 3
                       || _value.e10 > static_cast<int>(4 + (_m10_digcount > 1)) );
    }

    int width(int) const override;

    void write(stringify::v0::output_buffer<CharT>&) const override;

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
    ( stringify::v0::output_buffer<CharT>& ob ) const
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
class fast_i18n_double_printer: public stringify::v0::printer<CharT>
{
public:

    template <typename FPack, typename FloatT>
    fast_i18n_double_printer(const FPack& fp, FloatT d)
        : _chars(get_facet<stringify::v0::numchars_c<CharT, 10>, FloatT>(fp))
        , _punct(get_facet<stringify::v0::numpunct_c<10>, FloatT>(fp))
        , _encoding(get_facet<stringify::v0::encoding_c<CharT>, FloatT>(fp))
        , _value(decode(d))
        , _m10_digcount(stringify::v0::detail::count_digits<10>(_value.m10))
        , _sci_notation( _value.e10 < - static_cast<int>(_m10_digcount) - 3
                      || _value.e10 > static_cast<int>(4 + (_m10_digcount > 1)) )
    {
    }

    int width(int) const override;

    void write(stringify::v0::output_buffer<CharT>&) const override;

    std::size_t necessary_size() const override;

private:

    unsigned _size_sci() const
    {
        return _value.negative + _m10_digcount + (_m10_digcount != 1) + 4
            + (_value.e10 > 99 || _value.e10 < -99);

    }

    const stringify::v0::numchars<CharT>& _chars;
    const stringify::v0::numpunct_base& _punct;
    stringify::v0::encoding<CharT> _encoding;
    const detail::double_dec _value;
    const unsigned _m10_digcount;
    bool _sci_notation ;

};

template <typename CharT>
std::size_t fast_i18n_double_printer<CharT>::necessary_size() const
{
    if (_value.infinity || _value.nan)
    {
        return 3 + (_value.negative && _value.infinity);
    }
    if (_sci_notation)
    {
        return _chars.scientific_notation_printsize
            ( _encoding
            , _m10_digcount
            , _punct.decimal_point()
            , _value.e10 + (int)_m10_digcount - 1
            , _value.negative, false );
    }
    std::size_t seps_size = 0;
    auto idigcount = (int)_m10_digcount + _value.e10;
    if (idigcount > 1 && ! _punct.no_group_separation(idigcount))
    {
        auto s = _encoding.validate(_punct.thousands_sep());
        if (s != (std::size_t)-1)
        {
            seps_size = s * _punct.thousands_sep_count(idigcount);
        }
    }
    if (_value.e10 < 0)
    {
        bool idig = _value.e10 > -(int)_m10_digcount;
        auto idigcount = !idig + idig * ((int)_m10_digcount + _value.e10);
        auto isize = _chars.integer_printsize( _encoding
                                             , idigcount
                                             , _value.negative
                                             , false );
        auto fsize = _chars.fractional_digits_printsize( _encoding
                                                       , _punct.decimal_point()
                                                       , -_value.e10 );
        return isize + fsize + seps_size;
    }
    return _chars.integer_printsize( _encoding
                                   , idigcount
                                   , _value.negative
                                   , false )
        + seps_size;
}

template <typename CharT>
int fast_i18n_double_printer<CharT>::width(int) const
{
    if (_value.infinity || _value.nan)
    {
        return 3 + (_value.negative && _value.infinity);
    }
    constexpr unsigned decpoint_width = 1;
    if (_sci_notation)
    {
        return _chars.scientific_notation_printwidth( _m10_digcount
                                                    , (int)_m10_digcount + _value.e10 - 1
                                                    , _value.negative )
            + decpoint_width * (_m10_digcount > 1);
    }
    if (_value.e10 < 0)
    {
        bool idig = _value.e10 > -(int)_m10_digcount;
        auto idigcount = !idig + idig * ((int)_m10_digcount + _value.e10);
        auto fdigcount = - _value.e10;
        auto iwidth = _chars.integer_printwidth(idigcount, _value.negative, false);
        auto fwidth = _chars.fractional_digits_printwidth(fdigcount);
        unsigned seps_width = _punct.thousands_sep_count((idigcount > 0) * idigcount);
        return iwidth + fwidth + decpoint_width + seps_width;
    }
    return _chars.integer_printwidth( _m10_digcount + _value.e10
                                    , _value.negative
                                    , false );
}

template <typename CharT>
void fast_i18n_double_printer<CharT>::write
    ( stringify::v0::output_buffer<CharT>& ob ) const
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
        if (_value.negative)
        {
            _chars.print_neg_sign(ob, _encoding);
        }
        _chars.print_scientific_notation
            ( ob, _encoding, _value.m10, _m10_digcount, _punct.decimal_point()
            , _value.e10 + static_cast<int>(_m10_digcount) - 1 );
    }
    else
    {
        if (_value.negative)
        {
            _chars.print_neg_sign(ob, _encoding);
        }
        std::uint8_t grps_pool[std::numeric_limits<double>::max_exponent10 + 1];
        if (_value.e10 >= 0)
        {
            if (_punct.no_group_separation(_value.m10))
            {
                _chars.print_amplified_integer( ob, _encoding, _value.m10
                                              , _m10_digcount, _value.e10 );
            }
            else
            {
                char buff[std::numeric_limits<double>::max_exponent10 + 1];
                auto digits = detail::write_int_dec_txtdigits_backwards
                    ( _value.m10, buff + sizeof(buff) );
                _chars.print_amplified_integer
                    ( ob, _encoding, _punct, grps_pool, digits
                    , _m10_digcount, _value.e10 );
            }
        }
        else
        {
            unsigned e10u = - _value.e10;
            if (e10u >= _m10_digcount)
            {
                _chars.print_single_digit(ob, _encoding, 0);
                _chars.print_fractional_digits
                    ( ob, _encoding, _value.m10, e10u, _punct.decimal_point() );
            }
            else
            {
                //auto v = std::lldiv(_value.m10, detail::pow10(e10u)); // todo test this
                auto p10 = stringify::v0::detail::pow10(e10u);
                auto integral_part = _value.m10 / p10;
                auto fractional_part = _value.m10 % p10;
                auto idigcount = _m10_digcount - e10u;
                BOOST_ASSERT(idigcount == detail::count_digits<10>(integral_part));

                if (_punct.no_group_separation(_value.m10))
                {
                    _chars.print_integer( ob, _encoding
                                        , integral_part, idigcount );
                }
                else
                {
                    _chars.print_integer( ob, _encoding, _punct, grps_pool
                                        , integral_part, idigcount );
                }
                _chars.print_fractional_digits
                    ( ob, _encoding, fractional_part, e10u
                    , _punct.decimal_point() );
            }
        }
    }
}


#if defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class i18n_double_printer<char8_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class double_printer<char8_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_double_printer<char8_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_i18n_double_printer<char8_t>;
#endif

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class i18n_double_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class i18n_double_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class i18n_double_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class i18n_double_printer<wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class double_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class double_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class double_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class double_printer<wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_double_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_double_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_double_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_double_printer<wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_i18n_double_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_i18n_double_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_i18n_double_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fast_i18n_double_printer<wchar_t>;

#endif // defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

} // namespace detail

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_i18n<CharT, FPack, float, 10>
    , stringify::v0::detail::fast_i18n_double_printer<CharT>
    , stringify::v0::detail::fast_double_printer<CharT> >::type
make_printer(const FPack& fp, float d)
{
    return {fp, d};
}

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_i18n<CharT, FPack, double, 10>
    , stringify::v0::detail::fast_i18n_double_printer<CharT>
    , stringify::v0::detail::fast_double_printer<CharT> >::type
make_printer(const FPack& fp, double d)
{
    return {fp, d};
}

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_i18n<CharT, FPack, float, 10>
    , stringify::v0::detail::i18n_double_printer<CharT>
    , stringify::v0::detail::double_printer<CharT> >::type
make_printer
    ( const FPack& fp
    , const stringify::v0::value_with_format
            < float
            , stringify::v0::decimal_float_format
            , stringify::v0::empty_alignment_format >& x )
{
    return {fp, x};
}

template <typename CharT, typename FPack>
inline typename std::conditional
    < stringify::v0::detail::has_i18n<CharT, FPack, double, 10>
    , stringify::v0::detail::i18n_double_printer<CharT>
    , stringify::v0::detail::double_printer<CharT> >::type
make_printer
    ( const FPack& fp
    , const stringify::v0::value_with_format
            < double
            , stringify::v0::decimal_float_format
            , stringify::v0::empty_alignment_format >& x )
{
    return {fp, x};
}

template <typename CharT, typename FPack>
inline stringify::v0::detail::fast_double_printer<CharT> make_printer
    ( const FPack&
    , const stringify::v0::value_with_format
            < float
            , stringify::v0::decimal_float_format
            , stringify::v0::alignment_format >& x )
{
    static_assert
        ( ! std::is_void<FPack>::value // aways true
        , "Alignment formatting for floating points not supported yet. Sorry." );
    return {x.value()};
}

template <typename CharT, typename FPack>
inline stringify::v0::detail::fast_double_printer<CharT> make_printer
    ( const FPack&
    , const stringify::v0::value_with_format
            < double
            , stringify::v0::decimal_float_format
            , stringify::v0::alignment_format >& x )
{
    static_assert
        ( ! std::is_void<FPack>::value // aways true
        , "Alignment formatting for floating points not supported yet. Sorry." );
    return {x.value()};
}


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_FLOAT_HPP

