#ifndef STRF_WIDTH_T_HPP
#define STRF_WIDTH_T_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

// TODO: Consider whether we need to make the constexpr global variables here STRF_HD.

#include <strf/detail/common.hpp>
#include <type_traits>
#include <cstdint>

STRF_NAMESPACE_BEGIN

class width_t
{
public:

    struct from_underlying_tag{};

    constexpr STRF_HD width_t() noexcept
        : _value(0)
    {
    }

    constexpr STRF_HD width_t(std::int16_t x) noexcept
        : _value(static_cast<std::int32_t>(static_cast<std::uint32_t>(x) << 16))
    {
    }

    // template < typename IntT
    //          , typename = std::enable_if_t< std::is_integral<IntT>::value
    //                                      && (sizeof(IntT) >= 4) >>
    // constexpr explicit width_t(IntT x) noexcept
    // {
    //     STRF_IF_CONSTEXPR ( std::is_unsigned<IntT>::value )
    //     {
    //         _value = x < 0x10000 ? static_cast<std::int32_t>(x) : INT_MAX;
    //     }
    //     else
    //     {
    //         if (x >= INT16_MAX)
    //         {
    //             _value = INT32_MAX
    //         }
    //         else if (x < INT16_MIN)
    //         {
    //             _value = INT32_MIN
    //         }
    //         _value =  static_cast<std::int32_t>(x < 16);
    //     }
    // }

    constexpr STRF_HD width_t(from_underlying_tag, std::int32_t v) noexcept
        : _value(v)
    {
    }

    constexpr static STRF_HD width_t from_underlying(std::int32_t v) noexcept
    {
        return {from_underlying_tag{}, v};
    }

    constexpr STRF_HD width_t(const width_t& other) noexcept
        : _value(other._value)
    {
    }

    constexpr STRF_HD width_t& operator=(const width_t& other) noexcept
    {
        _value = other._value;
        return *this;
    }
    constexpr STRF_HD width_t& operator=(std::int16_t& x) noexcept
    {
        _value = static_cast<std::int32_t>(static_cast<std::uint32_t>(x) << 16);
        return *this;
    }
    constexpr STRF_HD bool operator==(const width_t& other) const noexcept
    {
        return _value == other._value;
    }
    constexpr STRF_HD bool operator!=(const width_t& other) const noexcept
    {
        return _value != other._value;
    }
    constexpr STRF_HD bool operator<(const width_t& other) const noexcept
    {
        return _value < other._value;
    }
    constexpr STRF_HD bool operator>(const width_t& other) const noexcept
    {
        return _value > other._value;
    }
    constexpr STRF_HD bool operator<=(const width_t& other) const noexcept
    {
        return _value <= other._value;
    }
    constexpr STRF_HD bool operator>=(const width_t& other) const noexcept
    {
        return _value >= other._value;
    }
    constexpr STRF_HD bool is_integral() const noexcept
    {
        return (_value & 0xFFFF) == 0;
    }
    constexpr STRF_HD std::int16_t floor() const noexcept
    {
        return static_cast<std::uint16_t>
            ( static_cast<std::uint32_t>(_value) >> 16 );
    }
    constexpr STRF_HD std::int16_t ceil() const noexcept
    {
        auto tmp = static_cast<std::uint32_t>(_value) + 0xFFFF;
        return static_cast<std::uint16_t>(tmp >> 16);
    }
    constexpr STRF_HD std::int16_t round() const noexcept
    {
        std::int16_t i = floor();
        std::uint16_t f = static_cast<std::int16_t>(_value & 0xFFFF);
        constexpr std::uint16_t half = 0x8000;
        return i + (f > half || ((i & 1) && f == half));
    }
    constexpr STRF_HD width_t operator-() const noexcept
    {
        return {from_underlying_tag{}, -_value};
    }
    constexpr STRF_HD width_t operator+() const noexcept
    {
        return *this;
    }
    constexpr STRF_HD width_t& operator+=(width_t other) noexcept

    {
        _value += other._value;
        return *this;
    }
    constexpr STRF_HD width_t& operator-=(width_t other) noexcept
    {
        _value -= other._value;
        return *this;
    }
    constexpr STRF_HD width_t& operator*=(std::int16_t m) noexcept
    {
        _value *= m;
        return *this;
    }
    constexpr STRF_HD width_t& operator/=(std::int16_t d) noexcept
    {
        auto v = static_cast<std::uint64_t>(static_cast<std::uint32_t>(_value));
        auto tmp = (static_cast<std::int64_t>(v << 32) / d);
        _value = static_cast<std::int32_t>(static_cast<std::uint64_t>(tmp) >> 32);
        return *this;
    }
    constexpr STRF_HD width_t& operator*=(width_t other) noexcept
    {
        std::int64_t tmp
            = static_cast<std::int64_t>(_value)
            * static_cast<std::int64_t>(other._value);

        _value = static_cast<std::int32_t>(tmp >> 16);
        return *this;
    }
    constexpr STRF_HD width_t& operator/=(width_t other) noexcept
    {
        auto v = static_cast<std::uint64_t>(static_cast<std::uint32_t>(_value));
        std::int64_t tmp = static_cast<std::int64_t>(v << 32) / other._value;

        _value = static_cast<std::int32_t>(static_cast<std::int64_t>(tmp) >> 16);
        return *this;
    }

    constexpr STRF_HD std::int32_t underlying_value() const noexcept
    {
        return _value;
    }

private:

    std::int32_t _value;
};

constexpr strf::width_t width_max
    = strf::width_t::from_underlying(INT32_MAX);

constexpr strf::width_t width_min
    = strf::width_t::from_underlying(INT32_MIN);


constexpr STRF_HD strf::width_t checked_add( strf::width_t lhs
                                            , strf::width_t rhs ) noexcept
{
    std::int64_t ulhs = lhs.underlying_value();
    std::int64_t urhs = rhs.underlying_value();
    std::int64_t tmp = ulhs + urhs;
    if (INT32_MIN <= tmp && tmp <= INT32_MAX) {
        return strf::width_t::from_underlying(static_cast<std::int32_t>(tmp));
    }
    if (tmp > INT32_MAX) {
        return strf::width_max;
    }
    return strf::width_min;
}

constexpr STRF_HD strf::width_t checked_subtract( strf::width_t lhs
                                                 , std::int64_t rhs ) noexcept
{
    std::int64_t ulhs = lhs.underlying_value();
    std::int64_t tmp = ulhs - rhs;
    if (INT32_MIN <= tmp && tmp <= INT32_MAX) {
        return strf::width_t::from_underlying(static_cast<std::int32_t>(tmp));
    }
    if (tmp < INT32_MIN) {
        return strf::width_min;
    }
    return strf::width_max;
}

constexpr STRF_HD strf::width_t checked_subtract( strf::width_t lhs
                                                 , strf::width_t rhs ) noexcept
{
    return checked_subtract(lhs, rhs.underlying_value());
}

constexpr STRF_HD strf::width_t checked_mul( strf::width_t w
                                            , std::uint32_t x ) noexcept
{
    std::int64_t tmp = x;
    tmp *= w.underlying_value();
    if (INT32_MIN <= tmp && tmp <= INT32_MAX) {
        return strf::width_t::from_underlying(static_cast<std::int32_t>(tmp));
    }
    if (tmp > INT32_MAX) {
        return strf::width_max;
    }
    return strf::width_min;
}

constexpr STRF_HD bool operator==(strf::width_t lhs, std::int16_t rhs ) noexcept
{
    return lhs == strf::width_t{rhs};
}
constexpr STRF_HD bool operator==(std::int16_t lhs, strf::width_t rhs ) noexcept
{
    return strf::width_t{lhs} == rhs;
}
constexpr STRF_HD bool operator!=(strf::width_t lhs, std::int16_t rhs ) noexcept
{
    return lhs != strf::width_t{rhs};
}
constexpr STRF_HD bool operator!=(std::int16_t lhs, strf::width_t rhs ) noexcept
{
    return strf::width_t{lhs} != rhs;
}
constexpr STRF_HD bool operator<(strf::width_t lhs, std::int16_t rhs ) noexcept
{
    return lhs < strf::width_t{rhs};
}
constexpr STRF_HD bool operator<(std::int16_t lhs, strf::width_t rhs ) noexcept
{
    return strf::width_t{lhs} < rhs;
}
constexpr STRF_HD bool operator<=(strf::width_t lhs, std::int16_t rhs ) noexcept
{
    return lhs <= strf::width_t{rhs};
}
constexpr STRF_HD bool operator<=(std::int16_t lhs, strf::width_t rhs ) noexcept
{
    return strf::width_t{lhs} <= rhs;
}
constexpr STRF_HD bool operator>(strf::width_t lhs, std::int16_t rhs ) noexcept
{
    return lhs > strf::width_t{rhs};
}
constexpr STRF_HD bool operator>(std::int16_t lhs, strf::width_t rhs ) noexcept
{
    return strf::width_t{lhs} > rhs;
}
constexpr STRF_HD bool operator>=(strf::width_t lhs, std::int16_t rhs ) noexcept
{
    return lhs >= strf::width_t{rhs};
}
constexpr STRF_HD bool operator>=(std::int16_t lhs, strf::width_t rhs ) noexcept
{
    return strf::width_t{lhs} >= rhs;
}
constexpr STRF_HD strf::width_t operator+( strf::width_t lhs
                                          , strf::width_t rhs ) noexcept
{
    return lhs += rhs;
}
constexpr STRF_HD strf::width_t operator+( std::int16_t lhs
                                          , strf::width_t rhs ) noexcept
{
    return rhs += lhs;
}
constexpr STRF_HD strf::width_t operator+( strf::width_t lhs
                                          , std::int16_t rhs ) noexcept
{
    return lhs += rhs;
}

constexpr STRF_HD strf::width_t operator-( strf::width_t lhs
                                          , strf::width_t rhs ) noexcept
{
    return lhs -= rhs;
}
constexpr STRF_HD strf::width_t operator-( std::int16_t lhs
                                          , strf::width_t rhs ) noexcept
{
    return strf::width_t(lhs) -= rhs;
}
constexpr STRF_HD strf::width_t operator-( strf::width_t lhs
                                          , std::int16_t rhs ) noexcept
{
    return lhs -= rhs;
}

constexpr STRF_HD strf::width_t operator*( strf::width_t lhs
                                          , strf::width_t rhs ) noexcept
{
    return lhs *= rhs;
}
constexpr STRF_HD strf::width_t operator*( std::int16_t lhs
                                          , strf::width_t rhs ) noexcept
{
    return strf::width_t(lhs) *= rhs;
}
constexpr STRF_HD strf::width_t operator*( strf::width_t lhs
                                          , std::int16_t rhs ) noexcept
{
    return lhs *= rhs;
}

constexpr STRF_HD strf::width_t operator/( strf::width_t lhs
                                          , strf::width_t rhs ) noexcept
{
    return lhs /= rhs;
}
constexpr STRF_HD strf::width_t operator/( std::int16_t lhs
                                          , strf::width_t rhs ) noexcept
{
    return strf::width_t(lhs) /= rhs;
}
constexpr STRF_HD strf::width_t operator/( strf::width_t lhs
                                          , std::int16_t rhs ) noexcept
{
    return lhs /= rhs;
}

namespace detail {

template <std::uint64_t X>
struct mp_pow10_impl
{
    static constexpr std::uint64_t value = 10 * mp_pow10_impl<X - 1>::value;
};
template <>
struct mp_pow10_impl<0>
{
    static constexpr std::uint64_t value = 1;
};
template <std::uint64_t x>
constexpr std::uint64_t mp_pow10 = mp_pow10_impl<x>::value;

template <bool preferUp, char ... D>
struct mp_round_up;

template <bool preferUp>
struct mp_round_up<preferUp>
{
    static constexpr bool value = false;
};

#if defined(__cpp_fold_expressions)

template <bool preferUp, char D0, char ... D>
struct mp_round_up<preferUp, D0, D...>
{
    static constexpr bool value
        =  D0 > '5'
        || (D0 == '5' && (preferUp || ((D != '0') || ...)));
};

#else // defined(__cpp_fold_expressions)

template <char ... D>
struct mp_contains_non_zero_digit;

template <>
struct mp_contains_non_zero_digit<>
{
    static constexpr bool value = false;
};

template <char D0, char ... D>
struct mp_contains_non_zero_digit<D0, D...>
{
    static constexpr bool value
    = (D0 != '0' || mp_contains_non_zero_digit<D...>::value);
};

template <bool preferUp, char D0, char ... D>
struct mp_round_up<preferUp, D0, D...>
{
    static constexpr bool value
        =  D0 > '5'
        || (D0 == '5' && (preferUp || mp_contains_non_zero_digit<D...>::value));
};

#endif // defined(__cpp_fold_expressions)

template <unsigned N, char ... D>
struct dec_frac_digits;

template <unsigned N>
struct dec_frac_digits<N>
{
    static constexpr std::uint64_t fvalue = 0;
    static constexpr std::uint64_t fdigcount = 0;
};

template <char D0, char...D>
struct dec_frac_digits<0, D0, D...>
{
    static constexpr std::uint64_t fvalue = 0;
    static constexpr std::uint64_t fdigcount = 0;
};

template <char D0, char...D>
struct dec_frac_digits<1, D0, D...>
{
    static constexpr unsigned _digit = (D0 - '0');
    static constexpr bool _round_up = mp_round_up<(_digit & 1), D...>::value;
    static constexpr std::uint64_t fvalue = _digit + _round_up;
    static constexpr std::uint64_t fdigcount = 1;
};

template <unsigned N, char D0, char...D>
struct dec_frac_digits<N, D0, D...>
{
    static_assert(N > 0, "");
    static_assert('0' <= D0 && D0 <= '9', "");

    using r = dec_frac_digits<N - 1, D...>;

    static constexpr std::uint64_t fdigcount = r::fdigcount + 1;
    static constexpr std::uint64_t fvalue
    = (D0 - '0') * mp_pow10<fdigcount - 1> + r::fvalue;
};

template <unsigned P, char ... Ch>
struct mp_fp_parser_2;

template <unsigned P>
struct mp_fp_parser_2<P>
{
    static constexpr std::uint64_t fvalue = 0;
    static constexpr std::uint64_t fdigcount = 0;
    static constexpr std::uint64_t ivalue = 0;
    static constexpr std::uint64_t idigcount = 0;
};

template <unsigned P>
struct mp_fp_parser_2<P, '.'>
{
    static constexpr std::uint64_t fvalue = 0;
    static constexpr std::uint64_t fdigcount = 0;
    static constexpr std::uint64_t ivalue = 0;
    static constexpr std::uint64_t idigcount = 0;
};

template <unsigned P, char ... F>
struct mp_fp_parser_2<P, '.', F...>
{
    static constexpr std::uint64_t fvalue = dec_frac_digits<P, F...>::fvalue;
    static constexpr std::uint64_t fdigcount = dec_frac_digits<P, F...>::fdigcount;
    static constexpr std::uint64_t ivalue = 0;
    static constexpr std::uint64_t idigcount = 0;
    static constexpr bool negative = false;
};

template <unsigned P, char ... C>
struct mp_fp_parser_2<P, '-', C...>: public mp_fp_parser_2<P, C...>
{
    static constexpr bool negative = true;
};
template <unsigned P, char ... C>
struct mp_fp_parser_2<P, '+', C...>: public mp_fp_parser_2<P, C...>
{
    static constexpr bool negative = false;
};

template <unsigned P, char C0, char ... Ch>
struct mp_fp_parser_2<P, C0, Ch...>
{
    static constexpr std::uint64_t idigcount = mp_fp_parser_2<P, Ch...>::idigcount + 1;
    static constexpr std::uint64_t fdigcount = mp_fp_parser_2<P, Ch...>::fdigcount;
    static constexpr std::uint64_t fvalue = mp_fp_parser_2<P, Ch...>::fvalue;
    static constexpr std::uint64_t ivalue
     = (C0 - '0') * mp_pow10<idigcount - 1> + mp_fp_parser_2<P, Ch...>::ivalue;
    static constexpr bool negative = false;
};

template <unsigned Q, char ... C>
class mp_fixed_point_parser
{
    using helper = detail::mp_fp_parser_2<Q, C...>;
    static constexpr auto divisor = detail::mp_pow10<helper::fdigcount>;
    static constexpr auto frac_ = (helper::fvalue << Q) / divisor;
    static constexpr auto rem_ = (helper::fvalue << Q) % divisor;
    static constexpr bool round_up = (rem_ > (divisor >> 1))
                           || (rem_ == (divisor >> 1) && (frac_ & 1) == 1);
    static constexpr auto frac = frac_ + round_up;

public:

    static constexpr bool negative = helper::negative;
    static constexpr std::uint64_t abs_value = (helper::ivalue << Q) + frac;
    static constexpr std::int64_t value = abs_value * (1 - (negative << 2));
};

} // namespace detail

namespace width_literal {

template <char ... C>
class mp_width_parser
{
    using helper = strf::detail::mp_fixed_point_parser<16, C...>;
    static_assert(helper::abs_value < 0x100000000, "width value too big");

public:

    constexpr static std::int64_t value = helper::value;
};

template <char ... C>
constexpr STRF_HD strf::width_t operator "" _w()
{
    return strf::width_t::from_underlying
        ( static_cast<std::int32_t>(mp_width_parser<C...>::value) );
}

} // namespace width_literal

STRF_NAMESPACE_END

#endif  // STRF_WIDTH_T_HPP

