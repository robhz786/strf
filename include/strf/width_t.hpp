#ifndef STRF_WIDTH_T_HPP
#define STRF_WIDTH_T_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/strf_def.hpp>
#include <type_traits>
#include <cstdint>

namespace strf {
namespace detail {

template <bool IsSigned, std::size_t Size, bool IsIntegral>
struct int_tag_
{
    static_assert(IsIntegral, "");
};

template <typename IntT>
using int_tag = int_tag_
    < std::is_signed<IntT>::value
    , sizeof(IntT)
    , std::is_integral<IntT>::value >;

using tag_i64 = int_tag_<true, 8, true>;
using tag_i32 = int_tag_<true, 4, true>;
using tag_i16 = int_tag_<true, 2, true>;
using tag_i8  = int_tag_<true, 1, true>;

using tag_u64 = int_tag_<false, 8, true>;
using tag_u32 = int_tag_<false, 4, true>;
using tag_u16 = int_tag_<false, 2, true>;
using tag_u8  = int_tag_<false, 1, true>;

} // namespace detail

class width_t
{
public:
    struct from_underlying_tag{};

    constexpr STRF_HD width_t() noexcept
        : value_(0)
    {
    }

    constexpr STRF_HD width_t(std::int16_t x) noexcept
        : value_(detail::cast_i32(detail::cast_u32(detail::cast_u16(x)) << 16))
    {
    }

    constexpr explicit STRF_HD width_t(std::uint16_t x) noexcept
        : width_t(static_cast<std::int16_t>(x))
    {
    }

    template < typename IntT
             , strf::detail::enable_if_t
                   < (2 < sizeof(IntT)) && std::is_integral<IntT>::value, int > = 0 >
    constexpr explicit STRF_HD width_t(IntT x) noexcept
        : width_t(static_cast<std::int16_t>(x))
    {
    }

    constexpr STRF_HD width_t(from_underlying_tag, std::int32_t v) noexcept
        : value_(v)
    {
    }

    constexpr static STRF_HD width_t from_underlying(std::int32_t v) noexcept
    {
        return {from_underlying_tag{}, v};
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD width_t& operator=(std::int16_t& x) noexcept
    {
        return operator=(strf::width_t(x));
    }

    constexpr STRF_HD bool operator==(const width_t& other) const noexcept
    {
        return value_ == other.value_;
    }
    constexpr STRF_HD bool operator!=(const width_t& other) const noexcept
    {
        return value_ != other.value_;
    }
    constexpr STRF_HD bool operator<(const width_t& other) const noexcept
    {
        return value_ < other.value_;
    }
    constexpr STRF_HD bool operator>(const width_t& other) const noexcept
    {
        return value_ > other.value_;
    }
    constexpr STRF_HD bool operator<=(const width_t& other) const noexcept
    {
        return value_ <= other.value_;
    }
    constexpr STRF_HD bool operator>=(const width_t& other) const noexcept
    {
        return value_ >= other.value_;
    }

    constexpr STRF_HD bool zero() const noexcept
    {
        return value_ == 0;
    }
    constexpr STRF_HD bool not_zero() const noexcept
    {
        return value_ != 0;
    }
    constexpr STRF_HD bool gt_zero() const noexcept
    {
        return value_ > 0;
    }
    constexpr STRF_HD bool ge_zero() const noexcept
    {
        return value_ >= 0;
    }
    constexpr STRF_HD bool lt_zero() const noexcept
    {
        return value_ < 0;
    }
    constexpr STRF_HD bool le_zero() const noexcept
    {
        return value_ <= 0;
    }
    constexpr STRF_HD std::int32_t underlying_value() const noexcept
    {
        return value_;
    }
    static constexpr STRF_HD width_t max() noexcept
    {
        return strf::width_t::from_underlying(0x7FFFFFFF);
    }
    static constexpr STRF_HD width_t min() noexcept
    {
        return strf::width_t::from_underlying(static_cast<std::int32_t>(0x80000000));
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD std::int16_t floor() const noexcept
    {
        using namespace strf::detail::cast_sugars;
        // equivalent of arithmetic shift right 16:
        const std::uint32_t leftbits = (value_ < 0 ? 0xFFFF0000U : 0);
        auto result = leftbits | (cast_u32(value_) >> 16);
        return cast_i16(cast_i32(result));
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD std::int32_t round() const noexcept
    {
        std::int32_t i = floor();
        const auto q = value_ & 0xFFFF;
        STRF_IF_LIKELY (value_ >= 0) {
            return q <= 0x8000 ? i : (i + 1);
        }
        return q >= 0x8000 ? (i + 1) : i;
    }
    constexpr STRF_HD width_t operator+() const noexcept
    {
        return *this;
    }
    constexpr STRF_HD width_t operator-() const noexcept
    {
        return from_underlying(-value_);
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD width_t& operator+=(width_t other) noexcept
    {
        value_ += other.value_;
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD width_t& operator-=(width_t other) noexcept
    {
        value_ -= other.value_;
        return *this;
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD width_t& sat_add(width_t x) noexcept
    {
        using namespace strf::detail::cast_sugars;
        auto result = cast_i64(value_) + cast_i64(x.value_);
        value_ = ( result >= max().value_ ? max().value_
                 : result <= min().value_ ? min().value_
                 : cast_i32(result) );
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD width_t& sat_add(detail::tag_i16, std::int16_t x) noexcept
    {
        return sat_add(detail::tag_i32(), strf::detail::cast_i32(x));
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD width_t& sat_add(detail::tag_i32, std::int32_t x) noexcept
    {
        using namespace strf::detail::cast_sugars;
        auto result = value_ + cast_i64(cast_u64(cast_i64(x)) << 16);
        value_ = ( result >= max().value_ ? max().value_
                 : result <= min().value_ ? min().value_
                 : cast_i32(result) );

        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD width_t& sat_add(detail::tag_i64, std::int64_t x) noexcept
    {
        using namespace strf::detail::cast_sugars;
        if (x >= 0x10000) {
            value_ = max().value_;
        } else if (x <= -0x10000) {
            value_ = min().value_;
        } else {
            auto result = value_ + cast_i64(cast_u64(x) << 16);
            value_ = ( result >= max().value_ ? max().value_
                     : result <= min().value_ ? min().value_
                     : cast_i32(result) );
        }
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD width_t& sat_sub(width_t x) noexcept
    {
        using namespace strf::detail::cast_sugars;
        auto result = cast_i64(value_) - cast_i64(x.value_);
        value_ = ( result >= max().value_ ? max().value_
                 : result <= min().value_ ? min().value_
                 : cast_i32(result) );
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD width_t& sat_sub(detail::tag_i16, std::int16_t x) noexcept
    {
        return sat_sub(detail::tag_i32(), strf::detail::cast_i32(x));
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD width_t& sat_sub(detail::tag_i32, std::int32_t x) noexcept
    {
        using namespace strf::detail::cast_sugars;
        auto result = value_ - cast_i64(cast_u64(cast_i64(x)) << 16);
        value_ = ( result >= max().value_ ? max().value_
                 : result <= min().value_ ? min().value_
                 : cast_i32(result) );
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD width_t& sat_sub(detail::tag_i64, std::int64_t x) noexcept
    {
        using namespace strf::detail::cast_sugars;
        if (x >= (cast_i64(max().value_) << 17))  {
            value_ = min().value_;
        } else if (x <= cast_i64(cast_u64(cast_i64(min().value_)) << 17)) {
            value_ = max().value_;
        } else {
            auto result = value_ - cast_i64(cast_u64(cast_i64(x)) << 16);
            value_ = ( result >= max().value_ ? max().value_
                     : result <= min().value_ ? min().value_
                     : cast_i32(result) );
        }
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD width_t& sat_mul(detail::tag_i16, std::int16_t x) noexcept
    {
        return sat_mul(width_t(x));
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD width_t& sat_mul(detail::tag_i64, std::int64_t x) noexcept
    {
        using namespace strf::detail::cast_sugars;
        if (value_ == 0) {
            return *this;
        }
        constexpr std::int64_t mini = width_t::min().underlying_value();
        constexpr std::int64_t maxi = width_t::max().underlying_value();
        if (mini <= x && x <= maxi) {
            const auto result = x * detail::cast_i64(value_);
            if (mini <= result && result <= maxi)
            {
                value_ = detail::cast_i32(result);
                return *this;
            }
        }
        const bool negative = (value_ < 0) ^ (x < 0);
        value_ = (negative ? width_t::min() : width_t::max()).underlying_value();
        return *this;
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD width_t& sat_mul(detail::tag_u64, std::uint64_t x) noexcept
    {
        using namespace strf::detail::cast_sugars;
        if (value_ == 0) {
            return *this;
        }
        constexpr std::int64_t mini = width_t::min().underlying_value();
        constexpr std::int64_t maxi = width_t::max().underlying_value();
        if (x <= cast_u64(maxi)) {
            const auto result = cast_i64(x) * cast_i64(value_);
            if (mini <= result && result <= maxi) {
                value_ = detail::cast_i32(result);
                return *this;
            }
        }
        value_ = (value_ >= 0 ? width_t::max() : width_t::min()).underlying_value();
        return *this;
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD width_t& sat_mul(detail::tag_i32, std::int32_t x) noexcept
    {
        using namespace strf::detail::cast_sugars;
        constexpr std::int64_t min = width_t::min().underlying_value();
        constexpr std::int64_t max = width_t::max().underlying_value();
        const auto result = detail::cast_i64(x) * detail::cast_i64(value_);
        if (min <= result && result <= max) {
            value_ = detail::cast_i32(result);
            return *this;
        }
        const bool negative = (value_ < 0) ^ (x < 0);
        return operator=(negative ? width_t::min() : width_t::max());
    }

    STRF_CONSTEXPR_IN_CXX14 STRF_HD width_t& sat_mul(detail::tag_u32, std::uint32_t x) noexcept
    {
        using namespace strf::detail::cast_sugars;
        constexpr std::int64_t min = width_t::min().underlying_value();
        constexpr std::int64_t max = width_t::max().underlying_value();
        const auto result = cast_i64(x) * cast_i64(cast_u64(value_));
        if (min <= result && result <= max) {
            value_ = detail::cast_i32(result);
            return *this;
        }
        return operator=(value_ >= 0 ? width_t::max() : width_t::min());
    }

    STRF_CONSTEXPR_IN_CXX14
    STRF_HD width_t& sat_mul(width_t other) noexcept
    {
        using namespace strf::detail::cast_sugars;
        const auto tmp =(cast_i64(value_) * cast_i64(other.value_));

        if (tmp >= (1LL << 47)) {
            value_ = 0x7FFFFFFF;
        } else if (tmp < cast_i64(cast_u64(cast_i64(min().underlying_value())) << 16)) {
            value_ = static_cast<std::int32_t>(0x80000000);
        } else {
            value_ = cast_i32(tmp / 0x10000);
        }
        return *this;
    }

    STRF_CONSTEXPR_IN_CXX14
    STRF_HD width_t& operator*=(width_t other) noexcept
    {
        return sat_mul(other);
    }

    STRF_CONSTEXPR_IN_CXX14
    STRF_HD width_t& operator/=(width_t other) noexcept
    {
        using namespace strf::detail::cast_sugars;
        const std::uint64_t vu64 = cast_u32(value_);
        const std::int64_t tmp = cast_i64(vu64 << 16) / other.value_;
        value_ = detail::cast_i32(cast_i64(tmp));
        return *this;
    }

    constexpr STRF_HD int compare(detail::tag_i16, std::int16_t x) const noexcept
    {
        using namespace strf::detail::cast_sugars;
        return value_ - cast_i32(cast_u32(cast_u16(x)) << 16);
    }
    constexpr STRF_HD int compare(detail::tag_u16, std::uint16_t x) const noexcept
    {
        using namespace strf::detail::cast_sugars;
        return x >= cast_u16(0x8000) ? -1 : (value_ - cast_i32(cast_u32(x) << 16));
    }

    constexpr STRF_HD std::int64_t compare(detail::tag_i32, std::int32_t x) const noexcept
    {
        using namespace strf::detail::cast_sugars;
        return cast_i64(value_) - cast_i64(cast_u64(cast_i64(x)) << 16 );
    }
    constexpr STRF_HD std::int64_t compare(detail::tag_u32, std::uint32_t x) const noexcept
    {
        using namespace strf::detail::cast_sugars;
        return cast_i64(value_) - cast_i64(cast_u64(x) << 16);
    }
    constexpr STRF_HD std::int64_t compare(detail::tag_i64, std::int64_t x) const noexcept
    {
        using namespace strf::detail::cast_sugars;
        return ( x >= 0x8000LL ? -1
               : x < -0x8000LL ? +1
               : (cast_i64(value_) - cast_i64(cast_u64(x) << 16)) );
    }
    constexpr STRF_HD std::int64_t compare(detail::tag_u64, std::uint64_t x) const noexcept
    {
        using namespace strf::detail::cast_sugars;
        return x >= 0x8000 ? -1 : (cast_i64(value_) - cast_i64(x << 16));
    }

private:

    std::int32_t value_;
};

constexpr strf::width_t width_max = strf::width_t::max();
constexpr strf::width_t width_min = strf::width_t::min();

STRF_CONSTEXPR_IN_CXX14
STRF_HD strf::width_t sat_sub(strf::width_t w, strf::width_t x) noexcept
{
    return w.sat_sub(x);
}
STRF_CONSTEXPR_IN_CXX14
STRF_HD strf::width_t sat_add(strf::width_t w, strf::width_t x) noexcept
{
    return w.sat_add(x);
}
STRF_CONSTEXPR_IN_CXX14
STRF_HD strf::width_t sat_mul(strf::width_t w, strf::width_t x) noexcept
{
    return w.sat_mul(x);
}

template <typename IntT, detail::enable_if_t<std::is_integral<IntT>::value, int> = 0>
STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::width_t sat_sub(strf::width_t w, IntT i) noexcept
{
    return w.sat_sub(detail::int_tag<IntT>(), i);
}
template <typename IntT, detail::enable_if_t<std::is_integral<IntT>::value, int> = 0>
STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::width_t sat_add(strf::width_t w, IntT i) noexcept
{
    return w.sat_add(detail::int_tag<IntT>(), i);
}
template <typename IntT, detail::enable_if_t<std::is_integral<IntT>::value, int> = 0>
STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::width_t sat_add(IntT i, strf::width_t w) noexcept
{
    return w.sat_add(detail::int_tag<IntT>(), i);
}
template <typename IntT, detail::enable_if_t<std::is_integral<IntT>::value, int> = 0>
STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::width_t sat_mul(strf::width_t w, IntT i) noexcept
{
    return w.sat_mul(detail::int_tag<IntT>(), i);
}
template <typename IntT, detail::enable_if_t<std::is_integral<IntT>::value, int> = 0>
STRF_CONSTEXPR_IN_CXX14 STRF_HD strf::width_t sat_mul(IntT i, strf::width_t w) noexcept
{
    return w.sat_mul(detail::int_tag<IntT>(), i);
}

template <typename IntT, detail::enable_if_t<std::is_integral<IntT>::value, int> = 0>
constexpr STRF_HD auto compare(strf::width_t w, IntT i) noexcept
{
    return w.compare(detail::int_tag<IntT>(), i);
}
template <typename IntT, detail::enable_if_t<std::is_integral<IntT>::value, int> = 0>
constexpr STRF_HD auto compare(IntT i, strf::width_t w) noexcept
{
    return -w.compare(detail::int_tag<IntT>(), i);
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
    return strf::compare(lhs, rhs) < 0;
}
constexpr STRF_HD bool operator<(std::int16_t lhs, strf::width_t rhs ) noexcept
{
    return strf::compare(rhs, lhs) > 0;
}
constexpr STRF_HD bool operator<=(strf::width_t lhs, std::int16_t rhs ) noexcept
{
    return strf::compare(lhs, rhs) <= 0;
}
constexpr STRF_HD bool operator<=(std::int16_t lhs, strf::width_t rhs ) noexcept
{
    return strf::compare(rhs, lhs) >= 0;
}

constexpr STRF_HD bool operator>(strf::width_t lhs, std::int16_t rhs ) noexcept
{
    return strf::compare(lhs, rhs) > 0;
}
constexpr STRF_HD bool operator>(std::int16_t lhs, strf::width_t rhs ) noexcept
{
    return strf::compare(rhs, lhs) < 0;
}
constexpr STRF_HD bool operator>=(strf::width_t lhs, std::int16_t rhs ) noexcept
{
    return strf::compare(lhs, rhs) >= 0;
}
constexpr STRF_HD bool operator>=(std::int16_t lhs, strf::width_t rhs ) noexcept
{
    return strf::compare(rhs, lhs) <= 0;
}


STRF_CONSTEXPR_IN_CXX14
STRF_HD strf::width_t operator+(strf::width_t lhs, strf::width_t rhs) noexcept
{
    return lhs += rhs;
}

STRF_CONSTEXPR_IN_CXX14
STRF_HD strf::width_t operator+(std::int16_t lhs, strf::width_t rhs) noexcept
{
    return rhs += lhs;
}

STRF_CONSTEXPR_IN_CXX14
STRF_HD strf::width_t operator+(strf::width_t lhs, std::int16_t rhs) noexcept
{
    return lhs += rhs;
}


STRF_CONSTEXPR_IN_CXX14
STRF_HD strf::width_t operator-(strf::width_t lhs, strf::width_t rhs) noexcept
{
    return lhs -= rhs;
}
STRF_CONSTEXPR_IN_CXX14
STRF_HD strf::width_t operator-(std::int16_t lhs, strf::width_t rhs) noexcept
{
    return strf::width_t(lhs) -= rhs;
}
STRF_CONSTEXPR_IN_CXX14
STRF_HD strf::width_t operator-(strf::width_t lhs, std::int16_t rhs) noexcept
{
    return lhs -= rhs;
}

#ifndef __CUDACC__
STRF_CONSTEXPR_IN_CXX14
#endif
inline
STRF_HD strf::width_t operator*(strf::width_t lhs, strf::width_t rhs) noexcept
{
    return lhs *= rhs;
}

#ifndef __CUDACC__
STRF_CONSTEXPR_IN_CXX14
#endif
inline
STRF_HD strf::width_t operator*(std::int16_t lhs, strf::width_t rhs) noexcept
{
    return strf::width_t(lhs) *= rhs;
}

#ifndef __CUDACC__
STRF_CONSTEXPR_IN_CXX14
#endif
inline
STRF_HD strf::width_t operator*(strf::width_t lhs, std::int16_t rhs) noexcept
{
    return lhs *= rhs;
}

#ifndef __CUDACC__
STRF_CONSTEXPR_IN_CXX14
#endif
inline
STRF_HD strf::width_t operator/(strf::width_t lhs, strf::width_t rhs) noexcept
{
    return lhs /= rhs;
}

#ifndef __CUDACC__
STRF_CONSTEXPR_IN_CXX14
#endif
inline
STRF_HD strf::width_t operator/(std::int16_t lhs, strf::width_t rhs) noexcept
{
    return strf::width_t(lhs) /= rhs;
}

#ifndef __CUDACC__
STRF_CONSTEXPR_IN_CXX14
#endif
inline
STRF_HD strf::width_t operator/(strf::width_t lhs, std::int16_t rhs) noexcept
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
constexpr STRF_HD std::uint64_t mp_pow10()
{
    return mp_pow10_impl<x>::value;
}

template <bool preferUp, char... D>
struct mp_round_up;

template <bool preferUp>
struct mp_round_up<preferUp>
{
    static constexpr bool value = false;
};

#if defined(__cpp_fold_expressions)

template <bool preferUp, char D0, char... D>
struct mp_round_up<preferUp, D0, D...>
{
    static constexpr bool value
        =  D0 > '5'
        || (D0 == '5' && (preferUp || ((D != '0') || ...)));
};

#else // defined(__cpp_fold_expressions)

template <char... D>
struct mp_contains_non_zero_digit;

template <>
struct mp_contains_non_zero_digit<>
{
    static constexpr bool value = false;
};

template <char D0, char... D>
struct mp_contains_non_zero_digit<D0, D...>
{
    static constexpr bool value
    = (D0 != '0' || mp_contains_non_zero_digit<D...>::value);
};

template <bool preferUp, char D0, char... D>
struct mp_round_up<preferUp, D0, D...>
{
    static constexpr bool value
        =  D0 > '5'
        || (D0 == '5' && (preferUp || mp_contains_non_zero_digit<D...>::value));
};

#endif // defined(__cpp_fold_expressions)

template <unsigned N, char... D>
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
    static constexpr unsigned digit_ = (D0 - '0');
    static constexpr bool round_up_ = mp_round_up<(digit_ & 1), D...>::value;
    static constexpr std::uint64_t fvalue = digit_ + round_up_;
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
    = (D0 - '0') * mp_pow10<fdigcount - 1>() + r::fvalue;
};

template <unsigned P, char... Ch>
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

template <unsigned P, char... F>
struct mp_fp_parser_2<P, '.', F...>
{
    static constexpr std::uint64_t fvalue = dec_frac_digits<P, F...>::fvalue;
    static constexpr std::uint64_t fdigcount = dec_frac_digits<P, F...>::fdigcount;
    static constexpr std::uint64_t ivalue = 0;
    static constexpr std::uint64_t idigcount = 0;
    static constexpr bool negative = false;
};

template <unsigned P, char... C>
struct mp_fp_parser_2<P, '-', C...>: public mp_fp_parser_2<P, C...>
{
    static constexpr bool negative = true;
};
template <unsigned P, char... C>
struct mp_fp_parser_2<P, '+', C...>: public mp_fp_parser_2<P, C...>
{
    static constexpr bool negative = false;
};

template <unsigned P, char C0, char... Ch>
struct mp_fp_parser_2<P, C0, Ch...>
{
    static constexpr std::uint64_t idigcount = mp_fp_parser_2<P, Ch...>::idigcount + 1;
    static constexpr std::uint64_t fdigcount = mp_fp_parser_2<P, Ch...>::fdigcount;
    static constexpr std::uint64_t fvalue = mp_fp_parser_2<P, Ch...>::fvalue;
    static constexpr std::uint64_t ivalue
      = (C0 - '0') * mp_pow10<idigcount - 1>() + mp_fp_parser_2<P, Ch...>::ivalue;
    static constexpr bool negative = false;
};

template <unsigned Q, char... C>
class mp_fixed_point_parser
{
    using helper = detail::mp_fp_parser_2<Q, C...>;
    static constexpr auto divisor = detail::mp_pow10<helper::fdigcount>();
    static constexpr auto frac_ = (helper::fvalue << Q) / divisor;
    static constexpr auto rem_ = (helper::fvalue << Q) % divisor;
    static constexpr bool round_up = (rem_ > (divisor >> 1))
                           || (rem_ == (divisor >> 1) && (frac_ & 1) == 1);
    static constexpr auto frac = frac_ + round_up;

public:

    static constexpr bool negative = helper::negative;
    static constexpr std::uint64_t abs_value = (helper::ivalue << Q) + frac;
};

} // namespace detail

namespace width_literal {

template <char... C>
class mp_width_parser
{
    using helper = strf::detail::mp_fixed_point_parser<16, C...>;
    static_assert(helper::abs_value < 0x100000000, "width value too big");
    static constexpr auto abs_value = static_cast<std::int32_t>(helper::abs_value);
    static constexpr auto negative = helper::negative;

public:

    constexpr static std::int32_t value =  negative ? -abs_value : abs_value;
};

template <char... C>
constexpr STRF_HD strf::width_t operator "" _w()
{
    return strf::width_t::from_underlying(mp_width_parser<C...>::value);
}

} // namespace width_literal

} // namespace strf

#endif  // STRF_WIDTH_T_HPP

