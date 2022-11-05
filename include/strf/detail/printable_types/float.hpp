#ifndef STRF_DETAIL_PRINTABLE_TYPES_FLOAT_HPP
#define STRF_DETAIL_PRINTABLE_TYPES_FLOAT_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/format_functions.hpp>
#include <strf/detail/facets/numpunct.hpp>
#include <strf/detail/facets/charset.hpp>
#include <strf/detail/dragonbox.hpp>

namespace strf {
namespace detail {

inline STRF_HD std::uint32_t to_bits(float f)
{
    return strf::detail::bit_cast<std::uint32_t>(f);
}

inline STRF_HD std::uint64_t to_bits(const double d)
{
    return strf::detail::bit_cast<std::uint64_t>(d);
}

struct double_dec
{
    std::uint64_t m10;
    std::int32_t e10;
    bool negative;
    bool infinity;
    bool nan;
};

#if ! defined(STRF_OMIT_IMPL)

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD detail::double_dec decode(float f)
{
    constexpr int e_size = 8;
    constexpr int m_size = 23;

    const std::uint32_t bits = strf::detail::to_bits(f);
    const auto exponent
        = static_cast<std::uint32_t>((bits << 1) >> (m_size + 1));
    const bool sign = (bits >> (m_size + e_size));
    const std::uint32_t mantissa = bits & 0x7FFFFF;

    if (exponent == 0 && mantissa == 0) {
        return {0, 0, sign, false, false};
    } else if (exponent == 0xFF) {
        if (mantissa == 0) {
            return {0, 0, sign, true, false};
        } else {
            return {0, 0, sign, false, true};
        }
    }
    namespace dragonbox = strf::detail::jkj::dragonbox;
    auto fdec = dragonbox::to_decimal<float>(exponent, mantissa);
    return {fdec.significand, fdec.exponent, sign, false, false};
}

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD detail::double_dec decode(double d)
{
    constexpr int e_size = 11; // bits in exponent
    constexpr int m_size = 52; // bits in matissa

    const std::uint64_t bits = strf::detail::to_bits(d);
    const auto exponent
        = static_cast<std::uint32_t>((bits << 1) >> (m_size + 1));
    const bool sign = (bits >> (m_size + e_size));
    const std::uint64_t mantissa = bits & 0xFFFFFFFFFFFFFULL;

    if (exponent == 0 && mantissa == 0) {
        return {0, 0, sign, false, false};
    } else if (exponent == 0x7FF) {
        if (mantissa == 0) {
            return {0, 0, sign, true, false};
        } else {
            return {0, 0, sign, false, true};
        }
    }
    namespace dragonbox = strf::detail::jkj::dragonbox;
    auto ddec = dragonbox::to_decimal<double>(exponent, mantissa);
    return {ddec.significand, ddec.exponent, sign, false, false};
}

#else  // ! defined(STRF_OMIT_IMPL)

STRF_HD detail::double_dec decode(double d);
STRF_HD detail::double_dec decode(float f);

#endif // ! defined(STRF_OMIT_IMPL)

using chars_count_t = unsigned;

} // namespace detail

enum class float_notation{fixed, scientific, general, hex};

struct float_format
{
#if __cplusplus < 201402L

    constexpr STRF_HD float_format
      ( detail::chars_count_t precision_ = (detail::chars_count_t)-1
      , detail::chars_count_t pad0width_ = 0
      , strf::float_notation notation_ = strf::float_notation::general
      , strf::showsign sign_ = strf::showsign::negative_only
      , bool showpoint_ = false
      , bool punctuate_ = false ) noexcept
      : precision(precision_)
      , pad0width(pad0width_)
      , notation(notation_)
      , sign(sign_)
      , showpoint(showpoint_)
      , punctuate(punctuate_)
    {
    }

#endif // __cplusplus < 201402L

    detail::chars_count_t precision = (detail::chars_count_t)-1;
    detail::chars_count_t pad0width = 0;
    strf::float_notation notation = strf::float_notation::general;
    strf::showsign sign = strf::showsign::negative_only;
    bool showpoint = false;
    bool punctuate = false;
};

struct float_format_no_punct
{
    detail::chars_count_t precision = (detail::chars_count_t)-1;
    detail::chars_count_t pad0width = 0;
    strf::float_notation notation = strf::float_notation::general;
    strf::showsign sign = strf::showsign::negative_only;
    bool showpoint = false;
    constexpr static bool punctuate = false;

    constexpr STRF_HD operator float_format () const noexcept
    {
        return {precision, pad0width, notation, sign, showpoint, false};
    }
};

struct default_float_format
{
    constexpr static detail::chars_count_t precision = (detail::chars_count_t)-1;
    constexpr static detail::chars_count_t pad0width = 0;
    constexpr static strf::float_notation notation = strf::float_notation::general;
    constexpr static strf::showsign sign = strf::showsign::negative_only;
    constexpr static bool showpoint = false;
    constexpr static bool punctuate = false;

    constexpr STRF_HD operator float_format () const noexcept
    {
        return {};
    }
    constexpr STRF_HD operator float_format_no_punct () const noexcept
    {
        return {};
    }
};

template <typename T>
class default_float_formatter_fn;

template <typename T>
class float_formatter_no_punct_fn;

template <typename T>
class float_formatter_full_dynamic_fn;

struct float_formatter
{
    template <typename T>
    using fn = default_float_formatter_fn<T>;

    constexpr static bool has_float_formatting = false;
    constexpr static bool has_punct = false;
};

struct float_formatter_no_punct
{
    template <typename T>
    using fn = float_formatter_no_punct_fn<T>;

    constexpr static bool has_float_formatting = true;
    constexpr static bool has_punct = false;
};

struct float_formatter_full_dynamic
{
    template <typename T>
    using fn = float_formatter_full_dynamic_fn<T>;

    constexpr static bool has_float_formatting = true;
    constexpr static bool has_punct = true;
};

template <typename T>
class default_float_formatter_fn
{
    using adapted_to_full_dynamic_ = strf::fmt_replace
        <T, float_formatter, float_formatter_full_dynamic>;

    using adapted_to_no_punct_ = strf::fmt_replace
        <T, float_formatter, float_formatter_no_punct>;

public:

    constexpr default_float_formatter_fn() noexcept = default;

    constexpr STRF_HD explicit default_float_formatter_fn
        ( const strf::default_float_format& ) noexcept
    {
    }
    template <typename U>
    constexpr STRF_HD explicit default_float_formatter_fn
        ( const default_float_formatter_fn<U>& ) noexcept
    {
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD adapted_to_no_punct_ operator+() const & noexcept
    {
        float_format format;
        format.sign = strf::showsign::positive_also;
        return { self_downcast_()
               , strf::tag<float_formatter_no_punct>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD adapted_to_no_punct_ fill_sign() const & noexcept
    {
        float_format format;
        format.sign = strf::showsign::fill_instead_of_positive;
        return { self_downcast_()
               , strf::tag<float_formatter_no_punct>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD adapted_to_no_punct_ operator~() const & noexcept
    {
        float_format format;
        format.sign = strf::showsign::fill_instead_of_positive;
        return { self_downcast_()
               , strf::tag<float_formatter_no_punct>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD adapted_to_no_punct_ operator*() const & noexcept
    {
        float_format format;
        format.showpoint = true;
        return { self_downcast_()
               , strf::tag<float_formatter_no_punct>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD adapted_to_full_dynamic_ operator!() const & noexcept
    {
        float_format format;
        format.punctuate = true;
        return { self_downcast_()
               , strf::tag<float_formatter_full_dynamic>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD adapted_to_full_dynamic_ punct() const & noexcept
    {
        float_format format;
        format.punctuate = true;
        return { self_downcast_()
               , strf::tag<float_formatter_full_dynamic>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD adapted_to_no_punct_ p(detail::chars_count_t _) const & noexcept
    {
        float_format format;
        format.precision = _;
        return { self_downcast_()
               , strf::tag<float_formatter_no_punct>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD adapted_to_no_punct_ pad0(detail::chars_count_t width) const & noexcept
    {
        float_format format;
        format.pad0width = width;
        return { self_downcast_()
               , strf::tag<float_formatter_no_punct>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD adapted_to_no_punct_
    float_notation(strf::float_notation n) const & noexcept
    {
        float_format format;
        format.notation = n;
        return { self_downcast_()
               , strf::tag<float_formatter_no_punct>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD adapted_to_no_punct_ sci() const & noexcept
    {
        float_format format;
        format.notation = strf::float_notation::scientific;
        return { self_downcast_()
               , strf::tag<float_formatter_no_punct>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD adapted_to_no_punct_ fixed() const & noexcept
    {
        float_format format;
        format.notation = strf::float_notation::fixed;
        return { self_downcast_()
               , strf::tag<float_formatter_no_punct>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD adapted_to_no_punct_ hex() const & noexcept
    {
        float_format format;
        format.notation = strf::float_notation::hex;
        return { self_downcast_()
               , strf::tag<float_formatter_no_punct>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD const T& gen() const & noexcept
    {
        return self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T& gen() & noexcept
    {
        return self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& gen() && noexcept
    {
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD adapted_to_no_punct_
    set_float_notation(strf::float_notation notation) const & noexcept
    {
        float_format format;
        format.notation = notation;
        return { self_downcast_()
               , strf::tag<float_formatter_no_punct>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD adapted_to_full_dynamic_
    set_float_format(strf::float_format format) && noexcept
    {
        return { self_downcast_()
               , strf::tag<float_formatter_full_dynamic>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD adapted_to_no_punct_
    set_float_format(strf::float_format_no_punct format) && noexcept
    {
        return { self_downcast_()
               , strf::tag<float_formatter_no_punct>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD const T& set_float_format(strf::default_float_format) const & noexcept
    {
        return self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T& set_float_format(strf::default_float_format) & noexcept
    {
        return self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& set_float_format(strf::default_float_format) && noexcept
    {
        return move_self_downcast_();
    }

    // observers

    constexpr STRF_HD strf::float_notation float_notation() const noexcept
    {
        return strf::float_notation::general;
    }
    constexpr STRF_HD strf::default_float_format get_float_format() const noexcept
    {
        return {};
    }
    constexpr STRF_HD detail::chars_count_t pad0width() const
    {
        return 0;
    }

private:

    STRF_HD STRF_CONSTEXPR_IN_CXX14 const T& self_downcast_() const
    {
        const T* base_ptr = static_cast<const T*>(this);
        return *base_ptr;
    }
    STRF_HD STRF_CONSTEXPR_IN_CXX14 T& self_downcast_()
    {
        T* base_ptr = static_cast<T*>(this);
        return *base_ptr;
    }
    STRF_HD STRF_CONSTEXPR_IN_CXX14 T&& move_self_downcast_()
    {
        T* base_ptr = static_cast<T*>(this);
        return static_cast<T&&>(*base_ptr);
    }
};

template <typename T>
class float_formatter_no_punct_fn
{
public:

    constexpr float_formatter_no_punct_fn() noexcept = default;

    constexpr STRF_HD explicit float_formatter_no_punct_fn
        ( const strf::float_format& data ) noexcept
        : data_(data)
    {
    }
    template <typename U>
    constexpr STRF_HD explicit float_formatter_no_punct_fn
        ( const float_formatter_no_punct_fn<U>& other ) noexcept
        : data_(other.get_float_format())
    {
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator+() && noexcept
    {
        data_.sign = strf::showsign::positive_also;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& fill_sign() && noexcept
    {
        data_.sign = strf::showsign::fill_instead_of_positive;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator~() && noexcept
    {
        data_.sign = strf::showsign::fill_instead_of_positive;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator*() && noexcept
    {
        data_.showpoint = true;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD
    strf::fmt_replace<T, float_formatter_no_punct, float_formatter_full_dynamic>
    punct() const & noexcept
    {
        float_format format{data_};
        format.punctuate = true;
        return { self_downcast_()
               , strf::tag<float_formatter_full_dynamic>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD
    strf::fmt_replace<T, float_formatter_no_punct, float_formatter_full_dynamic>
    operator!() const & noexcept
    {
        float_format format{data_};
        format.punctuate = true;
        return { self_downcast_()
               , strf::tag<float_formatter_full_dynamic>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& p(detail::chars_count_t _) && noexcept
    {
        data_.precision = _;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& pad0(detail::chars_count_t width) && noexcept
    {
        data_.pad0width = width;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& sci() && noexcept
    {
        data_.notation = strf::float_notation::scientific;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& fixed() && noexcept
    {
        data_.notation = strf::float_notation::fixed;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& gen() && noexcept
    {
        data_.notation = strf::float_notation::general;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& hex() && noexcept
    {
        data_.notation = strf::float_notation::hex;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& float_notation(strf::float_notation n) && noexcept
    {
        data_.notation = n;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&&
    set_float_notation(strf::float_notation notation) && noexcept
    {
        data_.notation = notation;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& set_float_format(strf::float_format_no_punct data) && noexcept
    {
        data_ = data;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD
    strf::fmt_replace<T, float_formatter_no_punct, float_formatter>
    set_float_format(strf::default_float_format format) const & noexcept
    {
        return { self_downcast_()
               , strf::tag<float_formatter>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD
    strf::fmt_replace<T, float_formatter_no_punct, float_formatter_full_dynamic>
    set_float_format(strf::float_format format) const & noexcept
    {
        return { self_downcast_()
               , strf::tag<float_formatter_full_dynamic>{}
               , format };
    }
    constexpr STRF_HD strf::float_notation float_notation() const noexcept
    {
        return data_.notation;
    }
    constexpr STRF_HD strf::float_format get_float_format() const noexcept
    {
        return data_;
    }
    constexpr STRF_HD detail::chars_count_t pad0width() const
    {
        return data_.pad0width;
    }

private:

    STRF_HD STRF_CONSTEXPR_IN_CXX14 const T& self_downcast_() const
    {
        const T* base_ptr = static_cast<const T*>(this);
        return *base_ptr;
    }
    STRF_HD STRF_CONSTEXPR_IN_CXX14 T& self_downcast_()
    {
        T* base_ptr = static_cast<T*>(this);
        return *base_ptr;
    }
    STRF_HD STRF_CONSTEXPR_IN_CXX14 T&& move_self_downcast_()
    {
        T* base_ptr = static_cast<T*>(this);
        return static_cast<T&&>(*base_ptr);
    }
    strf::float_format data_;
};

template <typename T>
class float_formatter_full_dynamic_fn
{
public:

    constexpr float_formatter_full_dynamic_fn() noexcept = default;

    constexpr STRF_HD explicit float_formatter_full_dynamic_fn
        ( const strf::float_format& data ) noexcept
        : data_(data)
    {
    }
    template <typename U>
    constexpr STRF_HD explicit float_formatter_full_dynamic_fn
        ( const float_formatter_full_dynamic_fn<U>& other ) noexcept
        : data_(other.get_float_format())
    {
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator+() && noexcept
    {
        data_.sign = strf::showsign::positive_also;
        T* base_ptr = static_cast<T*>(this); // work around UBSan bug
        return static_cast<T&&>(*base_ptr);
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& fill_sign() && noexcept
    {
        data_.sign = strf::showsign::fill_instead_of_positive;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator~() && noexcept
    {
        data_.sign = strf::showsign::fill_instead_of_positive;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator*() && noexcept
    {
        data_.showpoint = true;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& operator!() && noexcept
    {
        data_.punctuate = true;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& punct() && noexcept
    {
        data_.punctuate = true;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& p(detail::chars_count_t _) && noexcept
    {
        data_.precision = _;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& pad0(detail::chars_count_t width) && noexcept
    {
        data_.pad0width = width;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& sci() && noexcept
    {
        data_.notation = strf::float_notation::scientific;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& fixed() && noexcept
    {
        data_.notation = strf::float_notation::fixed;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& gen() && noexcept
    {
        data_.notation = strf::float_notation::general;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& hex() && noexcept
    {
        data_.notation = strf::float_notation::hex;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&&
    set_float_notation(strf::float_notation notation) && noexcept
    {
        data_.notation = notation;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& float_notation(strf::float_notation n) && noexcept
    {
        data_.notation = n;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& set_float_format(strf::float_format data) && noexcept
    {
        data_ = data;
        return move_self_downcast_();
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD
    strf::fmt_replace<T, float_formatter_full_dynamic, float_formatter_no_punct>
    set_float_format(strf::float_format_no_punct format) && noexcept
    {
        return { self_downcast_()
               , strf::tag<float_formatter_no_punct>{}
               , format };
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD
    strf::fmt_replace<T, float_formatter_full_dynamic, float_formatter>
    set_float_format(strf::default_float_format format) const & noexcept
    {
        return { self_downcast_()
               , strf::tag<float_formatter>{}
               , format };
    }
    constexpr STRF_HD strf::float_notation float_notation() const noexcept
    {
        return data_.notation;
    }
    constexpr STRF_HD strf::float_format get_float_format() const noexcept
    {
        return data_;
    }
    constexpr STRF_HD detail::chars_count_t pad0width() const
    {
        return data_.pad0width;
    }

private:

    STRF_HD STRF_CONSTEXPR_IN_CXX14 const T& self_downcast_() const
    {
        const T* base_ptr = static_cast<const T*>(this);
        return *base_ptr;
    }
    STRF_HD STRF_CONSTEXPR_IN_CXX14 T& self_downcast_()
    {
        T* base_ptr = static_cast<T*>(this);
        return *base_ptr;
    }
    STRF_HD STRF_CONSTEXPR_IN_CXX14 T&& move_self_downcast_()
    {
        T* base_ptr = static_cast<T*>(this);
        return static_cast<T&&>(*base_ptr);
    }
    strf::float_format data_;
};

namespace detail {

template <typename> class fast_double_printer;
// template <typename> class fast_punct_double_printer;
template <typename> class punct_double_printer;

template <typename> struct float_printing;

template<typename FloatT, typename FloatFormatter, bool Align>
using float_with_formatters = strf::printable_with_fmt
    < strf::printable_traits<FloatT>
    , FloatFormatter
    , strf::alignment_formatter_q<Align> >;

template<typename FloatT>
using float_with_default_formatters = strf::printable_with_fmt
    < strf::printable_traits<FloatT>
    , strf::float_formatter
    , strf::alignment_formatter >;

template <typename CharT, typename PrePrinting, typename FloatT>
struct fast_double_printer_input
{
    using printer_type = strf::detail::fast_double_printer<CharT>;

    template <typename FPack>
    STRF_HD fast_double_printer_input(PrePrinting& pre_, const FPack& fp_, FloatT arg_)
        : pre(pre_)
        , value(arg_)
        , lcase(strf::use_facet<strf::lettercase_c, float>(fp_))
    {
    }

    template <typename FPack>
    STRF_HD fast_double_printer_input
        ( PrePrinting& pre_
        , const FPack& fp_
        , strf::detail::float_with_default_formatters<FloatT> input )
        : pre(pre_)
        , value(input.value())
        , lcase(strf::use_facet<strf::lettercase_c, float>(fp_))
    {
    }

    PrePrinting& pre;
    FloatT value;
    strf::lettercase lcase;
};


// template <typename CharT, typename PrePrinting, typename FPack, typename FloatT>
// using fast_punct_double_printer_input =
//     strf::usual_printer_input< CharT, PrePrinting, FPack, FloatT
//                              , strf::detail::fast_punct_double_printer<CharT> >;

template < typename CharT, typename PrePrinting, typename FPack
         , typename FloatT, typename FloatFormatter, bool HasAlignment >
using fmt_double_printer_input =
    strf::usual_printer_input
        < CharT, PrePrinting, FPack
        , strf::detail::float_with_formatters<FloatT, FloatFormatter, HasAlignment>
        , strf::detail::punct_double_printer<CharT> >;

template <typename FloatT>
struct float_printing
{
    using representative_type = FloatT;
    using forwarded_type = FloatT;
    using formatters = strf::tag<strf::float_formatter, strf::alignment_formatter>;
    using is_overridable = std::true_type;

    template <typename CharT, typename PrePrinting, typename FPack>
    STRF_HD constexpr static auto make_input
        ( strf::tag<CharT>, PrePrinting& pre, const FPack& fp, FloatT x ) noexcept
        -> strf::detail::fast_double_printer_input<CharT, PrePrinting, FloatT>
    {
        return {pre, fp, x};
    }

    template < typename CharT, typename PrePrinting, typename FPack
             , typename FloatFormatter, bool HasAlignment >
    STRF_HD constexpr static auto make_input
        ( strf::tag<CharT>
        , PrePrinting& pre
        , const FPack& fp
        , strf::detail::float_with_formatters
            < FloatT, FloatFormatter, HasAlignment > x ) noexcept
        -> strf::detail::conditional_t
            < HasAlignment || FloatFormatter::has_float_formatting
            , strf::detail::fmt_double_printer_input
                < CharT, PrePrinting, FPack, FloatT, FloatFormatter, HasAlignment >
            , fast_double_printer_input<CharT, PrePrinting, FloatT> >
    {
        return {pre, fp, x};
    }
};

} // namespace detail

template <> struct printable_traits<float>:  public strf::detail::float_printing<float> {};
template <> struct printable_traits<double>: public strf::detail::float_printing<double> {};

STRF_HD constexpr auto tag_invoke(strf::printable_tag, float)
    -> strf::printable_traits<float>
    { return {}; }

STRF_HD constexpr auto tag_invoke(strf::printable_tag, double)
    -> strf::printable_traits<double>
    { return {}; }

void tag_invoke(strf::printable_tag, long double) = delete;

namespace detail {

template <int Base, typename CharT, typename IntT>
inline STRF_HD void write_int_with_leading_zeros
    ( strf::destination<CharT>& dest
    , IntT value
    , unsigned digcount
    , strf::lettercase lc )
{
    dest.ensure(digcount);
    auto *p = dest.buffer_ptr();
    auto *end = p + digcount;
    using writer = detail::intdigits_backwards_writer<Base>;
    auto *p2 = writer::write_txtdigits_backwards(value, end, lc);
    if (p != p2) {
        strf::detail::str_fill_n<CharT>(p, p2 - p, '0');
    }
    dest.advance_to(end);
}

template <typename CharT>
STRF_HD void print_amplified_integer_small_separator_1
    ( strf::destination<CharT>& dest
    , unsigned long long value
    , unsigned num_digits
    , strf::digits_distribution dist
    , CharT separator )
{
    STRF_ASSERT(num_digits <= dist.highest_group);

    dest.ensure(num_digits);
    auto *ptr = dest.buffer_ptr() + num_digits;
    strf::detail::write_int_dec_txtdigits_backwards(value, ptr);
    dest.advance_to(ptr);
    dist.highest_group -= num_digits;
    if (dist.highest_group != 0) {
        strf::detail::write_fill(dest, dist.highest_group, (CharT)'0');
    }
    if (dist.middle_groups_count) {
        auto middle_groups = dist.low_groups.highest_group();
        dist.low_groups.pop_high();
        do {
            dest.ensure(middle_groups + 1);
            auto *oit = dest.buffer_ptr();
            *oit = separator;
            strf::detail::str_fill_n<CharT>(++oit, middle_groups, '0');
            dest.advance_to(oit + middle_groups);
        } while (--dist.middle_groups_count);
    }
    while ( ! dist.low_groups.empty()) {
        auto grp = dist.low_groups.highest_group();
        dest.ensure(grp + 1);
        auto *oit = dest.buffer_ptr();
        *oit = separator;
        strf::detail::str_fill_n<CharT>(++oit, grp, '0');
        dest.advance_to(oit + grp);
        dist.low_groups.pop_high();
    }
}

template <typename CharT>
STRF_HD void print_amplified_integer_small_separator_2
    ( strf::destination<CharT>& dest
    , unsigned long long value
    , unsigned num_digits
    , strf::digits_distribution dist
    , CharT separator )
{
    STRF_ASSERT(dist.highest_group < num_digits);

    constexpr std::size_t size_after_recycle = strf::min_destination_buffer_size;
    (void) size_after_recycle;

    constexpr auto max_digits = detail::max_num_digits<unsigned long long, 10>();
    char digits_buff[max_digits];
    auto *digits = strf::detail::write_int_dec_txtdigits_backwards
        (value, digits_buff + max_digits);

    unsigned grp_size = 0;

    dest.ensure(dist.highest_group);
    strf::detail::copy_n(digits, dist.highest_group, dest.buffer_ptr());
    num_digits -= dist.highest_group;
    digits += dist.highest_group;
    dest.advance(dist.highest_group);

    if (dist.middle_groups_count) {
        auto middle_groups = dist.low_groups.highest_group();
        dist.low_groups.pop_high();
        while (num_digits >= middle_groups) {
            dest.ensure(1 + middle_groups);
            auto *oit = dest.buffer_ptr();
            *oit = separator;
            strf::detail::copy_n(digits, middle_groups, oit + 1);
            dest.advance(1 + middle_groups);
            num_digits -= middle_groups;
            digits += middle_groups;
            if (--dist.middle_groups_count == 0) {
                goto lower_groups;
            }
        }
        STRF_ASSERT(dist.middle_groups_count != 0);
        STRF_ASSERT(num_digits < middle_groups);
        if (num_digits != 0) {
            dest.ensure(1 + num_digits);
            auto *oit = dest.buffer_ptr();
            *oit = separator;
            strf::detail::copy_n(digits, num_digits, oit + 1);
            dest.advance(1 + num_digits);
            auto remaining = middle_groups - num_digits;
            num_digits = 0;
            strf::detail::write_fill(dest, remaining, (CharT)'0');
            -- dist.middle_groups_count;
        }
        STRF_ASSERT(num_digits == 0);
        while (dist.middle_groups_count) {
            strf::put(dest, separator);
            strf::detail::write_fill(dest, middle_groups, (CharT)'0');
            -- dist.middle_groups_count;
        }
        STRF_ASSERT(dist.middle_groups_count == 0);
        goto lower_groups_in_trailing_zeros;
    }
    lower_groups:
    if (num_digits != 0) {
        STRF_ASSERT(dist.middle_groups_count == 0);
        grp_size = dist.low_groups.highest_group();
        dist.low_groups.pop_high();
        while (num_digits > grp_size) {
            STRF_ASSERT(! dist.low_groups.empty());
            STRF_ASSERT(grp_size + 1 <= size_after_recycle);
            dest.ensure(grp_size + 1);
            auto *oit = dest.buffer_ptr();
            *oit = separator;
            strf::detail::copy_n(digits, grp_size, oit + 1);
            digits += grp_size;
            dest.advance(grp_size + 1);
            num_digits -= grp_size;
            grp_size = dist.low_groups.highest_group();
            dist.low_groups.pop_high();
        }
        STRF_ASSERT(num_digits != 0);
        STRF_ASSERT(num_digits + 1 <= size_after_recycle);
        dest.ensure(num_digits + 1);
        auto *oit = dest.buffer_ptr();
        *oit = separator;
        strf::detail::copy_n(digits, num_digits, oit + 1);
        dest.advance(num_digits + 1);
        if (grp_size > num_digits) {
            grp_size -= num_digits;
            STRF_ASSERT(grp_size <= size_after_recycle);
            dest.ensure(grp_size + (num_digits == 0));
            oit = dest.buffer_ptr();
            strf::detail::str_fill_n<CharT>(oit, grp_size, '0');
            dest.advance_to(oit + grp_size);
        }
    }
    lower_groups_in_trailing_zeros:
    while (! dist.low_groups.empty()) {
        grp_size = dist.low_groups.highest_group();
        dist.low_groups.pop_high();
        STRF_ASSERT(grp_size + 1 <= size_after_recycle);
        dest.ensure(grp_size + 1);
        auto *it = dest.buffer_ptr();
        *it = separator;
        strf::detail::str_fill_n<CharT>(it + 1, grp_size, '0');
        dest.advance(grp_size + 1);
    }
}


template <typename CharT>
inline STRF_HD void print_amplified_integer_small_separator
    ( strf::destination<CharT>& dest
    , unsigned long long value
    , strf::digits_grouping grouping
    , unsigned num_digits
    , unsigned num_trailing_zeros
    , CharT separator )
{
    auto dist = grouping.distribute(num_digits + num_trailing_zeros);
    if (num_digits <= dist.highest_group) {
        print_amplified_integer_small_separator_1
            ( dest, value, num_digits, dist, separator );
    } else {
        print_amplified_integer_small_separator_2
            ( dest, value, num_digits, dist, separator );
    }
}

template <typename CharT>
STRF_HD void print_amplified_integer_big_separator_1
    ( strf::destination<CharT>& dest
    , strf::encode_char_f<CharT> encode_char
    , unsigned long long value
    , unsigned num_digits
    , strf::digits_distribution dist
    , char32_t separator
    , unsigned separator_size )
{
    STRF_ASSERT(num_digits <= dist.highest_group);

    dest.ensure(num_digits);
    auto *ptr = dest.buffer_ptr() + num_digits;
    strf::detail::write_int_dec_txtdigits_backwards(value, ptr);
    dest.advance_to(ptr);
    dist.highest_group -= num_digits;
    if (dist.highest_group != 0) {
        strf::detail::write_fill(dest, dist.highest_group, (CharT)'0');
    }
    if (dist.middle_groups_count) {
        auto middle_groups = dist.low_groups.highest_group();
        dist.low_groups.pop_high();
        do {
            dest.ensure(separator_size + middle_groups);
            auto *oit = encode_char(dest.buffer_ptr(), separator);
            strf::detail::str_fill_n<CharT>(oit, middle_groups, '0');
            dest.advance_to(oit + middle_groups);
        } while (--dist.middle_groups_count);
    }
    while ( ! dist.low_groups.empty()) {
        auto grp = dist.low_groups.highest_group();
        dist.low_groups.pop_high();
        dest.ensure(separator_size + grp);
        auto *oit = encode_char(dest.buffer_ptr(), separator);
        strf::detail::str_fill_n<CharT>(oit, grp, '0');
        dest.advance(separator_size + grp);
    }
}

template <typename CharT>
STRF_HD void print_amplified_integer_big_separator_2
    ( strf::destination<CharT>& dest
    , strf::encode_char_f<CharT> encode_char
    , unsigned long long value
    , unsigned num_digits
    , strf::digits_distribution dist
    , char32_t separator
    , unsigned separator_size )
{
    STRF_ASSERT(dist.highest_group < num_digits);

    constexpr std::size_t size_after_recycle = strf::min_destination_buffer_size;
    (void) size_after_recycle;

    constexpr auto max_digits = detail::max_num_digits<unsigned long long, 10>();
    char digits_buff[max_digits];
    auto *digits = strf::detail::write_int_dec_txtdigits_backwards
        (value, digits_buff + max_digits);

    unsigned grp_size = 0;

    dest.ensure(dist.highest_group);
    strf::detail::copy_n(digits, dist.highest_group, dest.buffer_ptr());
    num_digits -= dist.highest_group;
    digits += dist.highest_group;
    dest.advance(dist.highest_group);

    if (dist.middle_groups_count) {
        auto middle_groups = dist.low_groups.highest_group();
        dist.low_groups.pop_high();
        while (num_digits >= middle_groups) {
            dest.ensure(separator_size + middle_groups);
            auto *oit = encode_char(dest.buffer_ptr(), separator);
            strf::detail::copy_n(digits, middle_groups, oit);
            dest.advance_to(oit + middle_groups);
            num_digits -= middle_groups;
            digits += middle_groups;
            if (--dist.middle_groups_count == 0) {
                goto lower_groups;
            }
        }
        STRF_ASSERT(dist.middle_groups_count != 0);
        STRF_ASSERT(num_digits < middle_groups);
        if (num_digits != 0) {
            dest.ensure(separator_size + middle_groups);
            const auto remaining = middle_groups - num_digits;
            auto *oit = encode_char(dest.buffer_ptr(), separator);
            strf::detail::copy_n(digits, num_digits, oit);
            strf::detail::str_fill_n<CharT>(oit + num_digits, remaining, '0');
            dest.advance_to(oit + middle_groups);
            num_digits = 0;
            --dist.middle_groups_count;
        }
        STRF_ASSERT(num_digits == 0);
        while (dist.middle_groups_count) {
            dest.ensure(separator_size + middle_groups);
            auto *oit = encode_char(dest.buffer_ptr(), separator);
            strf::detail::str_fill_n<CharT>(oit, middle_groups, '0');
            dest.advance_to(oit + middle_groups);
            -- dist.middle_groups_count;
        }
        STRF_ASSERT(dist.middle_groups_count == 0);
        goto lower_groups_in_trailing_zeros;
    }

    lower_groups:
    if (num_digits) {
        STRF_ASSERT(dist.middle_groups_count == 0);
        grp_size = dist.low_groups.highest_group();
        dist.low_groups.pop_high();
        while (num_digits > grp_size) {
            STRF_ASSERT(! dist.low_groups.empty());
            // `-> otherwise (num_digits > grp_size) should be false
            STRF_ASSERT(grp_size + separator_size <= size_after_recycle);
            dest.ensure(separator_size + grp_size);
            auto *oit = encode_char(dest.buffer_ptr(), separator);
            strf::detail::copy_n(digits, grp_size, oit);
            dest.advance_to(oit + grp_size);
            digits += grp_size;
            num_digits -= grp_size;
            grp_size = dist.low_groups.highest_group();
            dist.low_groups.pop_high();
        }
        STRF_ASSERT(num_digits + separator_size <= size_after_recycle);
        dest.ensure(separator_size + num_digits);
        auto *oit = encode_char(dest.buffer_ptr(), separator);
        strf::detail::copy_n(digits, num_digits, oit);
        dest.advance_to(oit + num_digits);
        if (grp_size > num_digits) {
            grp_size -= num_digits;
            STRF_ASSERT(grp_size <= size_after_recycle);
            dest.ensure(grp_size);
            oit = dest.buffer_ptr();
            strf::detail::str_fill_n<CharT>(oit, grp_size, '0');
            dest.advance_to(oit + grp_size);
        }
    }
    lower_groups_in_trailing_zeros:
    while (! dist.low_groups.empty()) {
        grp_size = dist.low_groups.highest_group();
        dist.low_groups.pop_high();
        STRF_ASSERT(separator_size + grp_size <= size_after_recycle);
        dest.ensure(separator_size + grp_size);
        auto *oit = encode_char(dest.buffer_ptr(), separator);
        strf::detail::str_fill_n<CharT>(oit, grp_size, '0');
        dest.advance_to(oit + grp_size);
    }
}

template <typename CharT>
STRF_HD void print_amplified_integer_big_separator
    ( strf::destination<CharT>& dest
    , strf::encode_char_f<CharT> encode_char
    , unsigned long long value
    , strf::digits_grouping grouping
    , unsigned num_digits
    , unsigned num_trailing_zeros
    , unsigned separator_size
    , char32_t separator )
{
    auto dist = grouping.distribute(num_digits + num_trailing_zeros);
    if (num_digits <= dist.highest_group) {
        print_amplified_integer_big_separator_1
            ( dest, encode_char, value, num_digits, dist, separator, separator_size );
    } else {
        print_amplified_integer_big_separator_2
            ( dest, encode_char, value, num_digits, dist, separator, separator_size );
    }
}

template <typename CharT>
STRF_HD void print_scientific_notation
    ( strf::destination<CharT>& dest
    , strf::encode_char_f<CharT> encode_char
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
    print_point |= num_digits != 1;
    dest.ensure(num_digits + print_point * decimal_point_size);
    if (num_digits == 1) {
        auto *it = dest.buffer_ptr();
        *it = static_cast<CharT>('0' + digits);
        ++it;
        if (print_point) {
            if (decimal_point < 0x80) {
                *it++ = static_cast<CharT>(decimal_point);
            } else {
                it = encode_char(it, decimal_point);
            }
        }
        dest.advance_to(it);
    } else {
       auto *it = dest.buffer_ptr();
       auto *end = it + num_digits + decimal_point_size;
       *it = *write_int_dec_txtdigits_backwards(digits, end);
       ++it;
       if (decimal_point < 0x80) {
           *it++ = static_cast<CharT>(decimal_point);
       } else {
           encode_char(it, decimal_point);
       }
       dest.advance_to(end);
    }

    // extra trailing zeros

    if (trailing_zeros != 0) {
        strf::detail::write_fill(dest, trailing_zeros, CharT('0'));
    }

    // exponent

    unsigned adv = 4;
    CharT* it = nullptr;
    unsigned e10u = std::abs(exponent);
    STRF_ASSERT(e10u < 1000);

    if (e10u >= 100) {
        dest.ensure(5);
        it = dest.buffer_ptr();
        it[4] = static_cast<CharT>('0' + e10u % 10);
        e10u /= 10;
        it[3] = static_cast<CharT>('0' + e10u % 10);
        it[2] = static_cast<CharT>('0' + e10u / 10);
        adv = 5;
    } else if (e10u >= 10) {
        dest.ensure(4);
        it = dest.buffer_ptr();
        it[3] = static_cast<CharT>('0' + e10u % 10);
        it[2] = static_cast<CharT>('0' + e10u / 10);
    } else {
        dest.ensure(4);
        it = dest.buffer_ptr();
        it[3] = static_cast<CharT>('0' + e10u);
        it[2] = '0';
    }
    it[0] = 'E' | ((lc != strf::uppercase) << 5);
    it[1] = static_cast<CharT>('+' + ((exponent < 0) << 1));
    dest.advance(adv);
}

template <typename CharT>
STRF_HD void print_nan(strf::destination<CharT>& dest, strf::lettercase lc)
{
    dest.ensure(3);
    auto *p = dest.buffer_ptr();
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
    dest.advance(3);

}
template <typename CharT>
STRF_HD void print_nan(strf::destination<CharT>& dest, strf::lettercase lc
                      , bool negative )
{
    dest.ensure(3 + negative);
    auto *p = dest.buffer_ptr();
    if (negative) {
        *p ++ = '-';
    }
    switch (lc) {
        case strf::mixedcase:
            *p++ = 'N';
            *p++ = 'a';
            *p++ = 'N';
            break;
        case strf::uppercase:
            *p++ = 'N';
            *p++ = 'A';
            *p++ = 'N';
            break;
        default:
            *p++ = 'n';
            *p++ = 'a';
            *p++ = 'n';
    }
    dest.advance_to(p);
}

template <typename CharT>
STRF_HD void print_inf(strf::destination<CharT>& dest, strf::lettercase lc)
{
    dest.ensure(3);
    auto *p = dest.buffer_ptr();
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
    dest.advance(3);
}

template <typename CharT>
STRF_HD void print_inf( strf::destination<CharT>& dest
                      , strf::lettercase lc
                      , bool negative )
{
    dest.ensure(3 + negative);
    auto *p = dest.buffer_ptr();
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
    dest.advance_to(p);
}

template <typename CharT>
class fast_double_printer: public strf::printer<CharT>
{
public:

    template <typename FloatT, typename PrePrinting>
    STRF_HD explicit fast_double_printer
        ( strf::detail::fast_double_printer_input<CharT, PrePrinting, FloatT> input) noexcept
        : fast_double_printer(input.value, input.lcase)
    {
        STRF_IF_CONSTEXPR (PrePrinting::width_required || PrePrinting::size_required) {
            const auto s = size();
            input.pre.subtract_width(s);
            input.pre.add_size(s);
        }
    }

    STRF_HD fast_double_printer(float f, strf::lettercase lc) noexcept
        : value_(decode(f))
        , m10_digcount_(strf::detail::count_digits<10>(value_.m10))
        , lettercase_(lc)

    {
        STRF_ASSERT(!value_.nan || !value_.infinity);
        sci_notation_ = (value_.e10 > 4 + (m10_digcount_ != 1))
            || (value_.e10 < -(int)m10_digcount_ - 2 - (m10_digcount_ != 1));
    }

    STRF_HD fast_double_printer(double d, strf::lettercase lc) noexcept
        : value_(decode(d))
        , m10_digcount_(strf::detail::count_digits<10>(value_.m10))
        , lettercase_(lc)

    {
        STRF_ASSERT(!value_.nan || !value_.infinity);
        sci_notation_ = (value_.e10 > 4 + (m10_digcount_ != 1))
            || (value_.e10 < -(int)m10_digcount_ - 2 - (m10_digcount_ != 1));
    }

    STRF_HD void print_to(strf::destination<CharT>&) const override;

    STRF_HD std::size_t size() const;

private:

    const detail::double_dec value_;
    bool sci_notation_ ;
    const unsigned m10_digcount_;
    strf::lettercase lettercase_;
};

template <typename CharT>
STRF_HD std::size_t fast_double_printer<CharT>::size() const
{
    const int sci_e10 = value_.e10 - 1 + (int)m10_digcount_;
    return ( value_.nan * 3
           + (value_.infinity * 3)
           + value_.negative
           + !(value_.infinity | value_.nan)
           * ( ( sci_notation_
               * ( 4 // e+xx
                 + (m10_digcount_ != 1) // decimal point
                 + m10_digcount_
                 + ((sci_e10 > 99) || (sci_e10 < -99))) )
             + ( !sci_notation_
               * ( (int)m10_digcount_
                 + (value_.e10 > 0) * value_.e10 // trailing zeros
                 + (value_.e10 <= -(int)m10_digcount_) * (2 -value_.e10 - (int)m10_digcount_) // leading zeros and point
                 + (-(int)m10_digcount_ < value_.e10 && value_.e10 < 0) ))));
}

template <typename CharT>
STRF_HD void fast_double_printer<CharT>::print_to
    ( strf::destination<CharT>& dest ) const
{
    if (value_.nan) {
        strf::detail::print_nan(dest, lettercase_, value_.negative);
    } else if (value_.infinity) {
        strf::detail::print_inf(dest, lettercase_, value_.negative);
    } else if (sci_notation_) {
        auto e10 = value_.e10 - 1 + (int)m10_digcount_;
        dest.ensure( value_.negative + m10_digcount_ + (m10_digcount_ != 1) + 4
                 + (e10 > 99 || e10 < -99) );
        CharT* it = dest.buffer_ptr();
        if (value_.negative) {
            * it = '-';
            ++it;
        }
        if (m10_digcount_ == 1) {
            * it = static_cast<CharT>('0' + value_.m10);
            ++ it;
        } else {
            auto *next = it + m10_digcount_ + 1;
            write_int_dec_txtdigits_backwards(value_.m10, next);
            it[0] = it[1];
            it[1] = '.';
            it = next;
        }
        it[0] = 'E' | ((lettercase_ != strf::uppercase) << 5);
        it[1] = static_cast<CharT>('+' + ((e10 < 0) << 1));
        unsigned e10u = std::abs(e10);
        if (e10u >= 100) {
            it[4] = static_cast<CharT>('0' + e10u % 10);
            e10u /= 10;
            it[3] = static_cast<CharT>('0' + e10u % 10);
            it[2] = static_cast<CharT>('0' + e10u / 10);
            it += 5;
        } else if (e10u >= 10) {
            it[3] = static_cast<CharT>('0' + e10u % 10);
            it[2] = static_cast<CharT>('0' + e10u / 10);
            it += 4;
        } else {
            it[3] = static_cast<CharT>('0' + e10u);
            it[2] = '0';
            it += 4;
        }
        dest.advance_to(it);
    } else {
        dest.ensure( value_.negative
                 + m10_digcount_ * (value_.e10 > - (int)m10_digcount_)
                 + (value_.e10 < - (int)m10_digcount_)
                 + (value_.e10 < 0) );
        auto *it = dest.buffer_ptr();
        if (value_.negative) {
            *it = '-';
            ++it;
        }
        if (value_.e10 >= 0) {
            it += m10_digcount_;
            write_int_dec_txtdigits_backwards(value_.m10, it);
            dest.advance_to(it);
            if (value_.e10 != 0) {
                detail::write_fill(dest, value_.e10, (CharT)'0');
            }
        } else {
            unsigned e10u = - value_.e10;
            if (e10u >= m10_digcount_) {
                it[0] = '0';
                it[1] = '.';
                dest.advance_to(it + 2);
                detail::write_fill(dest, e10u - m10_digcount_, (CharT)'0');

                dest.ensure(m10_digcount_);
                auto *end = dest.buffer_ptr() + m10_digcount_;
                write_int_dec_txtdigits_backwards(value_.m10, end);
                dest.advance_to(end);
            } else {
                const char* const arr = strf::detail::chars_00_to_99();
                auto m = value_.m10;
                it += m10_digcount_ + 1;
                CharT* const end = it;
                while(e10u >= 2) {
                    auto index = (m % 100) << 1;
                    it[-2] = arr[index];
                    it[-1] = arr[index + 1];
                    it -= 2;
                    m /= 100;
                    e10u -= 2;
                }
                if (e10u != 0) {
                    *--it = static_cast<CharT>('0' + (m % 10));
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
                    *--it = static_cast<CharT>('0' + m);
                }
                dest.advance_to(end);
            }
        }
    }
}

inline STRF_HD detail::chars_count_t exponent_hex_digcount(long exponent)
{
    if (exponent < 0) {
        exponent = -exponent;
    }
    return 1 + (exponent >= 1000) + (exponent >= 100) + (exponent >= 10);
}

inline STRF_HD detail::chars_count_t mantissa_hex_digcount(std::uint64_t mantissa)
{
    STRF_ASSERT(mantissa != 0);
    STRF_ASSERT(mantissa == (mantissa & 0xFFFFFFFFFFFFFULL));

#if defined(STRF_HAS_COUNTR_ZERO)
    return 13 - (strf::detail::countr_zero_ll(mantissa) >> 2);

#else
    if ((mantissa & 0xFFFFFFFFull) == 0) {
        if ((mantissa & 0xFFFFFFFFFFull) == 0) {
            if ((mantissa & 0xFFFFFFFFFFFFull) == 0) {
                return 1;
            }
            return ((mantissa & 0xFFFFFFFFFFFull) == 0) ? 2 : 3;
        }
        return ((mantissa & 0xFFFFFFFFFull) == 0) ? 4 : 5;
    }
    if ((mantissa & 0xFFFFull) == 0) {
         if ((mantissa & 0xFFFFFFull) == 0) {
             return ((mantissa & 0xFFFFFFFull) == 0) ? 6 : 7;
         }
         return ((mantissa & 0xFFFFFull) == 0) ? 8 : 9;
    }
    if ((mantissa & 0xFFull) == 0) {
        return ((mantissa & 0xFFFull) == 0) ? 10 : 11;
    }
    return ((mantissa & 0xFull) == 0) ? 12 : 13;

#endif
}


enum class float_form : std::uint8_t { nan, inf, fixed, sci, hex };

constexpr bool inf_or_nan(float_form f)
{
    return static_cast<std::uint8_t>(f) < 2;
}

struct double_printer_data
{
    strf::detail::float_form form;
    char sign;
    bool showsign;
    bool showpoint;
    bool subnormal;

    detail::chars_count_t sub_chars_count;
    detail::chars_count_t pad0width;
    detail::chars_count_t extra_zeros;
    union {
        detail::chars_count_t m10_digcount;
        detail::chars_count_t mantissa_digcount;
    };
    union {
        detail::chars_count_t sep_count;
        detail::chars_count_t exponent_digcount;
    };
    detail::chars_count_t left_fillcount;
    detail::chars_count_t right_fillcount;
    char32_t fillchar;
    union {
        std::uint64_t m10;
        std::uint64_t mantissa;
    };
    union {
        std::int32_t e10;
        std::int32_t exponent;
    };
};

struct float_init_result {
    detail::chars_count_t content_width;
    detail::chars_count_t fillcount;
};

#if ! defined(STRF_OMIT_IMPL)

inline STRF_HD strf::detail::float_init_result init_double_printer_data_fill
    ( strf::detail::double_printer_data& data
    , unsigned rounded_fmt_width
    , detail::chars_count_t content_width
    , strf::text_alignment alignment ) noexcept
{
    const bool fill_sign_space = data.sign == ' ';
    detail::chars_count_t fillcount = 0;
    if (rounded_fmt_width <= content_width) {
        data.left_fillcount = 0;
        data.right_fillcount = 0;
        if (fill_sign_space && data.fillchar != ' ') {
            goto adapt_fill_sign_space;
        }
    } else {
        fillcount = static_cast<detail::chars_count_t>(rounded_fmt_width - content_width);
        switch(alignment) {
            case strf::text_alignment::left:
                data.left_fillcount = 0;
                data.right_fillcount = fillcount;
                break;
            case strf::text_alignment::right:
                data.left_fillcount = fillcount;
                data.right_fillcount = 0;
                break;
            default:
                STRF_ASSERT(alignment == strf::text_alignment::center);
                auto half_fillcount = fillcount >> 1;
                data.left_fillcount = half_fillcount;
                data.right_fillcount = half_fillcount + (fillcount & 1);
        }
        if (fill_sign_space) {
            adapt_fill_sign_space:
            data.showsign = false;
            ++data.left_fillcount;
            ++fillcount;
            --data.sub_chars_count;
            --content_width;
            if (data.pad0width) {
                --data.pad0width;
            }
        }
    }
    return {content_width, fillcount};
}

inline STRF_HD strf::detail::float_init_result init_hex_double_printer_data
    ( strf::detail::double_printer_data& data
    , strf::float_format fdata
    , unsigned rounded_fmt_width
    , strf::text_alignment alignment ) noexcept
{
    data.form = detail::float_form::hex;
    data.sub_chars_count += 5; // "0x0p+"
    if (data.subnormal && data.mantissa == 0) {
        data.mantissa_digcount = 0;
        data.exponent_digcount = 1;
        data.exponent = 0;
        data.extra_zeros = (fdata.precision != (detail::chars_count_t)-1) * fdata.precision;
        data.showpoint = data.extra_zeros || fdata.showpoint;
        data.sub_chars_count += 1 + data.showpoint;
    } else {
        data.exponent_digcount = strf::detail::exponent_hex_digcount(data.exponent);
        data.sub_chars_count += data.exponent_digcount;
        if (data.mantissa == 0){
            data.mantissa_digcount = 0;
            data.extra_zeros = (fdata.precision != (detail::chars_count_t)-1) * fdata.precision;
            data.showpoint = data.extra_zeros || fdata.showpoint;
            data.sub_chars_count += data.showpoint;
        } else {
            data.mantissa_digcount = strf::detail::mantissa_hex_digcount(data.mantissa);
            data.extra_zeros = 0;
            if (fdata.precision >= data.mantissa_digcount) {
                static_assert(std::is_unsigned<detail::chars_count_t>::value, "");
                data.showpoint = true;
                data.sub_chars_count += 1 + data.mantissa_digcount;
                if (fdata.precision != (detail::chars_count_t)-1) {
                    data.extra_zeros = fdata.precision - data.mantissa_digcount;
                }
            } else {
                // round mantissa if necessary
                const unsigned s = (13 - fdata.precision) << 2;
                auto d = 1ULL << s;
                auto mask = d - 1;
                auto mantissa_low = data.mantissa & mask;
                if ( mantissa_low > (d >> 1)) {
                    data.mantissa += d;
                } else if (mantissa_low == (d >> 1) && (data.mantissa & d)) {
                    data.mantissa += d;
                }
                data.mantissa_digcount = fdata.precision;
                data.showpoint = fdata.precision || fdata.showpoint;
                data.sub_chars_count += data.showpoint + data.mantissa_digcount;
            }
        }
    }
    const detail::chars_count_t content_width =
        (detail::max)(data.sub_chars_count + data.extra_zeros, data.pad0width);

    return init_double_printer_data_fill
        ( data, rounded_fmt_width, content_width, alignment );
}

inline STRF_HD strf::detail::float_init_result init_double_data_with_precision_general_1digit
    ( strf::detail::double_printer_data& data
    , strf::digits_grouping grouping
    , detail::chars_count_t precision
    , unsigned rounded_fmt_width
    , strf::text_alignment alignment
    , bool alt_form ) noexcept
{
    if (precision == 0) {
        precision = 1;
    }
    if (data.e10 < -4 || (int)precision <= data.e10) {
        data.form = detail::float_form::sci;
        data.sub_chars_count = data.showsign + 5 + (data.e10 > 99 || data.e10 < -99);
        if (alt_form) {
            data.showpoint = true;
            data.extra_zeros = precision - 1;
            data.sub_chars_count += data.extra_zeros + 1;
        } else {
            data.showpoint = false;
        }
    } else { // -4 <= data.e10 && data.e10 < precision
        data.form = detail::float_form::fixed;
        data.sub_chars_count = data.showsign;
        if (data.e10 < 0) {
            data.showpoint = true;
            data.sub_chars_count = 2 - data.e10;
            if (alt_form) {
                data.extra_zeros = precision - 1;
                data.sub_chars_count += data.extra_zeros;
            }
        } else {
            data.showpoint = alt_form;
            const unsigned int_digcount = 1 + data.e10;
            data.sep_count = grouping.separators_count(int_digcount);
            data.sub_chars_count = 1 + data.e10 + data.sep_count + data.showpoint;
            if (alt_form && precision > int_digcount) {
                data.extra_zeros = precision - int_digcount;
                data.sub_chars_count += data.extra_zeros;
            }
        }
    }
    return init_double_printer_data_fill
        ( data, rounded_fmt_width
        , (detail::max)(data.sub_chars_count, data.pad0width)
        , alignment );
}

inline STRF_HD strf::detail::float_init_result init_double_data_with_precision_general
    ( strf::detail::double_printer_data& data
    , strf::digits_grouping grouping
    , detail::chars_count_t precision
    , unsigned rounded_fmt_width
    , strf::text_alignment alignment
    , bool showpoint ) noexcept
{
    data.sep_count = 0;
    data.m10_digcount = static_cast<detail::chars_count_t>
        (strf::detail::count_digits<10>(data.m10));

    // As in printf:
    // - Select the scientific notation if the resulting exponent
    //   is less than -4 or greater than or equal to the precision
    // - The precision specifies the number of significant digits.
    // - If the precision is 0, it is treated as 1
    // - Trailing fractional zeros are removed when showpoint is false.

    int xz = 0; // number of zeros to be added or ( if negative ) digits to be removed
    const int p = precision != 0 ? precision : 1;
    int int_digcount = (int)data.m10_digcount + data.e10;
    // equivalent to:
    // const int sci_notation_exp = (int)data.m10_digcount + data.e10 - 1;
    // if (sci_notation_exp < -4 || sci_notation_exp >= p);
    if (int_digcount < -3 || int_digcount > p) {
        data.form = detail::float_form::sci;
        const int sci_notation_exp = (int)data.m10_digcount + data.e10 - 1;
        data.sub_chars_count += 4 + (sci_notation_exp > 99 || sci_notation_exp < -99);
        xz = p - (int)data.m10_digcount;
        if (xz < 0) {
            int_digcount = 1;
            data.sub_chars_count += p;
            goto remove_fractional_trailing_digits;
        }
        STRF_ASSERT(p >= (int)data.m10_digcount);
        if (showpoint) {
            data.extra_zeros = static_cast<detail::chars_count_t>(xz);
            data.sub_chars_count += p;
            data.showpoint = true;
        } else {
            data.sub_chars_count += (int)data.m10_digcount;
            data.showpoint = data.m10_digcount > 1;
        }
        goto end;
    } else {
        data.form = detail::float_form::fixed;
        STRF_ASSERT (p >= int_digcount);
        if (grouping.any_separator(int_digcount)) {
            data.sep_count = grouping.separators_count(int_digcount);
            data.sub_chars_count += data.sep_count;
        }
        if (data.e10 >= 0) {
            data.sub_chars_count += static_cast<detail::chars_count_t>(int_digcount);
            if (showpoint && p > int_digcount) {
                xz = p - int_digcount;
                data.extra_zeros = static_cast<detail::chars_count_t>(xz);
                data.sub_chars_count += data.extra_zeros;
            }
            data.showpoint = showpoint;
            goto end;
        } else {
            data.sub_chars_count +=
                ( data.e10 <= -(int)data.m10_digcount
                ? static_cast<detail::chars_count_t>(1 - data.e10)
                : data.m10_digcount );
            xz = p - (int)data.m10_digcount;
            if (xz < 0) {
                data.sub_chars_count -= static_cast<detail::chars_count_t>(-xz);
                goto remove_fractional_trailing_digits;
            }
            data.showpoint = true;
            if (showpoint) {
                data.extra_zeros = static_cast<detail::chars_count_t>(xz);
                data.sub_chars_count += data.extra_zeros;
            }
            goto end;
        }
    }
    {
        remove_fractional_trailing_digits:

        const unsigned dp = -xz;
        data.m10_digcount -= static_cast<detail::chars_count_t>(dp);
        data.e10 += dp;
        auto p10 = strf::detail::pow10(dp);
        auto remainer = data.m10 % p10;
        data.m10 = data.m10 / p10;
        auto middle = p10 >> 1;
        if (remainer > middle || (remainer == middle && (data.m10 & 1) == 1)) {
            // round up
            ++ data.m10;
            STRF_IF_UNLIKELY (data.m10 == strf::detail::pow10(data.m10_digcount)) {
                data.m10 = 1;
                data.e10 += data.m10_digcount;
                data.m10_digcount = 1;
                return init_double_data_with_precision_general_1digit
                    ( data, grouping, precision, rounded_fmt_width
                    , alignment, showpoint );
            }
        }
        bool has_fractional_digits = true;
        while (true) {
            if ((int)data.m10_digcount <= int_digcount) {
                has_fractional_digits = false;
                break;
            }
            if (data.m10 % 10 != 0) {
                break;
            }
            data.m10 /= 10;
            -- data.m10_digcount;
            -- data.sub_chars_count;
            ++ data.e10;
        }
        data.showpoint = showpoint || has_fractional_digits;
    }
    end:
    data.sub_chars_count += data.showpoint;
    return init_double_printer_data_fill
        ( data, rounded_fmt_width
        , (detail::max)(data.sub_chars_count, data.pad0width)
        , alignment );
}

inline STRF_HD strf::detail::float_init_result init_double_data_with_precision_scientific
    ( strf::detail::double_printer_data& data
    , detail::chars_count_t precision
    , unsigned rounded_fmt_width
    , strf::text_alignment alignment
    , bool showpoint ) noexcept
{
    data.sep_count = 0;
    data.form = detail::float_form::sci;
    data.m10_digcount = static_cast<detail::chars_count_t>
        (strf::detail::count_digits<10>(data.m10));

    const int sci_notation_exp = (int)data.m10_digcount + data.e10 - 1;
    const unsigned frac_digits = data.m10_digcount - 1;
    data.showpoint = showpoint || (precision != 0);
    data.sub_chars_count += data.showpoint;
    data.sub_chars_count += 5 + precision;
    data.sub_chars_count += (sci_notation_exp > 99 || sci_notation_exp < -99);

    const int xz = (precision - frac_digits);
    if (xz >= 0) {
        data.extra_zeros = static_cast<detail::chars_count_t>(xz);
    } else {
        data.extra_zeros = 0;
        const unsigned dp = -xz;
        data.m10_digcount -= static_cast<detail::chars_count_t>(dp);
        data.e10 += dp;
        auto p10 = strf::detail::pow10(dp);
        auto remainer = data.m10 % p10;
        data.m10 = data.m10 / p10;
        auto middle = p10 >> 1;
        if (remainer > middle || (remainer == middle && (data.m10 & 1) == 1)) {
            // round up
            ++ data.m10;
            if (data.m10 == strf::detail::pow10(data.m10_digcount)) {
                data.e10 += data.m10_digcount;
                if (data.e10 == 100) {
                    ++ data.sub_chars_count;
                } else if (data.e10 == -99) {
                    -- data.sub_chars_count;
                }
                data.m10 = 1;
                data.m10_digcount = 1;
                data.extra_zeros = precision;
            }
        }
    }
    return init_double_printer_data_fill
        ( data, rounded_fmt_width
        , (detail::max)(data.sub_chars_count, data.pad0width)
        , alignment );
}

inline STRF_HD strf::detail::float_init_result init_double_data_with_precision_fixed
    ( strf::detail::double_printer_data& data
    , strf::digits_grouping grouping
    , detail::chars_count_t precision
    , unsigned rounded_fmt_width
    , strf::text_alignment alignment
    , bool showpoint ) noexcept
{
    data.sep_count = 0;
    data.m10_digcount = static_cast<detail::chars_count_t>
        (strf::detail::count_digits<10>(data.m10));
    data.showpoint = showpoint || (precision != 0);
    data.form = detail::float_form::fixed;
    data.sub_chars_count += data.showpoint;

    const int frac_digits = data.e10 < 0 ? -data.e10 : 0;
    const int xz = (precision - frac_digits);
    if (xz <= -(int)data.m10_digcount) {
        STRF_ASSERT(data.e10 <= -(int)(precision + data.m10_digcount));
        data.sub_chars_count += 1 + precision;
        if ( xz == -(int)data.m10_digcount
          && data.m10 > (strf::detail::pow10(data.m10_digcount) >> 1) ) {
            // round up ( to something like 0.0001 )
            data.e10 += data.m10_digcount;
            data.m10 = 1;
            data.m10_digcount = 1;
        } else {
            // round down to zero
            data.extra_zeros = precision;
            data.m10_digcount = 1;
            data.m10 = 0;
            data.e10 = 0;
            data.sep_count = 0;
        }
    } else {
        auto int_digcount = ( (int)data.m10_digcount > -data.e10
                            ? (int)data.m10_digcount + data.e10
                            : 1 );
        if (grouping.any_separator(int_digcount)) {
            data.sep_count = static_cast<detail::chars_count_t>
                (grouping.separators_count(int_digcount));
            data.sub_chars_count += static_cast<detail::chars_count_t>(data.sep_count);
        }
        data.sub_chars_count += static_cast<detail::chars_count_t>(int_digcount + precision);
        if (xz >= 0) {
            data.extra_zeros = static_cast<detail::chars_count_t>(xz);
        } else {
            data.extra_zeros = 0;
            const unsigned dp = -xz;
            data.m10_digcount -= static_cast<detail::chars_count_t>(dp);
            data.e10 += dp;
            auto p10 = strf::detail::pow10(dp);
            auto remainer = data.m10 % p10;
            data.m10 = data.m10 / p10;
            auto middle = p10 >> 1;
            if (remainer > middle || (remainer == middle && (data.m10 & 1) == 1)) {
                // round up
                ++data.m10;
                if (data.m10 == strf::detail::pow10(data.m10_digcount)) {
                    data.e10 += data.m10_digcount;
                    data.m10 = 1;
                    data.m10_digcount = 1;
                    if (data.e10 <= 0) {
                        STRF_ASSERT((int)precision >= -data.e10);
                        data.extra_zeros = (int)precision + data.e10;
                    } else {
                        data.extra_zeros = precision;
                        ++data.sub_chars_count;
                    }
                }
            }

        }
    }
    return init_double_printer_data_fill
        ( data, rounded_fmt_width
        , (detail::max)(data.sub_chars_count, data.pad0width)
        , alignment );
}

inline STRF_HD strf::detail::float_init_result init_double_data_without_precision_general
    ( strf::detail::double_printer_data& data
    , strf::digits_grouping grouping
    , unsigned rounded_fmt_width
    , strf::text_alignment alignment
    , bool showpoint ) noexcept
{
    data.sep_count = 0;
    data.m10_digcount = static_cast<detail::chars_count_t>
        (strf::detail::count_digits<10>(data.m10));

    const int sci_showpoint = showpoint || data.m10_digcount != 1;
    const int sci_sub_width = 4 + sci_showpoint;
    const int digcount_plus_e10 = (int)data.m10_digcount + data.e10;
    int fixed_sub_width =  ( data.e10 >= 0
                           ? data.e10 + showpoint
                           : ( digcount_plus_e10 <= 0
                             ? 2 - digcount_plus_e10
                             : 1 ) );
    if (sci_sub_width < fixed_sub_width) {
        scientific:
        data.form = detail::float_form::sci;
        data.showpoint = sci_showpoint;
        data.sub_chars_count += data.m10_digcount;
        data.sub_chars_count += static_cast<detail::chars_count_t>(sci_sub_width);
        const int sci_notation_exp = (int)data.m10_digcount + data.e10 - 1;
        if (sci_notation_exp > 99 || sci_notation_exp < -99) {
          ++data.sub_chars_count;
        }
    } else {
        const auto int_digcount = digcount_plus_e10;
        if (grouping.any_separator(int_digcount)) {
            auto sep_count = grouping.separators_count(int_digcount);
            fixed_sub_width += sep_count;
            if (fixed_sub_width > sci_sub_width) {
                goto scientific;
            }
            data.sep_count = static_cast<decltype(data.sep_count)>(sep_count);
        }
        data.form = detail::float_form::fixed;
        data.showpoint = showpoint || (data.e10 < 0);
        data.sub_chars_count += static_cast<detail::chars_count_t>(fixed_sub_width);
        data.sub_chars_count += data.m10_digcount;
    }
    return init_double_printer_data_fill
        ( data, rounded_fmt_width
        , (detail::max)(data.sub_chars_count, data.pad0width)
        , alignment );
}

inline STRF_HD strf::detail::float_init_result init_double_data_without_precision_scientific
    ( strf::detail::double_printer_data& data
    , unsigned rounded_fmt_width
    , strf::text_alignment alignment
    , bool showpoint ) noexcept
{
    data.sep_count = 0;
    data.form = detail::float_form::sci;
    data.m10_digcount = static_cast<detail::chars_count_t>
        (strf::detail::count_digits<10>(data.m10));

    const int sci_notation_exp = (int)data.m10_digcount + data.e10 - 1;
    data.showpoint = showpoint || (data.m10_digcount != 1);
    data.sub_chars_count += 4 + data.showpoint;
    data.sub_chars_count += (sci_notation_exp > 99 || sci_notation_exp < -99);
    data.sub_chars_count += data.m10_digcount;

    return init_double_printer_data_fill
        ( data, rounded_fmt_width
        , (detail::max)(data.sub_chars_count, data.pad0width)
        , alignment );
}

inline STRF_HD strf::detail::float_init_result init_double_data_without_precision_fixed
    ( strf::detail::double_printer_data& data
    , strf::digits_grouping grouping
    , unsigned rounded_fmt_width
    , strf::text_alignment alignment
    , bool showpoint ) noexcept
{
    data.sep_count = 0;
    data.form = detail::float_form::fixed;
    data.m10_digcount = static_cast<detail::chars_count_t>
        (strf::detail::count_digits<10>(data.m10));
    data.showpoint = showpoint || (data.e10 < 0);
    auto int_digcount = (int)data.m10_digcount + data.e10;
    if (grouping.any_separator(int_digcount)) {
        data.sep_count
            = static_cast<detail::chars_count_t>(grouping.separators_count(int_digcount));
        data.sub_chars_count += data.sep_count;
    }
    data.sub_chars_count +=
        ( data.e10 >= 0
        ? static_cast<detail::chars_count_t>(int_digcount + showpoint)
        : data.e10 <= -(int)data.m10_digcount
        ? static_cast<detail::chars_count_t>(2 - data.e10)
        : 1 + data.m10_digcount );
    return init_double_printer_data_fill
        ( data, rounded_fmt_width
        , (detail::max)(data.sub_chars_count, data.pad0width)
        , alignment );
}


// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD strf::detail::float_init_result init_float_printer_data
    ( strf::detail::double_printer_data& data
    , double d
    , strf::digits_grouping grp
    , strf::float_format ffmt
    , strf::alignment_format afmt ) noexcept
{
    constexpr int bias = 1023;
    constexpr int e_size = 11; // bits in exponent
    constexpr int m_size = 52; // bits in matissa

    const std::uint64_t bits = strf::detail::to_bits(d);
    const auto bits_exponent = static_cast<std::uint32_t>((bits << 1) >> (m_size + 1));
    const std::uint64_t bits_mantissa = bits & 0xFFFFFFFFFFFFFULL;
    const bool negative = (bits >> (m_size + e_size));

    chars_count_t rounded_fmt_width = afmt.width.round();

    data.sign = negative ? '-' : static_cast<char>(ffmt.sign);
    data.showsign = negative || ffmt.sign != strf::showsign::negative_only;
    data.sub_chars_count = data.showsign;
    data.pad0width = ffmt.pad0width;
    data.fillchar = afmt.fill;
    data.sep_count = 0;
    data.extra_zeros = 0;
    if (bits_exponent == 0x7FF) {
        // infinit or nan
        data.subnormal = false;
        data.form = static_cast<detail::float_form>(bits_mantissa == 0);
        data.sub_chars_count += 3;
        data.showpoint = false;
        if (data.pad0width > rounded_fmt_width) {
            rounded_fmt_width = data.pad0width;
        }
        return init_double_printer_data_fill
            ( data, rounded_fmt_width, 3 + data.showsign, afmt.alignment );
    }
    data.subnormal = bits_exponent == 0;
    if (ffmt.notation == strf::float_notation::hex) {
        data.mantissa = bits_mantissa;
        data.exponent = bits_exponent - bias + data.subnormal;
        return init_hex_double_printer_data
            ( data, ffmt, rounded_fmt_width, afmt.alignment);
    }
    if (bits_exponent == 0 && bits_mantissa == 0) {
        data.m10 = 0;
        data.e10 = 0;
    } else {
        namespace dragonbox = strf::detail::jkj::dragonbox;
        auto res = dragonbox::to_decimal<double>(bits_exponent, bits_mantissa);
        data.m10 = res.significand;
        data.e10 = res.exponent;
    }
    switch (ffmt.notation) {
        case strf::float_notation::general:
            if (ffmt.precision == (detail::chars_count_t)-1) {
                return init_double_data_without_precision_general
                    ( data, grp, rounded_fmt_width, afmt.alignment, ffmt.showpoint);
            }
            return init_double_data_with_precision_general
                ( data, grp, ffmt.precision, rounded_fmt_width, afmt.alignment, ffmt.showpoint );

        case strf::float_notation::scientific:
            if (ffmt.precision == (detail::chars_count_t)-1) {
                return init_double_data_without_precision_scientific
                    ( data, rounded_fmt_width, afmt.alignment, ffmt.showpoint);
            }
            return init_double_data_with_precision_scientific
                ( data, ffmt.precision, rounded_fmt_width, afmt.alignment, ffmt.showpoint );

        default:
            STRF_ASSERT(ffmt.notation == strf::float_notation::fixed);
            if (ffmt.precision == (detail::chars_count_t)-1) {
                return init_double_data_without_precision_fixed
                    ( data, grp, rounded_fmt_width, afmt.alignment, ffmt.showpoint);
            }
            return init_double_data_with_precision_fixed
                ( data, grp, ffmt.precision, rounded_fmt_width, afmt.alignment, ffmt.showpoint );
    }
}

// NOLINTNEXTLINE(misc-definitions-in-headers)
STRF_FUNC_IMPL STRF_HD strf::detail::float_init_result init_float_printer_data
    ( strf::detail::double_printer_data& data
    , float f
    , strf::digits_grouping grp
    , strf::float_format ffmt
    , strf::alignment_format afmt ) noexcept
{
    constexpr int bias = 127;
    constexpr int e_size = 8;
    constexpr int m_size = 23;

    const std::uint32_t bits = strf::detail::to_bits(f);
    const std::uint32_t bits_mantissa = bits & 0x7FFFFF;
    const auto bits_exponent = static_cast<std::uint32_t>((bits << 1) >> (m_size + 1));
    const bool negative = (bits >> (m_size + e_size));

    chars_count_t rounded_fmt_width = afmt.width.round();

    data.sign = negative ? '-' : static_cast<char>(ffmt.sign);
    data.showsign = negative || ffmt.sign != strf::showsign::negative_only;
    data.sub_chars_count = data.showsign;
    data.pad0width = ffmt.pad0width;
    data.fillchar = afmt.fill;
    data.sep_count = 0;
    data.extra_zeros = 0;
    if (bits_exponent == 0xFF) {
        // infinit or nan
        data.subnormal = false;
        data.form = static_cast<detail::float_form>(bits_mantissa == 0);
        data.sub_chars_count += 3;
        data.showpoint = false;
        if (data.pad0width > rounded_fmt_width) {
            rounded_fmt_width = data.pad0width;
        }
        return init_double_printer_data_fill
            ( data, rounded_fmt_width, 3 + data.showsign, afmt.alignment );
    }
    data.subnormal = bits_exponent == 0;
    if (ffmt.notation == strf::float_notation::hex) {
        data.mantissa = (std::uint64_t)bits_mantissa << 29;
        data.exponent = bits_exponent - bias + data.subnormal;
        return init_hex_double_printer_data
            ( data, ffmt, rounded_fmt_width, afmt.alignment);
    }
    if (bits_exponent == 0 && bits_mantissa == 0) {
        data.m10 = 0;
        data.e10 = 0;
    } else {
        namespace dragonbox = strf::detail::jkj::dragonbox;
        auto res = dragonbox::to_decimal<float>(bits_exponent, bits_mantissa);
        data.m10 = res.significand;
        data.e10 = res.exponent;
    }

    switch (ffmt.notation) {
        case strf::float_notation::general:
            if (ffmt.precision == (detail::chars_count_t)-1) {
                return init_double_data_without_precision_general
                    ( data, grp, rounded_fmt_width, afmt.alignment, ffmt.showpoint);
            }
            return init_double_data_with_precision_general
                ( data, grp, ffmt.precision, rounded_fmt_width, afmt.alignment, ffmt.showpoint );

        case strf::float_notation::scientific:
            if (ffmt.precision == (detail::chars_count_t)-1) {
                return init_double_data_without_precision_scientific
                    ( data, rounded_fmt_width, afmt.alignment, ffmt.showpoint);
            }
            return init_double_data_with_precision_scientific
                ( data, ffmt.precision, rounded_fmt_width, afmt.alignment, ffmt.showpoint );

        default:
            STRF_ASSERT(ffmt.notation == strf::float_notation::fixed);
            if (ffmt.precision == (detail::chars_count_t)-1) {
                return init_double_data_without_precision_fixed
                    ( data, grp, rounded_fmt_width, afmt.alignment, ffmt.showpoint);
            }
            return init_double_data_with_precision_fixed
                ( data, grp, ffmt.precision, rounded_fmt_width, afmt.alignment, ffmt.showpoint );
    }
}


#else // ! defined(STRF_OMIT_IMPL)

STRF_HD strf::detail::float_init_result init_float_printer_data
    ( strf::detail::double_printer_data& data
    , double d
    , strf::digits_grouping grp
    , strf::float_format ffmt
    , strf::alignment_format afmt ) noexcept;

STRF_HD strf::detail::float_init_result init_float_printer_data
    ( strf::detail::double_printer_data& data
    , float f
    , strf::digits_grouping grp
    , strf::float_format ffmt
    , strf::alignment_format afmt ) noexcept;

#endif // ! defined(STRF_OMIT_IMPL)


template <typename CharT>
class punct_double_printer: public strf::printer<CharT>
{
public:

    template < typename PrePrinting, typename FPack, typename FloatT, bool HasAlignment>
    STRF_HD explicit punct_double_printer
        ( const strf::detail::fmt_double_printer_input
            < CharT, PrePrinting, FPack, FloatT
            , strf::float_formatter_full_dynamic, HasAlignment >& input )
        : lettercase_(strf::use_facet<strf::lettercase_c, FloatT>(input.facets))
    {
        auto charset = use_facet<strf::charset_c<CharT>, FloatT>(input.facets);
        encode_fill_ = charset.encode_fill_func();
        encode_char_ = charset.encode_char_func();

        auto notation = input.arg.float_notation();
        if (input.arg.get_float_format().punctuate) {
            auto punct_dec = strf::use_facet<strf::numpunct_c<10>, FloatT>(input.facets);
            auto punct_hex = strf::use_facet<strf::numpunct_c<16>, FloatT>(input.facets);
            grouping_ = punct_dec.grouping();
            thousands_sep_ = punct_dec.thousands_sep();
            decimal_point_ = ( notation != strf::float_notation::hex
                             ? punct_dec.decimal_point()
                             : punct_hex.decimal_point() );
            auto ps = charset.validate(thousands_sep_);
            if (ps == strf::invalid_char_len) {
                grouping_ = strf::digits_grouping{};
            } else {
                sep_size_ = static_cast<detail::chars_count_t>(ps);
            }
        }
        auto r = strf::detail::init_float_printer_data
            ( data_, input.arg.value(), grouping_, input.arg.get_float_format()
            , input.arg.get_alignment_format() );
        if (data_.showpoint) {
            auto size = charset.encoded_char_size(decimal_point_);
            decimal_point_size_ = static_cast<detail::chars_count_t>(size);
        } else {
            decimal_point_size_ = 0;
        }
        input.pre.subtract_width(r.fillcount);
        input.pre.subtract_width(r.content_width);
        STRF_IF_CONSTEXPR (PrePrinting::size_required) {
            input.pre.add_size(r.content_width);
            if (r.fillcount) {
                const std::size_t fillchar_size = charset.encoded_char_size(data_.fillchar);
                input.pre.add_size(fillchar_size * r.fillcount);
            }
            if (notation != strf::float_notation::hex && data_.sep_count){
                input.pre.add_size(data_.sep_count * (sep_size_ - 1));
            }
            if (data_.showpoint) {
                input.pre.add_size(decimal_point_size_ - 1);
            }
        }
    }

    template < typename PrePrinting, typename FPack, typename FloatT
             , typename FloatFormatter, bool HasAlignment
             , strf::detail::enable_if_t<!FloatFormatter::has_punct, int> = 0 >
    STRF_HD explicit punct_double_printer
        ( const strf::detail::fmt_double_printer_input
            < CharT, PrePrinting, FPack, FloatT, FloatFormatter, HasAlignment >& input )
        : lettercase_(strf::use_facet<strf::lettercase_c, FloatT>(input.facets))
    {
        auto charset = use_facet<strf::charset_c<CharT>, FloatT>(input.facets);
        encode_fill_ = charset.encode_fill_func();
        auto r = strf::detail::init_float_printer_data
            ( data_, input.arg.value(), grouping_, input.arg.get_float_format()
            , input.arg.get_alignment_format() );
        decimal_point_size_ = data_.showpoint;
        input.pre.subtract_width(r.fillcount);
        input.pre.subtract_width(r.content_width);
        STRF_IF_CONSTEXPR (PrePrinting::size_required) {
            input.pre.add_size(r.content_width);
            if (r.fillcount) {
                const std::size_t fillchar_size = charset.encoded_char_size(data_.fillchar);
                input.pre.add_size(fillchar_size * r.fillcount);
            }
        }
    }

    STRF_HD void print_to(strf::destination<CharT>&) const override;

private:

    STRF_HD void print_fixed_
        ( strf::destination<CharT>& dest ) const noexcept;

    STRF_HD void print_scientific_
        ( strf::destination<CharT>& dest ) const noexcept;

    STRF_HD void print_hexadecimal_
        ( strf::destination<CharT>& dest ) const noexcept;

    STRF_HD void print_inf_or_nan_
        ( strf::destination<CharT>& dest ) const noexcept;

    strf::encode_char_f<CharT> encode_char_ = nullptr;
    strf::encode_fill_f<CharT> encode_fill_;
    strf::digits_grouping grouping_;
    unsigned sep_size_ = 1;
    unsigned decimal_point_size_ = 1;
    char32_t decimal_point_ = '.';
    char32_t thousands_sep_ = ',';
    strf::lettercase lettercase_;
    strf::detail::double_printer_data data_;
};

template <typename CharT>
STRF_HD void punct_double_printer<CharT>::print_to
    (strf::destination<CharT>& dest) const
{
    if (data_.left_fillcount != 0) {
        encode_fill_(dest, data_.left_fillcount, data_.fillchar);
    }
    switch (data_.form) {
        case detail::float_form::fixed:
            print_fixed_(dest);
            break;
        case detail::float_form::sci:
            print_scientific_(dest);
            break;
        case detail::float_form::hex:
            print_hexadecimal_(dest);
            break;
        default:
            print_inf_or_nan_(dest);
    }
    if (data_.right_fillcount != 0) {
        encode_fill_(dest, data_.right_fillcount, data_.fillchar);
    }
}

template <typename CharT>
STRF_HD void punct_double_printer<CharT>::print_fixed_
    ( strf::destination<CharT>& dest ) const noexcept
{
    if (data_.showsign) {
        put(dest, static_cast<CharT>(data_.sign));
    }
    if (data_.pad0width > data_.sub_chars_count) {
        auto count = data_.pad0width - data_.sub_chars_count;
        strf::detail::write_fill(dest, count, (CharT)'0');
    }
    if (data_.e10 >= 0) {
        if (data_.sep_count == 0) {
            strf::detail::write_int<10>( dest, data_.m10, data_.m10_digcount
                                       , strf::lowercase );
            strf::detail::write_fill(dest, data_.e10, (CharT)'0');
        } else if (sep_size_ == 1) {
            strf::detail::print_amplified_integer_small_separator
                ( dest, data_.m10, grouping_, data_.m10_digcount, data_.e10
                , static_cast<CharT>(thousands_sep_) );
        } else {
            strf::detail::print_amplified_integer_big_separator
                ( dest, encode_char_, data_.m10, grouping_, data_.m10_digcount
                , data_.e10, sep_size_, thousands_sep_ );
        }
        if (data_.showpoint) {
            dest.ensure(decimal_point_size_);
            if (decimal_point_ < 0x80) {
                *dest.buffer_ptr() = static_cast<CharT>(decimal_point_);
                dest.advance();
            } else {
                dest.advance_to(encode_char_(dest.buffer_ptr(), decimal_point_));
            }
        }
        if (data_.extra_zeros) {
            detail::write_fill(dest, data_.extra_zeros,  (CharT)'0');
        }
    } else {
        STRF_ASSERT(data_.e10 < 0);

        const detail::chars_count_t e10u = - data_.e10;
        if (e10u >= data_.m10_digcount) {
            dest.ensure(1 + decimal_point_size_);
            auto *it = dest.buffer_ptr();
            *it++ = static_cast<CharT>('0');
            if (decimal_point_ < 0x80) {
                *it++ = static_cast<CharT>(decimal_point_);
            } else {
                STRF_ASSERT(decimal_point_size_ != 0);
                it = encode_char_(it, decimal_point_);
            }
            dest.advance_to(it);

            if (e10u > data_.m10_digcount) {
                strf::detail::write_fill(dest, e10u - data_.m10_digcount, (CharT)'0');
            }
            strf::detail::write_int<10>( dest, data_.m10, data_.m10_digcount
                                       , strf::lowercase);
            if (data_.extra_zeros != 0) {
                strf::detail::write_fill(dest, data_.extra_zeros,  (CharT)'0');
            }
        } else {
            //auto v = std::lldiv(data_.m10, detail::pow10(e10u)); // todo test this
            auto p10 = strf::detail::pow10(e10u);
            auto integral_part = data_.m10 / p10;
            auto fractional_part = data_.m10 % p10;
            auto idigcount = data_.m10_digcount - e10u;

            STRF_ASSERT(idigcount == detail::count_digits<10>(integral_part));

            if (data_.sep_count == 0) {
                strf::detail::write_int<10>(dest, integral_part, idigcount, strf::lowercase);
            } else if (sep_size_ == 1) {
                CharT sep;
                if (thousands_sep_ < 0x80) {
                    sep = static_cast<CharT>(thousands_sep_);
                } else {
                    encode_char_(&sep, thousands_sep_);
                }
                strf::detail::write_int_little_sep<10>
                    ( dest, integral_part, grouping_, idigcount, data_.sep_count, sep );
            } else {
                strf::detail::write_int_big_sep<10>
                    ( dest, encode_char_, integral_part, grouping_, thousands_sep_
                    , sep_size_, idigcount );
            }

            dest.ensure(decimal_point_size_);
            auto *it = dest.buffer_ptr();
            if (decimal_point_ < 0x80) {
                *it++ = static_cast<CharT>(decimal_point_);
            } else {
                STRF_ASSERT(decimal_point_size_ != 0);
                it = encode_char_(it, decimal_point_);
            }
            dest.advance_to(it);

            strf::detail::write_int_with_leading_zeros<10>
                (dest, fractional_part, e10u, strf::lowercase);
            if (data_.extra_zeros) {
                detail::write_fill(dest, data_.extra_zeros,  (CharT)'0');
            }
        }
    }
}

template <typename CharT>
STRF_HD void punct_double_printer<CharT>::print_scientific_
    ( strf::destination<CharT>& dest ) const noexcept
{
    if (data_.showsign) {
        put(dest, static_cast<CharT>(data_.sign));
    }
    if (data_.pad0width > data_.sub_chars_count) {
        auto count = data_.pad0width - data_.sub_chars_count;
        strf::detail::write_fill(dest, count, (CharT)'0');
    }
    strf::detail::print_scientific_notation
        ( dest, encode_char_, data_.m10, data_.m10_digcount
        , decimal_point_, decimal_point_size_
        , data_.e10 + data_.m10_digcount - 1
        , data_.showpoint, data_.extra_zeros, lettercase_ );
}

template <typename CharT>
STRF_HD void punct_double_printer<CharT>::print_hexadecimal_
    ( strf::destination<CharT>& dest ) const noexcept
{
    const std::size_t sub_size = data_.sub_chars_count + decimal_point_size_ - data_.showpoint;
    dest.ensure(data_.sub_chars_count);
    auto *it = dest.buffer_ptr();
    if (data_.showsign)  {
        *it++ = static_cast<CharT>(data_.sign);
    }
    *it++ = '0';
    *it++ = 'X' | ((lettercase_ != strf::uppercase) << 5);
    auto content_width = data_.sub_chars_count + data_.extra_zeros;
    if (data_.pad0width > content_width) {
        dest.advance_to(it);
        strf::detail::write_fill(dest, data_.pad0width - content_width, (CharT)'0');
        dest.ensure(sub_size - 2 - data_.showsign);
        it = dest.buffer_ptr();
    }
    *it ++ = 0x30 | int(!data_.subnormal); // '0' or  '1'
    if (data_.showpoint) {
        if (decimal_point_ < 0x80) {
            *it ++ = static_cast<CharT>(decimal_point_);
        } else {
            it = encode_char_(it, decimal_point_);
        }
    }
    if (data_.mantissa != 0) {
        const std::uint8_t digits[13] =
            { static_cast<std::uint8_t>((data_.mantissa & (0xFULL << 48)) >> 48)
            , static_cast<std::uint8_t>((data_.mantissa & (0xFULL << 44)) >> 44)
            , static_cast<std::uint8_t>((data_.mantissa & (0xFULL << 40)) >> 40)
            , static_cast<std::uint8_t>((data_.mantissa & (0xFULL << 36)) >> 36)
            , static_cast<std::uint8_t>((data_.mantissa & (0xFULL << 32)) >> 32)
            , static_cast<std::uint8_t>((data_.mantissa & (0xFULL << 28)) >> 28)
            , static_cast<std::uint8_t>((data_.mantissa & (0xFULL << 24)) >> 24)
            , static_cast<std::uint8_t>((data_.mantissa & (0xFULL << 20)) >> 20)
            , static_cast<std::uint8_t>((data_.mantissa & (0xFULL << 16)) >> 16)
            , static_cast<std::uint8_t>((data_.mantissa & (0xFULL << 12)) >> 12)
            , static_cast<std::uint8_t>((data_.mantissa & (0xFULL <<  8)) >>  8)
            , static_cast<std::uint8_t>((data_.mantissa & (0xFULL <<  4)) >>  4)
            , static_cast<std::uint8_t>(data_.mantissa & 0xFULL) };

        const char offset_digit_a = ('A' | ((lettercase_ == strf::lowercase) << 5)) - 10;
        for(detail::chars_count_t i = 0; i < data_.mantissa_digcount; ++i) {
            auto digit = digits[i];
            *it ++ = ( digit < 10
                     ? ('0' + digit)
                     : (offset_digit_a + digit) );
        }
    }
    if (data_.extra_zeros) {
        dest.advance_to(it);
        detail::write_fill(dest, data_.extra_zeros,  (CharT)'0');
        dest.ensure(2 + data_.exponent_digcount);
        it = dest.buffer_ptr();
    }
    it[0] = 'P' | ((lettercase_ != strf::uppercase) << 5);
    it[1] = static_cast<CharT>('+') + ((data_.exponent < 0) << 1);
    it += 2 + data_.exponent_digcount;
    strf::detail::write_int_dec_txtdigits_backwards
        ( strf::detail::unsigned_abs(data_.exponent), it );
    dest.advance_to(it);
}

template <typename CharT>
STRF_HD void punct_double_printer<CharT>::print_inf_or_nan_
    ( strf::destination<CharT>& dest ) const noexcept
{
    if (data_.showsign) {
        put(dest, static_cast<CharT>(data_.sign));
    }
    if (data_.form == detail::float_form::nan) {
        strf::detail::print_nan(dest, lettercase_);
    } else {
        strf::detail::print_inf(dest, lettercase_);
    }
}

#if defined(STRF_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class punct_double_printer<char8_t>;
STRF_EXPLICIT_TEMPLATE class fast_double_printer<char8_t>;
// STRF_EXPLICIT_TEMPLATE class fast_punct_double_printer<char8_t>;
#endif

STRF_EXPLICIT_TEMPLATE class punct_double_printer<char>;
STRF_EXPLICIT_TEMPLATE class punct_double_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE class punct_double_printer<char32_t>;
STRF_EXPLICIT_TEMPLATE class punct_double_printer<wchar_t>;

STRF_EXPLICIT_TEMPLATE class fast_double_printer<char>;
STRF_EXPLICIT_TEMPLATE class fast_double_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE class fast_double_printer<char32_t>;
STRF_EXPLICIT_TEMPLATE class fast_double_printer<wchar_t>;

// STRF_EXPLICIT_TEMPLATE class fast_punct_double_printer<char>;
// STRF_EXPLICIT_TEMPLATE class fast_punct_double_printer<char16_t>;
// STRF_EXPLICIT_TEMPLATE class fast_punct_double_printer<char32_t>;
// STRF_EXPLICIT_TEMPLATE class fast_punct_double_printer<wchar_t>;

#endif // defined(STRF_SEPARATE_COMPILATION)

} // namespace detail

} // namespace strf

#endif // STRF_DETAIL_PRINTABLE_TYPES_FLOAT_HPP

