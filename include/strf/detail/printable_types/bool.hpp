#ifndef STRF_DETAIL_PRINTABLE_TYPES_BOOL_HPP
#define STRF_DETAIL_PRINTABLE_TYPES_BOOL_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/printer.hpp>
#include <strf/detail/format_functions.hpp>
#include <strf/detail/facets/lettercase.hpp>
#include <strf/detail/facets/charset.hpp>

namespace strf {
namespace detail {

template <typename CharT>
struct bool_printer
{
    bool value_;
    strf::lettercase lettercase_;

    void STRF_HD print_to(strf::destination<CharT>& dst) const;
};

template <typename CharT>
void STRF_HD bool_printer<CharT>::print_to(strf::destination<CharT>& dst) const
{
    auto size = 5 - (int)value_;
    dst.ensure(size);
    auto *p = dst.buffer_ptr();
    const unsigned mask_first_char = static_cast<unsigned>(lettercase_) >> 8;
    const unsigned mask_others_chars = static_cast<unsigned>(lettercase_) & 0x20;
    if (value_) {
        p[0] = static_cast<CharT>('T' | mask_first_char);
        p[1] = static_cast<CharT>('R' | mask_others_chars);
        p[2] = static_cast<CharT>('U' | mask_others_chars);
        p[3] = static_cast<CharT>('E' | mask_others_chars);
    } else {
        p[0] = static_cast<CharT>('F' | mask_first_char);
        p[1] = static_cast<CharT>('A' | mask_others_chars);
        p[2] = static_cast<CharT>('L' | mask_others_chars);
        p[3] = static_cast<CharT>('S' | mask_others_chars);
        p[4] = static_cast<CharT>('E' | mask_others_chars);
    }
    dst.advance(size);
}

template <typename CharT>
struct fmt_bool_printer
{
    void STRF_HD print_to(strf::destination<CharT>& dst) const;

    strf::encode_fill_f<CharT> encode_fill_ = nullptr;
    int fillcount_;
    bool value_;
    strf::alignment_format afmt_;
    strf::lettercase lettercase_;
};

template <typename CharT>
void fmt_bool_printer<CharT>::print_to
    ( strf::destination<CharT>& dst ) const
{
    decltype(fillcount_) right_fillcount = 0;
    if (fillcount_ > 0) {
        decltype(fillcount_) left_fillcount;
        switch (afmt_.alignment) {
            case strf::text_alignment::left:
                right_fillcount = fillcount_;
                goto print_value;
            case strf::text_alignment::right:
                left_fillcount = fillcount_;
                break;
            default:
                left_fillcount = fillcount_ >> 1;
                right_fillcount = fillcount_ - left_fillcount;
        }
        encode_fill_(dst, left_fillcount, afmt_.fill);
    }
    print_value:
    auto size = 5 - (int)value_;
    dst.ensure(size);
    auto *p = dst.buffer_ptr();
    const unsigned mask_first_char = static_cast<unsigned>(lettercase_) >> 8;
    const unsigned mask_others_chars = static_cast<unsigned>(lettercase_) & 0x20;
    if (value_) {
        p[0] = static_cast<CharT>('T' | mask_first_char);
        p[1] = static_cast<CharT>('R' | mask_others_chars);
        p[2] = static_cast<CharT>('U' | mask_others_chars);
        p[3] = static_cast<CharT>('E' | mask_others_chars);
    } else {
        p[0] = static_cast<CharT>('F' | mask_first_char);
        p[1] = static_cast<CharT>('A' | mask_others_chars);
        p[2] = static_cast<CharT>('L' | mask_others_chars);
        p[3] = static_cast<CharT>('S' | mask_others_chars);
        p[4] = static_cast<CharT>('E' | mask_others_chars);
    }
    dst.advance(size);

    if (right_fillcount > 0) {
        encode_fill_(dst, right_fillcount, afmt_.fill);
    }
}

#if defined(STRF_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE struct bool_printer<char8_t>;
STRF_EXPLICIT_TEMPLATE struct fmt_bool_printer<char8_t>;
#endif

STRF_EXPLICIT_TEMPLATE struct bool_printer<char>;
STRF_EXPLICIT_TEMPLATE struct bool_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE struct bool_printer<char32_t>;
STRF_EXPLICIT_TEMPLATE struct bool_printer<wchar_t>;

STRF_EXPLICIT_TEMPLATE struct fmt_bool_printer<char>;
STRF_EXPLICIT_TEMPLATE struct fmt_bool_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE struct fmt_bool_printer<char32_t>;
STRF_EXPLICIT_TEMPLATE struct fmt_bool_printer<wchar_t>;

#endif // defined(STRF_SEPARATE_COMPILATION)

} // namespace detail

template <>
struct printable_traits<bool>
{
    using representative_type = bool;
    using forwarded_type = bool;
    using formatters = strf::tag<strf::alignment_formatter>;
    using is_overridable = std::true_type;

    template <typename CharT, typename PreMeasurements, typename FPack>
    STRF_HD static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& fp
        , bool x ) noexcept
        -> strf::detail::bool_printer<CharT>
    {
        pre->subtract_width(static_cast<strf::width_t>(5 - x));
        pre->add_size(5 - (int)x);
        return detail::bool_printer<CharT>{x, strf::use_facet<strf::lettercase_c, bool>(fp)};
    }

    template <typename CharT, typename PreMeasurements, typename FPack, typename... T>
    STRF_HD static auto make_printer
        ( strf::tag<CharT>
        , PreMeasurements* pre
        , const FPack& fp
        , strf::printable_with_fmt<T...> x ) noexcept
        -> strf::detail::fmt_bool_printer<CharT>
    {
        const bool value = x.value();
        const int w = 5 - (int) value;
        const auto afmt = x.get_alignment_format();
        const auto fmt_width = afmt.width.round();
        const auto lcase = strf::use_facet<strf::lettercase_c, bool>(fp);

        if (fmt_width > w) {
            const int fillcount = fmt_width - w;
            auto charset = strf::use_facet<charset_c<CharT>, bool>(fp);

            pre->subtract_width(static_cast<strf::width_t>(fmt_width));
            pre->add_size(w + fillcount * charset.encoded_char_size(afmt.fill));

            return {charset.encode_fill_func(), fmt_width - w, value, afmt, lcase};
        }
        pre->subtract_width(static_cast<strf::width_t>(w));
        pre->add_size(w);
        return {nullptr, 0, value, afmt, lcase};
    }
};

constexpr STRF_HD strf::printable_traits<bool>
tag_invoke(strf::printable_tag, bool) noexcept { return {}; }

} // namespace strf

#endif // STRF_DETAIL_PRINTABLE_TYPES_BOOL_HPP

