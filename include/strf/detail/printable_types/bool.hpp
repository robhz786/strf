#ifndef STRF_DETAIL_INPUT_TYPES_BOOL_HPP
#define STRF_DETAIL_INPUT_TYPES_BOOL_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/printer.hpp>
#include <strf/detail/facets/lettercase.hpp>

namespace strf {
namespace detail {

template <typename CharT> class bool_stringifier;
template <typename CharT> class fmt_bool_stringifier;

} // namespace detail

template <>
struct printable_traits<bool>
{
    using representative_type = bool;
    using forwarded_type = bool;
    using formatters = strf::tag<strf::alignment_formatter>;
    using is_overridable = std::true_type;

    template <typename CharT, typename PrePrinting, typename FPack>
    constexpr STRF_HD static auto make_input
        ( strf::tag<CharT>
        , PrePrinting& pre
        , const FPack& fp
        , bool x ) noexcept
        -> strf::usual_stringifier_input
            < CharT, PrePrinting, FPack, bool, strf::detail::bool_stringifier<CharT> >
    {
        return {pre, fp, x};
    }

    template <typename CharT, typename PrePrinting, typename FPack, typename... T>
    constexpr STRF_HD static auto make_input
        ( strf::tag<CharT>
        , PrePrinting& pre
        , const FPack& fp
        , strf::value_with_formatters<T...> x ) noexcept
        -> strf::usual_stringifier_input
            < CharT, PrePrinting, FPack
            , strf::value_with_formatters<T...>
            , strf::detail::fmt_bool_stringifier<CharT> >
    {
        return {pre, fp, x};
    }
};

constexpr STRF_HD strf::printable_traits<bool>
tag_invoke(strf::printable_tag, bool) noexcept { return {}; }

namespace detail {

template <typename CharT>
class bool_stringifier: public stringifier<CharT>
{
public:

    template <typename... T>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD bool_stringifier
        ( const strf::usual_stringifier_input<T...>& input )
        : value_(input.arg)
        , lettercase_(strf::use_facet<strf::lettercase_c, bool>(input.facets))
    {
        input.pre.subtract_width(5u - (int)input.arg);
        input.pre.add_size(5 - (int)input.arg);
    }

    void STRF_HD print_to(strf::destination<CharT>& dest) const override;

private:

    bool value_;
    strf::lettercase lettercase_;
};

template <typename CharT>
void STRF_HD bool_stringifier<CharT>::print_to(strf::destination<CharT>& dest) const
{
    auto size = 5 - (int)value_;
    dest.ensure(size);
    auto p = dest.buffer_ptr();
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
    dest.advance(size);
}

template <typename CharT>
class fmt_bool_stringifier: public stringifier<CharT>
{
    using this_type_ = fmt_bool_stringifier<CharT>;

public:

    template <typename... T>
    STRF_HD fmt_bool_stringifier
        ( const strf::usual_stringifier_input<CharT, T...>& input )
        : value_(input.arg.value())
        , afmt_(input.arg.get_alignment_format())
        , lettercase_(strf::use_facet<strf::lettercase_c, bool>(input.facets))
    {
        auto charset = strf::use_facet<charset_c<CharT>, bool>(input.facets);
        std::uint16_t w = 5 - (int)input.arg.value();
        auto fmt_width = afmt_.width.round();
        if (fmt_width > w) {
            encode_fill_ = charset.encode_fill_func();
            fillcount_ = static_cast<std::uint16_t>(fmt_width - w);
            input.pre.subtract_width(fmt_width);
            input.pre.add_size(w + fillcount_ * charset.encoded_char_size(afmt_.fill));
        } else {
            fillcount_ = 0;
            input.pre.subtract_width(w);
            input.pre.add_size(w);
        }
    }

    void STRF_HD print_to(strf::destination<CharT>& dest) const override;

private:

    strf::encode_fill_f<CharT> encode_fill_;
    std::uint16_t fillcount_;
    bool value_;
    strf::alignment_format afmt_;
    strf::lettercase lettercase_;
};

template <typename CharT>
void fmt_bool_stringifier<CharT>::print_to
    ( strf::destination<CharT>& dest ) const
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
        encode_fill_(dest, left_fillcount, afmt_.fill);
    }
    print_value:
    auto size = 5 - (int)value_;
    dest.ensure(size);
    auto p = dest.buffer_ptr();
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
    dest.advance(size);

    if (right_fillcount != 0) {
        encode_fill_(dest, right_fillcount, afmt_.fill);
    }
}

#if defined(STRF_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class bool_stringifier<char8_t>;
STRF_EXPLICIT_TEMPLATE class  fmt_bool_stringifier<char8_t>;
#endif

STRF_EXPLICIT_TEMPLATE class bool_stringifier<char>;
STRF_EXPLICIT_TEMPLATE class bool_stringifier<char16_t>;
STRF_EXPLICIT_TEMPLATE class bool_stringifier<char32_t>;
STRF_EXPLICIT_TEMPLATE class bool_stringifier<wchar_t>;

#endif // defined(STRF_SEPARATE_COMPILATION)

} // namespace detail
} // namespace strf

#endif  // STRF_DETAIL_INPUT_TYPES_BOOL_HPP

