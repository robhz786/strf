#ifndef STRF_DETAIL_INPUT_TYPES_CHAR_HPP_INCLUDED
#define STRF_DETAIL_INPUT_TYPES_CHAR_HPP_INCLUDED

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/format_functions.hpp>
#include <strf/detail/facets/width_calculator.hpp>

namespace strf {

namespace detail {
template <typename> class char_stringifier;
template <typename> class fmt_char_stringifier;
template <typename> class conv_char32_stringifier;
template <typename> class fmt_conv_char32_stringifier;
} // namespace detail

template <typename SrcCharT>
struct char_printing
{
    using representative_type = SrcCharT;
    using forwarded_type = SrcCharT;
    using formatters = strf::tag<strf::quantity_formatter, strf::alignment_formatter>;
    using is_overridable = std::false_type;

    template <typename DestCharT, typename PrePrinting, typename FPack>
    constexpr STRF_HD static auto make_input
        ( strf::tag<DestCharT>
        , PrePrinting& pre
        , const FPack& fp
        , SrcCharT x ) noexcept
        -> strf::usual_stringifier_input
            < DestCharT, PrePrinting, FPack, SrcCharT, strf::detail::char_stringifier<DestCharT> >
    {
        static_assert( std::is_same<SrcCharT, DestCharT>::value, "Character type mismatch.");
        return {pre, fp, x};
    }

    template <typename DestCharT, typename PrePrinting, typename FPack, typename... T>
    constexpr STRF_HD static auto make_input
        ( strf::tag<DestCharT>
        , PrePrinting& pre
        , const FPack& fp
        , strf::value_with_formatters<T...> x ) noexcept
        -> strf::usual_stringifier_input
            < DestCharT, PrePrinting, FPack
            , strf::value_with_formatters<T...>
            , strf::detail::fmt_char_stringifier<DestCharT> >
    {
        static_assert( std::is_same<SrcCharT, DestCharT>::value, "Character type mismatch.");
        return {pre, fp, x};
    }
};

#if defined(__cpp_char8_t)
template <> struct printable_traits<char8_t> : public char_printing <char8_t> {};
#endif // defined(__cpp_char8_t)
template <> struct printable_traits<char>     : public char_printing <char> {};
template <> struct printable_traits<char16_t> : public char_printing <char16_t> {};
template <> struct printable_traits<wchar_t>  : public char_printing <wchar_t> {};

template <>
struct printable_traits<char32_t>
{
    using representative_type = char32_t;
    using forwarded_type = char32_t;
    using formatters = strf::tag<strf::quantity_formatter, strf::alignment_formatter>;
    using is_overridable = std::false_type;

    template <typename DestCharT, typename PrePrinting, typename FPack>
    constexpr STRF_HD static auto make_input
        ( strf::tag<DestCharT>
        , PrePrinting& pre
        , const FPack& fp
        , char32_t x ) noexcept
        -> strf::usual_stringifier_input
            < DestCharT, PrePrinting, FPack, char32_t
            , strf::detail::conditional_t
                < std::is_same<DestCharT, char32_t>::value
                , strf::detail::char_stringifier<DestCharT>
                , strf::detail::conv_char32_stringifier<DestCharT> > >
    {
        return {pre, fp, x};
    }

    template <typename DestCharT, typename PrePrinting, typename FPack, typename... F>
    constexpr STRF_HD static auto make_input
        ( strf::tag<DestCharT>
        , PrePrinting& pre
        , const FPack& fp
        , strf::value_with_formatters<F...> x ) noexcept
        -> strf::usual_stringifier_input
            < DestCharT, PrePrinting, FPack, strf::value_with_formatters<F...>
            , strf::detail::conditional_t
                < std::is_same<DestCharT, char32_t>::value
                , strf::detail::fmt_char_stringifier<DestCharT>
                , strf::detail::fmt_conv_char32_stringifier<DestCharT> > >
    {
        return {pre, fp, x};
    }
};

#if defined(__cpp_char8_t)

constexpr STRF_HD auto tag_invoke(strf::printable_tag, char8_t) noexcept
    -> strf::char_printing<char8_t>
    { return {}; }

#endif // defined(__cpp_char8_t)

constexpr STRF_HD auto tag_invoke(strf::printable_tag, char) noexcept
    -> strf::char_printing<char>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::printable_tag, char16_t) noexcept
    -> strf::char_printing<char16_t>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::printable_tag, char32_t) noexcept
    -> strf::printable_traits<char32_t>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::printable_tag, wchar_t) noexcept
    -> strf::char_printing<wchar_t>
    { return {}; }

namespace detail {

template <typename CharT>
class char_stringifier: public strf::stringifier<CharT>
{
public:
    template <typename... T>
    STRF_HD char_stringifier
        ( const strf::usual_stringifier_input<CharT, T...>& input )
        : ch_(static_cast<CharT>(input.arg))
    {
        input.pre.add_size(1);
        using pre_t = typename strf::usual_stringifier_input<CharT, T...>::preprinting_type;
        STRF_IF_CONSTEXPR(pre_t::width_required) {
            auto&& wcalc = use_facet<strf::width_calculator_c, CharT>(input.facets);
            auto charset = use_facet<strf::charset_c<CharT>, CharT>(input.facets);
            auto w = wcalc.char_width(charset, static_cast<CharT>(ch_));
            input.pre.subtract_width(w);
        }
    }

    STRF_HD void print_to(strf::destination<CharT>& dest) const override;

private:

    CharT ch_;
};

template <typename CharT>
STRF_HD void char_stringifier<CharT>::print_to
    ( strf::destination<CharT>& dest ) const
{
    dest.ensure(1);
    *dest.buffer_ptr() = ch_;
    dest.advance();
}


template <typename CharT>
class fmt_char_stringifier: public strf::stringifier<CharT>
{
public:
    template <typename... T>
    STRF_HD fmt_char_stringifier
        ( const usual_stringifier_input<CharT, T...>& input )
        : count_(input.arg.count())
        , afmt_(input.arg.get_alignment_format())
        , ch_(static_cast<CharT>(input.arg.value()))
    {
        auto charset = use_facet_<strf::charset_c<CharT>>(input.facets);
        auto&& wcalc = use_facet_<strf::width_calculator_c>(input.facets);
        encode_fill_fn_ = charset.encode_fill_func();
        init_(input.pre, wcalc, charset);
    }

    STRF_HD void print_to(strf::destination<CharT>& dest) const override;

private:

    strf::encode_fill_f<CharT> encode_fill_fn_;
    std::size_t count_;
    const strf::alignment_format afmt_;
    std::uint16_t left_fillcount_;
    std::uint16_t right_fillcount_;
    CharT ch_;

    template <typename Category, typename FPack>
    static STRF_HD
    STRF_DECLTYPE_AUTO((strf::use_facet<Category, CharT>(std::declval<FPack>())))
    use_facet_(const FPack& fp)
    {
        return fp.template use_facet<Category, CharT>();
    }

    template <typename PrePrinting, typename WCalc, typename Charset>
    STRF_HD void init_(PrePrinting& pre, const WCalc& wc, Charset charset);
};

template <typename CharT>
template <typename PrePrinting, typename WCalc, typename Charset>
STRF_HD void fmt_char_stringifier<CharT>::init_
    ( PrePrinting& pre, const WCalc& wc, Charset charset )
{
    auto ch_width = wc.char_width(charset, ch_);
    auto content_width = checked_mul(ch_width, count_);
    std::uint16_t fillcount = 0;
    if (content_width < afmt_.width) {
        fillcount = static_cast<std::uint16_t>((afmt_.width - content_width).round());
        pre.subtract_width(content_width + fillcount);
    } else {
        fillcount = 0;
        pre.subtract_width(content_width);
    }
    switch(afmt_.alignment) {
        case strf::text_alignment::left:
            left_fillcount_ = 0;
            right_fillcount_ = fillcount;
            break;
        case strf::text_alignment::center: {
            std::uint16_t halfcount = fillcount >> 1;
            left_fillcount_ = halfcount;
            right_fillcount_ = fillcount - halfcount;
            break;
        }
        default:
            left_fillcount_ = fillcount;
            right_fillcount_ = 0;
    }
    STRF_IF_CONSTEXPR (PrePrinting::size_required) {
        if (fillcount > 0) {
            pre.add_size(count_ + fillcount * charset.encoded_char_size(afmt_.fill));
        } else {
            pre.add_size(count_);
        }
    }
}


template <typename CharT>
STRF_HD void fmt_char_stringifier<CharT>::print_to
    ( strf::destination<CharT>& dest ) const
{
    if (left_fillcount_ != 0) {
        encode_fill_fn_(dest, left_fillcount_, afmt_.fill);
    }
    if (count_ == 1) {
        dest.ensure(1);
        * dest.buffer_ptr() = ch_;
        dest.advance();
    } else {
        std::size_t count = count_;
        while(true) {
            std::size_t space = dest.buffer_space();
            if (count <= space) {
                strf::detail::str_fill_n(dest.buffer_ptr(), count, ch_);
                dest.advance(count);
                break;
            }
            strf::detail::str_fill_n(dest.buffer_ptr(), space, ch_);
            count -= space;
            dest.advance_to(dest.buffer_end());
            dest.flush();
        }
    }
    if (right_fillcount_ != 0) {
        encode_fill_fn_(dest, right_fillcount_, afmt_.fill);
    }
}

template <typename DestCharT>
class conv_char32_stringifier: public strf::stringifier<DestCharT>
{
public:
    template <typename... T>
    STRF_HD conv_char32_stringifier(strf::usual_stringifier_input<T...> input)
        : ch_(input.arg)
    {
        auto encoding = strf::use_facet<charset_c<DestCharT>, char32_t>(input.facets);
        STRF_MAYBE_UNUSED(encoding);
        encode_char_f_ = encoding.encode_char_func();
        encoded_char_size_ = encoding.encoded_char_size(input.arg);
        input.pre.add_size(encoded_char_size_);
        using pre_t = typename strf::usual_stringifier_input<T...>::preprinting_type;
        STRF_IF_CONSTEXPR (pre_t::width_required) {
            auto&& wcalc = use_facet<strf::width_calculator_c, char32_t>(input.facets);
            input.pre.subtract_width(wcalc.char_width(strf::utf_t<char32_t>{}, ch_));
        }
    }

    void STRF_HD print_to(strf::destination<DestCharT>& dest) const override;

private:
    strf::encode_char_f<DestCharT> encode_char_f_;
    std::size_t encoded_char_size_;
    char32_t ch_;
};

template <typename DestCharT>
void STRF_HD conv_char32_stringifier<DestCharT>::print_to(strf::destination<DestCharT>& dest) const
{
    dest.ensure(encoded_char_size_);
    encode_char_f_(dest.buffer_ptr(), ch_);
    dest.advance(encoded_char_size_);
}

template <typename DestCharT>
class fmt_conv_char32_stringifier: public strf::stringifier<DestCharT>
{
public:

    template <typename... T>
    STRF_HD fmt_conv_char32_stringifier(strf::usual_stringifier_input<T...> input)
        : count_(input.arg.count())
        , ch_(input.arg.value())
    {
        auto charset = strf::use_facet<charset_c<DestCharT>, char32_t>(input.facets);
        auto&& wcalc = use_facet<strf::width_calculator_c, char32_t>(input.facets);
        auto char_width = wcalc.char_width(strf::utf_t<char32_t>{}, ch_);
        init_(input.pre, charset, input.arg.get_alignment_format(), char_width);
    }

    void STRF_HD print_to(strf::destination<DestCharT>& dest) const override;

private:

    template <typename PrePrinting, typename Charset>
    void STRF_HD init_
        ( PrePrinting& pre
        , Charset charset
        , strf::alignment_format afmt
        , strf::width_t ch_width );

    strf::encode_fill_f<DestCharT> encode_fill_f_;
    std::size_t count_;
    char32_t ch_;
    char32_t fillchar_;
    strf::text_alignment alignment_;
    std::uint16_t fillcount_;
};

template <typename DestCharT>
template <typename PrePrinting, typename Charset>
void STRF_HD fmt_conv_char32_stringifier<DestCharT>::init_
    ( PrePrinting& pre
    , Charset charset
    , strf::alignment_format afmt
    , strf::width_t ch_width )
{
    encode_fill_f_ = charset.encode_fill_func();
    auto content_width = checked_mul(ch_width, count_);
    fillchar_ = afmt.fill;
    alignment_ = afmt.alignment;
    if (content_width < afmt.width) {
        fillcount_ = static_cast<std::uint16_t>((afmt.width - content_width).round());
        pre.subtract_width(content_width + fillcount_);
    } else {
        fillcount_ = 0;
        pre.subtract_width(content_width);
    }
    STRF_IF_CONSTEXPR (PrePrinting::size_required) {
        pre.add_size(count_ * charset.encoded_char_size(ch_));
        if (fillcount_ > 0) {
            pre.add_size(fillcount_ * charset.encoded_char_size(afmt.fill));
        }
    }
}

template <typename DestCharT>
void STRF_HD fmt_conv_char32_stringifier<DestCharT>::print_to(strf::destination<DestCharT>& dest) const
{
    if(fillcount_ == 0) {
        encode_fill_f_(dest, count_, ch_);
    } else {
        switch(alignment_) {
            case strf::text_alignment::left:
                encode_fill_f_(dest, count_, ch_);
                encode_fill_f_(dest, fillcount_, fillchar_);
                break;
            case strf::text_alignment::center: {
                auto left_fillcount = fillcount_ >> 1;
                encode_fill_f_(dest, left_fillcount, fillchar_);
                encode_fill_f_(dest, count_, ch_);
                encode_fill_f_(dest, fillcount_ - left_fillcount, fillchar_);
                break;
            }
            default:
                encode_fill_f_(dest, fillcount_, fillchar_);
                encode_fill_f_(dest, count_, ch_);
        }
    }
}

#if defined(STRF_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class fmt_char_stringifier<char8_t>;
STRF_EXPLICIT_TEMPLATE class char_stringifier<char8_t>;
STRF_EXPLICIT_TEMPLATE class fmt_conv_char32_stringifier<char8_t>;
STRF_EXPLICIT_TEMPLATE class conv_char32_stringifier<char8_t>;
#endif

STRF_EXPLICIT_TEMPLATE class char_stringifier<char>;
STRF_EXPLICIT_TEMPLATE class char_stringifier<char16_t>;
STRF_EXPLICIT_TEMPLATE class char_stringifier<char32_t>;
STRF_EXPLICIT_TEMPLATE class char_stringifier<wchar_t>;

STRF_EXPLICIT_TEMPLATE class fmt_char_stringifier<char>;
STRF_EXPLICIT_TEMPLATE class fmt_char_stringifier<char16_t>;
STRF_EXPLICIT_TEMPLATE class fmt_char_stringifier<char32_t>;
STRF_EXPLICIT_TEMPLATE class fmt_char_stringifier<wchar_t>;

STRF_EXPLICIT_TEMPLATE class conv_char32_stringifier<char>;
STRF_EXPLICIT_TEMPLATE class conv_char32_stringifier<char16_t>;
STRF_EXPLICIT_TEMPLATE class conv_char32_stringifier<wchar_t>;

STRF_EXPLICIT_TEMPLATE class fmt_conv_char32_stringifier<char>;
STRF_EXPLICIT_TEMPLATE class fmt_conv_char32_stringifier<char16_t>;
STRF_EXPLICIT_TEMPLATE class fmt_conv_char32_stringifier<wchar_t>;

#endif // defined(STRF_SEPARATE_COMPILATION)

} // namespace detail

} // namespace strf

#endif // STRF_DETAIL_INPUT_TYPES_CHAR_HPP_INCLUDED
