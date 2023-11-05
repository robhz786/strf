#ifndef STRF_DETAIL_PRINTABLE_TYPES_CHAR_HPP
#define STRF_DETAIL_PRINTABLE_TYPES_CHAR_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/format_functions.hpp>
#include <strf/detail/facets/width_calculator.hpp>

namespace strf {

namespace detail {
template <typename> class char_printer;
template <typename> class fmt_char_printer;
template <typename> class conv_char32_printer;
template <typename> class fmt_conv_char32_printer;
} // namespace detail

template <typename SrcCharT>
struct char_printing
{
    using representative_type = SrcCharT;
    using forwarded_type = SrcCharT;
    using formatters = strf::tag<strf::quantity_formatter, strf::alignment_formatter>;
    using is_overridable = std::false_type;

    template <typename DstCharT, typename PrePrinting, typename FPack>
    constexpr STRF_HD static auto make_input
        ( strf::tag<DstCharT>
        , PrePrinting* pre
        , const FPack& fp
        , SrcCharT x ) noexcept
        -> strf::usual_printer_input
            < DstCharT, PrePrinting, FPack, SrcCharT, strf::detail::char_printer<DstCharT> >
    {
        static_assert( std::is_same<SrcCharT, DstCharT>::value, "Character type mismatch.");
        return {pre, fp, x};
    }

    template <typename DstCharT, typename PrePrinting, typename FPack, typename... T>
    constexpr STRF_HD static auto make_input
        ( strf::tag<DstCharT>
        , PrePrinting* pre
        , const FPack& fp
        , strf::printable_with_fmt<T...> x ) noexcept
        -> strf::usual_printer_input
            < DstCharT, PrePrinting, FPack
            , strf::printable_with_fmt<T...>
            , strf::detail::fmt_char_printer<DstCharT> >
    {
        static_assert( std::is_same<SrcCharT, DstCharT>::value, "Character type mismatch.");
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

    template <typename DstCharT, typename PrePrinting, typename FPack>
    constexpr STRF_HD static auto make_input
        ( strf::tag<DstCharT>
        , PrePrinting* pre
        , const FPack& fp
        , char32_t x ) noexcept
        -> strf::usual_printer_input
            < DstCharT, PrePrinting, FPack, char32_t
            , strf::detail::conditional_t
                < std::is_same<DstCharT, char32_t>::value
                , strf::detail::char_printer<DstCharT>
                , strf::detail::conv_char32_printer<DstCharT> > >
    {
        return {pre, fp, x};
    }

    template <typename DstCharT, typename PrePrinting, typename FPack, typename... F>
    constexpr STRF_HD static auto make_input
        ( strf::tag<DstCharT>
        , PrePrinting* pre
        , const FPack& fp
        , strf::printable_with_fmt<F...> x ) noexcept
        -> strf::usual_printer_input
            < DstCharT, PrePrinting, FPack, strf::printable_with_fmt<F...>
            , strf::detail::conditional_t
                < std::is_same<DstCharT, char32_t>::value
                , strf::detail::fmt_char_printer<DstCharT>
                , strf::detail::fmt_conv_char32_printer<DstCharT> > >
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
class char_printer: public strf::printer<CharT>
{
public:
    template <typename... T>
    STRF_HD explicit char_printer
        ( const strf::usual_printer_input<CharT, T...>& input )
        : ch_(static_cast<CharT>(input.arg))
    {
        input.pre->add_size(1);
        if (input.pre->has_remaining_width()) {
            auto&& wcalc = use_facet<strf::width_calculator_c, CharT>(input.facets);
            auto charset = use_facet<strf::charset_c<CharT>, CharT>(input.facets);
            auto w = wcalc.char_width(charset, static_cast<CharT>(ch_));
            input.pre->subtract_width(w);
        }
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override;

private:

    CharT ch_;
};

template <typename CharT>
STRF_HD void char_printer<CharT>::print_to
    ( strf::destination<CharT>& dst ) const
{
    dst.ensure(1);
    *dst.buffer_ptr() = ch_;
    dst.advance();
}


template <typename CharT>
class fmt_char_printer: public strf::printer<CharT>
{
public:
    template <typename... T>
    STRF_HD explicit fmt_char_printer
        ( const usual_printer_input<CharT, T...>& input )
        : count_(input.arg.scount())
        , afmt_(input.arg.get_alignment_format())
        , ch_(static_cast<CharT>(input.arg.value()))
    {
        auto charset = use_facet_<strf::charset_c<CharT>>(input.facets);
        auto&& wcalc = use_facet_<strf::width_calculator_c>(input.facets);
        encode_fill_fn_ = charset.encode_fill_func();
        init_(input.pre, wcalc, charset);
    }

    STRF_HD void print_to(strf::destination<CharT>& dst) const override;

private:

    strf::encode_fill_f<CharT> encode_fill_fn_;
    std::ptrdiff_t count_;
    const strf::alignment_format afmt_;
    int left_fillcount_{};
    int right_fillcount_{};
    CharT ch_;

    template <typename Category, typename FPack>
    static STRF_HD
    STRF_DECLTYPE_AUTO((strf::use_facet<Category, CharT>(std::declval<FPack>())))
    use_facet_(const FPack& fp)
    {
        return fp.template use_facet<Category, CharT>();
    }

    template <typename PrePrinting, typename WCalc, typename Charset>
    STRF_HD void init_(PrePrinting* pre, const WCalc& wc, Charset charset);
};

template <typename CharT>
template <typename PrePrinting, typename WCalc, typename Charset>
STRF_HD void fmt_char_printer<CharT>::init_
    ( PrePrinting* pre, const WCalc& wc, Charset charset )
{
    auto ch_width = wc.char_width(charset, ch_);
    auto content_width = strf::sat_mul(ch_width, count_);
    int fillcount = 0;
    if (content_width < afmt_.width) {
        fillcount = (afmt_.width - content_width).round();
        pre->subtract_width(content_width + static_cast<strf::width_t>(fillcount));
    } else {
        fillcount = 0;
        pre->subtract_width(content_width);
    }
    switch(afmt_.alignment) {
        case strf::text_alignment::left:
            left_fillcount_ = 0;
            right_fillcount_ = fillcount;
            break;
        case strf::text_alignment::center: {
            const auto halfcount = fillcount >> 1;
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
            pre->add_size(count_ + fillcount * charset.encoded_char_size(afmt_.fill));
        } else {
            pre->add_size(count_);
        }
    }
}


template <typename CharT>
STRF_HD void fmt_char_printer<CharT>::print_to
    ( strf::destination<CharT>& dst ) const
{
    if (left_fillcount_ > 0) {
        encode_fill_fn_(dst, left_fillcount_, afmt_.fill);
    }
    if (count_ == 1) {
        dst.ensure(1);
        * dst.buffer_ptr() = ch_;
        dst.advance();
    } else {
        std::ptrdiff_t count = count_;
        while(true) {
            const auto space = dst.buffer_sspace();
            if (count <= space) {
                strf::detail::str_fill_n(dst.buffer_ptr(), count, ch_);
                dst.advance(count);
                break;
            }
            strf::detail::str_fill_n(dst.buffer_ptr(), space, ch_);
            count -= space;
            dst.advance_to(dst.buffer_end());
            dst.flush();
        }
    }
    if (right_fillcount_ > 0) {
        encode_fill_fn_(dst, right_fillcount_, afmt_.fill);
    }
}

template <typename DstCharT>
class conv_char32_printer: public strf::printer<DstCharT>
{
public:
    template <typename... T>
    STRF_HD explicit conv_char32_printer(strf::usual_printer_input<T...> input)
        : ch_(input.arg)
    {
        auto encoding = strf::use_facet<charset_c<DstCharT>, char32_t>(input.facets);
        STRF_MAYBE_UNUSED(encoding);
        encode_char_f_ = encoding.encode_char_func();
        encoded_char_size_ = encoding.encoded_char_size(input.arg);
        input.pre->add_size(encoded_char_size_);
        if (input.pre->has_remaining_width()) {
            auto&& wcalc = use_facet<strf::width_calculator_c, char32_t>(input.facets);
            input.pre->subtract_width(wcalc.char_width(strf::utf_t<char32_t>{}, ch_));
        }
    }

    void STRF_HD print_to(strf::destination<DstCharT>& dst) const override;

private:
    strf::encode_char_f<DstCharT> encode_char_f_;
    int encoded_char_size_;
    char32_t ch_;
};

template <typename DstCharT>
void STRF_HD conv_char32_printer<DstCharT>::print_to(strf::destination<DstCharT>& dst) const
{
    dst.ensure(encoded_char_size_);
    encode_char_f_(dst.buffer_ptr(), ch_);
    dst.advance(encoded_char_size_);
}

template <typename DstCharT>
class fmt_conv_char32_printer: public strf::printer<DstCharT>
{
public:

    template <typename... T>
    STRF_HD explicit fmt_conv_char32_printer(strf::usual_printer_input<T...> input)
        : count_(input.arg.scount())
        , ch_(input.arg.value())
    {
        auto charset = strf::use_facet<charset_c<DstCharT>, char32_t>(input.facets);
        auto&& wcalc = use_facet<strf::width_calculator_c, char32_t>(input.facets);
        auto char_width = wcalc.char_width(strf::utf_t<char32_t>{}, ch_);
        init_(input.pre, charset, input.arg.get_alignment_format(), char_width);
    }

    void STRF_HD print_to(strf::destination<DstCharT>& dst) const override;

private:

    template <typename PrePrinting, typename Charset>
    void STRF_HD init_
        ( PrePrinting* pre
        , Charset charset
        , strf::alignment_format afmt
        , strf::width_t ch_width );

    strf::encode_fill_f<DstCharT> encode_fill_f_;
    std::ptrdiff_t count_;
    char32_t ch_;
    char32_t fillchar_{};
    strf::text_alignment alignment_{strf::text_alignment::right};
    int fillcount_{};
};

template <typename DstCharT>
template <typename PrePrinting, typename Charset>
void STRF_HD fmt_conv_char32_printer<DstCharT>::init_
    ( PrePrinting* pre
    , Charset charset
    , strf::alignment_format afmt
    , strf::width_t ch_width )
{
    encode_fill_f_ = charset.encode_fill_func();
    const auto content_width = strf::sat_mul(ch_width, count_);
    fillchar_ = afmt.fill;
    alignment_ = afmt.alignment;
    if (content_width < afmt.width) {
        fillcount_ = (afmt.width - content_width).round();
        pre->subtract_width(content_width + static_cast<strf::width_t>(fillcount_));
    } else {
        fillcount_ = 0;
        pre->subtract_width(content_width);
    }
    STRF_IF_CONSTEXPR (PrePrinting::size_required) {
        pre->add_size(count_ * charset.encoded_char_size(ch_));
        if (fillcount_ > 0) {
            pre->add_size(fillcount_ * charset.encoded_char_size(afmt.fill));
        }
    }
}

template <typename DstCharT>
void STRF_HD fmt_conv_char32_printer<DstCharT>::print_to(strf::destination<DstCharT>& dst) const
{
    if(fillcount_ <= 0) {
        encode_fill_f_(dst, count_, ch_);
    } else {
        switch(alignment_) {
            case strf::text_alignment::left:
                encode_fill_f_(dst, count_, ch_);
                encode_fill_f_(dst, fillcount_, fillchar_);
                break;
            case strf::text_alignment::center: {
                auto left_fillcount = fillcount_ >> 1;
                encode_fill_f_(dst, left_fillcount, fillchar_);
                encode_fill_f_(dst, count_, ch_);
                encode_fill_f_(dst, fillcount_ - left_fillcount, fillchar_);
                break;
            }
            default:
                encode_fill_f_(dst, fillcount_, fillchar_);
                encode_fill_f_(dst, count_, ch_);
        }
    }
}

#if defined(STRF_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class fmt_char_printer<char8_t>;
STRF_EXPLICIT_TEMPLATE class char_printer<char8_t>;
STRF_EXPLICIT_TEMPLATE class fmt_conv_char32_printer<char8_t>;
STRF_EXPLICIT_TEMPLATE class conv_char32_printer<char8_t>;
#endif

STRF_EXPLICIT_TEMPLATE class char_printer<char>;
STRF_EXPLICIT_TEMPLATE class char_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE class char_printer<char32_t>;
STRF_EXPLICIT_TEMPLATE class char_printer<wchar_t>;

STRF_EXPLICIT_TEMPLATE class fmt_char_printer<char>;
STRF_EXPLICIT_TEMPLATE class fmt_char_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE class fmt_char_printer<char32_t>;
STRF_EXPLICIT_TEMPLATE class fmt_char_printer<wchar_t>;

STRF_EXPLICIT_TEMPLATE class conv_char32_printer<char>;
STRF_EXPLICIT_TEMPLATE class conv_char32_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE class conv_char32_printer<wchar_t>;

STRF_EXPLICIT_TEMPLATE class fmt_conv_char32_printer<char>;
STRF_EXPLICIT_TEMPLATE class fmt_conv_char32_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE class fmt_conv_char32_printer<wchar_t>;

#endif // defined(STRF_SEPARATE_COMPILATION)

} // namespace detail

} // namespace strf

#endif // STRF_DETAIL_PRINTABLE_TYPES_CHAR_HPP
