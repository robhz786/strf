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

template <typename CharT>
struct char_printer
{
    STRF_HD void operator()(strf::destination<CharT>& dst) const
    {
        dst.ensure(1);
        *dst.buffer_ptr() = ch_;
        dst.advance();
    }

    CharT ch_;
};

template <typename CharT>
class fmt_char_printer
{
public:

    template <typename PreMeasurements, typename FPack, typename... T>
    STRF_HD fmt_char_printer
        ( PreMeasurements* pre
        , const FPack& facets
        , strf::printable_with_fmt<T...> arg )
        : count_(arg.scount())
        , afmt_(arg.get_alignment_format())
        , ch_(static_cast<CharT>(arg.value()))
    {
        auto charset = use_facet_<strf::charset_c<CharT>>(facets);
        auto&& wcalc = use_facet_<strf::width_calculator_c>(facets);
        encode_fill_fn_ = charset.encode_fill_func();
        init_(pre, wcalc, charset);
    }

    STRF_HD void operator()(strf::destination<CharT>& dst) const;

private:

    strf::encode_fill_f<CharT> encode_fill_fn_;
    std::ptrdiff_t count_;
    strf::alignment_format afmt_;
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

    template <typename PreMeasurements, typename WCalc, typename Charset>
    STRF_HD void init_(PreMeasurements* pre, const WCalc& wc, Charset charset);
};

template <typename CharT>
template <typename PreMeasurements, typename WCalc, typename Charset>
STRF_HD void fmt_char_printer<CharT>::init_
    ( PreMeasurements* pre, const WCalc& wc, Charset charset )
{
    auto ch_width = wc.char_width(charset, ch_);
    auto content_width = strf::sat_mul(ch_width, count_);
    int fillcount = 0;
    if (content_width < afmt_.width) {
        fillcount = (afmt_.width - content_width).round();
        pre->add_width(content_width + static_cast<strf::width_t>(fillcount));
    } else {
        fillcount = 0;
        pre->add_width(content_width);
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
    STRF_IF_CONSTEXPR (PreMeasurements::size_demanded) {
        if (fillcount > 0) {
            pre->add_size(count_ + fillcount * charset.encoded_char_size(afmt_.fill));
        } else {
            pre->add_size(count_);
        }
    }
}


template <typename CharT>
STRF_HD void fmt_char_printer<CharT>::operator()
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
struct conv_char32_printer
{
    void STRF_HD operator()(strf::destination<DstCharT>& dst) const;

    strf::encode_char_f<DstCharT> encode_char_f_;
    int encoded_char_size_;
    char32_t ch_;
};

template <typename DstCharT>
void STRF_HD conv_char32_printer<DstCharT>::operator()(strf::destination<DstCharT>& dst) const
{
    dst.ensure(encoded_char_size_);
    encode_char_f_(dst.buffer_ptr(), ch_);
    dst.advance(encoded_char_size_);
}

template <typename DstCharT>
class fmt_conv_char32_printer
{
public:

    template <typename PreMeasurements, typename FPack, typename... F>
    STRF_HD fmt_conv_char32_printer
        ( PreMeasurements* pre
        , const FPack& facets
        , strf::printable_with_fmt<F...> arg )
        : count_(arg.scount())
        , ch_(arg.value())
    {
        auto charset = strf::use_facet<charset_c<DstCharT>, char32_t>(facets);
        auto&& wcalc = use_facet<strf::width_calculator_c, char32_t>(facets);
        auto char_width = wcalc.char_width(strf::utf_t<char32_t>{}, ch_);
        init_(pre, charset, arg.get_alignment_format(), char_width);
    }

    void STRF_HD operator()(strf::destination<DstCharT>& dst) const;

private:

    template <typename PreMeasurements, typename Charset>
    void STRF_HD init_
        ( PreMeasurements* pre
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
template <typename PreMeasurements, typename Charset>
void STRF_HD fmt_conv_char32_printer<DstCharT>::init_
    ( PreMeasurements* pre
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
        pre->add_width(content_width + static_cast<strf::width_t>(fillcount_));
    } else {
        fillcount_ = 0;
        pre->add_width(content_width);
    }
    STRF_IF_CONSTEXPR (PreMeasurements::size_demanded) {
        pre->add_size(count_ * charset.encoded_char_size(ch_));
        if (fillcount_ > 0) {
            pre->add_size(fillcount_ * charset.encoded_char_size(afmt.fill));
        }
    }
}

template <typename DstCharT>
void STRF_HD fmt_conv_char32_printer<DstCharT>::operator()(strf::destination<DstCharT>& dst) const
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
STRF_EXPLICIT_TEMPLATE struct char_printer<char8_t>;
STRF_EXPLICIT_TEMPLATE class fmt_conv_char32_printer<char8_t>;
STRF_EXPLICIT_TEMPLATE struct conv_char32_printer<char8_t>;
#endif

STRF_EXPLICIT_TEMPLATE struct char_printer<char>;
STRF_EXPLICIT_TEMPLATE struct char_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE struct char_printer<char32_t>;
STRF_EXPLICIT_TEMPLATE struct char_printer<wchar_t>;

STRF_EXPLICIT_TEMPLATE class fmt_char_printer<char>;
STRF_EXPLICIT_TEMPLATE class fmt_char_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE class fmt_char_printer<char32_t>;
STRF_EXPLICIT_TEMPLATE class fmt_char_printer<wchar_t>;

STRF_EXPLICIT_TEMPLATE struct conv_char32_printer<char>;
STRF_EXPLICIT_TEMPLATE struct conv_char32_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE struct conv_char32_printer<wchar_t>;

STRF_EXPLICIT_TEMPLATE class fmt_conv_char32_printer<char>;
STRF_EXPLICIT_TEMPLATE class fmt_conv_char32_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE class fmt_conv_char32_printer<wchar_t>;

#endif // defined(STRF_SEPARATE_COMPILATION)

template <typename SrcCharT>
struct char_printing
{
    using representative_type = SrcCharT;
    using forwarded_type = SrcCharT;
    using format_specifiers = strf::tag
        < strf::quantity_format_specifier
        , strf::alignment_format_specifier >;
    using is_overridable = std::false_type;

    template <typename DstCharT, typename PreMeasurements, typename FPack>
    constexpr STRF_HD static auto make_printer
        ( strf::tag<DstCharT>
        , PreMeasurements* pre
        , const FPack& facets
        , SrcCharT x ) noexcept
        -> strf::detail::char_printer<DstCharT>
    {
        static_assert( std::is_same<SrcCharT, DstCharT>::value, "Character type mismatch.");

        pre->add_size(1);
        if (pre->has_remaining_width()) {
            auto&& wcalc = use_facet<strf::width_calculator_c, DstCharT>(facets);
            auto charset = use_facet<strf::charset_c<DstCharT>, SrcCharT>(facets);
            auto w = wcalc.char_width(charset, static_cast<DstCharT>(x));
            pre->add_width(w);
        }

        return strf::detail::char_printer<DstCharT>{x};
    }

    template <typename DstCharT, typename PreMeasurements, typename FPack, typename... T>
    constexpr STRF_HD static auto make_printer
        ( strf::tag<DstCharT>
        , PreMeasurements* pre
        , const FPack& facets
        , strf::printable_with_fmt<T...> arg ) noexcept
        -> strf::detail::fmt_char_printer<DstCharT>
    {
        return {pre, facets, arg};
    }
};

} // namespace detail

#if defined(__cpp_char8_t)
template <> struct printable_def<char8_t> : public detail::char_printing <char8_t> {};
#endif // defined(__cpp_char8_t)
template <> struct printable_def<char>     : public detail::char_printing <char> {};
template <> struct printable_def<char16_t> : public detail::char_printing <char16_t> {};
template <> struct printable_def<wchar_t>  : public detail::char_printing <wchar_t> {};

template <>
struct printable_def<char32_t>
{
    using representative_type = char32_t;
    using forwarded_type = char32_t;
    using format_specifiers = strf::tag
        < strf::quantity_format_specifier
        , strf::alignment_format_specifier >;
    using is_overridable = std::false_type;

    template < typename DstCharT, typename PreMeasurements, typename FPack
             , detail::enable_if_t<std::is_same<DstCharT, char32_t>::value, int> = 0 >
    constexpr STRF_HD static auto make_printer
        ( strf::tag<DstCharT>
        , PreMeasurements* pre
        , const FPack& facets
        , char32_t x ) noexcept
    {
        return detail::char_printing<char32_t>::make_printer
            ( strf::tag<char32_t>(), pre, facets, x );
    }

    template < typename DstCharT, typename PreMeasurements, typename FPack
             , detail::enable_if_t< ! std::is_same<DstCharT, char32_t>::value, int> = 0 >
    constexpr STRF_HD static auto make_printer
        ( strf::tag<DstCharT>
        , PreMeasurements* pre
        , const FPack& facets
        , char32_t x ) noexcept
        -> strf::detail::conv_char32_printer<DstCharT>
    {
        auto encoding = strf::use_facet<charset_c<DstCharT>, char32_t>(facets);
        STRF_MAYBE_UNUSED(encoding);
        auto encoded_char_size = encoding.encoded_char_size(x);
        pre->add_size(encoded_char_size);
        if (pre->has_remaining_width()) {
            auto&& wcalc = use_facet<strf::width_calculator_c, char32_t>(facets);
            pre->add_width(wcalc.char_width(strf::utf_t<char32_t>{}, x));
        }
        return strf::detail::conv_char32_printer<DstCharT>
            { encoding.encode_char_func(), encoded_char_size, x };
    }

    template <typename DstCharT, typename PreMeasurements, typename FPack, typename... F>
    constexpr STRF_HD static auto make_printer
        ( strf::tag<DstCharT>
        , PreMeasurements* pre
        , const FPack& fp
        , strf::printable_with_fmt<F...> x ) noexcept
        -> detail::conditional_t
            < std::is_same<DstCharT, char32_t>::value
            , strf::detail::fmt_char_printer<DstCharT>
            , strf::detail::fmt_conv_char32_printer<DstCharT> >
    {
        return {pre, fp, x};
    }
};

#if defined(__cpp_char8_t)

constexpr STRF_HD auto tag_invoke(strf::printable_tag, char8_t) noexcept
    -> strf::detail::char_printing<char8_t>
    { return {}; }

#endif // defined(__cpp_char8_t)

constexpr STRF_HD auto tag_invoke(strf::printable_tag, char) noexcept
    -> strf::detail::char_printing<char>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::printable_tag, char16_t) noexcept
    -> strf::detail::char_printing<char16_t>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::printable_tag, char32_t) noexcept
    -> strf::printable_def<char32_t>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::printable_tag, wchar_t) noexcept
    -> strf::detail::char_printing<wchar_t>
    { return {}; }


} // namespace strf

#endif // STRF_DETAIL_PRINTABLE_TYPES_CHAR_HPP
