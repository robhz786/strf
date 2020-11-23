#ifndef STRF_DETAIL_INPUT_TYPES_CHAR_HPP_INCLUDED
#define STRF_DETAIL_INPUT_TYPES_CHAR_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/printer.hpp>
#include <strf/facets_pack.hpp>
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
    using facet_tag = SrcCharT;
    using forwarded_type = SrcCharT;
    using formatters = strf::tag<strf::quantity_formatter, strf::alignment_formatter>;

    template <typename DestCharT, typename Preview, typename FPack>
    constexpr STRF_HD static auto make_printer_input
        ( Preview& preview, const FPack& fp, SrcCharT x ) noexcept
        -> strf::usual_printer_input
            < DestCharT, Preview, FPack, SrcCharT, strf::detail::char_printer<DestCharT> >
    {
        static_assert( std::is_same<SrcCharT, DestCharT>::value, "Character type mismatch.");
        return {preview, fp, x};
    }

    template <typename DestCharT, typename Preview, typename FPack, typename... T>
    constexpr STRF_HD static auto make_printer_input
    ( Preview& preview, const FPack& fp, strf::value_with_formatters<T...> x ) noexcept
        -> strf::usual_printer_input
            < DestCharT, Preview, FPack
            , strf::value_with_formatters<T...>
            , strf::detail::fmt_char_printer<DestCharT> >
    {
        static_assert( std::is_same<SrcCharT, DestCharT>::value, "Character type mismatch.");
        return {preview, fp, x};
    }
};

#if defined(__cpp_char8_t)
template <> struct print_traits<char8_t> : public char_printing <char8_t> {};
#endif // defined(__cpp_char8_t)
template <> struct print_traits<char>     : public char_printing <char> {};
template <> struct print_traits<char16_t> : public char_printing <char16_t> {};
template <> struct print_traits<wchar_t>  : public char_printing <wchar_t> {};

template <>
struct print_traits<char32_t>
{
    using facet_tag = char32_t;
    using forwarded_type = char32_t;
    using formatters = strf::tag<strf::quantity_formatter, strf::alignment_formatter>;

    template <typename DestCharT, typename Preview, typename FPack>
    constexpr STRF_HD static auto make_printer_input
        ( Preview& preview, const FPack& fp, char32_t x ) noexcept
        -> strf::usual_printer_input
            < DestCharT, Preview, FPack, char32_t
            , std::conditional_t< std::is_same<DestCharT, char32_t>::value
                                , strf::detail::char_printer<DestCharT>
                                , strf::detail::conv_char32_printer<DestCharT> > >
    {
        return {preview, fp, x};
    }

    template <typename DestCharT, typename Preview, typename FPack, typename... F>
    constexpr STRF_HD static auto make_printer_input
        ( Preview& preview, const FPack& fp, strf::value_with_formatters<F...> x ) noexcept
        -> strf::usual_printer_input
            < DestCharT, Preview, FPack, strf::value_with_formatters<F...>
            , std::conditional_t< std::is_same<DestCharT, char32_t>::value
                                , strf::detail::fmt_char_printer<DestCharT>
                                , strf::detail::fmt_conv_char32_printer<DestCharT> > >
    {
        return {preview, fp, x};
    }
};

#if defined(__cpp_char8_t)

constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, char8_t) noexcept
    -> strf::char_printing<char8_t>
    { return {}; }

#endif // defined(__cpp_char8_t)

constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, char) noexcept
    -> strf::char_printing<char>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, char16_t) noexcept
    -> strf::char_printing<char16_t>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, char32_t) noexcept
    -> strf::print_traits<char32_t>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, wchar_t) noexcept
    -> strf::char_printing<wchar_t>
    { return {}; }

namespace detail {

template <typename CharT>
class char_printer: public strf::printer<CharT>
{
public:
    template <typename... T>
    STRF_HD char_printer
        ( const strf::usual_printer_input<CharT, T...>& input )
        : ch_(static_cast<CharT>(input.arg))
    {
        input.preview.add_size(1);
        using preview_type = typename strf::usual_printer_input<CharT, T...>::preview_type;
        STRF_IF_CONSTEXPR(preview_type::width_required) {
            decltype(auto) wcalc = get_facet<strf::width_calculator_c, CharT>(input.fp);
            auto enc = get_facet<strf::char_encoding_c<CharT>, CharT>(input.fp);
            auto w = wcalc.char_width(enc, static_cast<CharT>(ch_));
            input.preview.subtract_width(w);
        }
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override;

private:

    CharT ch_;
};

template <typename CharT>
STRF_HD void char_printer<CharT>::print_to
    ( strf::basic_outbuff<CharT>& ob ) const
{
    ob.ensure(1);
    *ob.pointer() = ch_;
    ob.advance();
}


template <typename CharT>
class fmt_char_printer: public strf::printer<CharT>
{
public:
    template <typename... T>
    STRF_HD fmt_char_printer
        ( const usual_printer_input<CharT, T...>& input )
        : count_(input.arg.count())
        , afmt_(input.arg.get_alignment_format())
        , ch_(static_cast<CharT>(input.arg.value()))
    {
        auto enc = get_facet_<strf::char_encoding_c<CharT>>(input.fp);
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c>(input.fp);
        encode_fill_fn_ = enc.encode_fill_func();
        init_(input.preview, wcalc, enc);
    }

    STRF_HD void print_to(strf::basic_outbuff<CharT>& ob) const override;

private:

    strf::encode_fill_f<CharT> encode_fill_fn_;
    std::size_t count_;
    const strf::alignment_format afmt_;
    std::uint16_t left_fillcount_;
    std::uint16_t right_fillcount_;
    CharT ch_;

    template <typename Category, typename FPack>
    static STRF_HD decltype(auto) get_facet_(const FPack& fp)
    {
        return fp.template get_facet<Category, CharT>();
    }

    template <typename Preview, typename WCalc, typename Encoding>
    STRF_HD void init_(Preview& preview, const WCalc& wc, Encoding enc);
};

template <typename CharT>
template <typename Preview, typename WCalc, typename Encoding>
STRF_HD void fmt_char_printer<CharT>::init_
    ( Preview& preview, const WCalc& wc, Encoding enc )
{
    auto ch_width = wc.char_width(enc, ch_);
    auto content_width = checked_mul(ch_width, count_);
    std::uint16_t fillcount = 0;
    if (content_width < afmt_.width) {
        fillcount = static_cast<std::uint16_t>((afmt_.width - content_width).round());
        preview.subtract_width(content_width + fillcount);
    } else {
        fillcount = 0;
        preview.subtract_width(content_width);
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
    STRF_IF_CONSTEXPR (Preview::size_required) {
        if (fillcount > 0) {
            preview.add_size(count_ + fillcount * enc.encoded_char_size(afmt_.fill));
        } else {
            preview.add_size(count_);
        }
    }
}


template <typename CharT>
STRF_HD void fmt_char_printer<CharT>::print_to
    ( strf::basic_outbuff<CharT>& ob ) const
{
    if (left_fillcount_ != 0) {
        encode_fill_fn_(ob, left_fillcount_, afmt_.fill);
    }
    if (count_ == 1) {
        ob.ensure(1);
        * ob.pointer() = ch_;
        ob.advance();
    } else {
        std::size_t count = count_;
        while(true) {
            std::size_t space = ob.space();
            if (count <= space) {
                strf::detail::str_fill_n(ob.pointer(), count, ch_);
                ob.advance(count);
                break;
            }
            strf::detail::str_fill_n(ob.pointer(), space, ch_);
            count -= space;
            ob.advance_to(ob.end());
            ob.recycle();
        }
    }
    if (right_fillcount_ != 0) {
        encode_fill_fn_(ob, right_fillcount_, afmt_.fill);
    }
}

template <typename DestCharT>
class conv_char32_printer: public strf::printer<DestCharT>
{
public:
    template <typename... T>
    STRF_HD conv_char32_printer(strf::usual_printer_input<T...> input)
        : ch_(input.arg)
    {
        auto encoding = strf::get_facet<char_encoding_c<DestCharT>, char32_t>(input.fp);
        encode_char_f_ = encoding.encode_char_func();
        encoded_char_size_ = encoding.encoded_char_size(input.arg);
        input.preview.add_size(encoded_char_size_);
        using preview_type = typename strf::usual_printer_input<T...>::preview_type;
        STRF_IF_CONSTEXPR (preview_type::width_required) {
            decltype(auto) wcalc = get_facet<strf::width_calculator_c, char32_t>(input.fp);
            input.preview.subtract_width(wcalc.char_width(strf::utf32<char32_t>{}, ch_));
        }
    }

    void STRF_HD print_to(strf::basic_outbuff<DestCharT>& dest) const override;

private:
    strf::encode_char_f<DestCharT> encode_char_f_;
    std::size_t encoded_char_size_;
    char32_t ch_;
};

template <typename DestCharT>
void STRF_HD conv_char32_printer<DestCharT>::print_to(strf::basic_outbuff<DestCharT>& dest) const
{
    dest.ensure(encoded_char_size_);
    encode_char_f_(dest.pointer(), ch_);
    dest.advance(encoded_char_size_);
}

template <typename DestCharT>
class fmt_conv_char32_printer: public strf::printer<DestCharT>
{
public:

    template <typename... T>
    STRF_HD fmt_conv_char32_printer(strf::usual_printer_input<T...> input)
        : count_(input.arg.count())
        , ch_(input.arg.value())
    {
        auto enc = strf::get_facet<char_encoding_c<DestCharT>, char32_t>(input.fp);
        decltype(auto) wcalc = get_facet<strf::width_calculator_c, char32_t>(input.fp);
        auto char_width = wcalc.char_width(strf::utf32<char32_t>{}, ch_);
        init_(input.preview, enc, input.arg.get_alignment_format(), char_width);
    }

    void STRF_HD print_to(strf::basic_outbuff<DestCharT>& dest) const override;

private:

    template <typename Preview, typename Encoding>
    void STRF_HD init_
        ( Preview& preview
        , Encoding enc
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
template <typename Preview, typename Encoding>
void STRF_HD fmt_conv_char32_printer<DestCharT>::init_
    ( Preview& preview
    , Encoding enc
    , strf::alignment_format afmt
    , strf::width_t ch_width )
{
    encode_fill_f_ = enc.encode_fill_func();
    auto content_width = checked_mul(ch_width, count_);
    if (content_width < afmt.width) {
        fillcount_ = static_cast<std::uint16_t>((afmt.width - content_width).round());
        fillchar_ = afmt.fill;
        alignment_ = afmt.alignment;
        preview.subtract_width(content_width + fillcount_);
    } else {
        fillcount_ = 0;
        preview.subtract_width(content_width);
    }
    STRF_IF_CONSTEXPR (Preview::size_required) {            
        preview.add_size(count_ * enc.encoded_char_size(ch_));
        if (fillcount_ > 0) {
            preview.add_size(fillcount_ * enc.encoded_char_size(afmt.fill));
        }
    }
}

template <typename DestCharT>
void STRF_HD fmt_conv_char32_printer<DestCharT>::print_to(strf::basic_outbuff<DestCharT>& dest) const
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

template <typename> struct is_char: public std::false_type {};

#if defined(__cpp_char8_t)
template <> struct is_char<char8_t>: public std::true_type {};
#endif
template <> struct is_char<char>: public std::true_type {};
template <> struct is_char<char16_t>: public std::true_type {};
template <> struct is_char<char32_t>: public std::true_type {};
template <> struct is_char<wchar_t>: public std::true_type {};

} // namespace strf

#endif // STRF_DETAIL_INPUT_TYPES_CHAR_HPP_INCLUDED
