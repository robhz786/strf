#ifndef STRF_DETAIL_INPUT_TYPES_CHAR_HPP_INCLUDED
#define STRF_DETAIL_INPUT_TYPES_CHAR_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/printer.hpp>
#include <strf/facets_pack.hpp>
#include <strf/detail/facets/width_calculator.hpp>
#include <strf/detail/format_functions.hpp>

namespace strf {
namespace detail {
template <typename> class char_printer;
template <typename> class fmt_char_printer;
} // namespace detail

template <typename CharT>
struct char_tag
{
    CharT ch;
};

template <typename CharT>
using char_with_format = strf::value_with_format
    < char_tag<CharT>
    , strf::quantity_format
    , strf::alignment_format >;

#if defined(__cpp_char8_t)

constexpr STRF_HD auto tag_invoke(strf::fmt_tag, char8_t c) noexcept
    -> strf::char_with_format<char8_t>
{
    return strf::char_with_format<char8_t>{strf::char_tag<char8_t>{c}};
}

#endif // defined(__cpp_char8_t)

constexpr STRF_HD auto tag_invoke(strf::fmt_tag, char c) noexcept
    -> strf::char_with_format<char>
{
    return strf::char_with_format<char>{strf::char_tag<char>{c}};
}
constexpr STRF_HD auto tag_invoke(strf::fmt_tag, char16_t c) noexcept
    -> strf::char_with_format<char16_t>
{
    return strf::char_with_format<char16_t>{strf::char_tag<char16_t>{c}};
}
constexpr STRF_HD auto tag_invoke(strf::fmt_tag, char32_t c) noexcept
    -> strf::char_with_format<char32_t>
{
    return strf::char_with_format<char32_t>{strf::char_tag<char32_t>{c}};
}
constexpr STRF_HD auto tag_invoke(strf::fmt_tag, wchar_t c) noexcept
    -> strf::char_with_format<wchar_t>
{
    return strf::char_with_format<wchar_t>{strf::char_tag<wchar_t>{c}};
}

template <typename CharT, typename Preview, typename FPack>
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>, char x, Preview& preview, const FPack& fp ) noexcept
    -> strf::usual_printer_input
        < CharT, CharT, Preview, FPack, strf::detail::char_printer<CharT>>
{
    static_assert( std::is_same<CharT, char>::value, "Character type mismatch.");
    return {x, preview, fp};
}

#if defined(__cpp_char8_t)

template <typename CharT, typename Preview, typename FPack>
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>, char8_t x, Preview& preview, const FPack& fp ) noexcept
    -> strf::usual_printer_input
        < CharT, CharT, Preview, FPack, strf::detail::char_printer<CharT> >
{
    static_assert( std::is_same<CharT, char8_t>::value, "Character type mismatch.");
    return {x, preview, fp};
}

#endif // defined(__cpp_char8_t)

template <typename CharT, typename Preview, typename FPack>
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>, char16_t x, Preview& preview, const FPack& fp ) noexcept
    -> strf::usual_printer_input
        < CharT, CharT, Preview, FPack, strf::detail::char_printer<CharT> >
{
    static_assert( std::is_same<CharT, char16_t>::value, "Character type mismatch.");
    return {x, preview, fp};
}

template <typename CharT, typename Preview, typename FPack>
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>, char32_t x, Preview& preview, const FPack& fp ) noexcept
    -> strf::usual_printer_input
        < CharT, CharT, Preview, FPack, strf::detail::char_printer<CharT> >
{
    static_assert( std::is_same<CharT, char32_t>::value, "Character type mismatch.");
    return {x, preview, fp};
}

template <typename CharT, typename Preview, typename FPack>
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<CharT>, wchar_t x, Preview& preview, const FPack& fp ) noexcept
    -> strf::usual_printer_input
        < CharT, CharT, Preview, FPack, strf::detail::char_printer<CharT>>
{
    static_assert( std::is_same<CharT, wchar_t>::value, "Character type mismatch.");
    return {x, preview, fp};
}

template <typename DestCharT, typename SrcCharT, typename Preview, typename FPack>
constexpr STRF_HD auto tag_invoke
    ( strf::printer_input_tag<DestCharT>
    , strf::char_with_format<SrcCharT> x
    , Preview& preview
    , const FPack& fp ) noexcept
    -> strf::usual_printer_input
        < DestCharT, strf::char_with_format<SrcCharT>, Preview, FPack
        , strf::detail::fmt_char_printer<DestCharT> >
{
    static_assert( std::is_same<DestCharT, SrcCharT>::value, "Character type mismatch.");
    return {x, preview, fp};
}

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
        , afmt_(input.arg.get_alignment_format_data())
        , ch_(static_cast<CharT>(input.arg.value().ch))
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
    const strf::alignment_format_data afmt_;
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
        preview.checked_subtract_width(content_width + fillcount);
    } else {
        fillcount = 0;
        preview.checked_subtract_width(content_width);
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
            std::size_t space = ob.size();
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

#if defined(STRF_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class fmt_char_printer<char8_t>;
STRF_EXPLICIT_TEMPLATE class char_printer<char8_t>;
#endif

STRF_EXPLICIT_TEMPLATE class char_printer<char>;
STRF_EXPLICIT_TEMPLATE class char_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE class char_printer<char32_t>;
STRF_EXPLICIT_TEMPLATE class char_printer<wchar_t>;

STRF_EXPLICIT_TEMPLATE class fmt_char_printer<char>;
STRF_EXPLICIT_TEMPLATE class fmt_char_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE class fmt_char_printer<char32_t>;
STRF_EXPLICIT_TEMPLATE class fmt_char_printer<wchar_t>;

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
