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

template <> struct fmt_traits<char8_t>
{
    using fmt_type = strf::char_with_format<char8_t>;
};

#endif // defined(__cpp_char8_t)

template <> struct fmt_traits<char>
{
    using fmt_type = strf::char_with_format<char>;
};
template <> struct fmt_traits<char16_t>
{
    using fmt_type = strf::char_with_format<char16_t>;
};
template <> struct fmt_traits<char32_t>
{
    using fmt_type = strf::char_with_format<char32_t>;
};
template <> struct fmt_traits<wchar_t>
{
    using fmt_type = strf::char_with_format<wchar_t>;
};

namespace detail {

template <std::size_t> class char_printer;
template <std::size_t> class fmt_char_printer;

template <typename DestCharT, typename FPack, typename SrcCharT>
struct char_printable_traits
    : strf::usual_printable_traits
        < DestCharT, FPack, strf::detail::char_printer<sizeof(DestCharT)> >
{
     static_assert( std::is_same<SrcCharT, DestCharT>::value
                  , "Character type mismatch.");
};

} // namespace detail

template <typename CharT, typename FPack, typename Preview>
struct printable_traits<CharT, FPack, Preview, char>
    : strf::detail::char_printable_traits<CharT, FPack, char>
{ };

#if defined(__cpp_char8_t)

template <typename CharT, typename FPack, typename Preview>
struct printable_traits<CharT, FPack, Preview, char8_t>
    : strf::detail::char_printable_traits<CharT, FPack, char8_t>
{ };

#endif // defined(__cpp_char8_t)

template <typename CharT, typename FPack, typename Preview>
struct printable_traits<CharT, FPack, Preview, char16_t>
    : strf::detail::char_printable_traits<CharT, FPack, char16_t>
{ };

template <typename CharT, typename FPack, typename Preview>
struct printable_traits<CharT, FPack, Preview, char32_t>
    : strf::detail::char_printable_traits<CharT, FPack, char32_t>
{ };

template <typename CharT, typename FPack, typename Preview>
struct printable_traits<CharT, FPack, Preview, wchar_t>
    : strf::detail::char_printable_traits<CharT, FPack, wchar_t>
{ };

template <typename DestCharT, typename FPack, typename Preview, typename SrcCharT>
struct printable_traits<DestCharT, FPack, Preview, strf::char_with_format<SrcCharT>>
    : strf::usual_printable_traits
        < DestCharT, FPack, strf::detail::fmt_char_printer<sizeof(DestCharT)> >
{
    static_assert( std::is_same<SrcCharT, DestCharT>::value
                 , "Character type mismatch.");
};

namespace detail {

template <std::size_t CharSize>
class char_printer: public strf::printer<CharSize>
{
public:
    using char_type = strf::underlying_char_type<CharSize>;

    template <typename CharT, typename FPack, typename Preview, typename... T>
    STRF_HD char_printer
        ( const strf::usual_printer_input<CharT, FPack, Preview, T...>& input )
        : ch_(static_cast<char_type>(input.arg))
    {
        static_assert(sizeof(CharT) == CharSize, "");
        input.preview.add_size(1);
        STRF_IF_CONSTEXPR(Preview::width_required) {
            decltype(auto) wcalc = get_facet<strf::width_calculator_c, CharT>(input.fp);
            auto enc = get_facet<strf::char_encoding_c<CharT>, CharT>(input.fp);
            auto w = wcalc.char_width(enc, static_cast<char_type>(ch_));
            input.preview.subtract_width(w);
        }
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    char_type ch_;
};

template <std::size_t CharSize>
STRF_HD void char_printer<CharSize>::print_to
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    ob.ensure(1);
    *ob.pointer() = ch_;
    ob.advance();
}


template <std::size_t CharSize>
class fmt_char_printer: public strf::printer<CharSize>
{
public:
    using char_type = strf::underlying_char_type<CharSize>;

    template <typename CharT, typename... T>
    STRF_HD fmt_char_printer
        ( const usual_printer_input<CharT, T...>& input )
        : count_(input.arg.count())
        , afmt_(input.arg.get_alignment_format_data())
        , ch_(static_cast<char_type>(input.arg.value().ch))
    {
        auto enc = get_facet_<strf::char_encoding_c<CharT>, CharT>(input.fp);
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c, CharT>(input.fp);
        encode_fill_fn_ = enc.encode_fill_func();
        init_(input.preview, wcalc, enc);
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    strf::encode_fill_f<CharSize> encode_fill_fn_;
    std::size_t count_;
    const strf::alignment_format_data afmt_;
    std::uint16_t left_fillcount_;
    std::uint16_t right_fillcount_;
    char_type ch_;

    template <typename Category, typename CharT, typename FPack>
    static STRF_HD decltype(auto) get_facet_(const FPack& fp)
    {
        return fp.template get_facet<Category, CharT>();
    }

    template <typename Preview, typename WCalc, typename Encoding>
    STRF_HD void init_(Preview& preview, const WCalc& wc, Encoding enc);
};

template <std::size_t CharSize>
template <typename Preview, typename WCalc, typename Encoding>
STRF_HD void fmt_char_printer<CharSize>::init_
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


template <std::size_t CharSize>
STRF_HD void fmt_char_printer<CharSize>::print_to
    ( strf::underlying_outbuf<CharSize>& ob ) const
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

STRF_EXPLICIT_TEMPLATE class char_printer<1>;
STRF_EXPLICIT_TEMPLATE class char_printer<2>;
STRF_EXPLICIT_TEMPLATE class char_printer<4>;
STRF_EXPLICIT_TEMPLATE class fmt_char_printer<1>;
STRF_EXPLICIT_TEMPLATE class fmt_char_printer<2>;
STRF_EXPLICIT_TEMPLATE class fmt_char_printer<4>;

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
