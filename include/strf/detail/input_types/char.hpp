#ifndef STRF_DETAIL_INPUT_TYPES_CHAR_HPP_INCLUDED
#define STRF_DETAIL_INPUT_TYPES_CHAR_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>
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

constexpr STRF_HD auto make_fmt(strf::rank<1>, char8_t ch) noexcept
{
    return strf::char_with_format<char8_t>{{ch}};
}

#endif

constexpr STRF_HD auto make_fmt(strf::rank<1>, char ch) noexcept
{
    return strf::char_with_format<char>{{ch}};
}
constexpr STRF_HD auto make_fmt(strf::rank<1>, wchar_t ch) noexcept
{
    return strf::char_with_format<wchar_t>{{ch}};
}
constexpr STRF_HD auto make_fmt(strf::rank<1>, char16_t ch) noexcept
{
    return strf::char_with_format<char16_t>{{ch}};
}
constexpr STRF_HD auto make_fmt(strf::rank<1>, char32_t ch) noexcept
{
    return strf::char_with_format<char32_t>{{ch}};
}

namespace detail {

template <std::size_t> class char_printer;
template <std::size_t> class fmt_char_printer;

template <typename DestCharT, typename FPack, typename SrcCharT>
struct char_printer_traits
    : strf::usual_printer_traits_by_val
        < DestCharT, FPack, strf::detail::char_printer<sizeof(DestCharT)> >
{
     static_assert( std::is_same<SrcCharT, DestCharT>::value
                  , "Character type mismatch.");
};

} // namespace detail

template <typename CharT, typename FPack, typename Preview>
struct printer_traits<CharT, FPack, Preview, char>
    : strf::detail::char_printer_traits<CharT, FPack, char>
{ };

#if defined(__cpp_char8_t)

template <typename CharT, typename FPack, typename Preview>
struct printer_traits<CharT, FPack, Preview, char8_t>
    : strf::detail::char_printer_traits<CharT, FPack, char8_t>
{ };

#endif // defined(__cpp_char8_t)

template <typename CharT, typename FPack, typename Preview>
struct printer_traits<CharT, FPack, Preview, char16_t>
    : strf::detail::char_printer_traits<CharT, FPack, char16_t>
{ };

template <typename CharT, typename FPack, typename Preview>
struct printer_traits<CharT, FPack, Preview, char32_t>
    : strf::detail::char_printer_traits<CharT, FPack, char32_t>
{ };

template <typename CharT, typename FPack, typename Preview>
struct printer_traits<CharT, FPack, Preview, wchar_t>
    : strf::detail::char_printer_traits<CharT, FPack, wchar_t>
{ };

template <typename DestCharT, typename FPack, typename Preview, typename SrcCharT>
struct printer_traits<DestCharT, FPack, Preview, strf::char_with_format<SrcCharT>>
    : strf::usual_printer_traits_by_val
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
            auto w = wcalc.char_width( get_facet<strf::charset_c<CharT>, CharT>(input.fp)
                                     , static_cast<char_type>(ch_) );
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
        , inv_seq_poli_(get_facet_<strf::invalid_seq_policy_c, CharT>(input.fp))
        , surr_poli_(get_facet_<strf::surrogate_policy_c, CharT>(input.fp))
        , ch_(static_cast<char_type>(input.arg.value().ch))
    {
        decltype(auto) cs = get_facet_<strf::charset_c<CharT>, CharT>(input.fp);
        encode_fill_fn_ = cs.encode_fill_func();
        init_( input.preview
             , get_facet_<strf::width_calculator_c, CharT>(input.fp)
             , cs );
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    strf::encode_fill_f<CharSize> encode_fill_fn_;
    std::size_t count_;
    const strf::alignment_format_data afmt_;
    const strf::invalid_seq_policy  inv_seq_poli_;
    const strf::surrogate_policy  surr_poli_;
    std::uint16_t left_fillcount_;
    std::uint16_t right_fillcount_;
    char_type ch_;

    template <typename Category, typename CharT, typename FPack>
    static STRF_HD decltype(auto) get_facet_(const FPack& fp)
    {
        return fp.template get_facet<Category, CharT>();
    }

    template <typename Preview, typename WCalc, typename Charset>
    STRF_HD void init_(Preview& preview, const WCalc& wc, const Charset& cs);
};

template <std::size_t CharSize>
template <typename Preview, typename WCalc, typename Charset>
STRF_HD void fmt_char_printer<CharSize>::init_
    ( Preview& preview, const WCalc& wc, const Charset& cs)
{
    auto ch_width = wc.char_width(cs, ch_);
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
            preview.add_size(count_ + fillcount * cs.encoded_char_size(afmt_.fill));
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
        encode_fill_fn_(ob, left_fillcount_, afmt_.fill, inv_seq_poli_, surr_poli_);
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
        encode_fill_fn_(ob, right_fillcount_, afmt_.fill, inv_seq_poli_, surr_poli_);
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
