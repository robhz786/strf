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

namespace detail {

template <std::size_t CharSize>
class char_printer: public strf::printer<CharSize>
{
public:
    using char_type = strf::underlying_outbuf_char_type<CharSize>;

    template <typename FPack, typename Preview, typename CharT>
    STRF_HD char_printer(const FPack& fp, Preview& preview, CharT ch)
        : _ch(static_cast<char_type>(ch))
    {
        static_assert(sizeof(CharT) == CharSize, "");
        preview.add_size(1);
        STRF_IF_CONSTEXPR(Preview::width_required) {
            decltype(auto) wcalc = get_facet<strf::width_calculator_c, CharT>(fp);
            auto w = wcalc.width( get_facet<strf::encoding_c<CharT>, CharT>(fp)
                                , static_cast<char_type>(ch) );
            preview.subtract_width(w);
        }
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    char_type _ch;
};

template <std::size_t CharSize>
STRF_HD void char_printer<CharSize>::print_to
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    ob.ensure(1);
    *ob.pos() = _ch;
    ob.advance();
}


template <std::size_t CharSize>
class fmt_char_printer: public strf::printer<CharSize>
{
public:
    using char_type = strf::underlying_outbuf_char_type<CharSize>;

    template <typename FPack, typename Preview, typename CharT>
    fmt_char_printer
        ( const FPack& fp
        , Preview& preview
        , const strf::char_with_format<CharT>& input ) noexcept
        : _count(input.count())
        , _afmt(input.get_alignment_format_data())
        , _enc_err(_get_facet<strf::encoding_error_c, CharT>(fp))
        , _allow_surr(_get_facet<strf::surrogate_policy_c, CharT>(fp))
        , _ch(static_cast<char_type>(input.value().ch))
    {
        decltype(auto) enc = _get_facet<strf::encoding_c<CharT>, CharT>(fp);
        _encode_fill_fn = enc.encode_fill;
        _init( preview
             , _get_facet<strf::width_calculator_c, CharT>(fp)
             , enc );
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    strf::encode_fill_func<CharSize> _encode_fill_fn;
    std::size_t _count;
    const strf::alignment_format_data _afmt;
    const strf::encoding_error  _enc_err;
    const strf::surrogate_policy  _allow_surr;
    std::int16_t _left_fillcount;
    std::int16_t _right_fillcount;
    char_type _ch;

    template <typename Category, typename CharT, typename FPack>
    static STRF_HD decltype(auto) _get_facet(const FPack& fp)
    {
        return fp.template get_facet<Category, CharT>();
    }

    template <typename Preview, typename WCalc, typename Encoding>
    STRF_HD void _init(Preview& preview, const WCalc& wc, const Encoding& enc);
};

template <std::size_t CharSize>
template <typename Preview, typename WCalc, typename Encoding>
STRF_HD void fmt_char_printer<CharSize>::_init
    ( Preview& preview, const WCalc& wc, const Encoding& enc)
{
    auto ch_width = wc.width(enc, _ch);
    auto content_width = checked_mul(ch_width, _count);
    unsigned fillcount = 0;
    if (content_width < _afmt.width) {
        fillcount = (_afmt.width - content_width).round();
        preview.checked_subtract_width(content_width + fillcount);
    } else {
        fillcount = 0;
        preview.checked_subtract_width(content_width);
    }
    switch(_afmt.alignment) {
        case strf::text_alignment::left:
            _left_fillcount = 0;
            _right_fillcount = fillcount;
            break;
        case strf::text_alignment::center: {
            auto halfcount = fillcount / 2;
            _left_fillcount = halfcount;
            _right_fillcount = fillcount - halfcount;
            break;
        }
        default:
            _left_fillcount = fillcount;
            _right_fillcount = 0;
    }
    STRF_IF_CONSTEXPR (Preview::size_required) {
        if (fillcount > 0) {
            preview.add_size(_count + fillcount * enc.encoded_char_size(_afmt.fill));
        } else {
            preview.add_size(_count);
        }
    }
}


template <std::size_t CharSize>
STRF_HD void fmt_char_printer<CharSize>::print_to
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (_left_fillcount != 0) {
        _encode_fill_fn(ob, _left_fillcount, _afmt.fill, _enc_err, _allow_surr);
    }
    if (_count == 1) {
        ob.ensure(1);
        * ob.pos() = _ch;
        ob.advance();
    } else {
        std::size_t count = _count;
        while(true) {
            std::size_t space = ob.size();
            if (count <= space) {
                strf::detail::str_fill_n(ob.pos(), count, _ch);
                ob.advance(count);
                break;
            }
            strf::detail::str_fill_n(ob.pos(), space, _ch);
            count -= space;
            ob.advance_to(ob.end());
            ob.recycle();
        }
    }
    if (_right_fillcount != 0) {
        _encode_fill_fn(ob, _right_fillcount, _afmt.fill, _enc_err, _allow_surr);
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

template <typename CharOut, typename FPack, typename Preview>
inline strf::detail::char_printer<sizeof(CharOut)>
STRF_HD make_printer(strf::rank<1>, const FPack& fp, Preview& preview, CharOut ch)
{
    return {fp, preview, ch};
}

#if defined(__cpp_char8_t)

template <typename CharOut, typename FPack, typename Preview>
inline strf::detail::char_printer<sizeof(CharOut)>
STRF_HD make_printer(strf::rank<1>, const FPack& fp, Preview& preview, char8_t ch)
{
    static_assert( std::is_same<CharOut, char8_t>::value
                 , "Character type mismatch." );
    return {fp, preview, ch};
}

#endif

template < typename CharOut
         , typename FPack
         , typename Preview
         , typename CharIn
         , std::enable_if_t<std::is_same<CharIn, CharOut>::value, int> = 0>
inline strf::detail::char_printer<sizeof(CharOut)>
STRF_HD make_printer(strf::rank<1>, const FPack& fp, Preview& preview, CharIn ch)
{
    return {fp, preview, ch};
}

template <typename CharOut, typename FPack, typename Preview>
inline strf::detail::char_printer<sizeof(CharOut)>
STRF_HD make_printer(strf::rank<1>, const FPack& fp, Preview& preview, char ch)
{
    static_assert( std::is_same<CharOut, char>::value
                 , "Character type mismatch." );
    return {fp, preview, ch};
}

template <typename CharOut, typename FPack, typename Preview>
inline strf::detail::char_printer<sizeof(CharOut)>
STRF_HD make_printer(strf::rank<1>, const FPack& fp, Preview& preview, wchar_t ch)
{
    static_assert( std::is_same<CharOut, wchar_t>::value
                 , "Character type mismatch." );
    return {fp, preview, ch};
}

template <typename CharOut, typename FPack, typename Preview>
inline strf::detail::char_printer<sizeof(CharOut)>
STRF_HD make_printer(strf::rank<1>, const FPack& fp, Preview& preview, char16_t ch)
{
    static_assert( std::is_same<CharOut, char16_t>::value
                 , "Character type mismatch." );
    return {fp, preview, ch};
}

template <typename CharOut, typename FPack, typename Preview>
inline strf::detail::char_printer<sizeof(CharOut)>
STRF_HD make_printer(strf::rank<1>, const FPack& fp, Preview& preview, char32_t ch )
{
    static_assert( std::is_same<CharOut, char32_t>::value
                 , "Character type mismatch." );
    return {fp, preview, ch};
}

template <typename CharOut, typename FPack, typename Preview, typename CharIn>
inline strf::detail::fmt_char_printer<sizeof(CharOut)>
STRF_HD make_printer(strf::rank<1>, const FPack& fp, Preview& preview, char_with_format<CharIn> ch)
{
    static_assert( std::is_same<CharOut, CharIn>::value
                 , "Character type mismatch." );
    return {fp, preview, ch};
}

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



