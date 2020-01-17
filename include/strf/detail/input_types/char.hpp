#ifndef STRF_DETAIL_INPUT_TYPES_CHAR_HPP_INCLUDED
#define STRF_DETAIL_INPUT_TYPES_CHAR_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>
#include <strf/printer.hpp>
#include <strf/facets_pack.hpp>
#include <strf/detail/facets/width_calculator.hpp>

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
        preview.add_size(1);
        _wcalc( preview
              , get_facet<strf::width_calculator_c<CharSize>, CharT>(fp)
              , get_facet<strf::encoding_c<CharT>, CharT>(fp).as_underlying() );
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    STRF_HD void _wcalc( strf::width_preview<false>&
                       , const strf::fast_width<CharSize>&
                       , strf::underlying_encoding<CharSize> ) noexcept
    {
    }
    STRF_HD void _wcalc( strf::width_preview<true>& wpreview
                       , const strf::fast_width<CharSize>&
                       , strf::underlying_encoding<CharSize> ) noexcept
    {
        wpreview.subtract_width(1);
    }
    STRF_HD void _wcalc( strf::width_preview<true>& wpreview
                       , const strf::width_as_u32len<CharSize>&
                       , strf::underlying_encoding<CharSize> ) noexcept
    {
        wpreview.subtract_width(1);
    }
    STRF_HD void _wcalc( strf::width_preview<true>& wpreview
                       , const strf::width_calculator<CharSize>& wc
                       , strf::underlying_encoding<CharSize> encoding )
    {
        wpreview.subtract_width(wc.width_of(_ch, encoding));
    }

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
        : _encoding(_get_facet<strf::encoding_c<CharT>, CharT>(fp).as_underlying())
        , _count(input.count())
        , _ch(static_cast<char_type>(input.value().ch))
        , _afmt(input.get_alignment_format_data())
        , _enc_err(_get_facet<strf::encoding_error_c, CharT>(fp))
        , _allow_surr(_get_facet<strf::surrogate_policy_c, CharT>(fp))
    {
        _init( preview
             , _get_facet<strf::width_calculator_c<CharSize>, CharT>(fp) );
        _calc_size(preview);
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    const strf::underlying_encoding<CharSize>& _encoding;
    std::size_t _count;
    char_type _ch;
    const strf::alignment_format_data _afmt;
    const strf::encoding_error  _enc_err;
    const strf::surrogate_policy  _allow_surr;
    strf::width_t _content_width = strf::width_max;
    std::int16_t _fillcount = 0;

    template <typename Category, typename CharT, typename FPack>
    static STRF_HD decltype(auto) _get_facet(const FPack& fp)
    {
        return fp.template get_facet<Category, CharT>();
    }

    template <bool RequireWidth>
    STRF_HD void _fast_init(strf::width_preview<RequireWidth>& wpreview)
    {
        if (_afmt.width > static_cast<std::ptrdiff_t>(_count)) {
            _fillcount = _afmt.width - static_cast<std::int16_t>(_count);
            wpreview.subtract_width(_afmt.width);
        } else {
            _fillcount = 0;
            wpreview.checked_subtract_width(_count);
        }
    }

    template <bool RequireWidth>
    STRF_HD void _init( strf::width_preview<RequireWidth>& wpreview
                      , const strf::fast_width<CharSize>&)
    {
        _fast_init(wpreview);
    }

    template <bool RequireWidth>
    STRF_HD void _init( strf::width_preview<RequireWidth>& wpreview
                      , const strf::width_as_u32len<CharSize>&)
    {
        _fast_init(wpreview);
    }

    template <bool RequireWidth>
    STRF_HD void _init( strf::width_preview<RequireWidth>& wpreview
              , const strf::width_calculator<CharSize>& wc)
    {
        auto ch_width = wc.wc.width_of(_ch, _encoding);
        auto content_width = checked_mul(ch_width, _count);
        if (content_width < _afmt.width) {
            _fillcount = (_afmt.width - content_width).round();
            wpreview.checked_subtract_width(content_width + _fillcount);
        } else {
            _fillcount = 0;
            wpreview.checked_subtract_width(content_width);
        }
    }

    STRF_HD void _calc_size(strf::size_preview<false>&) const
    {
    }

    STRF_HD void _calc_size(strf::size_preview<true>& spreview) const
    {
        std::size_t s = _count
                      * _encoding.char_size(_ch);
        if (_fillcount > 0) {
            s += _fillcount * _encoding.char_size(_afmt.fill);
        }
        spreview.add_size(s);
    }

    STRF_HD void _write_body(strf::underlying_outbuf<CharSize>& ob) const;

    STRF_HD void _write_fill
        ( strf::underlying_outbuf<CharSize>& ob
        , unsigned count ) const;
};

template <std::size_t CharSize>
STRF_HD void fmt_char_printer<CharSize>::print_to
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (_fillcount == 0) {
        return _write_body(ob);
    } else {
        switch(_afmt.alignment) {
            case strf::text_alignment::left: {
                _write_body(ob);
                _write_fill(ob, _fillcount);
                break;
            }
            case strf::text_alignment::center: {
                auto halfcount = _fillcount / 2;
                _write_fill(ob, halfcount);
                _write_body(ob);
                _write_fill(ob, _fillcount - halfcount);
                break;
            }
            default:
            {
                _write_fill(ob, _fillcount);
                _write_body(ob);
            }
        }
    }
}

template <std::size_t CharSize>
STRF_HD void fmt_char_printer<CharSize>::_write_body
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
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

}

template <std::size_t CharSize>
STRF_HD void fmt_char_printer<CharSize>::_write_fill
    ( strf::underlying_outbuf<CharSize>& ob
    , unsigned count ) const
{
    _encoding.encode_fill(ob, count, _afmt.fill, _enc_err, _allow_surr);
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



