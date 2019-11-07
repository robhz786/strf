#ifndef BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_CHAR_HPP_INCLUDED
#define BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_CHAR_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>
#include <boost/stringify/v0/printer.hpp>
#include <boost/stringify/v0/facets_pack.hpp>
#include <boost/stringify/v0/detail/facets/width_calculator.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT>
struct char_tag
{
    CharT ch;
};

template <typename CharT>
using char_with_format = stringify::v0::value_with_format
    < char_tag<CharT>
    , stringify::v0::quantity_format
    , stringify::v0::alignment_format >;

namespace detail {

template <typename CharT>
class char_printer: public printer<CharT>
{
public:

    template <typename FPack, typename Preview>
    char_printer (const FPack& fp, Preview& preview, CharT ch)
        : _ch(ch)
    {
        preview.add_size(1);
        _wcalc( preview
              , get_facet<stringify::v0::width_calculator_c<CharT>, CharT>(fp)
              , get_facet<stringify::v0::encoding_c<CharT>, CharT>(fp) );
    }

    void print_to(stringify::v0::basic_outbuf<CharT>& ob) const override;

private:

    void _wcalc( stringify::v0::width_preview<false>&
               , const stringify::v0::width_as_len<CharT>&
               , stringify::v0::encoding<CharT> ) noexcept
    {
    }
    void _wcalc( stringify::v0::width_preview<true>& wpreview
               , const stringify::v0::width_as_len<CharT>&
               , stringify::v0::encoding<CharT> ) noexcept
    {
        wpreview.subtract_width(1);
    }
    void _wcalc( stringify::v0::width_preview<true>& wpreview
               , const stringify::v0::width_as_u32len<CharT>&
               , stringify::v0::encoding<CharT> ) noexcept
    {
        wpreview.subtract_width(1);
    }
    void _wcalc( stringify::v0::width_preview<true>& wpreview
               , const stringify::v0::width_calculator<CharT>& wc
               , stringify::v0::encoding<CharT> encoding )
    {
        wpreview.subtract_width(wc.width_of(_ch, encoding));
    }

    CharT _ch;
};

template <typename CharT>
void char_printer<CharT>::print_to
    ( stringify::v0::basic_outbuf<CharT>& ob ) const
{
    ob.ensure(1);
    *ob.pos() = _ch;
    ob.advance();
}



template <typename CharT>
class fmt_char_printer: public printer<CharT>
{
    using input_type = CharT;

public:

    template <typename FPack, typename Preview>
    fmt_char_printer
        ( const FPack& fp
        , Preview& preview
        , const stringify::v0::char_with_format<CharT>& input ) noexcept
        : _encoding(_get_facet<stringify::v0::encoding_c<CharT>>(fp))
        , _enc_err(_get_facet<stringify::v0::encoding_error_c>(fp))
        , _fmt(input)
        , _allow_surr(_get_facet<stringify::v0::surrogate_policy_c>(fp))
    {
        _init( preview
             , _get_facet<stringify::v0::width_calculator_c<CharT>>(fp) );
        _calc_size(preview);
    }

    void print_to(stringify::v0::basic_outbuf<CharT>& ob) const override;

    stringify::v0::width_t width(stringify::v0::width_t) const;

private:

    const stringify::v0::encoding<CharT> _encoding;
    const stringify::v0::encoding_error  _enc_err;
    const stringify::v0::char_with_format<CharT> _fmt;
    const stringify::v0::surrogate_policy  _allow_surr;
    stringify::v0::width_t _content_width = stringify::v0::width_t_max;
    std::int16_t _fillcount = 0;

    template <typename Category, typename FPack>
    static decltype(auto) _get_facet(const FPack& fp)
    {
        return fp.template get_facet<Category, input_type>();
    }

    template <bool RequireWidth>
    void _fast_init(stringify::v0::width_preview<RequireWidth>& wpreview)
    {
        if (_fmt.width() > static_cast<std::ptrdiff_t>(_fmt.count()))
        {
            _fillcount = _fmt.width() - static_cast<std::int16_t>(_fmt.count());
            wpreview.subtract_width(_fmt.width());
        }
        else
        {
            _fillcount = 0;
            wpreview.checked_subtract_width(_fmt.count());
        }
    }

    template <bool RequireWidth>
    void _init( stringify::v0::width_preview<RequireWidth>& wpreview
              , const stringify::v0::width_as_len<CharT>&)
    {
        _fast_init(wpreview);
    }

    template <bool RequireWidth>
    void _init( stringify::v0::width_preview<RequireWidth>& wpreview
              , const stringify::v0::width_as_u32len<CharT>&)
    {
        _fast_init(wpreview);
    }

    template <bool RequireWidth>
    void _init( stringify::v0::width_preview<RequireWidth>& wpreview
              , const stringify::v0::width_calculator<CharT>& wc)
    {
        auto ch_width = wc.wc.width_of(_fmt.value().ch, _encoding);
        auto content_width = checked_mul(ch_width, _fmt.count());
        if (content_width < _fmt.width())
        {
            _fillcount = (_fmt.width() - content_width).round();
            wpreview.checked_subtract_width(content_width + _fillcount);
        }
        else
        {
            _fillcount = 0;
            wpreview.checked_subtract_width(content_width);
        }
    }

    void _calc_size(stringify::v0::size_preview<false>&) const
    {
    }

    void _calc_size(stringify::v0::size_preview<true>& spreview) const
    {
        std::size_t s = _fmt.count()
                      * _encoding.char_size(_fmt.value().ch, _enc_err);
        if (_fillcount > 0)
        {
            s += _fillcount * _encoding.char_size(_fmt.fill(), _enc_err);
        }
        spreview.add_size(s);
    }

    void _write_body(stringify::v0::basic_outbuf<CharT>& ob) const;

    void _write_fill
        ( stringify::v0::basic_outbuf<CharT>& ob
        , unsigned count ) const;
};

template <typename CharT>
void fmt_char_printer<CharT>::print_to
    ( stringify::v0::basic_outbuf<CharT>& ob ) const
{
    if (_fillcount == 0)
    {
        return _write_body(ob);
    }
    else
    {
        switch(_fmt.alignment())
        {
            case stringify::v0::text_alignment::left:
            {
                _write_body(ob);
                _write_fill(ob, _fillcount);
                break;
            }
            case stringify::v0::text_alignment::center:
            {
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

template <typename CharT>
void fmt_char_printer<CharT>::_write_body
    ( stringify::v0::basic_outbuf<CharT>& ob ) const
{
    if (_fmt.count() == 1)
    {
        ob.ensure(1);
        * ob.pos() = _fmt.value().ch;
        ob.advance();
    }
    else
    {
        std::size_t count = _fmt.count();
        while(true)
        {
            std::size_t space = ob.size();
            if (count <= space)
            {
                std::fill_n(ob.pos(), count, _fmt.value().ch);
                ob.advance(count);
                break;
            }
            std::fill_n(ob.pos(), space, _fmt.value().ch);
            count -= space;
            ob.advance_to(ob.end());
            ob.recycle();
        }
    }

}

template <typename CharT>
void fmt_char_printer<CharT>::_write_fill
    ( stringify::v0::basic_outbuf<CharT>& ob
    , unsigned count ) const
{
    _encoding.encode_fill(ob, count, _fmt.fill(), _enc_err, _allow_surr);
}

#if defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_printer<char8_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_char_printer<char8_t>;
#endif
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_printer<wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_char_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_char_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_char_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_char_printer<wchar_t>;

#endif // defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

// template <typename A, typename B>
// struct enable_if_same_impl
// {
// };

// template <typename A>
// struct enable_if_same_impl<A, A>
// {
//     using type = void;
// };

// template <typename A, typename B>
// using enable_if_same = typename stringify::v0::detail::enable_if_same<A, B>::type;

} // namespace detail

template <typename CharOut, typename FPack, typename Preview>
inline stringify::v0::detail::char_printer<CharOut>
make_printer(const FPack& fp, Preview& preview, CharOut ch)
{
    return {fp, preview, ch};
}

#if defined(__cpp_char8_t)

template <typename CharOut, typename FPack, typename Preview>
inline stringify::v0::detail::char_printer<CharOut>
make_printer(const FPack& fp, Preview& preview, char8_t ch)
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
inline stringify::v0::detail::char_printer<CharOut>
make_printer(const FPack& fp, Preview& preview, CharIn ch)
{
    return {fp, preview, ch};
}

template <typename CharOut, typename FPack, typename Preview>
inline stringify::v0::detail::char_printer<CharOut>
make_printer(const FPack& fp, Preview& preview, char ch)
{
    static_assert( std::is_same<CharOut, char>::value
                 , "Character type mismatch." );
    return {fp, preview, ch};
}

template <typename CharOut, typename FPack, typename Preview>
inline stringify::v0::detail::char_printer<CharOut>
make_printer(const FPack& fp, Preview& preview, wchar_t ch)
{
    static_assert( std::is_same<CharOut, wchar_t>::value
                 , "Character type mismatch." );
    return {fp, preview, ch};
}

template <typename CharOut, typename FPack, typename Preview>
inline stringify::v0::detail::char_printer<CharOut>
make_printer(const FPack& fp, Preview& preview, char16_t ch)
{
    static_assert( std::is_same<CharOut, char16_t>::value
                 , "Character type mismatch." );
    return {fp, preview, ch};
}

template <typename CharOut, typename FPack, typename Preview>
inline stringify::v0::detail::char_printer<CharOut>
make_printer(const FPack& fp, Preview& preview, char32_t ch )
{
    static_assert( std::is_same<CharOut, char32_t>::value
                 , "Character type mismatch." );
    return {fp, preview, ch};
}

template <typename CharOut, typename FPack, typename Preview, typename CharIn>
inline stringify::v0::detail::fmt_char_printer<CharOut>
make_printer(const FPack& fp, Preview& preview, char_with_format<CharIn> ch)
{
    static_assert( std::is_same<CharOut, CharIn>::value
                 , "Character type mismatch." );
    return {fp, preview, ch};
}

#if defined(__cpp_char8_t)

constexpr auto make_fmt(stringify::v0::tag, char8_t ch) noexcept
{
    return stringify::v0::char_with_format<char8_t>{{ch}};
}

#endif

constexpr auto make_fmt(stringify::v0::tag, char ch) noexcept
{
    return stringify::v0::char_with_format<char>{{ch}};
}
constexpr auto make_fmt(stringify::v0::tag, wchar_t ch) noexcept
{
    return stringify::v0::char_with_format<wchar_t>{{ch}};
}
constexpr auto make_fmt(stringify::v0::tag, char16_t ch) noexcept
{
    return stringify::v0::char_with_format<char16_t>{{ch}};
}
constexpr auto make_fmt(stringify::v0::tag, char32_t ch) noexcept
{
    return stringify::v0::char_with_format<char32_t>{{ch}};
}

template <typename> struct is_char: public std::false_type {};

#if defined(__cpp_char8_t)
template <> struct is_char<char8_t>: public std::true_type {};
#endif
template <> struct is_char<char>: public std::true_type {};
template <> struct is_char<char16_t>: public std::true_type {};
template <> struct is_char<char32_t>: public std::true_type {};
template <> struct is_char<wchar_t>: public std::true_type {};

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif // BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_CHAR_HPP_INCLUDED



