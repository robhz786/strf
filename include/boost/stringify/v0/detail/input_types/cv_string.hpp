#ifndef BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_CV_STRING_HPP
#define BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_CV_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/format_functions.hpp>
#include <boost/stringify/v0/detail/facets/encoding.hpp>
#include <boost/utility/string_view.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename CharIn>
class cv_string
{
public:

    constexpr cv_string(const CharIn* str, std::size_t len) noexcept
       : _str(str, len)
    {
    }

    constexpr const CharIn* begin() const
    {
        return _str.begin();
    }
    constexpr const CharIn* end() const
    {
        return _str.end();
    }
    constexpr std::size_t length() const
    {
        return _str.size();
    }
    constexpr std::size_t size() const
    {
        return _str.size();
    }

private:

    stringify::v0::detail::simple_string_view<CharIn> _str;
};


template <typename CharIn>
class cv_string_with_encoding: public stringify::v0::detail::cv_string<CharIn>
{
public:

    cv_string_with_encoding
        ( const CharIn* str
        , std::size_t len
        , stringify::v0::encoding<CharIn> enc ) noexcept
        : stringify::v0::detail::cv_string<CharIn>(str, len)
        , _enc(enc)
    {
    }

    constexpr stringify::v0::encoding<CharIn> encoding() const
    {
        return _enc;
    }
    constexpr void set_encoding(stringify::v0::encoding<CharIn> enc)
    {
        _enc = enc;
    }

    constexpr stringify::v0::encoding<CharIn> get_encoding() const
    {
        return _enc;
    }

private:

    stringify::v0::encoding<CharIn> _enc;
};


} // namespace detail


BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string<char> cv(const char* str)
{
    return {str, std::char_traits<char>::length(str)};
}
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string<char16_t> cv(const char16_t* str)
{
    return {str, std::char_traits<char16_t>::length(str)};
}
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string<char32_t> cv(const char32_t* str)
{
    return {str, std::char_traits<char32_t>::length(str)};
}
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string<wchar_t> cv(const wchar_t* str)
{
    return {str, std::char_traits<wchar_t>::length(str)};
}

template <typename CharIn>
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string_with_encoding<CharIn> cv
    ( const CharIn* str
    , stringify::v0::encoding<CharIn> enc )
{
    return {str, std::char_traits<CharIn>::length(str), enc};
}

template <typename CharIn, typename Traits>
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string<CharIn> cv
    ( const std::basic_string<CharIn, Traits>& str )
{
    return {str.data(), str.size()};
}

template <typename CharIn, typename Traits>
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string_with_encoding<CharIn> cv
    ( const std::basic_string<CharIn, Traits>& str
    , stringify::v0::encoding<CharIn> enc )
{
    return {str.data(), str.size(), enc};
}

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

template <typename CharIn, typename Traits>
constexpr stringify::v0::detail::cv_string<CharIn> cv
    ( const std::basic_string_view<CharIn, Traits>& str )
{
    return {str.data(), str.size()};
}

template <typename CharIn, typename Traits>
constexpr stringify::v0::detail::cv_string_with_encoding<CharIn> cv
    ( const std::basic_string_view<CharIn, Traits>& str
    , stringify::v0::encoding<CharIn> enc )
{
    return { str.data(), str.size(), &enc };
}

#endif

namespace detail {

template <typename CharIn>
using cv_string_with_format = stringify::v0::value_with_format
    < stringify::v0::detail::cv_string<CharIn>
    , stringify::v0::alignment_format >;

template <typename CharIn>
using cv_string_with_format_and_encoding = stringify::v0::value_with_format
    < stringify::v0::detail::cv_string_with_encoding<CharIn>
    , stringify::v0::alignment_format >;

template <typename CharIn>
constexpr auto make_fmt
    ( stringify::v0::tag
    , const stringify::v0::detail::cv_string<CharIn>& cv_str )
{
    return stringify::v0::detail::cv_string_with_format<char>{cv_str};
}

} // namespace detail

BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string_with_format<char> fmt_cv(const char* str)
{
    stringify::v0::detail::cv_string<char> cv_str
        { str, std::char_traits<char>::length(str) };
    return stringify::v0::detail::cv_string_with_format<char>{cv_str};
}
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string_with_format<char16_t> fmt_cv(const char16_t* str)
{
    stringify::v0::detail::cv_string<char16_t> cv_str
        { str, std::char_traits<char16_t>::length(str) };
    return stringify::v0::detail::cv_string_with_format<char16_t>{cv_str};
}
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string_with_format<char32_t> fmt_cv(const char32_t* str)
{
    stringify::v0::detail::cv_string<char32_t> cv_str
        { str, std::char_traits<char32_t>::length(str) };
    return stringify::v0::detail::cv_string_with_format<char32_t>{cv_str};
}
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string_with_format<wchar_t> fmt_cv(const wchar_t* str)
{
    stringify::v0::detail::cv_string<wchar_t> cv_str
        { str, std::char_traits<wchar_t>::length(str) };
    return stringify::v0::detail::cv_string_with_format<wchar_t>{cv_str};

}

template <typename CharIn>
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string_with_format<CharIn> fmt_cv
    ( const CharIn* str
    , stringify::v0::encoding<CharIn> enc )
{
    stringify::v0::detail::cv_string<CharIn> cv_str{str, enc};
    return stringify::v0::detail::cv_string_with_format<CharIn>{cv_str};
}

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

template <typename CharIn, typename Traits>
constexpr stringify::v0::detail::cv_string_with_format<CharIn> fmt_cv
    ( const std::basic_string_view<CharIn, Traits>& str )
{
    stringify::v0::detail::cv_string<CharIn> cv_str{str.data(), str.size()};
    return stringify::v0::detail::cv_string_with_format<CharIn>{cv_str};
}

template <typename CharIn, typename Traits>
constexpr stringify::v0::detail::cv_string_with_format<CharIn> fmt_cv
    ( const std::basic_string_view<CharIn, Traits>& str
    , stringify::v0::encoding<CharIn> enc )
{
    stringify::v0::detail::cv_string<CharIn> cv_str
        { str.data(), str.size(), &enc };
    return stringify::v0::detail::cv_string_with_format<CharIn>{cv_str};
}

#endif

template <typename CharIn, typename Traits>
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string_with_format<CharIn> fmt_cv
    ( const std::basic_string<CharIn, Traits>& str )
{
    stringify::v0::detail::cv_string<CharIn> cv_str{str.data(), str.length()};
    return stringify::v0::detail::cv_string_with_format<CharIn>{cv_str};
}

template <typename CharIn, typename Traits>
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string_with_format<CharIn> fmt_cv
    ( const std::basic_string<CharIn, Traits>& str
    , stringify::v0::encoding<CharIn> enc )
{
    stringify::v0::detail::cv_string<CharIn> cv_str{str.data(), str.length(), &enc};
    return stringify::v0::detail::cv_string_with_format<CharIn>{cv_str};
}

// template <typename T>
// inline auto ascii(const T& x)
// {
//     return stringify::v0::cv(x, stringify::v0::ascii());
// }
// template <typename T>
// inline auto iso_8859_1(const T& x)
// {
//     return stringify::v0::cv(x, stringify::v0::iso_8859_1());
// }
// template <typename T>
// inline auto iso_8859_15(const T& x)
// {
//     return stringify::v0::cv(x, stringify::v0::iso_8859_15());
// }
// template <typename T>
// inline auto windows_1252(const T& x)
// {
//     return stringify::v0::cv(x, stringify::v0::windows_1252());
// }
template <typename T>
inline auto utf8(const T& x)
{
    return stringify::v0::cv(x, stringify::v0::utf8());
}
// template <typename T>
// inline auto mutf8(const T& x)
// {
//     return stringify::v0::cv(x, stringify::v0::mutf8());
// }
// template <typename T>
// inline auto utfw(const T& x)
// {
//     return stringify::v0::cv(x, stringify::v0::utfw());
// }

template <typename T>
inline auto utf16(const T& x)
{
    return stringify::v0::cv(x, stringify::v0::utf16());
}
template <typename T>
inline auto utf32(const T& x)
{
    return stringify::v0::cv(x, stringify::v0::utf32());
}

namespace detail {

template<typename CharIn, typename CharOut>
class cv_string_printer: public stringify::v0::printer<CharOut>
{
public:

    template <typename FPack>
    cv_string_printer
        ( const FPack& fp
        , const CharIn* str
        , std::size_t len
        , stringify::v0::encoding<CharIn> src_enc ) noexcept
        : cv_string_printer
            ( str
            , len
            , _get_facet<stringify::v0::width_calculator_category>(fp)
            , src_enc
            , _get_facet<stringify::v0::encoding_category<CharOut>>(fp)
            , _get_facet<stringify::v0::encoding_policy_category>(fp) )
    {
    }

    cv_string_printer
        ( const CharIn* str
        , std::size_t len
        , const stringify::v0::width_calculator& wcalc
        , stringify::v0::encoding<CharIn> src_enc
        , stringify::v0::encoding<CharOut> dest_enc
        , const stringify::v0::encoding_policy epoli ) noexcept;

    ~cv_string_printer() = default;

    std::size_t necessary_size() const override;

    bool write(stringify::v0::output_buffer<CharOut>& ob) const override;

    int remaining_width(int w) const override;

private:

    const CharIn* const _str;
    const std::size_t _len;
    const stringify::v0::width_calculator _wcalc;
    const stringify::v0::encoding<CharIn>  _src_encoding;
    const stringify::v0::encoding<CharOut> _dest_encoding;
    const stringify::v0::transcoder_impl_type<CharIn, CharOut>* _transcoder_impl;
    const stringify::v0::encoding_policy _epoli;

    template <typename Category, typename FPack>
    static decltype(auto) _get_facet(const FPack& fp)
    {
        using input_tag = stringify::v0::string_input_tag<CharIn>;
        return fp.template get_facet<Category, input_tag>();
    }
};

template<typename CharIn, typename CharOut>
cv_string_printer<CharIn, CharOut>::cv_string_printer
    ( const CharIn* str
    , std::size_t len
    , const stringify::v0::width_calculator& wcalc
    , stringify::v0::encoding<CharIn> src_enc
    , stringify::v0::encoding<CharOut> dest_enc
    , const stringify::v0::encoding_policy epoli ) noexcept
    : _str(str)
    , _len(len)
    , _wcalc(wcalc)
    , _src_encoding(src_enc)
    , _dest_encoding(dest_enc)
    , _transcoder_impl(stringify::v0::get_transcoder_impl(src_enc, dest_enc))
    , _epoli(epoli)
{
}

template<typename CharIn, typename CharOut>
std::size_t cv_string_printer<CharIn, CharOut>::necessary_size() const
{
    if (_transcoder_impl)
    {
        stringify::v0::transcoder<CharIn, CharOut> transcoder(*_transcoder_impl);
        return transcoder.necessary_size
            ( _str, _str + _len, _epoli.err_hdl(), _epoli.allow_surr() );
    }
    return stringify::v0::detail::decode_encode_size
        ( _str, _str + _len
        , _src_encoding, _dest_encoding
        , _epoli.err_hdl(), _epoli.allow_surr() );
}

template<typename CharIn, typename CharOut>
bool cv_string_printer<CharIn, CharOut>::write
    ( stringify::v0::output_buffer<CharOut>& ob ) const
{
    if (_transcoder_impl)
    {
        stringify::v0::transcoder<CharIn, CharOut> transcoder(*_transcoder_impl);
        return transcoder.transcode( ob, _str, _str + _len
                                   , _epoli.err_hdl(), _epoli.allow_surr() );
    }
    return stringify::v0::detail::decode_encode( ob
                                               , _str, _str + _len
                                               , _src_encoding, _dest_encoding
                                               , _epoli.err_hdl()
                                               , _epoli.allow_surr() );
}

template<typename CharIn, typename CharOut>
int cv_string_printer<CharIn, CharOut>::remaining_width(int w) const
{
    return _wcalc.remaining_width(w, _str, _len, _src_encoding, _epoli);
}

template<typename CharIn, typename CharOut>
class fmt_cv_string_printer: public printer<CharOut>
{
public:

    template <typename FPack>
    fmt_cv_string_printer
        ( const FPack& fp
        , const stringify::v0::detail::cv_string_with_format<CharIn>& input
        , const stringify::v0::encoding<CharIn>& src_enc ) noexcept
        : _fmt(input)
        , _src_encoding(src_enc)
        , _dest_encoding(_get_facet<stringify::v0::encoding_category<CharOut>>(fp))
        , _wcalc(_get_facet<stringify::v0::width_calculator_category>(fp))
        , _epoli(_get_facet<stringify::v0::encoding_policy_category>(fp))
    {
        _init();
    }

    std::size_t necessary_size() const override;

    bool write(stringify::v0::output_buffer<CharOut>& ob) const override;

    int remaining_width(int w) const override;

private:

    stringify::v0::detail::cv_string_with_format<CharIn> _fmt;
    const stringify::v0::transcoder_impl_type<CharIn, CharOut>* _transcoder_impl;
    const stringify::v0::encoding<CharIn> _src_encoding;
    const stringify::v0::encoding<CharOut> _dest_encoding;
    const stringify::v0::width_calculator _wcalc;
    const stringify::v0::encoding_policy  _epoli;
    int _fillcount = 0;

    template <typename Category, typename FPack>
    static decltype(auto) _get_facet(const FPack& fp)
    {
        using input_tag = stringify::v0::string_input_tag<CharIn>;
        return fp.template get_facet<Category, input_tag>();
    }

    void _init();

    bool _write_str(stringify::v0::output_buffer<CharOut>& ob) const;

    bool _write_fill
        ( stringify::v0::output_buffer<CharOut>& ob
        , unsigned count ) const;
};

template<typename CharIn, typename CharOut>
void fmt_cv_string_printer<CharIn, CharOut>::_init()
{
    _fillcount = ( _fmt.width() > 0
                 ? _wcalc.remaining_width
                     ( _fmt.width()
                     , _fmt.value().begin()
                     , _fmt.value().length()
                     , _src_encoding
                     , _epoli )
                 : 0 );
    _transcoder_impl = stringify::v0::get_transcoder_impl(_src_encoding, _dest_encoding);
}

template<typename CharIn, typename CharOut>
std::size_t fmt_cv_string_printer<CharIn, CharOut>::necessary_size() const
{
    std::size_t size;
    if(_transcoder_impl)
    {
        stringify::v0::transcoder<CharIn, CharOut> transcoder(*_transcoder_impl);
        size = transcoder.necessary_size( _fmt.value().begin(), _fmt.value().end()
                                        , _epoli.err_hdl(), _epoli.allow_surr() );
    }
    else
    {
        size = stringify::v0::detail::decode_encode_size
            ( _fmt.value().begin(), _fmt.value().end()
            , _src_encoding, _dest_encoding
            , _epoli.err_hdl(), _epoli.allow_surr() );
    }

    if (_fillcount > 0)
    {
        size += _fillcount * _dest_encoding.char_size( _fmt.fill()
                                                     , _epoli.err_hdl() );
    }

    return size;
}


template<typename CharIn, typename CharOut>
bool fmt_cv_string_printer<CharIn, CharOut>::write
    ( stringify::v0::output_buffer<CharOut>& ob ) const
{
    if (_fillcount > 0)
    {
        switch(_fmt.alignment())
        {
            case stringify::v0::alignment::left:
            {
                return _write_str(ob)
                    && _write_fill(ob, _fillcount);
            }
            case stringify::v0::alignment::center:
            {
                int halfcount = _fillcount / 2;
                return _write_fill(ob, halfcount)
                    && _write_str(ob)
                    && _write_fill(ob, _fillcount - halfcount);
            }
            default:
            {
                return _write_fill(ob, _fillcount)
                    && _write_str(ob);
            }
        }
    }
    return _write_str(ob);
}


template<typename CharIn, typename CharOut>
bool fmt_cv_string_printer<CharIn, CharOut>::_write_str
    ( stringify::v0::output_buffer<CharOut>& ob ) const
{
    if (_transcoder_impl)
    {
        stringify::v0::transcoder<CharIn, CharOut> transcoder(*_transcoder_impl);
        return transcoder.transcode( ob
                                   , _fmt.value().begin()
                                   , _fmt.value().end()
                                   , _epoli.err_hdl()
                                   , _epoli.allow_surr() );
    }
    return stringify::v0::detail::decode_encode( ob
                                               , _fmt.value().begin()
                                               , _fmt.value().end()
                                               , _src_encoding
                                               , _dest_encoding
                                               , _epoli.err_hdl()
                                               , _epoli.allow_surr() );
}

template<typename CharIn, typename CharOut>
bool fmt_cv_string_printer<CharIn, CharOut>::_write_fill
    ( stringify::v0::output_buffer<CharOut>& ob
    , unsigned count ) const
{
    return _dest_encoding.encode_fill
        ( ob, count, _fmt.fill(), _epoli.err_hdl(), _epoli.allow_surr() );
}

template<typename CharIn, typename CharOut>
int fmt_cv_string_printer<CharIn, CharOut>::remaining_width(int w) const
{
    if (_fillcount > 0)
    {
        return w > _fmt.width() ? w - _fmt.width() : 0;
    }
    return _wcalc.remaining_width
        ( w
        , _fmt.value().begin()
        , _fmt.value().length()
        , _src_encoding
        , _epoli );
}


#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char16_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char16_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char16_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char16_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char32_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char32_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char32_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char32_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<wchar_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<wchar_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<wchar_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<wchar_t, wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char16_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char16_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char16_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char16_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char32_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char32_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char32_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char32_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<wchar_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<wchar_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<wchar_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<wchar_t, wchar_t>;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

template <typename CharOut, typename FPack, typename CharIn>
inline stringify::v0::detail::cv_string_printer<CharIn, CharOut>
make_printer( const FPack& fp
            , stringify::v0::detail::cv_string<CharIn> str )
{
    using enc_cat = stringify::v0::encoding_category<CharIn>;
    using input_tag = stringify::v0::string_input_tag<CharIn>;
    return { fp
           , str.begin()
           , str.size()
           , stringify::v0::get_facet<enc_cat, input_tag>(fp) };
}

template <typename CharOut, typename FPack, typename CharIn>
inline stringify::v0::detail::cv_string_printer<CharIn, CharOut>
make_printer( const FPack& fp
            , stringify::v0::detail::cv_string_with_encoding<CharIn> str )
{
    return {fp, str.begin(), str.size(), str.get_encoding()};
}

template <typename CharOut, typename FPack, typename CharIn>
inline stringify::v0::detail::fmt_cv_string_printer<CharIn, CharOut>
make_printer
    ( const FPack& fp
    , stringify::v0::detail::cv_string_with_format<CharIn> str )
{
    using enc_cat = stringify::v0::encoding_category<CharIn>;
    using input_tag = stringify::v0::string_input_tag<CharIn>;
    return {fp, str, stringify::v0::get_facet<enc_cat, input_tag>(fp) };
}

template <typename CharOut, typename FPack, typename CharIn>
inline stringify::v0::detail::fmt_cv_string_printer<CharIn, CharOut>
make_printer
    ( const FPack& fp
    , stringify::v0::detail::cv_string_with_format_and_encoding<CharIn> str )
{
    return {fp, str, str.get_encoding()};
}

} // namespace detail

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_CV_STRING_HPP

