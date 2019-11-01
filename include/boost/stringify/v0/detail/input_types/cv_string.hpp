#ifndef BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_CV_STRING_HPP
#define BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_CV_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/format_functions.hpp>
#include <boost/stringify/v0/detail/facets/encoding.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharIn>
class cv_string
{
public:

    constexpr cv_string(const CharIn* str, std::size_t len) noexcept
       : _str(str, len)
    {
    }
    constexpr cv_string(const cv_string&) noexcept = default;

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
class cv_string_with_encoding: public stringify::v0::cv_string<CharIn>
{
public:

    cv_string_with_encoding
        ( const CharIn* str
        , std::size_t len
        , stringify::v0::encoding<CharIn> enc ) noexcept
        : stringify::v0::cv_string<CharIn>(str, len)
        , _enc(enc)
    {
    }

    cv_string_with_encoding(const cv_string_with_encoding&) noexcept = default;

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

#if defined(__cpp_char8_t)

BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::cv_string<char8_t> cv(const char8_t* str)
{
    return {str, std::char_traits<char8_t>::length(str)};
}

#endif

BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::cv_string<char> cv(const char* str)
{
    return {str, std::char_traits<char>::length(str)};
}
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::cv_string<char16_t> cv(const char16_t* str)
{
    return {str, std::char_traits<char16_t>::length(str)};
}
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::cv_string<char32_t> cv(const char32_t* str)
{
    return {str, std::char_traits<char32_t>::length(str)};
}
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::cv_string<wchar_t> cv(const wchar_t* str)
{
    return {str, std::char_traits<wchar_t>::length(str)};
}

template <typename CharIn>
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::cv_string_with_encoding<CharIn> cv
    ( const CharIn* str
    , stringify::v0::encoding<CharIn> enc )
{
    return {str, std::char_traits<CharIn>::length(str), enc};
}

template <typename CharIn, typename Traits, typename Allocator>
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::cv_string<CharIn> cv
    ( const std::basic_string<CharIn, Traits, Allocator>& str )
{
    return {str.data(), str.size()};
}

template <typename CharIn, typename Traits, typename Allocator>
stringify::v0::cv_string_with_encoding<CharIn> cv
    ( const std::basic_string<CharIn, Traits, Allocator>& str
    , stringify::v0::encoding<CharIn> enc )
{
    return {str.data(), str.size(), enc};
}

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

template <typename CharIn, typename Traits>
constexpr stringify::v0::cv_string<CharIn> cv
    ( std::basic_string_view<CharIn, Traits> str )
{
    return {str.data(), str.size()};
}

template <typename CharIn, typename Traits>
constexpr stringify::v0::cv_string_with_encoding<CharIn> cv
    ( std::basic_string_view<CharIn, Traits> str
    , stringify::v0::encoding<CharIn> enc )
{
    return { str.data(), str.size(), &enc };
}

#endif

template <typename CharIn>
using cv_string_with_format = stringify::v0::value_with_format
    < stringify::v0::cv_string<CharIn>
    , stringify::v0::alignment_format >;

template <typename CharIn>
using cv_string_with_format_and_encoding = stringify::v0::value_with_format
    < stringify::v0::cv_string_with_encoding<CharIn>
    , stringify::v0::alignment_format >;

template <typename CharIn>
constexpr auto make_fmt(stringify::v0::tag, stringify::v0::cv_string<CharIn>& cv_str) noexcept
{
    return stringify::v0::cv_string_with_format<char>{cv_str};
}

#if defined(__cpp_char8_t)

BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::cv_string_with_format<char8_t> fmt_cv(const char8_t* str) noexcept
{
    stringify::v0::cv_string<char8_t> cv_str
        { str, std::char_traits<char8_t>::length(str) };
    return stringify::v0::cv_string_with_format<char8_t>{cv_str};
}

#endif

BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::cv_string_with_format<char> fmt_cv(const char* str) noexcept
{
    stringify::v0::cv_string<char> cv_str
        { str, std::char_traits<char>::length(str) };
    return stringify::v0::cv_string_with_format<char>{cv_str};
}
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::cv_string_with_format<char16_t> fmt_cv(const char16_t* str) noexcept
{
    stringify::v0::cv_string<char16_t> cv_str
        { str, std::char_traits<char16_t>::length(str) };
    return stringify::v0::cv_string_with_format<char16_t>{cv_str};
}
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::cv_string_with_format<char32_t> fmt_cv(const char32_t* str) noexcept
{
    stringify::v0::cv_string<char32_t> cv_str
        { str, std::char_traits<char32_t>::length(str) };
    return stringify::v0::cv_string_with_format<char32_t>{cv_str};
}
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::cv_string_with_format<wchar_t> fmt_cv(const wchar_t* str) noexcept
{
    stringify::v0::cv_string<wchar_t> cv_str
        { str, std::char_traits<wchar_t>::length(str) };
    return stringify::v0::cv_string_with_format<wchar_t>{cv_str};

}

template <typename CharIn>
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::cv_string_with_format_and_encoding<CharIn> fmt_cv
    ( const CharIn* str
    , stringify::v0::encoding<CharIn> enc ) noexcept
{
    stringify::v0::cv_string_with_encoding<CharIn> cv_str{str, enc};
    return stringify::v0::cv_string_with_format_and_encoding<CharIn>{cv_str};
}

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

template <typename CharIn, typename Traits>
constexpr stringify::v0::cv_string_with_format<CharIn> fmt_cv
    ( std::basic_string_view<CharIn, Traits> str ) noexcept
{
    stringify::v0::cv_string<CharIn> cv_str{str.data(), str.size()};
    return stringify::v0::cv_string_with_format<CharIn>{cv_str};
}

template <typename CharIn, typename Traits>
constexpr stringify::v0::cv_string_with_format_and_encoding<CharIn> fmt_cv
    ( std::basic_string_view<CharIn, Traits> str
    , stringify::v0::encoding<CharIn> enc ) noexcept
{
    stringify::v0::cv_string_and_encoding<CharIn> cv_str
        { str.data(), str.size(), &enc };
    return stringify::v0::cv_string_with_format_and_encoding<CharIn>{cv_str};
}

#endif

template <typename CharIn, typename Traits, typename Allocator>
stringify::v0::cv_string_with_format<CharIn> fmt_cv
    ( const std::basic_string<CharIn, Traits, Allocator>& str )
{
    stringify::v0::cv_string<CharIn> cv_str{str.data(), str.length()};
    return stringify::v0::cv_string_with_format<CharIn>{cv_str};
}

template <typename CharIn, typename Traits, typename Allocator>
stringify::v0::cv_string_with_format_and_encoding<CharIn> fmt_cv
    ( const std::basic_string<CharIn, Traits, Allocator>& str
    , stringify::v0::encoding<CharIn> enc ) noexcept
{
    stringify::v0::cv_string_with_encoding<CharIn> cv_str_e
        {str.data(), str.length(), &enc};
    return stringify::v0::cv_string_with_format_and_encoding<CharIn>{cv_str_e};
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
            , src_enc
            , _get_facet<stringify::v0::encoding_c<CharOut>>(fp)
            , _get_facet<stringify::v0::encoding_error_c>(fp)
            , _get_facet<stringify::v0::surrogate_policy_c>(fp) )
    {
        _init_wcalc(_get_facet<stringify::v0::width_calculator_c<CharIn>>(fp));
    }

    cv_string_printer
        ( const CharIn* str
        , std::size_t len
        , stringify::v0::encoding<CharIn> src_enc
        , stringify::v0::encoding<CharOut> dest_enc
        , stringify::v0::encoding_error enc_err
        , stringify::v0::surrogate_policy allow_surr ) noexcept;

    ~cv_string_printer() = default;

    std::size_t necessary_size() const override;

    void print_to(stringify::v0::basic_outbuf<CharOut>& ob) const override;

    stringify::v0::width_t width(stringify::v0::width_t limit) const override;

private:

    void _init_wcalc(const stringify::v0::width_as_len<CharIn>&)
    {
        _wcalc = nullptr;
    }
    void _init_wcalc(const stringify::v0::width_calculator<CharIn>& wc)
    {
        _wcalc = &wc;
    }

    const CharIn* const _str;
    const std::size_t _len;
    const stringify::v0::width_calculator<CharIn>* _wcalc;
    const stringify::v0::encoding<CharIn>  _src_encoding;
    const stringify::v0::encoding<CharOut> _dest_encoding;
    const stringify::v0::transcoder_engine<CharIn, CharOut>* _transcoder_eng;
    const stringify::v0::encoding_error _enc_err;
    const stringify::v0::surrogate_policy _allow_surr;

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
    , stringify::v0::encoding<CharIn> src_enc
    , stringify::v0::encoding<CharOut> dest_enc
    , stringify::v0::encoding_error enc_err
    , stringify::v0::surrogate_policy allow_surr ) noexcept
    : _str(str)
    , _len(len)
    , _src_encoding(src_enc)
    , _dest_encoding(dest_enc)
    , _transcoder_eng(stringify::v0::get_transcoder(src_enc, dest_enc))
    , _enc_err(enc_err)
    , _allow_surr(allow_surr)
{
}

template<typename CharIn, typename CharOut>
std::size_t cv_string_printer<CharIn, CharOut>::necessary_size() const
{
    if (_transcoder_eng)
    {
        stringify::v0::transcoder<CharIn, CharOut> transcoder(*_transcoder_eng);
        return transcoder.necessary_size
            ( _str, _str + _len, _enc_err, _allow_surr );
    }
    return stringify::v0::decode_encode_size( _str, _str + _len
                                            , _src_encoding, _dest_encoding
                                            , _enc_err, _allow_surr );
}

template<typename CharIn, typename CharOut>
void cv_string_printer<CharIn, CharOut>::print_to
    ( stringify::v0::basic_outbuf<CharOut>& ob ) const
{
    if (_transcoder_eng != nullptr)
    {
        stringify::v0::transcoder<CharIn, CharOut> transcoder(*_transcoder_eng);
        transcoder.transcode(ob, _str, _str + _len, _enc_err, _allow_surr );
    }
    else
    {
        stringify::v0::decode_encode( ob, _str, _str + _len, _src_encoding
                                    , _dest_encoding, _enc_err, _allow_surr );
    }
}

template<typename CharIn, typename CharOut>
stringify::v0::width_t cv_string_printer<CharIn, CharOut>::width(stringify::v0::width_t limit) const
{
    if (_wcalc == nullptr)
    {
        if (static_cast<std::ptrdiff_t>(_len) <= limit.floor())
        {
            return static_cast<std::int16_t>(_len);
        }
        return limit;
    }
    return _wcalc->width(limit, _str, _len, _src_encoding, _enc_err, _allow_surr);
}

template<typename CharIn, typename CharOut>
class fmt_cv_string_printer: public printer<CharOut>
{
public:

    template <typename FPack>
    fmt_cv_string_printer
        ( const FPack& fp
        , const stringify::v0::cv_string_with_format<CharIn>& input
        , const stringify::v0::encoding<CharIn>& src_enc ) noexcept
        : _fmt(input)
        , _src_encoding(src_enc)
        , _dest_encoding(_get_facet<stringify::v0::encoding_c<CharOut>>(fp))
        , _enc_err(_get_facet<stringify::v0::encoding_error_c>(fp))
        , _allow_surr(_get_facet<stringify::v0::surrogate_policy_c>(fp))
    {
        _init(_get_facet<stringify::v0::width_calculator_c<CharIn>>(fp));
    }

    std::size_t necessary_size() const override;

    void print_to(stringify::v0::basic_outbuf<CharOut>& ob) const override;

    stringify::v0::width_t width(stringify::v0::width_t limit) const override;

private:

    stringify::v0::cv_string_with_format<CharIn> _fmt;
    const stringify::v0::transcoder_engine<CharIn, CharOut>* _transcoder_eng;
    const stringify::v0::encoding<CharIn> _src_encoding;
    const stringify::v0::encoding<CharOut> _dest_encoding;
    const stringify::v0::width_calculator<CharIn>* _wcalc;
    const stringify::v0::encoding_error _enc_err;
    const stringify::v0::surrogate_policy  _allow_surr;
    std::uint16_t _fillcount = 0;
    bool _width_from_fmt = false;

    template <typename Category, typename FPack>
    static decltype(auto) _get_facet(const FPack& fp)
    {
        using input_tag = stringify::v0::string_input_tag<CharIn>;
        return fp.template get_facet<Category, input_tag>();
    }

    void _init(const stringify::v0::width_as_len<CharIn>&);
    void _init(const stringify::v0::width_as_u32len<CharIn>&);
    void _init(const stringify::v0::width_calculator<CharIn>&);

    void _write_str(stringify::v0::basic_outbuf<CharOut>& ob) const;

    void _write_fill
        ( stringify::v0::basic_outbuf<CharOut>& ob
        , unsigned count ) const;
};

template<typename CharIn, typename CharOut>
void fmt_cv_string_printer<CharIn, CharOut>::_init
    ( const stringify::v0::width_as_len<CharIn>&)
{
    auto len = _fmt.value().length();
    if (_fmt.width() > static_cast<std::ptrdiff_t>(len))
    {
        _fillcount = _fmt.width() - static_cast<std::int16_t>(len);
        _width_from_fmt = true;
    }
    _wcalc = nullptr;
    _transcoder_eng =
        stringify::v0::get_transcoder(_src_encoding, _dest_encoding);
}

template<typename CharIn, typename CharOut>
void fmt_cv_string_printer<CharIn, CharOut>::_init
    ( const stringify::v0::width_as_u32len<CharIn>& wc)
{
    auto str_width = _src_encoding.codepoints_count( _fmt.value().begin()
                                                   , _fmt.value().end()
                                                   , _fmt.width() );
    if (_fmt.width() > static_cast<std::ptrdiff_t>(str_width))
    {
        _fillcount = _fmt.width() - static_cast<std::int16_t>(str_width);
        _width_from_fmt = true;
    }
    _wcalc = &wc;
    _transcoder_eng =
        stringify::v0::get_transcoder(_src_encoding, _dest_encoding);
}

template<typename CharIn, typename CharOut>
void fmt_cv_string_printer<CharIn, CharOut>::_init
    ( const stringify::v0::width_calculator<CharIn>& wc)
{
    auto str_width = wc.width( _fmt.width()
                             , _fmt.value().begin(), _fmt.value().length()
                             , _src_encoding, _enc_err, _allow_surr );
    stringify::v0::width_t fmt_width{_fmt.width()};
    if (fmt_width > str_width)
    {
        auto wdiff = (fmt_width - str_width);
        _fillcount = wdiff.round();
        _width_from_fmt = wdiff.is_integral();
    }
    _wcalc = &wc;
    _transcoder_eng =
        stringify::v0::get_transcoder(_src_encoding, _dest_encoding);
}

template<typename CharIn, typename CharOut>
stringify::v0::width_t fmt_cv_string_printer<CharIn, CharOut>::width
    ( stringify::v0::width_t limit ) const
{
    if (_width_from_fmt)
    {
        return _fmt.width();
    }
    if (_wcalc == nullptr)
    {
        auto len = _fmt.value().length();
        if (static_cast<std::ptrdiff_t>(len) <= limit.floor())
        {
            return static_cast<std::int16_t>(len);
        }
        return limit;
    }
    return _wcalc->width( limit, _fmt.value().begin(), _fmt.value().length()
                        , _src_encoding, _enc_err, _allow_surr );
}

template<typename CharIn, typename CharOut>
std::size_t fmt_cv_string_printer<CharIn, CharOut>::necessary_size() const
{
    std::size_t size;
    if(_transcoder_eng)
    {
        stringify::v0::transcoder<CharIn, CharOut> transcoder(*_transcoder_eng);
        size = transcoder.necessary_size( _fmt.value().begin()
                                        , _fmt.value().end()
                                        , _enc_err, _allow_surr );
    }
    else
    {
        size = stringify::v0::decode_encode_size
            ( _fmt.value().begin(), _fmt.value().end()
            , _src_encoding, _dest_encoding
            , _enc_err, _allow_surr );
    }
    if (_fillcount > 0)
    {
        size += _fillcount * _dest_encoding.char_size(_fmt.fill(), _enc_err);
    }
    return size;
}


template<typename CharIn, typename CharOut>
void fmt_cv_string_printer<CharIn, CharOut>::print_to
    ( stringify::v0::basic_outbuf<CharOut>& ob ) const
{
    if (_fillcount > 0)
    {
        switch(_fmt.alignment())
        {
            case stringify::v0::text_alignment::left:
            {
                _write_str(ob);
                _write_fill(ob, _fillcount);
                break;
            }
            case stringify::v0::text_alignment::center:
            {
                auto halfcount = _fillcount >> 1;
                _write_fill(ob, halfcount);
                _write_str(ob);
                _write_fill(ob, _fillcount - halfcount);;
                break;
            }
            default:
            {
                _write_fill(ob, _fillcount);
                _write_str(ob);
            }
        }
    }
    else
    {
        _write_str(ob);
    }
}


template<typename CharIn, typename CharOut>
void fmt_cv_string_printer<CharIn, CharOut>::_write_str
    ( stringify::v0::basic_outbuf<CharOut>& ob ) const
{
    if (_transcoder_eng)
    {
        stringify::v0::transcoder<CharIn, CharOut> transcoder(*_transcoder_eng);
        transcoder.transcode( ob, _fmt.value().begin(), _fmt.value().end()
                            , _enc_err, _allow_surr );
    }
    else
    {
        stringify::v0::decode_encode( ob
                                    , _fmt.value().begin(), _fmt.value().end()
                                    , _src_encoding, _dest_encoding
                                    , _enc_err, _allow_surr );
    }
}

template<typename CharIn, typename CharOut>
void fmt_cv_string_printer<CharIn, CharOut>::_write_fill
    ( stringify::v0::basic_outbuf<CharOut>& ob
    , unsigned count ) const
{
    _dest_encoding.encode_fill(ob, count, _fmt.fill(), _enc_err, _allow_surr);
}

#if defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char8_t, char8_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char8_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char8_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char8_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char8_t, wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char, char8_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char16_t, char8_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<char32_t, char8_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class cv_string_printer<wchar_t, char8_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char8_t, char8_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char8_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char8_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char8_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char8_t, wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char, char8_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char16_t, char8_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char32_t, char8_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_cv_string_printer<wchar_t, char8_t>;

#endif

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

#endif // defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

} // namespace detail

template <typename CharOut, typename FPack, typename CharIn>
inline stringify::v0::detail::cv_string_printer<CharIn, CharOut>
make_printer( const FPack& fp
            , stringify::v0::cv_string<CharIn> str )
{
    using enc_cat = stringify::v0::encoding_c<CharIn>;
    using input_tag = stringify::v0::string_input_tag<CharIn>;
    return { fp
           , str.begin()
           , str.size()
           , stringify::v0::get_facet<enc_cat, input_tag>(fp) };
}

template <typename CharOut, typename FPack, typename CharIn>
inline stringify::v0::detail::cv_string_printer<CharIn, CharOut>
make_printer( const FPack& fp
            , stringify::v0::cv_string_with_encoding<CharIn> str )
{
    return {fp, str.begin(), str.size(), str.get_encoding()};
}

template <typename CharOut, typename FPack, typename CharIn>
inline stringify::v0::detail::fmt_cv_string_printer<CharIn, CharOut>
make_printer( const FPack& fp
            , stringify::v0::cv_string_with_format<CharIn> str )
{
    using enc_cat = stringify::v0::encoding_c<CharIn>;
    using input_tag = stringify::v0::string_input_tag<CharIn>;
    return {fp, str, stringify::v0::get_facet<enc_cat, input_tag>(fp) };
}

template <typename CharOut, typename FPack, typename CharIn>
inline stringify::v0::detail::fmt_cv_string_printer<CharIn, CharOut>
make_printer( const FPack& fp
            , stringify::v0::cv_string_with_format_and_encoding<CharIn> str )
{
    return {fp, str, str.get_encoding()};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_CV_STRING_HPP

