#ifndef STRF_V0_DETAIL_INPUT_TYPES_CV_STRING_HPP
#define STRF_V0_DETAIL_INPUT_TYPES_CV_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/detail/format_functions.hpp>
#include <boost/stringify/v0/detail/facets/encoding.hpp>

STRF_NAMESPACE_BEGIN

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

    strf::detail::simple_string_view<CharIn> _str;
};


template <typename CharIn>
class cv_string_with_encoding: public strf::cv_string<CharIn>
{
public:

    cv_string_with_encoding
        ( const CharIn* str
        , std::size_t len
        , strf::encoding<CharIn> enc ) noexcept
        : strf::cv_string<CharIn>(str, len)
        , _enc(enc)
    {
    }

    cv_string_with_encoding(const cv_string_with_encoding&) noexcept = default;

    constexpr strf::encoding<CharIn> encoding() const
    {
        return _enc;
    }
    constexpr void set_encoding(strf::encoding<CharIn> enc)
    {
        _enc = enc;
    }

    constexpr strf::encoding<CharIn> get_encoding() const
    {
        return _enc;
    }

private:

    strf::encoding<CharIn> _enc;
};

#if defined(__cpp_char8_t)

STRF_CONSTEXPR_CHAR_TRAITS
strf::cv_string<char8_t> cv(const char8_t* str)
{
    return {str, std::char_traits<char8_t>::length(str)};
}

#endif

STRF_CONSTEXPR_CHAR_TRAITS
strf::cv_string<char> cv(const char* str)
{
    return {str, std::char_traits<char>::length(str)};
}
STRF_CONSTEXPR_CHAR_TRAITS
strf::cv_string<char16_t> cv(const char16_t* str)
{
    return {str, std::char_traits<char16_t>::length(str)};
}
STRF_CONSTEXPR_CHAR_TRAITS
strf::cv_string<char32_t> cv(const char32_t* str)
{
    return {str, std::char_traits<char32_t>::length(str)};
}
STRF_CONSTEXPR_CHAR_TRAITS
strf::cv_string<wchar_t> cv(const wchar_t* str)
{
    return {str, std::char_traits<wchar_t>::length(str)};
}

template <typename CharIn>
STRF_CONSTEXPR_CHAR_TRAITS
strf::cv_string_with_encoding<CharIn> cv
    ( const CharIn* str
    , strf::encoding<CharIn> enc )
{
    return {str, std::char_traits<CharIn>::length(str), enc};
}

template <typename CharIn, typename Traits, typename Allocator>
STRF_CONSTEXPR_CHAR_TRAITS
strf::cv_string<CharIn> cv
    ( const std::basic_string<CharIn, Traits, Allocator>& str )
{
    return {str.data(), str.size()};
}

template <typename CharIn, typename Traits, typename Allocator>
strf::cv_string_with_encoding<CharIn> cv
    ( const std::basic_string<CharIn, Traits, Allocator>& str
    , strf::encoding<CharIn> enc )
{
    return {str.data(), str.size(), enc};
}

#if defined(STRF_HAS_STD_STRING_VIEW)

template <typename CharIn, typename Traits>
constexpr strf::cv_string<CharIn> cv
    ( std::basic_string_view<CharIn, Traits> str )
{
    return {str.data(), str.size()};
}

template <typename CharIn, typename Traits>
constexpr strf::cv_string_with_encoding<CharIn> cv
    ( std::basic_string_view<CharIn, Traits> str
    , strf::encoding<CharIn> enc )
{
    return { str.data(), str.size(), &enc };
}

#endif

template <typename CharIn>
using cv_string_with_format = strf::value_with_format
    < strf::cv_string<CharIn>
    , strf::alignment_format >;

template <typename CharIn>
using cv_string_with_format_and_encoding = strf::value_with_format
    < strf::cv_string_with_encoding<CharIn>
    , strf::alignment_format >;

template <typename CharIn>
constexpr auto make_fmt(strf::tag, strf::cv_string<CharIn>& cv_str) noexcept
{
    return strf::cv_string_with_format<char>{cv_str};
}

#if defined(__cpp_char8_t)

STRF_CONSTEXPR_CHAR_TRAITS
strf::cv_string_with_format<char8_t> fmt_cv(const char8_t* str) noexcept
{
    strf::cv_string<char8_t> cv_str
        { str, std::char_traits<char8_t>::length(str) };
    return strf::cv_string_with_format<char8_t>{cv_str};
}

#endif

STRF_CONSTEXPR_CHAR_TRAITS
strf::cv_string_with_format<char> fmt_cv(const char* str) noexcept
{
    strf::cv_string<char> cv_str
        { str, std::char_traits<char>::length(str) };
    return strf::cv_string_with_format<char>{cv_str};
}
STRF_CONSTEXPR_CHAR_TRAITS
strf::cv_string_with_format<char16_t> fmt_cv(const char16_t* str) noexcept
{
    strf::cv_string<char16_t> cv_str
        { str, std::char_traits<char16_t>::length(str) };
    return strf::cv_string_with_format<char16_t>{cv_str};
}
STRF_CONSTEXPR_CHAR_TRAITS
strf::cv_string_with_format<char32_t> fmt_cv(const char32_t* str) noexcept
{
    strf::cv_string<char32_t> cv_str
        { str, std::char_traits<char32_t>::length(str) };
    return strf::cv_string_with_format<char32_t>{cv_str};
}
STRF_CONSTEXPR_CHAR_TRAITS
strf::cv_string_with_format<wchar_t> fmt_cv(const wchar_t* str) noexcept
{
    strf::cv_string<wchar_t> cv_str
        { str, std::char_traits<wchar_t>::length(str) };
    return strf::cv_string_with_format<wchar_t>{cv_str};

}

template <typename CharIn>
STRF_CONSTEXPR_CHAR_TRAITS
strf::cv_string_with_format_and_encoding<CharIn> fmt_cv
    ( const CharIn* str
    , strf::encoding<CharIn> enc ) noexcept
{
    strf::cv_string_with_encoding<CharIn> cv_str{str, enc};
    return strf::cv_string_with_format_and_encoding<CharIn>{cv_str};
}

#if defined(STRF_HAS_STD_STRING_VIEW)

template <typename CharIn, typename Traits>
constexpr strf::cv_string_with_format<CharIn> fmt_cv
    ( std::basic_string_view<CharIn, Traits> str ) noexcept
{
    strf::cv_string<CharIn> cv_str{str.data(), str.size()};
    return strf::cv_string_with_format<CharIn>{cv_str};
}

template <typename CharIn, typename Traits>
constexpr strf::cv_string_with_format_and_encoding<CharIn> fmt_cv
    ( std::basic_string_view<CharIn, Traits> str
    , strf::encoding<CharIn> enc ) noexcept
{
    strf::cv_string_and_encoding<CharIn> cv_str
        { str.data(), str.size(), &enc };
    return strf::cv_string_with_format_and_encoding<CharIn>{cv_str};
}

#endif

template <typename CharIn, typename Traits, typename Allocator>
strf::cv_string_with_format<CharIn> fmt_cv
    ( const std::basic_string<CharIn, Traits, Allocator>& str )
{
    strf::cv_string<CharIn> cv_str{str.data(), str.length()};
    return strf::cv_string_with_format<CharIn>{cv_str};
}

template <typename CharIn, typename Traits, typename Allocator>
strf::cv_string_with_format_and_encoding<CharIn> fmt_cv
    ( const std::basic_string<CharIn, Traits, Allocator>& str
    , strf::encoding<CharIn> enc ) noexcept
{
    strf::cv_string_with_encoding<CharIn> cv_str_e
        {str.data(), str.length(), &enc};
    return strf::cv_string_with_format_and_encoding<CharIn>{cv_str_e};
}

namespace detail {

template<typename CharIn, typename CharOut>
class cv_string_printer: public strf::printer<CharOut>
{
public:

    template <typename FPack, typename Preview>
    cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , const CharIn* str
        , std::size_t len
        , strf::encoding<CharIn> src_enc ) noexcept
        : cv_string_printer
            ( str
            , len
            , src_enc
            , _get_facet<strf::encoding_c<CharOut>>(fp)
            , _get_facet<strf::encoding_error_c>(fp)
            , _get_facet<strf::surrogate_policy_c>(fp) )
    {
        STRF_IF_CONSTEXPR (Preview::width_required)
        {
            const auto& wc = _get_facet<strf::width_calculator_c<CharIn>>(fp);
            _calc_width(preview, wc);
        }
        STRF_IF_CONSTEXPR (Preview::size_required)
        {
            preview.add_size(necessary_size());
        }
    }

    cv_string_printer
        ( const CharIn* str
        , std::size_t len
        , strf::encoding<CharIn> src_enc
        , strf::encoding<CharOut> dest_enc
        , strf::encoding_error enc_err
        , strf::surrogate_policy allow_surr ) noexcept;

    ~cv_string_printer() = default;

    std::size_t necessary_size() const;

    void print_to(strf::basic_outbuf<CharOut>& ob) const override;

private:

    constexpr void _calc_width
        ( strf::width_preview<false>&
        , const strf::width_calculator<CharIn>& ) noexcept
    {
    }

    constexpr void _calc_width
        ( strf::width_preview<true>& wpreview
        , const strf::width_as_len<CharIn>& ) noexcept
    {
        auto remaining_width = wpreview.remaining_width().floor();
        if (static_cast<std::ptrdiff_t>(_len) <= remaining_width)
        {
            wpreview.subtract_width(static_cast<std::int16_t>(_len));
        }
        else
        {
            wpreview.clear_remaining_width();
        }
    }

    constexpr void _calc_width
        ( strf::width_preview<true>& wpreview
        , const strf::width_as_u32len<CharIn>& )
    {
        auto limit = wpreview.remaining_width();
        if (limit > 0)
        {
            auto count = _src_encoding.codepoints_count
                ( _str, _str + _len, limit.ceil() );

            if (static_cast<std::ptrdiff_t>(count) < limit.ceil())
            {
                wpreview.subtract_width(static_cast<std::int16_t>(count));
            }
            else
            {
                wpreview.clear_remaining_width();
            }
        }
    }

    constexpr void _calc_width
        ( strf::width_preview<true>& wpreview
        , const strf::width_calculator<CharIn>& wcalc )
    {
        auto limit = wpreview.remaining_width();
        if (limit > 0)
        {
            auto w = wcalc.width( limit, _str, _len, _src_encoding
                                , _enc_err, _allow_surr );
            if (w < limit)
            {
                wpreview.subtract_width(w);
            }
            else
            {
                wpreview.clear_remaining_width();
            }
            // wcalc.subtract_width( wpreview, _str, _len
            //                      , encoding, enc_err, allow_surr );
        }
    }

    const CharIn* const _str;
    const std::size_t _len;
    const strf::encoding<CharIn>  _src_encoding;
    const strf::encoding<CharOut> _dest_encoding;
    const strf::transcoder_engine<CharIn, CharOut>* _transcoder_eng;
    const strf::encoding_error _enc_err;
    const strf::surrogate_policy _allow_surr;

    template <typename Category, typename FPack>
    static decltype(auto) _get_facet(const FPack& fp)
    {
        using input_tag = strf::string_input_tag<CharIn>;
        return fp.template get_facet<Category, input_tag>();
    }
};

template<typename CharIn, typename CharOut>
cv_string_printer<CharIn, CharOut>::cv_string_printer
    ( const CharIn* str
    , std::size_t len
    , strf::encoding<CharIn> src_enc
    , strf::encoding<CharOut> dest_enc
    , strf::encoding_error enc_err
    , strf::surrogate_policy allow_surr ) noexcept
    : _str(str)
    , _len(len)
    , _src_encoding(src_enc)
    , _dest_encoding(dest_enc)
    , _transcoder_eng(strf::get_transcoder(src_enc, dest_enc))
    , _enc_err(enc_err)
    , _allow_surr(allow_surr)
{
}

template<typename CharIn, typename CharOut>
std::size_t cv_string_printer<CharIn, CharOut>::necessary_size() const
{
    if (_transcoder_eng)
    {
        strf::transcoder<CharIn, CharOut> transcoder(*_transcoder_eng);
        return transcoder.necessary_size
            ( _str, _str + _len, _enc_err, _allow_surr );
    }
    return strf::decode_encode_size( _str, _str + _len
                                            , _src_encoding, _dest_encoding
                                            , _enc_err, _allow_surr );
}

template<typename CharIn, typename CharOut>
void cv_string_printer<CharIn, CharOut>::print_to
    ( strf::basic_outbuf<CharOut>& ob ) const
{
    if (_transcoder_eng != nullptr)
    {
        strf::transcoder<CharIn, CharOut> transcoder(*_transcoder_eng);
        transcoder.transcode(ob, _str, _str + _len, _enc_err, _allow_surr );
    }
    else
    {
        strf::decode_encode( ob, _str, _str + _len, _src_encoding
                                    , _dest_encoding, _enc_err, _allow_surr );
    }
}

template<typename CharIn, typename CharOut>
class fmt_cv_string_printer: public printer<CharOut>
{
public:

    template <typename FPack, typename Preview>
    fmt_cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , const strf::cv_string_with_format<CharIn>& input
        , const strf::encoding<CharIn>& src_enc ) noexcept
        : _fmt(input)
        , _src_encoding(src_enc)
        , _dest_encoding(_get_facet<strf::encoding_c<CharOut>>(fp))
        , _enc_err(_get_facet<strf::encoding_error_c>(fp))
        , _allow_surr(_get_facet<strf::surrogate_policy_c>(fp))
    {
        _init(preview, _get_facet<strf::width_calculator_c<CharIn>>(fp));
        _calc_size(preview);
    }

    std::size_t necessary_size() const;

    void print_to(strf::basic_outbuf<CharOut>& ob) const override;

    strf::width_t width(strf::width_t limit) const;

private:

    strf::cv_string_with_format<CharIn> _fmt;
    const strf::transcoder_engine<CharIn, CharOut>* _transcoder_eng;
    const strf::encoding<CharIn> _src_encoding;
    const strf::encoding<CharOut> _dest_encoding;
    const strf::width_calculator<CharIn>* _wcalc;
    const strf::encoding_error _enc_err;
    const strf::surrogate_policy  _allow_surr;
    std::uint16_t _fillcount = 0;
    bool _width_from_fmt = false;

    template <typename Category, typename FPack>
    static decltype(auto) _get_facet(const FPack& fp)
    {
        using input_tag = strf::string_input_tag<CharIn>;
        return fp.template get_facet<Category, input_tag>();
    }

    template <bool RequiringWidth>
    void _init( strf::width_preview<RequiringWidth>& preview
              , const strf::width_as_len<CharIn>&);

    template <bool RequiringWidth>
    void _init( strf::width_preview<RequiringWidth>& preview
              , const strf::width_as_u32len<CharIn>&);

    template <bool RequiringWidth>
    void _init( strf::width_preview<RequiringWidth>& preview
              , const strf::width_calculator<CharIn>&);

    constexpr void _calc_size(strf::size_preview<false>&) const
    {
    }

    void _calc_size(strf::size_preview<true>& preview) const;

    void _write_str(strf::basic_outbuf<CharOut>& ob) const;

    void _write_fill
        ( strf::basic_outbuf<CharOut>& ob
        , unsigned count ) const;
};

template <typename CharIn, typename CharOut>
template <bool RequiringWidth>
void fmt_cv_string_printer<CharIn, CharOut>::_init
    ( strf::width_preview<RequiringWidth>& preview
    , const strf::width_as_len<CharIn>&)
{
    auto len = _fmt.value().length();
    if (_fmt.width() > static_cast<std::ptrdiff_t>(len))
    {
        _fillcount = _fmt.width() - static_cast<std::int16_t>(len);
        preview.subtract_width(_fmt.width());
    }
    else
    {
        preview.checked_subtract_width(len);
    }
    _transcoder_eng =
        strf::get_transcoder(_src_encoding, _dest_encoding);
}

template <typename CharIn, typename CharOut>
template <bool RequiringWidth>
void fmt_cv_string_printer<CharIn, CharOut>::_init
    ( strf::width_preview<RequiringWidth>& preview
    , const strf::width_as_u32len<CharIn>&)
{
    auto cp_count = _src_encoding.codepoints_count( _fmt.value().begin()
                                                  , _fmt.value().end()
                                                  , _fmt.width() );
    if (_fmt.width() > static_cast<std::ptrdiff_t>(cp_count))
    {
        _fillcount = _fmt.width() - static_cast<std::int16_t>(cp_count);
        preview.subtract_width(_fmt.width());
    }
    else
    {
        preview.checked_subtract_width(cp_count);
    }
    _transcoder_eng =
        strf::get_transcoder(_src_encoding, _dest_encoding);
}

template <typename CharIn, typename CharOut>
template <bool RequiringWidth>
void fmt_cv_string_printer<CharIn, CharOut>::_init
    ( strf::width_preview<RequiringWidth>& preview
    , const strf::width_calculator<CharIn>& wc)
{
    strf::width_t wmax = _fmt.width();
    strf::width_t wdiff = 0;
    if (preview.remaining_width() > _fmt.width())
    {
        wmax = preview.remaining_width();
        wdiff = preview.remaining_width() - _fmt.width();
    }
    auto str_width = wc.width( wmax
                             , _fmt.value().begin(), _fmt.value().length()
                             , _src_encoding, _enc_err, _allow_surr );

    strf::width_t fmt_width{_fmt.width()};
    if (fmt_width > str_width)
    {
        auto wfill = (fmt_width - str_width);
        _fillcount = wfill.round();
        preview.subtract_width(wfill + _fillcount);
    }
    else
    {
        preview.subtract_width(str_width);
    }
    _transcoder_eng =
        strf::get_transcoder(_src_encoding, _dest_encoding);
}

template<typename CharIn, typename CharOut>
strf::width_t fmt_cv_string_printer<CharIn, CharOut>::width
    ( strf::width_t limit ) const
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
void fmt_cv_string_printer<CharIn, CharOut>::_calc_size
    ( strf::size_preview<true>& preview ) const
{
    std::size_t size;
    if(_transcoder_eng)
    {
        strf::transcoder<CharIn, CharOut> transcoder(*_transcoder_eng);
        size = transcoder.necessary_size( _fmt.value().begin()
                                        , _fmt.value().end()
                                        , _enc_err, _allow_surr );
    }
    else
    {
        size = strf::decode_encode_size
            ( _fmt.value().begin(), _fmt.value().end()
            , _src_encoding, _dest_encoding
            , _enc_err, _allow_surr );
    }
    if (_fillcount > 0)
    {
        size += _fillcount * _dest_encoding.char_size(_fmt.fill(), _enc_err);
    }
    preview.add_size(size);
}


template<typename CharIn, typename CharOut>
void fmt_cv_string_printer<CharIn, CharOut>::print_to
    ( strf::basic_outbuf<CharOut>& ob ) const
{
    if (_fillcount > 0)
    {
        switch(_fmt.alignment())
        {
            case strf::text_alignment::left:
            {
                _write_str(ob);
                _write_fill(ob, _fillcount);
                break;
            }
            case strf::text_alignment::center:
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
    ( strf::basic_outbuf<CharOut>& ob ) const
{
    if (_transcoder_eng)
    {
        strf::transcoder<CharIn, CharOut> transcoder(*_transcoder_eng);
        transcoder.transcode( ob, _fmt.value().begin(), _fmt.value().end()
                            , _enc_err, _allow_surr );
    }
    else
    {
        strf::decode_encode( ob
                                    , _fmt.value().begin(), _fmt.value().end()
                                    , _src_encoding, _dest_encoding
                                    , _enc_err, _allow_surr );
    }
}

template<typename CharIn, typename CharOut>
void fmt_cv_string_printer<CharIn, CharOut>::_write_fill
    ( strf::basic_outbuf<CharOut>& ob
    , unsigned count ) const
{
    _dest_encoding.encode_fill(ob, count, _fmt.fill(), _enc_err, _allow_surr);
}

#if defined(STRF_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)

STRF_EXPLICIT_TEMPLATE class cv_string_printer<char8_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<char8_t, char>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<char8_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<char8_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<char8_t, wchar_t>;

STRF_EXPLICIT_TEMPLATE class cv_string_printer<char, char8_t>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<char16_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<char32_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<wchar_t, char8_t>;

STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char8_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char8_t, char>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char8_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char8_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char8_t, wchar_t>;

STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char, char8_t>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char16_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char32_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<wchar_t, char8_t>;

#endif

STRF_EXPLICIT_TEMPLATE class cv_string_printer<char, char>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<char, char16_t>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<char, char32_t>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<char, wchar_t>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<char16_t, char>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<char16_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<char16_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<char16_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<char32_t, char>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<char32_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<char32_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<char32_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<wchar_t, char>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<wchar_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<wchar_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<wchar_t, wchar_t>;

STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char, char>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char, char16_t>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char, char32_t>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char, wchar_t>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char16_t, char>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char16_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char16_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char16_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char32_t, char>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char32_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char32_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<char32_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<wchar_t, char>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<wchar_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<wchar_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<wchar_t, wchar_t>;

#endif // defined(STRF_SEPARATE_COMPILATION)

} // namespace detail

template <typename CharOut, typename FPack, typename Preview, typename CharIn>
inline strf::detail::cv_string_printer<CharIn, CharOut>
make_printer( const FPack& fp
            , Preview& preview
            , strf::cv_string<CharIn> str )
{
    using enc_cat = strf::encoding_c<CharIn>;
    using input_tag = strf::string_input_tag<CharIn>;
    return { fp
           , preview
           , str.begin()
           , str.size()
           , strf::get_facet<enc_cat, input_tag>(fp) };
}

template <typename CharOut, typename FPack, typename Preview, typename CharIn>
inline strf::detail::cv_string_printer<CharIn, CharOut>
make_printer( const FPack& fp
            , Preview& preview
            , strf::cv_string_with_encoding<CharIn> str )
{
    return {fp, preview, str.begin(), str.size(), str.get_encoding()};
}

template <typename CharOut, typename FPack, typename Preview, typename CharIn>
inline strf::detail::fmt_cv_string_printer<CharIn, CharOut>
make_printer( const FPack& fp
            , Preview& preview
            , strf::cv_string_with_format<CharIn> str )
{
    using enc_cat = strf::encoding_c<CharIn>;
    using input_tag = strf::string_input_tag<CharIn>;
    return {fp, preview, str, strf::get_facet<enc_cat, input_tag>(fp) };
}

template <typename CharOut, typename FPack, typename Preview, typename CharIn>
inline strf::detail::fmt_cv_string_printer<CharIn, CharOut>
make_printer( const FPack& fp
            , Preview& preview
            , strf::cv_string_with_format_and_encoding<CharIn> str )
{
    return {fp, preview, str, str.get_encoding()};
}

STRF_NAMESPACE_END

#endif  // STRF_V0_DETAIL_INPUT_TYPES_CV_STRING_HPP

