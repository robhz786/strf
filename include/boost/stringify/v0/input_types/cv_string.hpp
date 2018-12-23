#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_CV_STRING_HPP
#define BOOST_STRINGIFY_V0_INPUT_TYPES_CV_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template<typename CharIn, typename CharOut>
class cv_string_printer: public stringify::v0::printer<CharOut>
{
public:

    template <typename FPack>
    cv_string_printer
        ( const FPack& fp
        , const CharIn* str
        , std::size_t len ) noexcept
        : cv_string_printer
            ( str
            , len
            , _get_facet<stringify::v0::width_calculator_category>(fp)
            , _get_facet<stringify::v0::encoding_category<CharIn>>(fp)
            , _get_facet<stringify::v0::encoding_category<CharOut>>(fp)
            , _get_facet<stringify::v0::encoding_policy_category>(fp) )
    {
    }

    cv_string_printer
        ( const CharIn* str
        , std::size_t len
        , const stringify::v0::width_calculator& wcalc
        , const stringify::v0::encoding<CharIn>& src_enc
        , const stringify::v0::encoding<CharOut>& dest_enc
        , const stringify::v0::encoding_policy epoli ) noexcept
        : _str(str)
        , _len(len)
        , _wcalc(wcalc)
        , _src_encoding(src_enc)
        , _dest_encoding(dest_enc)
        , _transcoder(stringify::v0::get_transcoder(src_enc, dest_enc))
        , _epoli(epoli)
    {
    }

    ~cv_string_printer() = default;

    std::size_t necessary_size() const override;

    stringify::v0::expected_output_buffer<CharOut> write
        ( stringify::v0::output_buffer<CharOut> buff
        , stringify::buffer_recycler<CharOut>& recycler ) const override;

    int remaining_width(int w) const override;

private:

    const CharIn* const _str;
    const std::size_t _len;
    const stringify::v0::width_calculator _wcalc;
    const stringify::v0::encoding<CharIn>&  _src_encoding;
    const stringify::v0::encoding<CharOut>& _dest_encoding;
    const stringify::v0::transcoder<CharIn, CharOut>* _transcoder;
    const stringify::v0::encoding_policy _epoli;

    template <typename Category, typename FPack>
    const auto& _get_facet(const FPack& fp) const
    {
        using input_tag = stringify::v0::string_input_tag<CharIn>;
        return fp.template get_facet<Category, input_tag>();
    }
};

template<typename CharIn, typename CharOut>
std::size_t cv_string_printer<CharIn, CharOut>::necessary_size() const
{
    if (_transcoder != nullptr)
    {
        return _transcoder->necessary_size
            ( _str, _str + _len, _epoli.err_hdl(), _epoli.allow_surr() );
    }
    return stringify::v0::detail::decode_encode_size
        ( _str, _str + _len
        , _src_encoding, _dest_encoding
        , _epoli );
}

template<typename CharIn, typename CharOut>
stringify::v0::expected_output_buffer<CharOut>
cv_string_printer<CharIn, CharOut>::write
    ( stringify::v0::output_buffer<CharOut> buff
    , stringify::buffer_recycler<CharOut>& recycler ) const
{
    if (_transcoder)
    {
        return stringify::v0::detail::transcode( buff, recycler
                                               , _str, _str + _len
                                               , *_transcoder, _epoli);
    }
    return stringify::v0::detail::decode_encode( buff, recycler
                                               , _str, _str + _len
                                               , _src_encoding, _dest_encoding
                                               , _epoli );
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
        , const stringify::v0::detail::cv_string_with_format<CharIn>& input ) noexcept
        : _fmt(input)
        , _dest_encoding(_get_facet<stringify::v0::encoding_category<CharOut>>(fp))
        , _wcalc(_get_facet<stringify::v0::width_calculator_category>(fp))
        , _epoli(_get_facet<stringify::v0::encoding_policy_category>(fp))
    {
        _init(_get_facet<stringify::v0::encoding_category<CharIn>>(fp));
    }

    std::size_t necessary_size() const override;

    stringify::v0::expected_output_buffer<CharOut> write
        ( stringify::v0::output_buffer<CharOut> buff
        , stringify::buffer_recycler<CharOut>& recycler ) const override;

    int remaining_width(int w) const override;

private:

    stringify::v0::detail::cv_string_with_format<CharIn> _fmt;
    const stringify::v0::transcoder<CharIn, CharOut>* _transcoder;
    const stringify::v0::encoding<CharOut>& _dest_encoding;
    const stringify::v0::width_calculator _wcalc;
    const stringify::v0::encoding_policy  _epoli;
    int _fillcount = 0;

    template <typename Category, typename FPack>
    const auto& _get_facet(const FPack& fp) const
    {
        using input_tag = stringify::v0::string_input_tag<CharIn>;
        return fp.template get_facet<Category, input_tag>();
    }
    
    void _init(const stringify::v0::encoding<CharIn>& src_enc);
    
    stringify::v0::expected_output_buffer<CharOut> _write_str
        ( stringify::v0::output_buffer<CharOut> buff
        , stringify::buffer_recycler<CharOut>& recycler ) const;

    stringify::v0::expected_output_buffer<CharOut> _write_fill
        ( stringify::v0::output_buffer<CharOut> buff
        , stringify::buffer_recycler<CharOut>& recycler
        , unsigned count ) const;
};

template<typename CharIn, typename CharOut>
void fmt_cv_string_printer<CharIn, CharOut>::_init
    ( const stringify::v0::encoding<CharIn>& src_enc )
{
    if( ! _fmt.has_encoding())
    {
        _fmt.value().set_encoding(src_enc);
    }
    _fillcount = ( _fmt.width() > 0
                 ? _wcalc.remaining_width
                     ( _fmt.width()
                     , _fmt.value().begin()
                     , _fmt.value().length()
                     , src_enc
                     , _epoli )
                 : 0 );
    _transcoder = stringify::v0::get_transcoder(src_enc, _dest_encoding);
}

template<typename CharIn, typename CharOut>
std::size_t fmt_cv_string_printer<CharIn, CharOut>::necessary_size() const
{
    std::size_t size
        = _transcoder != nullptr
        ? _transcoder->necessary_size( _fmt.value().begin(), _fmt.value().end()
                                    , _epoli.err_hdl(), _epoli.allow_surr() )
        : stringify::v0::detail::decode_encode_size
            ( _fmt.value().begin(), _fmt.value().end()
            , _fmt.value().encoding(), _dest_encoding
            , _epoli );

    if (_fillcount > 0)
    {
        size += _fillcount * _dest_encoding.char_size( _fmt.fill()
                                                     , _epoli.err_hdl() );
    }

    return size;
}


template<typename CharIn, typename CharOut>
stringify::v0::expected_output_buffer<CharOut>
fmt_cv_string_printer<CharIn, CharOut>::write
    ( stringify::v0::output_buffer<CharOut> buff
    , stringify::buffer_recycler<CharOut>& recycler ) const
{
    if (_fillcount > 0)
    {
        switch(_fmt.alignment())
        {
            case stringify::v0::alignment::left:
            {
                auto x = _write_str(buff, recycler);
                return x ? _write_fill(*x, recycler, _fillcount) : x;
            }
            case stringify::v0::alignment::center:
            {
                int halfcount = _fillcount / 2;
                auto x = _write_fill(buff, recycler, halfcount);
                if(x) x = _write_str(*x, recycler);
                return x ? _write_fill(*x, recycler, _fillcount - halfcount) : x;
            }
            default:
            {
                auto x = _write_fill(buff, recycler, _fillcount);
                return x ? _write_str(*x, recycler) : x;
            }
        }
    }
    return _write_str(buff, recycler);
}


template<typename CharIn, typename CharOut>
stringify::v0::expected_output_buffer<CharOut>
fmt_cv_string_printer<CharIn, CharOut>::_write_str
    ( stringify::v0::output_buffer<CharOut> buff
    , stringify::buffer_recycler<CharOut>& recycler ) const
{
    if (_transcoder)
    {
        return stringify::v0::detail::transcode( buff, recycler
                                               , _fmt.value().begin()
                                               , _fmt.value().end()
                                               , *_transcoder
                                               , _epoli);
    }
    return stringify::v0::detail::decode_encode( buff, recycler
                                               , _fmt.value().begin()
                                               , _fmt.value().end()
                                               , _fmt.value().encoding()
                                               , _dest_encoding
                                               , _epoli );
}

template<typename CharIn, typename CharOut>
stringify::v0::expected_output_buffer<CharOut>
fmt_cv_string_printer<CharIn, CharOut>::_write_fill
    ( stringify::v0::output_buffer<CharOut> buff
    , stringify::buffer_recycler<CharOut>& recycler
    , unsigned count ) const
{
    return stringify::v0::detail::write_fill
        ( _dest_encoding, buff, recycler, count, _fmt.fill(), _epoli.err_hdl() );
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
        , _fmt.value().encoding()
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
    return {fp, str.begin(), str.size()};
}

template <typename CharOut, typename FPack, typename CharIn>
inline stringify::v0::detail::fmt_cv_string_printer<CharIn, CharOut>
make_printer( const FPack& fp
            , stringify::v0::detail::cv_string_with_format<CharIn> str )
{
    return {fp, str};
}

} // namespace detail

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_INPUT_TYPES_CV_STRING_HPP

