#ifndef STRF_DETAIL_INPUT_TYPES_CV_STRING_HPP
#define STRF_DETAIL_INPUT_TYPES_CV_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/input_types/string.hpp>
#include <strf/detail/printer_variant.hpp>

STRF_NAMESPACE_BEGIN

namespace detail {

template<typename CharIn, typename CharOut>
class cv_string_printer: public strf::printer<CharOut>
{
public:

    template <typename FPack, typename Preview>
    cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<CharIn> str
        , strf::encoding<CharIn> src_enc ) noexcept
        : cv_string_printer( str
                           , src_enc
                           , _get_facet<strf::encoding_c<CharOut>>(fp)
                           , _get_facet<strf::encoding_error_c>(fp)
                           , _get_facet<strf::surrogate_policy_c>(fp) )
    {
        _preview(fp, preview);
    }

    template <typename FPack, typename Preview>
    cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<CharIn> str ) noexcept
        : cv_string_printer( str
                           , _get_facet<strf::encoding_c<CharIn>>(fp)
                           , _get_facet<strf::encoding_c<CharOut>>(fp)
                           , _get_facet<strf::encoding_error_c>(fp)
                           , _get_facet<strf::surrogate_policy_c>(fp) )
    {
        _preview(fp, preview);
    }

    cv_string_printer
        ( strf::detail::simple_string_view<CharIn> str
        , strf::encoding<CharIn> src_enc
        , strf::encoding<CharOut> dest_enc
        , strf::encoding_error enc_err
        , strf::surrogate_policy allow_surr ) noexcept;

    ~cv_string_printer() = default;

    std::size_t necessary_size() const;

    void print_to(strf::basic_outbuf<CharOut>& ob) const override;

private:

    template <typename FPack, typename Preview>
    void _preview(const FPack& fp, Preview& preview)
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
        (void)fp;
    }

    constexpr void _calc_width
        ( strf::width_preview<false>&
        , const strf::width_calculator<CharIn>& ) noexcept
    {
    }

    constexpr void _calc_width
        ( strf::width_preview<true>& wpreview
        , const strf::fast_width<CharIn>& ) noexcept
    {
        _count_codepoints(wpreview);
    }

    constexpr void _calc_width
        ( strf::width_preview<true>& wpreview
        , const strf::width_as_u32len<CharIn>& )
    {
        _count_codepoints(wpreview);
    }

    constexpr void _count_codepoints(strf::width_preview<true>& wpreview)
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
    ( strf::detail::simple_string_view<CharIn> str
    , strf::encoding<CharIn> src_enc
    , strf::encoding<CharOut> dest_enc
    , strf::encoding_error enc_err
    , strf::surrogate_policy allow_surr ) noexcept
    : _str(str.begin())
    , _len(str.size())
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
        return transcoder.necessary_size(_str, _str + _len, _allow_surr);
    }
    return strf::decode_encode_size( _str, _str + _len
                                   , _src_encoding, _dest_encoding
                                   , _allow_surr );
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
        , strf::detail::simple_string_view<CharIn> str
        , strf::alignment_format_data text_alignment
        , strf::encoding<CharIn> src_enc ) noexcept
        : _str(str)
        , _afmt(text_alignment)
        , _src_encoding(src_enc)
        , _dest_encoding(_get_facet<strf::encoding_c<CharOut>>(fp))
        , _enc_err(_get_facet<strf::encoding_error_c>(fp))
        , _allow_surr(_get_facet<strf::surrogate_policy_c>(fp))
    {
        _init(preview, _get_facet<strf::width_calculator_c<CharIn>>(fp));
        _calc_size(preview);
    }

    template <typename FPack, typename Preview>
    fmt_cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<CharIn> str
        , strf::alignment_format_data text_alignment ) noexcept
        : _str(str)
        , _afmt(text_alignment)
        , _src_encoding(_get_facet<strf::encoding_c<CharIn>>(fp))
        , _dest_encoding(_get_facet<strf::encoding_c<CharOut>>(fp))
        , _enc_err(_get_facet<strf::encoding_error_c>(fp))
        , _allow_surr(_get_facet<strf::surrogate_policy_c>(fp))
    {
        _init(preview, _get_facet<strf::width_calculator_c<CharIn>>(fp));
        _calc_size(preview);
    }

    void print_to(strf::basic_outbuf<CharOut>& ob) const override;

private:

    strf::detail::simple_string_view<CharIn> _str;
    strf::alignment_format_data _afmt;
    const strf::encoding<CharIn> _src_encoding;
    const strf::encoding<CharOut> _dest_encoding;
    const strf::transcoder_engine<CharIn, CharOut>* _transcoder_eng = nullptr;
    const strf::width_calculator<CharIn>* _wcalc = nullptr;
    const strf::encoding_error _enc_err;
    const strf::surrogate_policy  _allow_surr;
    std::uint16_t _fillcount = 0;

    template <typename Category, typename FPack>
    static decltype(auto) _get_facet(const FPack& fp)
    {
        using input_tag = strf::string_input_tag<CharIn>;
        return fp.template get_facet<Category, input_tag>();
    }

    template <bool RequiringWidth>
    void _init( strf::width_preview<RequiringWidth>& );

    template <bool RequiringWidth>
    void _init( strf::width_preview<RequiringWidth>& preview
              , const strf::fast_width<CharIn>&)
    {
        _init<RequiringWidth>(preview);
    }

    template <bool RequiringWidth>
    void _init( strf::width_preview<RequiringWidth>& preview
              , const strf::width_as_u32len<CharIn>&)
    {
        _init<RequiringWidth>(preview);
    }

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
    ( strf::width_preview<RequiringWidth>& preview )
{
    auto limit = std::max( _afmt.width
                         , preview.remaining_width().ceil() );
    auto cp_count = _src_encoding.codepoints_count( _str.begin()
                                                  , _str.end()
                                                  , limit );
    if (_afmt.width > static_cast<std::ptrdiff_t>(cp_count))
    {
        _fillcount = _afmt.width - static_cast<std::int16_t>(cp_count);
        preview.subtract_width(_afmt.width);
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
    strf::width_t wmax = _afmt.width;
    strf::width_t wdiff = 0;
    if (preview.remaining_width() > _afmt.width)
    {
        wmax = preview.remaining_width();
        wdiff = preview.remaining_width() - _afmt.width;
    }
    auto str_width = wc.width( wmax
                             , _str.begin(), _str.length()
                             , _src_encoding, _enc_err, _allow_surr );

    strf::width_t fmt_width{_afmt.width};
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
void fmt_cv_string_printer<CharIn, CharOut>::_calc_size
    ( strf::size_preview<true>& preview ) const
{
    std::size_t size;
    if(_transcoder_eng)
    {
        strf::transcoder<CharIn, CharOut> transcoder(*_transcoder_eng);
        size = transcoder.necessary_size(_str.begin(), _str.end(), _allow_surr);
    }
    else
    {
        size = strf::decode_encode_size
            ( _str.begin(), _str.end()
            , _src_encoding, _dest_encoding
            , _allow_surr );
    }
    if (_fillcount > 0)
    {
        size += _fillcount * _dest_encoding.char_size(_afmt.fill);
    }
    preview.add_size(size);
}


template<typename CharIn, typename CharOut>
void fmt_cv_string_printer<CharIn, CharOut>::print_to
    ( strf::basic_outbuf<CharOut>& ob ) const
{
    if (_fillcount > 0)
    {
        switch(_afmt.alignment)
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
        transcoder.transcode( ob, _str.begin(), _str.end()
                            , _enc_err, _allow_surr );
    }
    else
    {
        strf::decode_encode( ob
                           , _str.begin(), _str.end()
                           , _src_encoding, _dest_encoding
                           , _enc_err, _allow_surr );
    }
}

template<typename CharIn, typename CharOut>
void fmt_cv_string_printer<CharIn, CharOut>::_write_fill
    ( strf::basic_outbuf<CharOut>& ob
    , unsigned count ) const
{
    _dest_encoding.encode_fill(ob, count, _afmt.fill, _enc_err, _allow_surr);
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

template <typename CharIn, typename CharOut, bool SameSize>
class cv_printer_maker_without_encoding;

template <typename CharT>
class cv_printer_maker_without_encoding<CharT, CharT, true>
{
    using CharIn = CharT;
    using CharOut = CharT;

public:

    template <typename FPack, typename Preview>
    inline static strf::detail::string_printer<CharOut> make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                 , strf::alignment_format_q<false>
                                 , strf::cv_format<CharIn> > input )
    {
        return {fp, preview, input.value()};
    }

    template <typename FPack, typename Preview>
    inline static strf::detail::fmt_string_printer<CharOut> make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                 , strf::alignment_format_q<true>
                                 , strf::cv_format<CharIn> > input )
    {
        return {fp, preview, input.value(), input.get_alignment_format_data()};
    }
};

template <typename CharIn, typename CharOut>
class cv_printer_maker_without_encoding<CharIn, CharOut, true>
{
public:

    template <typename FPack, typename Preview>
    inline static strf::detail::printer_variant
        < strf::detail::string_printer<CharOut>
        , strf::detail::cv_string_printer<CharIn, CharOut> >
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                 , strf::alignment_format_q<false>
                                 , strf::cv_format<CharIn> > input )
    {
        using enc_cat_in = strf::encoding_c<CharIn>;
        using input_tag_in = strf::string_input_tag<CharIn>;
        auto encoding_in = strf::get_facet<enc_cat_in, input_tag_in>(fp);

        using enc_cat_out = strf::encoding_c<CharOut>;
        using input_tag_out = strf::string_input_tag<CharOut>;
        auto encoding_out = strf::get_facet<enc_cat_out, input_tag_out>(fp);

        if (encoding_in.id() == encoding_out.id())
        {
            return { strf::tag<strf::detail::string_printer<CharOut>>()
                   , fp
                   , preview
                   , strf::detail::simple_string_view<CharOut>
                       { reinterpret_cast<const CharOut*>(input.value().begin())
                       , input.value().size() } };
        }
        else
        {
            return { strf::tag<strf::detail::cv_string_printer<CharIn, CharOut>>()
                   , fp, preview, input.value(), encoding_in };
        }
    }

    template <typename FPack, typename Preview>
    inline static strf::detail::printer_variant
        < strf::detail::fmt_string_printer<CharOut>
        , strf::detail::fmt_cv_string_printer<CharIn, CharOut> >
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                 , strf::alignment_format_q<true>
                                 , strf::cv_format_with_encoding<CharIn> > input )
    {
        using enc_cat_in = strf::encoding_c<CharIn>;
        using input_tag_in = strf::string_input_tag<CharIn>;
        auto encoding_in = strf::get_facet<enc_cat_in, input_tag_in>(fp);

        using enc_cat_out = strf::encoding_c<CharOut>;
        using input_tag_out = strf::string_input_tag<CharOut>;
        auto encoding_out = strf::get_facet<enc_cat_out, input_tag_out>(fp);

        if (encoding_in.id() == encoding_out.id())
        {
            return { strf::tag<strf::detail::string_printer<CharOut>>()
                   , fp
                   , preview
                   , strf::detail::simple_string_view<CharOut>
                       { reinterpret_cast<const CharOut*>(input.value().begin())
                       , input.value().size() }
                   , input.get_alignment_format_data() };
        }
        else
        {
            return { strf::tag<strf::detail::cv_string_printer<CharIn, CharOut>>()
                   , fp, preview, input.value()
                   , input.get_alignment_format_data()
                   , encoding_in };
        }
    }
};

template <typename CharIn, typename CharOut>
class cv_printer_maker_without_encoding<CharIn, CharOut, false>
{
public:

    template <typename FPack, typename Preview>
    inline static strf::detail::cv_string_printer<CharIn, CharOut> make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                 , strf::alignment_format_q<false>
                                 , strf::cv_format<CharIn> > input )
    {
        return {fp, preview, input.value()};
    }

    template <typename FPack, typename Preview>
    inline static strf::detail::fmt_cv_string_printer<CharIn, CharOut> make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                 , strf::alignment_format_q<true>
                                 , strf::cv_format<CharIn> > input )
    {
        return {fp, preview, input.value(), input.get_alignment_format_data()};
    }
};



template <typename CharIn, typename CharOut, bool SameSize>
class cv_printer_maker_with_encoding;

template <typename CharIn, typename CharOut>
class cv_printer_maker_with_encoding<CharIn, CharOut, true>
{
public:

    template <typename FPack, typename Preview>
    inline static strf::detail::printer_variant
        < strf::detail::string_printer<CharOut>
        , strf::detail::cv_string_printer<CharIn, CharOut> >
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                 , strf::alignment_format_q<false>
                                 , strf::cv_format_with_encoding<CharIn> > input )
    {
        using enc_cat = strf::encoding_c<CharOut>;
        using input_tag = strf::string_input_tag<CharOut>;
        auto encoding_from_facets = strf::get_facet<enc_cat, input_tag>(fp);
        if (input.get_encoding().id() == encoding_from_facets.id())
        {
            return { strf::tag<strf::detail::string_printer<CharOut>>()
                   , fp
                   , preview
                   , strf::detail::simple_string_view<CharOut>
                       { reinterpret_cast<const CharOut*>(input.value().begin())
                       , input.value().size() } };
        }
        else
        {
            return { strf::tag<strf::detail::cv_string_printer<CharIn, CharOut>>()
                   , fp, preview, input.value(), input.get_encoding() };
        }
    }

    template <typename FPack, typename Preview>
    inline static strf::detail::printer_variant
        < strf::detail::fmt_string_printer<CharOut>
        , strf::detail::fmt_cv_string_printer<CharIn, CharOut> >
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                 , strf::alignment_format_q<true>
                                 , strf::cv_format_with_encoding<CharIn> > input )
    {
        using enc_cat = strf::encoding_c<CharOut>;
        using input_tag = strf::string_input_tag<CharOut>;
        auto encoding_from_facets = strf::get_facet<enc_cat, input_tag>(fp);
        if (input.get_encoding().id() == encoding_from_facets.id())
        {
            return { strf::tag<strf::detail::fmt_string_printer<CharOut>>()
                   , fp
                   , preview
                   , strf::detail::simple_string_view<CharOut>
                       { reinterpret_cast<const CharOut*>(input.value().begin())
                       , input.value().size() }
                   , input.get_alignment_format_data() };
        }
        else
        {
            return { strf::tag<strf::detail::fmt_cv_string_printer<CharIn, CharOut>>()
                   , fp
                   , preview
                   , input.value()
                   , input.get_alignment_format_data()
                   , input.get_encoding() };
        }
    }
};

template <typename CharIn, typename CharOut>
class cv_printer_maker_with_encoding<CharIn, CharOut, false>
{
public:
    template <typename FPack, typename Preview>
    inline strf::detail::cv_string_printer<CharIn, CharOut> make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                 , strf::alignment_format_q<false>
                                 , strf::cv_format_with_encoding<CharIn> > input )
    {
        return {fp, preview, input.value(), input.get_encoding()};
    }

    template <typename FPack, typename Preview>
    inline static strf::detail::fmt_cv_string_printer<CharIn, CharOut> make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                 , strf::alignment_format_q<true>
                                 , strf::cv_format_with_encoding<CharIn> > input )
    {
        return { fp
               , preview
               , input.value()
               , input.get_alignment_format_data()
               , input.get_encoding() };
    }
};

} // namespace detail

template < typename CharOut
         , typename FPack
         , typename Preview
         , typename CharIn
         , bool WithAlignment >
inline auto make_printer
    ( strf::rank<1>
    , const FPack& fp
    , Preview& preview
    , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                             , strf::alignment_format_q<WithAlignment>
                             , strf::cv_format<CharIn> > input )
{
    constexpr bool ss = sizeof(CharIn) == sizeof(CharOut);
    return strf::detail::cv_printer_maker_without_encoding<CharIn, CharOut, ss>
        :: make_printer(fp, preview, input);
}

template < typename CharOut
         , typename FPack
         , typename Preview
         , typename CharIn
         , bool WithAlignment >
inline auto make_printer
    ( strf::rank<1>
    , const FPack& fp
    , Preview& preview
    , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                             , strf::alignment_format_q<WithAlignment>
                             , strf::cv_format_with_encoding<CharIn> > input )
{
    constexpr bool ss = sizeof(CharIn) == sizeof(CharOut);
    return strf::detail::cv_printer_maker_with_encoding<CharIn, CharOut, ss>
        :: make_printer(fp, preview, input);
}

template <typename CharOut, typename FPack, typename Preview, typename CharIn>
inline strf::detail::cv_string_printer<CharIn, CharOut>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                     , strf::alignment_format_q<false>
                                     , strf::sani_format<CharIn> > input )
{
    return {fp, preview, input.value()};
}

template <typename CharOut, typename FPack, typename Preview, typename CharIn>
inline strf::detail::cv_string_printer<CharIn, CharOut>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                     , strf::alignment_format_q<false>
                                     , strf::sani_format_with_encoding<CharIn> > input )
{
    return {fp, preview, input.value(), input.get_encoding()};
}

template <typename CharOut, typename FPack, typename Preview, typename CharIn>
inline strf::detail::fmt_cv_string_printer<CharIn, CharOut>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                     , strf::alignment_format_q<true>
                                     , strf::sani_format<CharIn> > input )
{
    return { fp
           , preview
           , input.value()
           , input.get_alignment_format_data() };
}

template <typename CharOut, typename FPack, typename Preview, typename CharIn>
inline strf::detail::fmt_cv_string_printer<CharIn, CharOut>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                     , strf::alignment_format_q<true>
                                     , strf::sani_format_with_encoding<CharIn> > input )
{
    return { fp
           , preview
           , input.value()
           , input.get_alignment_format_data()
           , input.get_encoding() };
}

STRF_NAMESPACE_END

#endif  // STRF_DETAIL_INPUT_TYPES_CV_STRING_HPP

