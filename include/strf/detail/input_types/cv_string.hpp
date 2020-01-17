#ifndef STRF_DETAIL_INPUT_TYPES_CV_STRING_HPP
#define STRF_DETAIL_INPUT_TYPES_CV_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/input_types/string.hpp>
#include <strf/detail/printer_variant.hpp>

namespace strf {

namespace detail {

template<std::size_t CharInSize, std::size_t CharOutSize>
class cv_string_printer: public strf::printer<CharOutSize>
{
public:

    using char_in_type = strf::underlying_outbuf_char_type<CharInSize>;

    template <typename FPack, typename Preview, typename CharIn, typename CharOut>
    STRF_HD cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<CharIn> str
        , strf::encoding<CharIn> src_enc
        , strf::tag<CharOut> ) noexcept
        : cv_string_printer( { reinterpret_cast<const char_in_type*>(str.begin())
                             , str.size() }
                           , src_enc.as_underlying()
                           , _get_facet<strf::encoding_c<CharOut>, CharIn>(fp).as_underlying()
                           , _get_facet<strf::encoding_error_c, CharIn>(fp)
                           , _get_facet<strf::surrogate_policy_c, CharIn>(fp) )
    {
        static_assert(sizeof(CharIn) == CharInSize);
        static_assert(sizeof(CharOut) == CharOutSize);
        _preview<CharIn>(fp, preview);
    }

    template <typename FPack, typename Preview, typename CharIn, typename CharOut>
    STRF_HD cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<CharIn> str
        , strf::tag<CharOut> ) noexcept
        : cv_string_printer( { reinterpret_cast<const char_in_type*>(str.begin())
                             , str.size()}
                           , _get_facet<strf::encoding_c<CharIn>, CharIn>(fp).as_underlying()
                           , _get_facet<strf::encoding_c<CharOut>, CharIn>(fp).as_underlying()
                           , _get_facet<strf::encoding_error_c, CharIn>(fp)
                           , _get_facet<strf::surrogate_policy_c, CharIn>(fp) )
    {
        static_assert(sizeof(CharIn) == CharInSize);
        static_assert(sizeof(CharOut) == CharOutSize);
        _preview<CharIn>(fp, preview);
    }

    STRF_HD cv_string_printer
        ( strf::detail::simple_string_view<char_in_type> str
        , const strf::underlying_encoding<CharInSize>& src_enc
        , const strf::underlying_encoding<CharOutSize>& dest_enc
        , strf::encoding_error enc_err
        , strf::surrogate_policy allow_surr ) noexcept;

    STRF_HD ~cv_string_printer() { }

    STRF_HD std::size_t necessary_size() const;

    STRF_HD void print_to(strf::underlying_outbuf<CharOutSize>& ob) const override;

private:

    template <typename CharIn, typename FPack, typename Preview>
    STRF_HD void _preview(const FPack& fp, Preview& preview)
    {
        STRF_IF_CONSTEXPR (Preview::width_required) {
            const auto& wc
                = _get_facet<strf::width_calculator_c<CharInSize>, CharIn>(fp);
            _calc_width(preview, wc);
        }
        STRF_IF_CONSTEXPR (Preview::size_required) {
            preview.add_size(necessary_size());
        }
        (void)fp;
    }

    STRF_HD constexpr void _calc_width
        ( strf::width_preview<false>&
        , const strf::width_calculator<CharInSize>& ) noexcept
    {
    }

    STRF_HD constexpr void _calc_width
        ( strf::width_preview<true>& wpreview
        , const strf::fast_width<CharInSize>& ) noexcept
    {
        _count_codepoints(wpreview);
    }

    STRF_HD constexpr void _calc_width
        ( strf::width_preview<true>& wpreview
        , const strf::width_as_u32len<CharInSize>& )
    {
        _count_codepoints(wpreview);
    }

    STRF_HD constexpr void _count_codepoints(strf::width_preview<true>& wpreview)
    {
        auto limit = wpreview.remaining_width();
        if (limit > 0) {
            auto count = _src_encoding.codepoints_count
                ( _str, _str + _len, limit.ceil() );

            if (static_cast<std::ptrdiff_t>(count) < limit.ceil()) {
                wpreview.subtract_width(static_cast<std::int16_t>(count));
            } else {
                wpreview.clear_remaining_width();
            }
        }
    }

    STRF_HD constexpr void _calc_width
        ( strf::width_preview<true>& wpreview
        , const strf::width_calculator<CharInSize>& wcalc )
    {
        auto limit = wpreview.remaining_width();
        if (limit > 0) {
            auto w = wcalc.width( limit, _str, _len, _src_encoding
                                , _enc_err, _allow_surr );
            if (w < limit) {
                wpreview.subtract_width(w);
            } else {
                wpreview.clear_remaining_width();
            }
        }
    }

    const char_in_type* const _str;
    const std::size_t _len;
    const strf::underlying_encoding<CharInSize>&  _src_encoding;
    const strf::underlying_encoding<CharOutSize>& _dest_encoding;
    const strf::underlying_transcoder<CharInSize, CharOutSize>* _transcoder;
    const strf::encoding_error _enc_err;
    const strf::surrogate_policy _allow_surr;

    template <typename Category, typename CharIn, typename FPack>
    static STRF_HD decltype(auto) _get_facet(const FPack& fp)
    {
        using input_tag = strf::string_input_tag<CharIn>;
        return fp.template get_facet<Category, input_tag>();
    }
};

template<std::size_t CharInSize, std::size_t CharOutSize>
STRF_HD cv_string_printer<CharInSize, CharOutSize>::cv_string_printer
    ( strf::detail::simple_string_view<char_in_type> str
    , const strf::underlying_encoding<CharInSize>& src_enc
    , const strf::underlying_encoding<CharOutSize>& dest_enc
    , strf::encoding_error enc_err
    , strf::surrogate_policy allow_surr ) noexcept
    : _str(str.begin())
    , _len(str.size())
    , _src_encoding(src_enc)
    , _dest_encoding(dest_enc)
    , _transcoder(strf::get_transcoder(src_enc, dest_enc))
    , _enc_err(enc_err)
    , _allow_surr(allow_surr)
{
}

template<std::size_t CharInSize, std::size_t CharOutSize>
STRF_HD std::size_t cv_string_printer<CharInSize, CharOutSize>::necessary_size() const
{
    if (_transcoder) {
        return _transcoder->necessary_size(_str, _str + _len, _allow_surr);
    }
    return strf::decode_encode_size( _str, _str + _len
                                   , _src_encoding, _dest_encoding
                                   , _allow_surr );
}

template<std::size_t CharInSize, std::size_t CharOutSize>
STRF_HD void cv_string_printer<CharInSize, CharOutSize>::print_to
    ( strf::underlying_outbuf<CharOutSize>& ob ) const
{
    if (_transcoder != nullptr) {
        _transcoder->transcode( ob, _str, _str + _len
                                  , _enc_err, _allow_surr );
    } else {
        strf::decode_encode( ob, _str, _str + _len, _src_encoding
                           , _dest_encoding, _enc_err, _allow_surr );
    }
}

template<std::size_t CharInSize, std::size_t CharOutSize>
class fmt_cv_string_printer: public printer<CharOutSize>
{
public:
    using char_in_type = strf::underlying_outbuf_char_type<CharInSize>;

    template <typename FPack, typename Preview, typename CharIn, typename CharOut>
    STRF_HD fmt_cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<CharIn> str
        , strf::alignment_format_data text_alignment
        , strf::encoding<CharIn> src_enc
        , strf::tag<CharOut> ) noexcept
        : _str{ reinterpret_cast<const char_in_type*>(str.begin())
              , str.size() }
        , _afmt(text_alignment)
        , _src_encoding(src_enc.as_underlying())
        , _dest_encoding(_get_facet<strf::encoding_c<CharOut>, CharIn>(fp).as_underlying())
        , _enc_err(_get_facet<strf::encoding_error_c, CharIn>(fp))
        , _allow_surr(_get_facet<strf::surrogate_policy_c, CharIn>(fp))
    {
        _init(preview, _get_facet<strf::width_calculator_c<CharInSize>, CharIn>(fp));
        _calc_size(preview);
    }

    template <typename FPack, typename Preview, typename CharIn, typename CharOut>
    STRF_HD fmt_cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<CharIn> str
        , strf::alignment_format_data text_alignment
        , strf::tag<CharOut> ) noexcept
        : _str{ reinterpret_cast<const char_in_type*>(str.begin())
              , str.size() }
        , _afmt(text_alignment)
        , _src_encoding(_get_facet<strf::encoding_c<CharIn>, CharIn>(fp).as_underlying())
        , _dest_encoding(_get_facet<strf::encoding_c<CharOut>, CharIn>(fp).as_underlying())
        , _enc_err(_get_facet<strf::encoding_error_c, CharIn>(fp))
        , _allow_surr(_get_facet<strf::surrogate_policy_c, CharIn>(fp))
    {
        _init(preview, _get_facet<strf::width_calculator_c<CharInSize>, CharIn>(fp));
        _calc_size(preview);
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharOutSize>& ob) const override;

private:

    strf::detail::simple_string_view<char_in_type> _str;
    strf::alignment_format_data _afmt;
    const strf::underlying_encoding<CharInSize>& _src_encoding;
    const strf::underlying_encoding<CharOutSize>& _dest_encoding;
    const strf::underlying_transcoder<CharInSize, CharOutSize>* _transcoder = nullptr;
    const strf::width_calculator<CharInSize>* _wcalc = nullptr;
    const strf::encoding_error _enc_err;
    const strf::surrogate_policy  _allow_surr;
    std::uint16_t _fillcount = 0;

    template <typename Category, typename CharIn, typename FPack>
    static STRF_HD decltype(auto) _get_facet(const FPack& fp)
    {
        using input_tag = strf::string_input_tag<CharIn>;
        return fp.template get_facet<Category, input_tag>();
    }

    template <bool RequiringWidth>
    STRF_HD void _init( strf::width_preview<RequiringWidth>& );

    template <bool RequiringWidth>
    STRF_HD void _init( strf::width_preview<RequiringWidth>& preview
                      , const strf::fast_width<CharInSize>&)
    {
        _init<RequiringWidth>(preview);
    }

    template <bool RequiringWidth>
    STRF_HD void _init( strf::width_preview<RequiringWidth>& preview
                      , const strf::width_as_u32len<CharInSize>&)
    {
        _init<RequiringWidth>(preview);
    }

    template <bool RequiringWidth>
    STRF_HD void _init( strf::width_preview<RequiringWidth>& preview
                      , const strf::width_calculator<CharInSize>&);

    constexpr void _calc_size(strf::size_preview<false>&) const
    {
    }

    STRF_HD void _calc_size(strf::size_preview<true>& preview) const;

    STRF_HD void _write_str(strf::underlying_outbuf<CharOutSize>& ob) const;

    STRF_HD void _write_fill
        ( strf::underlying_outbuf<CharOutSize>& ob
        , unsigned count ) const;
};

template <std::size_t CharInSize, std::size_t CharOutSize>
template <bool RequiringWidth>
void STRF_HD fmt_cv_string_printer<CharInSize, CharOutSize>::_init
    ( strf::width_preview<RequiringWidth>& preview )
{
    auto limit = std::max( _afmt.width
                         , preview.remaining_width().ceil() );
    auto cp_count = _src_encoding.codepoints_count( _str.begin()
                                                  , _str.end()
                                                  , limit );
    if (_afmt.width > static_cast<std::ptrdiff_t>(cp_count)) {
        _fillcount = _afmt.width - static_cast<std::int16_t>(cp_count);
        preview.subtract_width(_afmt.width);
    } else {
        preview.checked_subtract_width(cp_count);
    }
    _transcoder =
        strf::get_transcoder(_src_encoding, _dest_encoding);
}

template <std::size_t CharInSize, std::size_t CharOutSize>
template <bool RequiringWidth>
void STRF_HD fmt_cv_string_printer<CharInSize, CharOutSize>::_init
    ( strf::width_preview<RequiringWidth>& preview
    , const strf::width_calculator<CharInSize>& wc)
{
    strf::width_t wmax = _afmt.width;
    strf::width_t wdiff = 0;
    if (preview.remaining_width() > _afmt.width) {
        wmax = preview.remaining_width();
        wdiff = preview.remaining_width() - _afmt.width;
    }
    auto str_width = wc.width( wmax
                             , _str.begin(), _str.length()
                             , _src_encoding, _enc_err, _allow_surr );

    strf::width_t fmt_width{_afmt.width};
    if (fmt_width > str_width) {
        auto wfill = (fmt_width - str_width);
        _fillcount = wfill.round();
        preview.subtract_width(wfill + _fillcount);
    } else {
        preview.subtract_width(str_width);
    }
    _transcoder =
        strf::get_transcoder(_src_encoding, _dest_encoding);
}

template<std::size_t CharInSize, std::size_t CharOutSize>
void STRF_HD fmt_cv_string_printer<CharInSize, CharOutSize>::_calc_size
    ( strf::size_preview<true>& preview ) const
{
    std::size_t size;
    if(_transcoder) {
        size = _transcoder->necessary_size(_str.begin(), _str.end(), _allow_surr);
    } else {
        size = strf::decode_encode_size
            ( _str.begin(), _str.end()
            , _src_encoding, _dest_encoding
            , _allow_surr );
    }
    if (_fillcount > 0) {
        size += _fillcount * _dest_encoding.char_size(_afmt.fill);
    }
    preview.add_size(size);
}


template<std::size_t CharInSize, std::size_t CharOutSize>
void STRF_HD fmt_cv_string_printer<CharInSize, CharOutSize>::print_to
    ( strf::underlying_outbuf<CharOutSize>& ob ) const
{
    if (_fillcount > 0) {
        switch(_afmt.alignment) {
            case strf::text_alignment::left: {
                _write_str(ob);
                _write_fill(ob, _fillcount);
                break;
            }
            case strf::text_alignment::center: {
                auto halfcount = _fillcount >> 1;
                _write_fill(ob, halfcount);
                _write_str(ob);
                _write_fill(ob, _fillcount - halfcount);;
                break;
            }
            default:
                _write_fill(ob, _fillcount);
                _write_str(ob);
        }
    } else {
        _write_str(ob);
    }
}


template<std::size_t CharInSize, std::size_t CharOutSize>
void STRF_HD fmt_cv_string_printer<CharInSize, CharOutSize>::_write_str
    ( strf::underlying_outbuf<CharOutSize>& ob ) const
{
    if (_transcoder) {
        _transcoder->transcode( ob, _str.begin(), _str.end()
                              , _enc_err, _allow_surr );
    } else {
        strf::decode_encode( ob
                           , _str.begin(), _str.end()
                           , _src_encoding, _dest_encoding
                           , _enc_err, _allow_surr );
    }
}

template<std::size_t CharInSize, std::size_t CharOutSize>
void STRF_HD fmt_cv_string_printer<CharInSize, CharOutSize>::_write_fill
    ( strf::underlying_outbuf<CharOutSize>& ob
    , unsigned count ) const
{
    _dest_encoding.encode_fill(ob, count, _afmt.fill, _enc_err, _allow_surr);
}

#if defined(STRF_SEPARATE_COMPILATION)

STRF_EXPLICIT_TEMPLATE class cv_string_printer<1, 1>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<1, 2>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<1, 4>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<2, 1>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<2, 2>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<2, 4>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<4, 1>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<4, 2>;
STRF_EXPLICIT_TEMPLATE class cv_string_printer<4, 4>;

STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<1, 1>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<1, 2>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<1, 4>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<2, 1>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<2, 2>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<2, 4>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<4, 1>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<4, 2>;
STRF_EXPLICIT_TEMPLATE class fmt_cv_string_printer<4, 4>;

#endif // defined(STRF_SEPARATE_COMPILATION)

template <typename CharIn, typename CharOut, bool SameSize>
class cv_printer_maker_without_encoding;

template <typename CharT>
class cv_printer_maker_without_encoding<CharT, CharT, true>
{

public:
    template <typename FPack, typename Preview>
    static inline STRF_HD strf::detail::string_printer<sizeof(CharT)> make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharT>
                                 , strf::alignment_format_q<false>
                                 , strf::cv_format<CharT> > input )
    {
        return {fp, preview, input.value(), strf::tag<CharT>()};
    }

    template <typename FPack, typename Preview>
    static inline STRF_HD strf::detail::fmt_string_printer<sizeof(CharT)> make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharT>
                                 , strf::alignment_format_q<true>
                                 , strf::cv_format<CharT> > input )
    {
        return { fp, preview, input.value(), input.get_alignment_format_data()
               , strf::tag<CharT>()};
    }
};

template <typename CharIn, typename CharOut>
class cv_printer_maker_without_encoding<CharIn, CharOut, true>
{
public:
    static_assert(sizeof(CharIn) == sizeof(CharOut), "");
    constexpr static std::size_t char_size = sizeof(CharIn);

    template <typename FPack, typename Preview>
    static inline STRF_HD strf::detail::printer_variant
        < strf::detail::string_printer<char_size>
        , strf::detail::cv_string_printer<char_size, char_size> >
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                 , strf::alignment_format_q<false>
                                 , strf::cv_format<CharIn> > input )
    {
        using enc_cat_in = strf::encoding_c<CharIn>;
        using input_tag_in = strf::string_input_tag<CharIn>;
        const auto& encoding_in = strf::get_facet<enc_cat_in, input_tag_in>(fp);

        using enc_cat_out = strf::encoding_c<CharOut>;
        using input_tag_out = strf::string_input_tag<CharOut>;
        const auto& encoding_out = strf::get_facet<enc_cat_out, input_tag_out>(fp);

        if (encoding_in.id() == encoding_out.id()) {
            return { strf::tag<strf::detail::string_printer<char_size>>()
                   , fp
                   , preview
                   , strf::detail::simple_string_view<CharOut>
                       { reinterpret_cast<const CharOut*>(input.value().begin())
                       , input.value().size() }
                   , strf::tag<CharOut>() };
        } else {
            return { strf::tag<strf::detail::cv_string_printer<char_size, char_size>>()
                   , fp, preview, input.value(), encoding_in, strf::tag<CharOut>() };
        }
    }

    template <typename FPack, typename Preview>
    static inline STRF_HD strf::detail::printer_variant
        < strf::detail::fmt_string_printer<char_size>
        , strf::detail::fmt_cv_string_printer<char_size, char_size> >
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                 , strf::alignment_format_q<true>
                                 , strf::cv_format_with_encoding<CharIn> > input )
    {
        using enc_cat_in = strf::encoding_c<CharIn>;
        using input_tag_in = strf::string_input_tag<CharIn>;
        const auto& encoding_in = strf::get_facet<enc_cat_in, input_tag_in>(fp);

        using enc_cat_out = strf::encoding_c<CharOut>;
        using input_tag_out = strf::string_input_tag<CharOut>;
        const auto& encoding_out = strf::get_facet<enc_cat_out, input_tag_out>(fp);

        if (encoding_in.id() == encoding_out.id()) {
            return { strf::tag<strf::detail::string_printer<char_size>>()
                   , fp
                   , preview
                   , strf::detail::simple_string_view<CharOut>
                       { reinterpret_cast<const CharOut*>(input.value().begin())
                       , input.value().size() }
                   , input.get_alignment_format_data() };
        } else {
            return { strf::tag<strf::detail::cv_string_printer<char_size, char_size>>()
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
    static inline STRF_HD
    strf::detail::cv_string_printer<sizeof(CharIn), sizeof(CharOut)>
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                 , strf::alignment_format_q<false>
                                 , strf::cv_format<CharIn> > input )
    {
        return {fp, preview, input.value(), strf::tag<CharOut>()};
    }

    template <typename FPack, typename Preview>
    static inline STRF_HD
    strf::detail::fmt_cv_string_printer<sizeof(CharIn), sizeof(CharOut)>
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                 , strf::alignment_format_q<true>
                                 , strf::cv_format<CharIn> > input )
    {
        return { fp, preview, input.value(), input.get_alignment_format_data()
               , strf::tag<CharOut>() };
    }
};


template <typename CharIn, typename CharOut, bool SameSize>
class cv_printer_maker_with_encoding;

template <typename CharIn, typename CharOut>
class cv_printer_maker_with_encoding<CharIn, CharOut, true>
{
public:
    static_assert(sizeof(CharIn) == sizeof(CharOut), "");

    template <typename FPack, typename Preview>
    static inline STRF_HD strf::detail::printer_variant
        < strf::detail::string_printer<sizeof(CharOut)>
        , strf::detail::cv_string_printer<sizeof(CharIn), sizeof(CharOut)> >
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                 , strf::alignment_format_q<false>
                                 , strf::cv_format_with_encoding<CharIn> > input )
    {
        using enc_cat = strf::encoding_c<CharOut>;
        using input_tag = strf::string_input_tag<CharOut>;
        const auto& encoding_from_facets
            = strf::get_facet<enc_cat, input_tag>(fp).as_underlying();
        if (input.get_encoding().id() == encoding_from_facets.id) {
            return { strf::tag<strf::detail::string_printer<sizeof(CharOut)>>()
                   , fp
                   , preview
                   , strf::detail::simple_string_view<CharOut>
                       { reinterpret_cast<const CharOut*>(input.value().begin())
                       , input.value().size() }
                   , strf::tag<CharOut>() };
        } else {
            return { strf::tag<strf::detail::cv_string_printer
                                  <sizeof(CharIn), sizeof(CharOut)>>()
                   , fp, preview, input.value(), input.get_encoding()
                   , strf::tag<CharOut>() };
        }
    }

    template <typename FPack, typename Preview>
    static inline STRF_HD strf::detail::printer_variant
        < strf::detail::fmt_string_printer<sizeof(CharOut)>
        , strf::detail::fmt_cv_string_printer<sizeof(CharIn), sizeof(CharOut)> >
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                 , strf::alignment_format_q<true>
                                 , strf::cv_format_with_encoding<CharIn> > input )
    {
        using enc_cat = strf::encoding_c<CharOut>;
        using input_tag = strf::string_input_tag<CharOut>;
        const auto& encoding_from_facets
            = strf::get_facet<enc_cat, input_tag>(fp).as_underlying();
        if (input.get_encoding().id() == encoding_from_facets.id) {
            return { strf::tag<strf::detail::fmt_string_printer<sizeof(CharOut)>>()
                   , fp
                   , preview
                   , strf::detail::simple_string_view<CharOut>
                       { reinterpret_cast<const CharOut*>(input.value().begin())
                       , input.value().size() }
                   , input.get_alignment_format_data()
                   , strf::tag<CharOut>() };
        } else {
            return { strf::tag<strf::detail::fmt_cv_string_printer
                                  <sizeof(CharIn), sizeof(CharOut)>>()
                   , fp
                   , preview
                   , input.value()
                   , input.get_alignment_format_data()
                   , input.get_encoding()
                   , strf::tag<CharOut>() };
        }
    }
};

template <typename CharIn, typename CharOut>
class cv_printer_maker_with_encoding<CharIn, CharOut, false>
{
public:
    template <typename FPack, typename Preview>
    inline STRF_HD
    strf::detail::cv_string_printer<sizeof(CharIn), sizeof(CharOut)>
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                 , strf::alignment_format_q<false>
                                 , strf::cv_format_with_encoding<CharIn> > input )
    {
        return { fp, preview, input.value(), input.get_encoding()
               , strf::tag<CharOut>()};
    }

    template <typename FPack, typename Preview>
    static inline STRF_HD
    strf::detail::fmt_cv_string_printer<sizeof(CharIn), sizeof(CharOut)>
    make_printer
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
               , input.get_encoding()
               , strf::tag<CharOut>() };
    }
};

} // namespace detail

template < typename CharOut
         , typename FPack
         , typename Preview
         , typename CharIn
         , bool WithAlignment >
inline STRF_HD auto make_printer
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
inline STRF_HD auto make_printer
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
inline STRF_HD strf::detail::cv_string_printer<sizeof(CharIn), sizeof(CharOut)>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                     , strf::alignment_format_q<false>
                                     , strf::sani_format<CharIn> > input )
{
    return {fp, preview, input.value(), strf::tag<CharOut>()};
}

template <typename CharOut, typename FPack, typename Preview, typename CharIn>
inline STRF_HD strf::detail::cv_string_printer<sizeof(CharIn), sizeof(CharOut)>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , strf::value_with_format< strf::detail::simple_string_view<CharIn>
                                     , strf::alignment_format_q<false>
                                     , strf::sani_format_with_encoding<CharIn> > input )
{
    return { fp, preview, input.value(), input.get_encoding()
           , strf::tag<CharOut>() };
}

template <typename CharOut, typename FPack, typename Preview, typename CharIn>
inline STRF_HD strf::detail::fmt_cv_string_printer<sizeof(CharIn), sizeof(CharOut)>
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
           , input.get_alignment_format_data()
           , strf::tag<CharOut>() };
}

template <typename CharOut, typename FPack, typename Preview, typename CharIn>
inline STRF_HD strf::detail::fmt_cv_string_printer<sizeof(CharIn), sizeof(CharOut)>
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
           , input.get_encoding()
           , strf::tag<CharOut>() };
}

} // namespace strf

#endif  // STRF_DETAIL_INPUT_TYPES_CV_STRING_HPP

