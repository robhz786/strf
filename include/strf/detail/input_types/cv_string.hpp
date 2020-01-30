#ifndef STRF_DETAIL_INPUT_TYPES_CV_STRING_HPP
#define STRF_DETAIL_INPUT_TYPES_CV_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/input_types/string.hpp>
#include <strf/detail/printer_variant.hpp>

namespace strf {

namespace detail {

template<std::size_t SrcCharSize, std::size_t DestCharSize>
class cv_string_printer: public strf::printer<DestCharSize>
{
public:

    using char_in_type = strf::underlying_outbuf_char_type<SrcCharSize>;

    template < typename FPack, typename Preview, typename SrcChar
             , typename DestChar, typename SrcEncoding >
    STRF_HD cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<SrcChar> str
        , const SrcEncoding& src_enc
        , strf::tag<DestChar> ) noexcept
        : _str(reinterpret_cast<const char_in_type*>(str.begin()))
        , _len(str.size())
        , _enc_err(_get_facet<strf::encoding_error_c, SrcChar>(fp))
        , _allow_surr(_get_facet<strf::surrogate_policy_c, SrcChar>(fp))
    {
        static_assert(sizeof(SrcChar) == SrcCharSize, "Incompatible char type");
        static_assert(sizeof(DestChar) == DestCharSize, "Incompatible char type");

        init_( preview
             , _get_facet<strf::width_calculator_c, SrcChar>(fp)
             , src_enc
             , _get_facet<strf::encoding_c<DestChar>, SrcChar>(fp) );
    }

    template <typename FPack, typename Preview, typename SrcChar, typename DestChar>
    STRF_HD cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<SrcChar> str
        , strf::tag<DestChar> char_tag ) noexcept
        : cv_string_printer
              ( fp, preview, str
              , _get_facet<strf::encoding_c<SrcChar>, SrcChar>(fp), char_tag )
    {
    }

    STRF_HD ~cv_string_printer() { }

    STRF_HD void print_to(strf::underlying_outbuf<DestCharSize>& ob) const override;

private:

    template < typename Preview, typename WCalc
             , typename SrcEncoding, typename DestEncoding >
    STRF_HD void init_
        ( Preview& preview, const WCalc& wcalc
        , const SrcEncoding& src_enc, const DestEncoding& dest_enc )
    {
        (void) wcalc;

        static_assert(SrcEncoding::char_size == SrcCharSize, "Incompatible char type");
        static_assert(DestEncoding::char_size == DestCharSize, "Incompatible char type");
        decltype(auto) transcoder = get_transcoder(src_enc, dest_enc);
        _transcode = transcoder.transcode;
        if (_transcode == nullptr) {
            _src_to_u32 = src_enc.to_u32().transcode;
            _u32_to_dest = dest_enc.from_u32().transcode;
        }
        STRF_IF_CONSTEXPR (Preview::width_required) {
            auto w = wcalc.width( src_enc, preview.remaining_width(), _str, _len
                                , _enc_err, _allow_surr );
            preview.subtract_width(w);
        }
        STRF_IF_CONSTEXPR (Preview::size_required) {
            strf::transcode_size_func<SrcCharSize>  transcode_size
                = transcoder.necessary_size;
            std::size_t s = 0;
            if (transcode_size != nullptr) {
                s = transcode_size(_str, _str + _len, _allow_surr);
            } else {
                s = strf::decode_encode_size<SrcCharSize>
                    ( src_enc.to_u32().transcode
                    , dest_enc.from_u32().necessary_size
                    , _str, _str + _len, _enc_err, _allow_surr );
            }
            preview.add_size(s);
        }
    }

    STRF_HD bool can_transcode_directly() const
    {
        return _u32_to_dest == nullptr;
    }

    const char_in_type* const _str;
    const std::size_t _len;
    union {
        strf::transcode_func<SrcCharSize, DestCharSize>  _transcode;
        strf::transcode_func<SrcCharSize, 4>  _src_to_u32;
    };
    strf::transcode_func<4, DestCharSize>  _u32_to_dest = nullptr;
    const strf::encoding_error _enc_err;
    const strf::surrogate_policy _allow_surr;
    template <typename Category, typename SrcChar, typename FPack>
    static STRF_HD decltype(auto) _get_facet(const FPack& fp)
    {
        using input_tag = strf::string_input_tag<SrcChar>;
        return fp.template get_facet<Category, input_tag>();
    }
};

template<std::size_t SrcCharSize, std::size_t DestCharSize>
STRF_HD void cv_string_printer<SrcCharSize, DestCharSize>::print_to
    ( strf::underlying_outbuf<DestCharSize>& ob ) const
{
    if (can_transcode_directly()) {
        _transcode(ob, _str, _str + _len, _enc_err, _allow_surr);
    } else {
        strf::decode_encode<SrcCharSize, DestCharSize>
            ( ob, _src_to_u32, _u32_to_dest, _str
            , _str + _len, _enc_err, _allow_surr );
    }
}

template<std::size_t SrcCharSize, std::size_t DestCharSize>
class fmt_cv_string_printer: public printer<DestCharSize>
{
public:
    using char_in_type = strf::underlying_outbuf_char_type<SrcCharSize>;

    template < typename FPack, typename Preview, typename SrcChar
             , typename DestChar, typename SrcEncoding >
    STRF_HD fmt_cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<SrcChar> str
        , strf::alignment_format_data text_alignment
        , const SrcEncoding& src_enc
        , strf::tag<DestChar> ) noexcept
        : _str(reinterpret_cast<const char_in_type*>(str.begin()))
        , _len(str.size())
        , _afmt(text_alignment)
        , _enc_err(_get_facet<strf::encoding_error_c, SrcChar>(fp))
        , _allow_surr(_get_facet<strf::surrogate_policy_c, SrcChar>(fp))
    {
        static_assert(sizeof(SrcChar) == SrcCharSize, "Incompatible char type");
        static_assert(sizeof(DestChar) == DestCharSize, "Incompatible char type");
        init_( preview
             , _get_facet<strf::width_calculator_c, SrcChar>(fp)
             , src_enc
             , _get_facet<strf::encoding_c<DestChar>, SrcChar>(fp) );
    }

    template <typename FPack, typename Preview, typename SrcChar, typename DestChar>
    STRF_HD fmt_cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<SrcChar> str
        , strf::alignment_format_data text_alignment
        , strf::tag<DestChar> ch_tag ) noexcept
        : fmt_cv_string_printer
            ( fp, preview, str, text_alignment
            , _get_facet<strf::encoding_c<SrcChar>, SrcChar>(fp), ch_tag )
    {
    }

    STRF_HD void print_to(strf::underlying_outbuf<DestCharSize>& ob) const override;

private:

    STRF_HD bool can_transcode_directly() const
    {
        return _u32_to_dest == nullptr;
    }

    const char_in_type* _str;
    std::size_t _len;
    strf::alignment_format_data _afmt;
    union {
        strf::transcode_func<SrcCharSize, DestCharSize>  _transcode;
        strf::transcode_func<SrcCharSize, 4>  _src_to_u32;
    };
    strf::transcode_func<4, DestCharSize>  _u32_to_dest = nullptr;
    strf::encode_fill_func<DestCharSize> _encode_fill = nullptr;
    const strf::encoding_error _enc_err;
    const strf::surrogate_policy  _allow_surr;
    std::uint16_t _left_fillcount = 0;
    std::uint16_t _right_fillcount = 0;

    template <typename Category, typename SrcChar, typename FPack>
    static STRF_HD decltype(auto) _get_facet(const FPack& fp)
    {
        using input_tag = strf::string_input_tag<SrcChar>;
        return fp.template get_facet<Category, input_tag>();
    }

    template < typename Preview, typename WCalc
             , typename SrcEnc, typename DestEnc>
    STRF_HD void init_
        ( Preview& preview, const WCalc& wcalc
        , const SrcEnc& src_enc, const DestEnc& dest_enc );
};

template <std::size_t SrcCharSize, std::size_t DestCharSize>
template <typename Preview, typename WCalc, typename SrcEnc, typename DestEnc>
void STRF_HD fmt_cv_string_printer<SrcCharSize, DestCharSize>::init_
    ( Preview& preview, const WCalc& wcalc
    , const SrcEnc& src_enc, const DestEnc& dest_enc )
{
    static_assert(SrcEnc::char_size == SrcCharSize, "Incompatible char type");
    static_assert(DestEnc::char_size == DestCharSize, "Incompatible char type");

    _encode_fill = dest_enc.encode_fill;
    decltype(auto) transcoder = get_transcoder(src_enc, dest_enc);
    _transcode = transcoder.transcode;
    if (_transcode == nullptr) {
        _src_to_u32 = src_enc.to_u32().transcode;
        _u32_to_dest = dest_enc.from_u32().transcode;
    }
    std::uint16_t fillcount = 0;
    strf::width_t fmt_width = _afmt.width;
    strf::width_t limit =
        ( Preview::width_required && preview.remaining_width() > fmt_width
        ? preview.remaining_width()
        : fmt_width );
    auto strw = wcalc.width(src_enc, limit, _str, _len , _enc_err, _allow_surr);
    if (fmt_width > strw) {
        fillcount = (fmt_width - strw).round();
        switch(_afmt.alignment) {
            case strf::text_alignment::left:
                _left_fillcount = 0;
                _right_fillcount = fillcount;
                break;
            case strf::text_alignment::center: {
                std::uint16_t halfcount = fillcount / 2;
                _left_fillcount = halfcount;
                _right_fillcount = fillcount - halfcount;
                break;
            }
            default:
                _left_fillcount = fillcount;
                _right_fillcount = 0;
        }
        preview.subtract_width(strw + fillcount);
    } else {
        _right_fillcount = 0;
        _left_fillcount = 0;
        preview.subtract_width(strw);
    }
    STRF_IF_CONSTEXPR (Preview::size_required) {
        std::size_t s = 0;
        strf::transcode_size_func<SrcCharSize>  transcode_size
                = transcoder.necessary_size;
        if (transcode_size != nullptr) {
            s = transcode_size(_str, _str + _len, _allow_surr);
        } else {
            s = strf::decode_encode_size<SrcCharSize>
                ( src_enc.to_u32().transcode, dest_enc.from_u32().necessary_size
                , _str, _str + _len, _enc_err, _allow_surr );
        }
        if (fillcount > 0) {
            s += dest_enc.encoded_char_size(_afmt.fill) * fillcount;
        }
        preview.add_size(s);
    }
}

template<std::size_t SrcCharSize, std::size_t DestCharSize>
void STRF_HD fmt_cv_string_printer<SrcCharSize, DestCharSize>::print_to
    ( strf::underlying_outbuf<DestCharSize>& ob ) const
{
    if (_left_fillcount > 0) {
        _encode_fill(ob, _left_fillcount, _afmt.fill, _enc_err, _allow_surr);
    }
    if (can_transcode_directly()) {
        _transcode(ob, _str, _str + _len, _enc_err, _allow_surr);
    } else {
        strf::decode_encode<SrcCharSize, DestCharSize>
            ( ob, _src_to_u32, _u32_to_dest, _str
            , _str + _len, _enc_err, _allow_surr );
    }
    if (_right_fillcount > 0) {
        _encode_fill(ob, _right_fillcount, _afmt.fill, _enc_err, _allow_surr);
    }
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

template <typename SrcChar, typename DestChar, bool SameSize>
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

template <typename SrcChar, typename DestChar>
class cv_printer_maker_without_encoding<SrcChar, DestChar, true>
{
public:
    static_assert(sizeof(SrcChar) == sizeof(DestChar), "");
    constexpr static std::size_t char_size = sizeof(SrcChar);

    template <typename FPack, typename Preview>
    static inline STRF_HD strf::detail::printer_variant
        < strf::detail::string_printer<char_size>
        , strf::detail::cv_string_printer<char_size, char_size> >
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<SrcChar>
                                 , strf::alignment_format_q<false>
                                 , strf::cv_format<SrcChar> > input )
    {
        using enc_cat_in = strf::encoding_c<SrcChar>;
        using input_tag_in = strf::string_input_tag<SrcChar>;
        const auto& encoding_in = strf::get_facet<enc_cat_in, input_tag_in>(fp);

        using enc_cat_out = strf::encoding_c<DestChar>;
        using input_tag_out = strf::string_input_tag<DestChar>;
        const auto& encoding_out = strf::get_facet<enc_cat_out, input_tag_out>(fp);

        if (encoding_in.id() == encoding_out.id()) {
            return { strf::tag<strf::detail::string_printer<char_size>>()
                   , fp
                   , preview
                   , strf::detail::simple_string_view<DestChar>
                       { reinterpret_cast<const DestChar*>(input.value().begin())
                       , input.value().size() }
                   , strf::tag<DestChar>() };
        } else {
            return { strf::tag<strf::detail::cv_string_printer<char_size, char_size>>()
                   , fp, preview, input.value(), encoding_in, strf::tag<DestChar>() };
        }
    }

    template <typename FPack, typename Preview, typename SrcEncoding>
    static inline STRF_HD strf::detail::printer_variant
        < strf::detail::fmt_string_printer<char_size>
        , strf::detail::fmt_cv_string_printer<char_size, char_size> >
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format
            < strf::detail::simple_string_view<SrcChar>
            , strf::alignment_format_q<true>
            , strf::cv_format_with_encoding<SrcChar, SrcEncoding > > input )
    {
        using enc_cat_in = strf::encoding_c<SrcChar>;
        using input_tag_in = strf::string_input_tag<SrcChar>;
        const auto& encoding_in = strf::get_facet<enc_cat_in, input_tag_in>(fp);

        using enc_cat_out = strf::encoding_c<DestChar>;
        using input_tag_out = strf::string_input_tag<DestChar>;
        const auto& encoding_out = strf::get_facet<enc_cat_out, input_tag_out>(fp);

        if (encoding_in.id() == encoding_out.id()) {
            return { strf::tag<strf::detail::string_printer<char_size>>()
                   , fp
                   , preview
                   , strf::detail::simple_string_view<DestChar>
                       { reinterpret_cast<const DestChar*>(input.value().begin())
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

template <typename SrcChar, typename DestChar>
class cv_printer_maker_without_encoding<SrcChar, DestChar, false>
{
public:

    template <typename FPack, typename Preview>
    static inline STRF_HD
    strf::detail::cv_string_printer<sizeof(SrcChar), sizeof(DestChar)>
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<SrcChar>
                                 , strf::alignment_format_q<false>
                                 , strf::cv_format<SrcChar> > input )
    {
        return {fp, preview, input.value(), strf::tag<DestChar>()};
    }

    template <typename FPack, typename Preview>
    static inline STRF_HD
    strf::detail::fmt_cv_string_printer<sizeof(SrcChar), sizeof(DestChar)>
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<SrcChar>
                                 , strf::alignment_format_q<true>
                                 , strf::cv_format<SrcChar> > input )
    {
        return { fp, preview, input.value(), input.get_alignment_format_data()
               , strf::tag<DestChar>() };
    }
};


template <typename SrcChar, typename DestChar, bool SameSize>
class cv_printer_maker_with_encoding;

template <typename SrcChar, typename DestChar>
class cv_printer_maker_with_encoding<SrcChar, DestChar, true>
{
public:
    static_assert(sizeof(SrcChar) == sizeof(DestChar), "");

    template <typename FPack, typename Preview, typename SrcEncoding>
    static inline STRF_HD strf::detail::printer_variant
        < strf::detail::string_printer<sizeof(DestChar)>
        , strf::detail::cv_string_printer<sizeof(SrcChar), sizeof(DestChar)> >
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format
            < strf::detail::simple_string_view<SrcChar>
            , strf::alignment_format_q<false>
            , strf::cv_format_with_encoding<SrcChar, SrcEncoding> > input )
    {
        using enc_cat = strf::encoding_c<DestChar>;
        using input_tag = strf::string_input_tag<DestChar>;
        const auto& encoding_from_facets
            = strf::get_facet<enc_cat, input_tag>(fp);
        if (input.get_encoding().id() == encoding_from_facets.id()) {
            return { strf::tag<strf::detail::string_printer<sizeof(DestChar)>>()
                   , fp
                   , preview
                   , strf::detail::simple_string_view<DestChar>
                       { reinterpret_cast<const DestChar*>(input.value().begin())
                       , input.value().size() }
                   , strf::tag<DestChar>() };
        } else {
            return { strf::tag<strf::detail::cv_string_printer
                                  <sizeof(SrcChar), sizeof(DestChar)>>()
                   , fp, preview, input.value(), input.get_encoding()
                   , strf::tag<DestChar>() };
        }
    }

    template <typename FPack, typename Preview, typename SrcEncoding>
    static inline STRF_HD strf::detail::printer_variant
        < strf::detail::fmt_string_printer<sizeof(DestChar)>
        , strf::detail::fmt_cv_string_printer<sizeof(SrcChar), sizeof(DestChar)> >
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format
            < strf::detail::simple_string_view<SrcChar>
            , strf::alignment_format_q<true>
            , strf::cv_format_with_encoding<SrcChar, SrcEncoding> > input )
    {
        using enc_cat = strf::encoding_c<DestChar>;
        using input_tag = strf::string_input_tag<DestChar>;
        const auto& encoding_from_facets
            = strf::get_facet<enc_cat, input_tag>(fp);
        if (input.get_encoding().id() == encoding_from_facets.id()) {
            return { strf::tag<strf::detail::fmt_string_printer<sizeof(DestChar)>>()
                   , fp
                   , preview
                   , strf::detail::simple_string_view<DestChar>
                       { reinterpret_cast<const DestChar*>(input.value().begin())
                       , input.value().size() }
                   , input.get_alignment_format_data()
                   , strf::tag<DestChar>() };
        } else {
            return { strf::tag<strf::detail::fmt_cv_string_printer
                                  <sizeof(SrcChar), sizeof(DestChar)>>()
                   , fp
                   , preview
                   , input.value()
                   , input.get_alignment_format_data()
                   , input.get_encoding()
                   , strf::tag<DestChar>() };
        }
    }
};

template <typename SrcChar, typename DestChar>
class cv_printer_maker_with_encoding<SrcChar, DestChar, false>
{
public:
    template <typename FPack, typename Preview, typename SrcEncoding>
    inline STRF_HD
    strf::detail::cv_string_printer<sizeof(SrcChar), sizeof(DestChar)>
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format
            < strf::detail::simple_string_view<SrcChar>
            , strf::alignment_format_q<false>
            , strf::cv_format_with_encoding<SrcChar, SrcEncoding> > input )
    {
        return { fp, preview, input.value(), input.get_encoding()
               , strf::tag<DestChar>()};
    }

    template <typename FPack, typename Preview, typename SrcEncoding>
    static inline STRF_HD
    strf::detail::fmt_cv_string_printer<sizeof(SrcChar), sizeof(DestChar)>
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format
            < strf::detail::simple_string_view<SrcChar>
            , strf::alignment_format_q<true>
            , strf::cv_format_with_encoding<SrcChar, SrcEncoding> > input )
    {
        return { fp
               , preview
               , input.value()
               , input.get_alignment_format_data()
               , input.get_encoding()
               , strf::tag<DestChar>() };
    }
};

} // namespace detail

template < typename DestChar, typename FPack, typename Preview
         , typename SrcChar, bool WithAlignment >
inline STRF_HD auto make_printer
    ( strf::rank<1>
    , const FPack& fp
    , Preview& preview
    , strf::value_with_format< strf::detail::simple_string_view<SrcChar>
                             , strf::alignment_format_q<WithAlignment>
                             , strf::cv_format<SrcChar> > input )
{
    constexpr bool ss = sizeof(SrcChar) == sizeof(DestChar);
    return strf::detail::cv_printer_maker_without_encoding<SrcChar, DestChar, ss>
        :: make_printer(fp, preview, input);
}

template < typename DestChar, typename FPack, typename Preview
         , typename SrcChar, typename SrcEncoding, bool WithAlignment >
inline STRF_HD auto make_printer
    ( strf::rank<1>
    , const FPack& fp
    , Preview& preview
    , strf::value_with_format
        < strf::detail::simple_string_view<SrcChar>
        , strf::alignment_format_q<WithAlignment>
        , strf::cv_format_with_encoding<SrcChar, SrcEncoding> > input )
{
    constexpr bool ss = sizeof(SrcChar) == sizeof(DestChar);
    return strf::detail::cv_printer_maker_with_encoding<SrcChar, DestChar, ss>
        :: make_printer(fp, preview, input);
}

template <typename DestChar, typename FPack, typename Preview, typename SrcChar>
inline STRF_HD strf::detail::cv_string_printer<sizeof(SrcChar), sizeof(DestChar)>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , strf::value_with_format< strf::detail::simple_string_view<SrcChar>
                                     , strf::alignment_format_q<false>
                                     , strf::sani_format<SrcChar> > input )
{
    return {fp, preview, input.value(), strf::tag<DestChar>()};
}

template < typename DestChar, typename FPack, typename Preview
         , typename SrcChar, typename SrcEncoding >
inline STRF_HD strf::detail::cv_string_printer<sizeof(SrcChar), sizeof(DestChar)>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , strf::value_with_format
                < strf::detail::simple_string_view<SrcChar>
                , strf::alignment_format_q<false>
                , strf::sani_format_with_encoding<SrcChar, SrcEncoding> > input )
{
    return { fp, preview, input.value(), input.get_encoding()
           , strf::tag<DestChar>() };
}

template <typename DestChar, typename FPack, typename Preview, typename SrcChar>
inline STRF_HD strf::detail::fmt_cv_string_printer<sizeof(SrcChar), sizeof(DestChar)>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , strf::value_with_format
                < strf::detail::simple_string_view<SrcChar>
                , strf::alignment_format_q<true>
                , strf::sani_format<SrcChar> > input )
{
    return { fp
           , preview
           , input.value()
           , input.get_alignment_format_data()
           , strf::tag<DestChar>() };
}

template < typename DestChar, typename FPack, typename Preview
         , typename SrcChar, typename SrcEncoding >
inline STRF_HD strf::detail::fmt_cv_string_printer<sizeof(SrcChar), sizeof(DestChar)>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , strf::value_with_format
                < strf::detail::simple_string_view<SrcChar>
                , strf::alignment_format_q<true>
                , strf::sani_format_with_encoding<SrcChar, SrcEncoding> > input )
{
    return { fp
           , preview
           , input.value()
           , input.get_alignment_format_data()
           , input.get_encoding()
           , strf::tag<DestChar>() };
}

} // namespace strf

#endif  // STRF_DETAIL_INPUT_TYPES_CV_STRING_HPP

