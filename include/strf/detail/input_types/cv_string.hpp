#ifndef STRF_DETAIL_INPUT_TYPES_CV_STRING_HPP
#define STRF_DETAIL_INPUT_TYPES_CV_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/input_types/string.hpp>

namespace strf {
namespace detail {

template < typename DestCharT, typename FPack, typename Preview
         , typename SrcCharT, bool HasP, bool HasA, typename SrcEncoding >
constexpr STRF_HD decltype(auto) get_src_encoding
    ( const strf::detail::fmt_string_printer_input
        < DestCharT, FPack, Preview, SrcCharT, HasP, HasA
        , strf::cv_format_with_encoding<SrcCharT, SrcEncoding> >& input )
{
    return input.vwf.get_encoding();
}

template < typename DestCharT, typename FPack, typename Preview
         , typename SrcCharT, bool HasP, bool HasA, typename SrcEncoding >
constexpr STRF_HD decltype(auto) get_src_encoding
    ( const strf::detail::fmt_string_printer_input
        < DestCharT, FPack, Preview, SrcCharT, HasP, HasA
        , strf::sani_format_with_encoding<SrcCharT, SrcEncoding> >& input )
{
    return input.vwf.get_encoding();
}

template < typename DestCharT, typename FPack, typename Preview
         , typename SrcCharT, bool HasP, bool HasA >
constexpr STRF_HD decltype(auto) get_src_encoding
    ( const strf::detail::fmt_string_printer_input
        < DestCharT, FPack, Preview, SrcCharT, HasP, HasA
        , strf::cv_format<SrcCharT> >& input )
{
    return strf::get_facet
        <strf::char_encoding_c<SrcCharT>, strf::string_input_tag<SrcCharT>>
        ( input.fp );
}

template < typename DestCharT, typename FPack, typename Preview
         , typename SrcCharT, bool HasP, bool HasA >
constexpr STRF_HD decltype(auto) get_src_encoding
    ( const strf::detail::fmt_string_printer_input
        < DestCharT, FPack, Preview, SrcCharT, HasP, HasA
        , strf::sani_format<SrcCharT> >& input )
{
    return strf::get_facet
        <strf::char_encoding_c<SrcCharT>, strf::string_input_tag<SrcCharT>>
        ( input.fp );
}

template<std::size_t SrcCharSize, std::size_t DestCharSize>
class cv_string_printer: public strf::printer<DestCharSize>
{
public:

    using char_in_type = strf::underlying_char_type<SrcCharSize>;

    template < typename DestCharT, typename FPack, typename Preview
             , typename SrcCharT, typename CvFormat >
    STRF_HD cv_string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, FPack, Preview, SrcCharT, false, false, CvFormat >&
            input )
        : str_(reinterpret_cast<const char_in_type*>(input.vwf.value().data()))
        , len_(input.vwf.value().size())
        , inv_seq_poli_(get_facet_<strf::invalid_seq_policy_c, SrcCharT>(input.fp))
        , surr_poli_(get_facet_<strf::surrogate_policy_c, SrcCharT>(input.fp))
    {
        static_assert(sizeof(SrcCharT) == SrcCharSize, "Incompatible char type");
        static_assert(sizeof(DestCharT) == DestCharSize, "Incompatible char type");

        auto src_enc  = strf::detail::get_src_encoding(input);
        auto dest_enc = get_facet_<strf::char_encoding_c<DestCharT>, SrcCharT>(input.fp);
        STRF_IF_CONSTEXPR (Preview::width_required) {
            decltype(auto) wcalc = get_facet_<strf::width_calculator_c, SrcCharT>(input.fp);
            auto w = wcalc.str_width( src_enc, input.preview.remaining_width()
                                    , str_, len_, surr_poli_);
            input.preview.subtract_width(w);
        }
        init_(input.preview, src_enc, dest_enc);
    }

   template < typename DestCharT, typename FPack, typename Preview
             , typename SrcCharT, typename CvFormat >
    STRF_HD cv_string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, FPack, Preview, SrcCharT, true, false, CvFormat >&
            input )
        : str_(reinterpret_cast<const char_in_type*>(input.vwf.value().data()))
        , inv_seq_poli_(get_facet_<strf::invalid_seq_policy_c, SrcCharT>(input.fp))
        , surr_poli_(get_facet_<strf::surrogate_policy_c, SrcCharT>(input.fp))
    {
        static_assert(sizeof(SrcCharT) == SrcCharSize, "Incompatible char type");
        static_assert(sizeof(DestCharT) == DestCharSize, "Incompatible char type");

        auto src_enc  = strf::detail::get_src_encoding(input);
        auto dest_enc = get_facet_<strf::char_encoding_c<DestCharT>, SrcCharT>(input.fp);
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c, SrcCharT>(input.fp);
        auto res = wcalc.str_width_and_pos
            ( src_enc, input.vwf.precision(), str_
            , input.vwf.value().size(), surr_poli_ );
        len_ = res.pos;
        input.preview.subtract_width(res.width);
        init_( input.preview, src_enc
             , get_facet_<strf::char_encoding_c<DestCharT>, SrcCharT>(input.fp));
    }

    STRF_HD ~cv_string_printer() { }

    STRF_HD void print_to(strf::underlying_outbuf<DestCharSize>& ob) const override;

private:

    template < typename Preview, typename SrcEncoding, typename DestEncoding >
    STRF_HD void init_(Preview& preview, SrcEncoding src_enc, DestEncoding dest_enc)
    {
        static_assert(SrcEncoding::char_size == SrcCharSize, "Incompatible char type");
        static_assert(DestEncoding::char_size == DestCharSize, "Incompatible char type");

        decltype(auto) transcoder = find_transcoder(src_enc, dest_enc);
        transcode_ = transcoder.transcode_func();
        if (transcode_ == nullptr) {
            src_to_u32_ = src_enc.to_u32().transcode_func();
            u32_to_dest_ = dest_enc.from_u32().transcode_func();
        }
        STRF_IF_CONSTEXPR (Preview::size_required) {
            strf::transcode_size_f<SrcCharSize>  transcode_size
                = transcoder.transcode_size_func();
            std::size_t s = 0;
            if (transcode_size != nullptr) {
                s = transcode_size(str_, len_, surr_poli_);
            } else {
                s = strf::decode_encode_size<SrcCharSize>
                    ( src_enc.to_u32().transcode_func()
                    , dest_enc.from_u32().transcode_size_func()
                    , str_, len_, inv_seq_poli_, surr_poli_ );
            }
            preview.add_size(s);
        }
    }

    STRF_HD bool can_transcode_directly() const
    {
        return u32_to_dest_ == nullptr;
    }

    const char_in_type* const str_;
    std::size_t len_;
    union {
        strf::transcode_f<SrcCharSize, DestCharSize>  transcode_;
        strf::transcode_f<SrcCharSize, 4>  src_to_u32_;
    };
    strf::transcode_f<4, DestCharSize>  u32_to_dest_ = nullptr;
    const strf::invalid_seq_policy inv_seq_poli_;
    const strf::surrogate_policy surr_poli_;
    template <typename Category, typename SrcChar, typename FPack>
    static STRF_HD decltype(auto) get_facet_(const FPack& fp)
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
        transcode_(ob, str_, len_, inv_seq_poli_, surr_poli_);
    } else {
        strf::decode_encode<SrcCharSize, DestCharSize>
            ( ob, src_to_u32_, u32_to_dest_, str_
            , len_, inv_seq_poli_, surr_poli_ );
    }
}

template<std::size_t SrcCharSize, std::size_t DestCharSize>
class aligned_cv_string_printer: public printer<DestCharSize>
{
public:
    using char_in_type = strf::underlying_char_type<SrcCharSize>;

    template < typename DestCharT, typename FPack, typename Preview
             , typename SrcCharT, typename CvFormat >
    aligned_cv_string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, FPack, Preview, SrcCharT, false, true, CvFormat >&
            input )
        : str_(reinterpret_cast<const char_in_type*>(input.vwf.value().data()))
        , len_(input.vwf.value().size())
        , afmt_(input.vwf.get_alignment_format_data())
        , inv_seq_poli_(get_facet_<strf::invalid_seq_policy_c, SrcCharT>(input.fp))
        , surr_poli_(get_facet_<strf::surrogate_policy_c, SrcCharT>(input.fp))
    {
        static_assert(sizeof(SrcCharT) == SrcCharSize, "Incompatible char type");
        static_assert(sizeof(DestCharT) == DestCharSize, "Incompatible char type");

        auto src_enc = strf::detail::get_src_encoding(input);
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c, SrcCharT>(input.fp);
        strf::width_t limit =
            ( Preview::width_required && input.preview.remaining_width() > afmt_.width
            ? input.preview.remaining_width()
            : afmt_.width );
        auto str_width = wcalc.str_width(src_enc, limit, str_, len_, surr_poli_);
        init_( input.preview, str_width, src_enc
             , get_facet_<strf::char_encoding_c<DestCharT>, SrcCharT>(input.fp) );
    }

    template < typename DestCharT, typename FPack, typename Preview
             , typename SrcCharT, typename CvFormat >
    aligned_cv_string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, FPack, Preview, SrcCharT, true, true, CvFormat >&
            input )
        : str_(reinterpret_cast<const char_in_type*>(input.vwf.value().data()))
        , len_(input.vwf.value().size())
        , afmt_(input.vwf.get_alignment_format_data())
        , inv_seq_poli_(get_facet_<strf::invalid_seq_policy_c, SrcCharT>(input.fp))
        , surr_poli_(get_facet_<strf::surrogate_policy_c, SrcCharT>(input.fp))
    {
        static_assert(sizeof(SrcCharT) == SrcCharSize, "Incompatible char type");
        static_assert(sizeof(DestCharT) == DestCharSize, "Incompatible char type");

        auto src_enc = strf::detail::get_src_encoding(input);
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c, SrcCharT>(input.fp);
        auto res = wcalc.str_width_and_pos
            ( src_enc, input.vwf.precision(), str_
            , input.vwf.value().size(), surr_poli_ );
        len_ = res.pos;
        init_( input.preview, res.width, src_enc
             , get_facet_<strf::char_encoding_c<DestCharT>, SrcCharT>(input.fp) );
    }

    STRF_HD void print_to(strf::underlying_outbuf<DestCharSize>& ob) const override;

private:

    STRF_HD bool can_transcode_directly() const
    {
        return u32_to_dest_ == nullptr;
    }

    const char_in_type* str_;
    std::size_t len_;
    strf::alignment_format_data afmt_;
    union {
        strf::transcode_f<SrcCharSize, DestCharSize>  transcode_;
        strf::transcode_f<SrcCharSize, 4>  src_to_u32_;
    };
    strf::transcode_f<4, DestCharSize>  u32_to_dest_ = nullptr;
    strf::encode_fill_f<DestCharSize> encode_fill_ = nullptr;
    const strf::invalid_seq_policy inv_seq_poli_;
    const strf::surrogate_policy  surr_poli_;
    std::uint16_t left_fillcount_ = 0;
    std::uint16_t right_fillcount_ = 0;

    template <typename Category, typename SrcChar, typename FPack>
    static STRF_HD decltype(auto) get_facet_(const FPack& fp)
    {
        using input_tag = strf::string_input_tag<SrcChar>;
        return fp.template get_facet<Category, input_tag>();
    }

    template < typename Preview, typename SrcEncoding, typename DestEncoding>
    STRF_HD void init_
        ( Preview& preview, strf::width_t str_width
        , SrcEncoding src_enc, DestEncoding dest_enc );
};

template <std::size_t SrcCharSize, std::size_t DestCharSize>
template <typename Preview, typename SrcEncoding, typename DestEncoding>
void STRF_HD aligned_cv_string_printer<SrcCharSize, DestCharSize>::init_
    ( Preview& preview, strf::width_t str_width
    , SrcEncoding src_enc, DestEncoding dest_enc )
{
    static_assert(SrcEncoding::char_size == SrcCharSize, "Incompatible char type");
    static_assert(DestEncoding::char_size == DestCharSize, "Incompatible char type");

    encode_fill_ = dest_enc.encode_fill_func();
    decltype(auto) transcoder = find_transcoder(src_enc, dest_enc);
    transcode_ = transcoder.transcode_func();
    if (transcode_ == nullptr) {
        src_to_u32_ = src_enc.to_u32().transcode_func();
        u32_to_dest_ = dest_enc.from_u32().transcode_func();
    }
    std::uint16_t fillcount = 0;
    if (afmt_.width > str_width) {
        fillcount = (afmt_.width - str_width).round();
        switch(afmt_.alignment) {
            case strf::text_alignment::left:
                left_fillcount_ = 0;
                right_fillcount_ = fillcount;
                break;
            case strf::text_alignment::center: {
                std::uint16_t halfcount = fillcount / 2;
                left_fillcount_ = halfcount;
                right_fillcount_ = fillcount - halfcount;
                break;
            }
            default:
                left_fillcount_ = fillcount;
                right_fillcount_ = 0;
        }
        preview.subtract_width(str_width + fillcount);
    } else {
        right_fillcount_ = 0;
        left_fillcount_ = 0;
        preview.subtract_width(str_width);
    }
    STRF_IF_CONSTEXPR (Preview::size_required) {
        std::size_t s = 0;
        strf::transcode_size_f<SrcCharSize> transcode_size
                = transcoder.transcode_size_func();
        if (transcode_size != nullptr) {
            s = transcode_size(str_, len_, surr_poli_);
        } else {
            s = strf::decode_encode_size<SrcCharSize>
                ( src_enc.to_u32().transcode
                , dest_enc.from_u32().transcode_size_func()
                , str_, len_, inv_seq_poli_, surr_poli_ );
        }
        if (fillcount > 0) {
            s += dest_enc.encoded_char_size(afmt_.fill) * fillcount;
        }
        preview.add_size(s);
    }
}

template<std::size_t SrcCharSize, std::size_t DestCharSize>
void STRF_HD aligned_cv_string_printer<SrcCharSize, DestCharSize>::print_to
    ( strf::underlying_outbuf<DestCharSize>& ob ) const
{
    if (left_fillcount_ > 0) {
        encode_fill_(ob, left_fillcount_, afmt_.fill, inv_seq_poli_, surr_poli_);
    }
    if (can_transcode_directly()) {
        transcode_(ob, str_, len_, inv_seq_poli_, surr_poli_);
    } else {
        strf::decode_encode<SrcCharSize, DestCharSize>
            ( ob, src_to_u32_, u32_to_dest_, str_
            , len_, inv_seq_poli_, surr_poli_ );
    }
    if (right_fillcount_ > 0) {
        encode_fill_(ob, right_fillcount_, afmt_.fill, inv_seq_poli_, surr_poli_);
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

STRF_EXPLICIT_TEMPLATE class aligned_cv_string_printer<1, 1>;
STRF_EXPLICIT_TEMPLATE class aligned_cv_string_printer<1, 2>;
STRF_EXPLICIT_TEMPLATE class aligned_cv_string_printer<1, 4>;
STRF_EXPLICIT_TEMPLATE class aligned_cv_string_printer<2, 1>;
STRF_EXPLICIT_TEMPLATE class aligned_cv_string_printer<2, 2>;
STRF_EXPLICIT_TEMPLATE class aligned_cv_string_printer<2, 4>;
STRF_EXPLICIT_TEMPLATE class aligned_cv_string_printer<4, 1>;
STRF_EXPLICIT_TEMPLATE class aligned_cv_string_printer<4, 2>;
STRF_EXPLICIT_TEMPLATE class aligned_cv_string_printer<4, 4>;

#endif // defined(STRF_SEPARATE_COMPILATION)

template<std::size_t CharSize>
class cv_string_printer_variant
{
public:

    template < typename DestCharT, typename FPack, typename Preview
             , typename SrcCharT, bool HasPrecision, typename CvFormat >
    cv_string_printer_variant
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, FPack, Preview, SrcCharT, HasPrecision, false, CvFormat >&
            input )
    {
        auto src_encoding  = strf::detail::get_src_encoding(input);
        using facet_tag = strf::string_input_tag<SrcCharT>;
        using dest_enc_cat = strf::char_encoding_c<DestCharT>;
        auto dest_encoding = strf::get_facet<dest_enc_cat, facet_tag>(input.fp);
        if (src_encoding.id() == dest_encoding.id()) {
            new ((void*)&pool_) strf::detail::string_printer<CharSize>(input);
        } else {
            new ((void*)&pool_) strf::detail::cv_string_printer<CharSize, CharSize>(input);
        }
    }

    ~cv_string_printer_variant()
    {
        const strf::printer<CharSize>& p = *this;
        p.~printer();
    }

    operator const strf::printer<CharSize>& () const
    {
        return * reinterpret_cast<const strf::printer<CharSize>*>(&pool_);
    }

private:

    static constexpr std::size_t pool_size_
        = std::max( sizeof(strf::detail::string_printer<CharSize>)
                  , sizeof(strf::detail::cv_string_printer<CharSize, CharSize>) );
    using storage_type_ = typename std::aligned_storage_t
        < pool_size_, alignof(strf::printer<CharSize>)>;

    storage_type_ pool_;
};

template<std::size_t CharSize>
class aligned_cv_string_printer_variant
{
public:

    template < typename DestCharT, typename FPack, typename Preview
             , typename SrcCharT, bool HasPrecision, typename CvFormat >
    aligned_cv_string_printer_variant
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, FPack, Preview, SrcCharT, HasPrecision, true, CvFormat >&
            input )
    {
        auto src_encoding  = strf::detail::get_src_encoding(input);
        using facet_tag = strf::string_input_tag<SrcCharT>;
        using dest_enc_cat = strf::char_encoding_c<DestCharT>;
        auto dest_encoding = strf::get_facet<dest_enc_cat, facet_tag>(input.fp);

        if (src_encoding.id() == dest_encoding.id()) {
            new ((void*)&pool_) strf::detail::aligned_string_printer<CharSize> (input);
        } else {
            new ((void*)&pool_)
                strf::detail::aligned_cv_string_printer<CharSize, CharSize>(input);
        }
    }

    ~aligned_cv_string_printer_variant()
    {
        const strf::printer<CharSize>& p = *this;
        p.~printer();
    }

    operator const strf::printer<CharSize>& () const
    {
        return * reinterpret_cast<const strf::printer<CharSize>*>(&pool_);
    }

private:

    static constexpr std::size_t pool_size_
        = std::max( sizeof(strf::detail::aligned_string_printer<CharSize>)
                  , sizeof(strf::detail::aligned_cv_string_printer<CharSize, CharSize>) );
    using storage_type_ = typename std::aligned_storage_t
        < pool_size_, alignof(strf::printer<CharSize>)>;

    storage_type_ pool_;
};

} // namespace detail

} // namespace strf

#endif  // STRF_DETAIL_INPUT_TYPES_CV_STRING_HPP

