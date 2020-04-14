#ifndef STRF_DETAIL_INPUT_TYPES_CV_STRING_HPP
#define STRF_DETAIL_INPUT_TYPES_CV_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/input_types/string.hpp>

namespace strf {
namespace detail {

template<std::size_t SrcCharSize, std::size_t DestCharSize>
class cv_string_printer: public strf::printer<DestCharSize>
{
public:

    using char_in_type = strf::underlying_char_type<SrcCharSize>;

    template < typename FPack, typename Preview, typename SrcChar
             , typename DestChar, bool HasPrecision >
    STRF_HD cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<SrcChar>
                                 , strf::string_precision_format<HasPrecision>
                                 , strf::alignment_format_q<false>
                                 , strf::cv_format<SrcChar> > input
        , strf::tag<DestChar> tag )
        : cv_string_printer( fp, preview, input.value(), input.get_string_precision()
                             , get_facet_<strf::charset_c<SrcChar>, SrcChar>(fp), tag )
    {
    }

    template < typename FPack, typename Preview, typename SrcChar
             , typename DestChar, typename SrcCharset, bool HasPrecision >
    STRF_HD cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format
            < strf::detail::simple_string_view<SrcChar>
            , strf::string_precision_format<HasPrecision>
            , strf::alignment_format_q<false>
            , strf::cv_format_with_charset<SrcChar, SrcCharset> > input
        , strf::tag<DestChar> tag )
        : cv_string_printer( fp, preview, input.value(), input.get_string_precision()
                           , input.get_charset(), tag )
    {
    }
    template < typename FPack, typename Preview, typename SrcChar
             , typename DestChar, bool HasPrecision >
    STRF_HD cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<SrcChar>
                                 , strf::string_precision_format<HasPrecision>
                                 , strf::alignment_format_q<false>
                                 , strf::sani_format<SrcChar> > input
        , strf::tag<DestChar> tag )
        : cv_string_printer( fp, preview, input.value(), input.get_string_precision()
                           , get_facet_<strf::charset_c<SrcChar>, SrcChar>(fp), tag )
    {
    }

    template < typename FPack, typename Preview, typename SrcChar
             , typename DestChar, typename SrcCharset, bool HasPrecision >
    STRF_HD cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format
            < strf::detail::simple_string_view<SrcChar>
            , strf::string_precision_format<HasPrecision>
            , strf::alignment_format_q<false>
            , strf::sani_format_with_charset<SrcChar, SrcCharset> > input
        , strf::tag<DestChar> tag )
        : cv_string_printer( fp, preview, input.value(), input.get_string_precision()
                           , input.get_charset(), tag )
    {
    }

    template < typename FPack, typename Preview, typename SrcChar
             , typename DestChar, typename SrcCharset >
    STRF_HD cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<SrcChar> str
        , strf::string_precision<false>
        , const SrcCharset& src_cs
        , strf::tag<DestChar> ) noexcept
        : str_(reinterpret_cast<const char_in_type*>(str.begin()))
        , len_(str.size())
        , inv_seq_poli_(get_facet_<strf::invalid_seq_policy_c, SrcChar>(fp))
        , surr_poli_(get_facet_<strf::surrogate_policy_c, SrcChar>(fp))
    {
        static_assert(sizeof(SrcChar) == SrcCharSize, "Incompatible char type");
        static_assert(sizeof(DestChar) == DestCharSize, "Incompatible char type");
        STRF_IF_CONSTEXPR (Preview::width_required) {
            decltype(auto) wcalc = get_facet_<strf::width_calculator_c, SrcChar>(fp);
            auto w = wcalc.str_width(src_cs, preview.remaining_width(), str_, len_, surr_poli_);
            preview.subtract_width(w);
        }
        init_(preview, src_cs, get_facet_<strf::charset_c<DestChar>, SrcChar>(fp));
    }

    template < typename FPack, typename Preview, typename SrcChar
             , typename DestChar, typename SrcCharset >
    STRF_HD cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<SrcChar> str
        , strf::string_precision<true> sp
        , const SrcCharset& src_cs
        , strf::tag<DestChar> ) noexcept
        : str_(reinterpret_cast<const char_in_type*>(str.begin()))
        , inv_seq_poli_(get_facet_<strf::invalid_seq_policy_c, SrcChar>(fp))
        , surr_poli_(get_facet_<strf::surrogate_policy_c, SrcChar>(fp))
    {
        static_assert(sizeof(SrcChar) == SrcCharSize, "Incompatible char type");
        static_assert(sizeof(DestChar) == DestCharSize, "Incompatible char type");

        decltype(auto) wcalc = get_facet_<strf::width_calculator_c, SrcChar>(fp);
        auto res = wcalc.str_width_and_pos(src_cs, sp.precision, str_, str.size(), surr_poli_);
        len_ = res.pos;
        preview.subtract_width(res.width);
        init_(preview, src_cs, get_facet_<strf::charset_c<DestChar>, SrcChar>(fp));
    }

    STRF_HD ~cv_string_printer() { }

    STRF_HD void print_to(strf::underlying_outbuf<DestCharSize>& ob) const override;

private:

    template < typename Preview, typename SrcCharset, typename DestCharset >
    STRF_HD void init_(Preview& preview, const SrcCharset& src_cs, const DestCharset& dest_cs)
    {
        static_assert(SrcCharset::char_size == SrcCharSize, "Incompatible char type");
        static_assert(DestCharset::char_size == DestCharSize, "Incompatible char type");

        decltype(auto) transcoder = find_transcoder(src_cs, dest_cs);
        transcode_ = transcoder.transcode_func();
        if (transcode_ == nullptr) {
            src_to_u32_ = src_cs.to_u32().transcode_func();
            u32_to_dest_ = dest_cs.from_u32().transcode_func();
        }
        STRF_IF_CONSTEXPR (Preview::size_required) {
            strf::transcode_size_f<SrcCharSize>  transcode_size
                = transcoder.transcode_size_func();
            std::size_t s = 0;
            if (transcode_size != nullptr) {
                s = transcode_size(str_, len_, surr_poli_);
            } else {
                s = strf::decode_encode_size<SrcCharSize>
                    ( src_cs.to_u32().transcode_func()
                    , dest_cs.from_u32().transcode_size_func()
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
    const std::size_t len_;
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

    template < typename FPack, typename Preview, typename SrcChar
             , typename DestChar, bool HasPrecision >
    STRF_HD aligned_cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format
            < strf::detail::simple_string_view<SrcChar>
            , strf::string_precision_format<HasPrecision>
            , strf::alignment_format_q<true>
            , strf::cv_format<SrcChar> > input
        , strf::tag<DestChar> t )
        : aligned_cv_string_printer
            ( fp, preview, input.value(), input.get_string_precision()
            , input.get_alignment_format_data()
            , get_facet_<strf::charset_c<SrcChar>, SrcChar>(fp), t )
    {
    }

    template < typename FPack, typename Preview, typename SrcChar
             , typename DestChar, bool HasPrecision >
    STRF_HD aligned_cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format
            < strf::detail::simple_string_view<SrcChar>
            , strf::string_precision_format<HasPrecision>
            , strf::alignment_format_q<true>
            , strf::sani_format<SrcChar> > input
        , strf::tag<DestChar> t )
        : aligned_cv_string_printer
            ( fp, preview, input.value(), input.get_string_precision()
            , input.get_alignment_format_data()
            , get_facet_<strf::charset_c<SrcChar>, SrcChar>(fp), t )
    {
    }
    template < typename FPack, typename Preview, typename SrcChar
             , typename SrcCharset, typename DestChar, bool HasPrecision >
    STRF_HD aligned_cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format
            < strf::detail::simple_string_view<SrcChar>
            , strf::string_precision_format<HasPrecision>
            , strf::alignment_format_q<true>
            , strf::cv_format_with_charset<SrcChar, SrcCharset> > input
        , strf::tag<DestChar> t )
        : aligned_cv_string_printer
            ( fp, preview, input.value(), input.get_string_precision()
            , input.get_alignment_format_data()
            , input.get_charset(), t )
    {
    }

    template < typename FPack, typename Preview, typename SrcChar
             , typename SrcCharset, typename DestChar, bool HasPrecision >
    STRF_HD aligned_cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format
            < strf::detail::simple_string_view<SrcChar>
            , strf::string_precision_format<HasPrecision>
            , strf::alignment_format_q<true>
            , strf::sani_format_with_charset<SrcChar, SrcCharset> > input
        , strf::tag<DestChar> t )
        : aligned_cv_string_printer
            ( fp, preview, input.value(), input.get_string_precision()
            , input.get_alignment_format_data()
            , input.get_charset(), t )
    {
    }

    template < typename FPack, typename Preview, typename SrcChar
             , typename DestChar, typename SrcCharset >
    STRF_HD aligned_cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<SrcChar> str
        , strf::string_precision<false>
        , strf::alignment_format_data text_alignment
        , const SrcCharset& src_cs
        , strf::tag<DestChar> ) noexcept
        : str_(reinterpret_cast<const char_in_type*>(str.begin()))
        , len_(str.size())
        , afmt_(text_alignment)
        , inv_seq_poli_(get_facet_<strf::invalid_seq_policy_c, SrcChar>(fp))
        , surr_poli_(get_facet_<strf::surrogate_policy_c, SrcChar>(fp))
    {
        static_assert(sizeof(SrcChar) == SrcCharSize, "Incompatible char type");
        static_assert(sizeof(DestChar) == DestCharSize, "Incompatible char type");
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c, SrcChar>(fp);
        strf::width_t limit =
            ( Preview::width_required && preview.remaining_width() > afmt_.width
            ? preview.remaining_width()
            : afmt_.width );
        auto str_width = wcalc.str_width(src_cs, limit, str_, len_, surr_poli_);
        init_( preview, str_width, src_cs
             , get_facet_<strf::charset_c<DestChar>, SrcChar>(fp) );
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

    template < typename Preview, typename SrcCharset, typename DestCharset>
    STRF_HD void init_
        ( Preview& preview, strf::width_t str_width
        , const SrcCharset& src_cs, const DestCharset& dest_cs );
};

template <std::size_t SrcCharSize, std::size_t DestCharSize>
template <typename Preview, typename SrcCharset, typename DestCharset>
void STRF_HD aligned_cv_string_printer<SrcCharSize, DestCharSize>::init_
    ( Preview& preview, strf::width_t str_width
    , const SrcCharset& src_cs, const DestCharset& dest_cs )
{
    static_assert(SrcCharset::char_size == SrcCharSize, "Incompatible char type");
    static_assert(DestCharset::char_size == DestCharSize, "Incompatible char type");

    encode_fill_ = dest_cs.encode_fill_func();
    decltype(auto) transcoder = find_transcoder(src_cs, dest_cs);
    transcode_ = transcoder.transcode_func();
    if (transcode_ == nullptr) {
        src_to_u32_ = src_cs.to_u32().transcode_func();
        u32_to_dest_ = dest_cs.from_u32().transcode_func();
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
                ( src_cs.to_u32().transcode
                , dest_cs.from_u32().transcode_size_func()
                , str_, len_, inv_seq_poli_, surr_poli_ );
        }
        if (fillcount > 0) {
            s += dest_cs.encoded_char_size(afmt_.fill) * fillcount;
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

    template < typename FPack, typename Preview
             , typename SrcChar, typename DestChar, bool HasPrecision >
    cv_string_printer_variant
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<SrcChar>
                                 , strf::string_precision_format<HasPrecision>
                                 , strf::alignment_format_q<false>
                                 , strf::cv_format<SrcChar> > input
                                 , strf::tag<DestChar> )
    {
        using facet_tag = strf::string_input_tag<SrcChar>;

        using enc_cat_in  = strf::charset_c<SrcChar>;
        using enc_cat_out = strf::charset_c<DestChar>;
        const auto& charset_in  = strf::get_facet<enc_cat_in,  facet_tag>(fp);
        const auto& charset_out = strf::get_facet<enc_cat_out, facet_tag>(fp);

        if (charset_in.id() == charset_out.id()) {
            new ((void*)&pool_) strf::detail::string_printer<CharSize>
                ( fp, preview
                , strf::detail::simple_string_view<DestChar>
                      { reinterpret_cast<const DestChar*>(input.value().begin())
                      , input.value().size() }
                , input.get_string_precision(), strf::tag<DestChar>() );
        } else {
            new ((void*)&pool_) strf::detail::cv_string_printer<CharSize, CharSize>
                ( fp, preview, input, strf::tag<DestChar>() );
        }
    }

    template < typename FPack, typename Preview, typename SrcChar
             , bool HasPrecision, typename SrcCharset, typename DestChar >
    cv_string_printer_variant
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format
            < strf::detail::simple_string_view<SrcChar>
            , strf::string_precision_format<HasPrecision>
            , strf::alignment_format_q<false>
            , strf::cv_format_with_charset<SrcChar, SrcCharset> > input
            , strf::tag<DestChar> )
    {
        using facet_tag = strf::string_input_tag<SrcChar>;
        using enc_cat_out = strf::charset_c<DestChar>;
        const auto& charset_out = strf::get_facet<enc_cat_out, facet_tag>(fp);
        if (input.get_charset().id() == charset_out.id()) {
            new ((void*)&pool_) strf::detail::string_printer<CharSize>
                ( fp, preview
                , strf::detail::simple_string_view<DestChar>
                      { reinterpret_cast<const DestChar*>(input.value().begin())
                      , input.value().size() }
                , input.get_string_precision(), strf::tag<DestChar>() );
        } else {
            new ((void*)&pool_) strf::detail::cv_string_printer<CharSize, CharSize>
                ( fp, preview, input, strf::tag<DestChar>() );
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
    template < typename FPack, typename Preview
             , typename SrcChar, typename DestChar, bool HasPrecision >
    aligned_cv_string_printer_variant
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<SrcChar>
                                 , strf::string_precision_format<HasPrecision>
                                 , strf::alignment_format_q<true>
                                 , strf::cv_format<SrcChar> > input
                                 , strf::tag<DestChar> )
    {
        using enc_cat_in = strf::charset_c<SrcChar>;
        using facets_tag = strf::string_input_tag<SrcChar>;
        const auto& charset_in = strf::get_facet<enc_cat_in, facets_tag>(fp);

        using enc_cat_out = strf::charset_c<DestChar>;
        const auto& charset_out = strf::get_facet<enc_cat_out, facets_tag>(fp);

        if (charset_in.id() == charset_out.id()) {
            new ((void*)&pool_) strf::detail::aligned_string_printer<CharSize>
                ( fp, preview
                , strf::detail::simple_string_view<DestChar>
                      { reinterpret_cast<const DestChar*>(input.value().begin())
                      , input.value().size() }
                , input.get_string_precision()
                , input.get_alignment_format_data()
                , strf::tag<DestChar>() );
        } else {
            new ((void*)&pool_) strf::detail::aligned_cv_string_printer<CharSize, CharSize>
                ( fp, preview, input, strf::tag<DestChar>() );
        }
    }

    template < typename FPack, typename Preview, typename SrcChar
             , typename SrcCharset, typename DestChar, bool HasPrecision >
    aligned_cv_string_printer_variant
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format< strf::detail::simple_string_view<SrcChar>
                                 , strf::string_precision_format<HasPrecision>
                                 , strf::alignment_format_q<true>
                                 , strf::cv_format_with_charset<SrcChar, SrcCharset> > input
                                 , strf::tag<DestChar> )
    {
        using facet_tag = strf::string_input_tag<SrcChar>;
        using enc_cat_out = strf::charset_c<DestChar>;
        const auto& charset_out = strf::get_facet<enc_cat_out, facet_tag>(fp);

        if (input.get_charset().id() == charset_out.id()) {
            new ((void*)&pool_) strf::detail::aligned_string_printer<CharSize>
                ( fp, preview
                , strf::detail::simple_string_view<DestChar>
                      { reinterpret_cast<const DestChar*>(input.value().begin())
                      , input.value().size() }
                , input.get_string_precision()
                , input.get_alignment_format_data()
                , strf::tag<DestChar>() );
        } else {
            new ((void*)&pool_) strf::detail::aligned_cv_string_printer<CharSize, CharSize>
                ( fp, preview, input, strf::tag<DestChar>() );
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

template < typename DestChar, typename SrcChar, bool HasPrecision >
class printer_traits
    < DestChar
    , strf::value_with_format
        < strf::detail::simple_string_view<SrcChar>
        , strf::string_precision_format<HasPrecision>
        , strf::alignment_format_q<true>
        , strf::cv_format<SrcChar> > >
{
public:

    template <typename>
    using printer_type = std::conditional_t
        < sizeof(SrcChar) == sizeof(DestChar)
        , strf::detail::aligned_cv_string_printer_variant<sizeof(SrcChar)>
        , strf::detail::aligned_cv_string_printer<sizeof(SrcChar), sizeof(DestChar)> >;
};

template < typename DestChar, typename SrcChar, typename SrcCharset, bool HasPrecision >
class printer_traits
    < DestChar
    , strf::value_with_format
        < strf::detail::simple_string_view<SrcChar>
        , strf::string_precision_format<HasPrecision>
        , strf::alignment_format_q<true>
        , strf::cv_format_with_charset<SrcChar, SrcCharset> > >
{
public:

    template <typename>
    using printer_type = std::conditional_t
        < sizeof(SrcChar) == sizeof(DestChar)
        , strf::detail::aligned_cv_string_printer_variant<sizeof(SrcChar)>
        , strf::detail::aligned_cv_string_printer<sizeof(SrcChar), sizeof(DestChar)> >;
};

template < typename DestChar, typename SrcChar, bool HasPrecision >
class printer_traits
    < DestChar
    , strf::value_with_format
        < strf::detail::simple_string_view<SrcChar>
        , strf::string_precision_format<HasPrecision>
        , strf::alignment_format_q<false>
        , strf::cv_format<SrcChar> > >
{
public:

    template <typename>
    using printer_type = std::conditional_t
        < sizeof(SrcChar) == sizeof(DestChar)
        , strf::detail::cv_string_printer_variant<sizeof(SrcChar)>
        , strf::detail::cv_string_printer<sizeof(SrcChar), sizeof(DestChar)> >;
};

template < typename DestChar, typename SrcChar, typename SrcCharset, bool HasPrecision >
class printer_traits
    < DestChar
    , strf::value_with_format
        < strf::detail::simple_string_view<SrcChar>
        , strf::string_precision_format<HasPrecision>
        , strf::alignment_format_q<false>
        , strf::cv_format_with_charset<SrcChar, SrcCharset> > >
{
public:

    template <typename>
    using printer_type = std::conditional_t
        < sizeof(SrcChar) == sizeof(DestChar)
        , strf::detail::cv_string_printer_variant<sizeof(SrcChar)>
        , strf::detail::cv_string_printer<sizeof(SrcChar), sizeof(DestChar)> >;
};

template < typename DestChar, typename SrcChar, bool HasPrecision >
class printer_traits
    < DestChar
    , strf::value_with_format
        < strf::detail::simple_string_view<SrcChar>
        , strf::string_precision_format<HasPrecision>
        , strf::alignment_format_q<false>
        , strf::sani_format<SrcChar> > >
{
public:
    template <typename>
    using printer_type
    = strf::detail::cv_string_printer<sizeof(SrcChar), sizeof(DestChar)>;
};

template <typename DestChar, typename SrcChar, typename SrcCharset, bool HasPrecision>
class printer_traits
    < DestChar
    , strf::value_with_format
         < strf::detail::simple_string_view<SrcChar>
         , strf::string_precision_format<HasPrecision>
         , strf::alignment_format_q<false>
         , strf::sani_format_with_charset<SrcChar, SrcCharset> > >
{
public:

    template <typename>
    using printer_type
    = strf::detail::cv_string_printer<sizeof(SrcChar), sizeof(DestChar)>;
};

template <typename DestChar, typename SrcChar, bool HasPrecision>
class printer_traits
    < DestChar
    , strf::value_with_format
        < strf::detail::simple_string_view<SrcChar>
        , strf::string_precision_format<HasPrecision>
        , strf::alignment_format_q<true>
        , strf::sani_format<SrcChar> > >
{
public:

    template <typename>
    using printer_type
    = strf::detail::aligned_cv_string_printer<sizeof(SrcChar), sizeof(DestChar)>;
};

template < typename DestChar, typename SrcChar, typename SrcCharset, bool HasPrecision >
class printer_traits
    < DestChar
    , strf::value_with_format
        < strf::detail::simple_string_view<SrcChar>
        , strf::string_precision_format<HasPrecision>
        , strf::alignment_format_q<true>
        , strf::sani_format_with_charset<SrcChar, SrcCharset> > >
{
public:

    template <typename>
    using printer_type
    = strf::detail::aligned_cv_string_printer<sizeof(SrcChar), sizeof(DestChar)>;
};

} // namespace strf

#endif  // STRF_DETAIL_INPUT_TYPES_CV_STRING_HPP

