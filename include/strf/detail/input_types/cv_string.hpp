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

    using char_in_type = strf::underlying_char_type<SrcCharSize>;

    template < typename FPack, typename Preview, typename SrcChar
             , typename DestChar, typename SrcCharset >
    STRF_HD cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<SrcChar> str
        , const SrcCharset& src_cs
        , strf::tag<DestChar> ) noexcept
        : str_(reinterpret_cast<const char_in_type*>(str.begin()))
        , len_(str.size())
        , inv_seq_poli_(get_facet_<strf::invalid_seq_policy_c, SrcChar>(fp))
        , surr_poli_(get_facet_<strf::surrogate_policy_c, SrcChar>(fp))
    {
        static_assert(sizeof(SrcChar) == SrcCharSize, "Incompatible char type");
        static_assert(sizeof(DestChar) == DestCharSize, "Incompatible char type");

        init_( preview
             , get_facet_<strf::width_calculator_c, SrcChar>(fp)
             , src_cs
             , get_facet_<strf::charset_c<DestChar>, SrcChar>(fp) );
    }

    template <typename FPack, typename Preview, typename SrcChar, typename DestChar>
    STRF_HD cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<SrcChar> str
        , strf::tag<DestChar> char_tag ) noexcept
        : cv_string_printer
              ( fp, preview, str
              , get_facet_<strf::charset_c<SrcChar>, SrcChar>(fp), char_tag )
    {
    }

    STRF_HD ~cv_string_printer() { }

    STRF_HD void print_to(strf::underlying_outbuf<DestCharSize>& ob) const override;

private:

    template < typename Preview, typename WCalc
             , typename SrcCharset, typename DestCharset >
    STRF_HD void init_
        ( Preview& preview, const WCalc& wcalc
        , const SrcCharset& src_cs, const DestCharset& dest_cs )
    {
        (void) wcalc;

        static_assert(SrcCharset::char_size == SrcCharSize, "Incompatible char type");
        static_assert(DestCharset::char_size == DestCharSize, "Incompatible char type");
        decltype(auto) transcoder = find_transcoder(src_cs, dest_cs);
        transcode_ = transcoder.transcode_func();
        if (transcode_ == nullptr) {
            src_to_u32_ = src_cs.to_u32().transcode_func();
            u32_to_dest_ = dest_cs.from_u32().transcode_func();
        }
        STRF_IF_CONSTEXPR (Preview::width_required) {
            auto w = wcalc.str_width(src_cs, preview.remaining_width(), str_, len_, surr_poli_);
            preview.subtract_width(w);
        }
        STRF_IF_CONSTEXPR (Preview::size_required) {
            strf::transcode_size_f<SrcCharSize>  transcode_size
                = transcoder.transcode_size_func();
            std::size_t s = 0;
            if (transcode_size != nullptr) {
                s = transcode_size(str_, str_ + len_, surr_poli_);
            } else {
                s = strf::decode_encode_size<SrcCharSize>
                    ( src_cs.to_u32().transcode_func()
                    , dest_cs.from_u32().transcode_size_func()
                    , str_, str_ + len_, inv_seq_poli_, surr_poli_ );
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
        transcode_(ob, str_, str_ + len_, inv_seq_poli_, surr_poli_);
    } else {
        strf::decode_encode<SrcCharSize, DestCharSize>
            ( ob, src_to_u32_, u32_to_dest_, str_
            , str_ + len_, inv_seq_poli_, surr_poli_ );
    }
}

template<std::size_t SrcCharSize, std::size_t DestCharSize>
class fmt_cv_string_printer: public printer<DestCharSize>
{
public:
    using char_in_type = strf::underlying_char_type<SrcCharSize>;

    template < typename FPack, typename Preview, typename SrcChar
             , typename DestChar, typename SrcCharset >
    STRF_HD fmt_cv_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<SrcChar> str
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
        init_( preview
             , get_facet_<strf::width_calculator_c, SrcChar>(fp)
             , src_cs
             , get_facet_<strf::charset_c<DestChar>, SrcChar>(fp) );
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
            , get_facet_<strf::charset_c<SrcChar>, SrcChar>(fp), ch_tag )
    {
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

    template < typename Preview, typename WCalc
             , typename SrcCharset, typename DestCharset>
    STRF_HD void init_
        ( Preview& preview, const WCalc& wcalc
        , const SrcCharset& src_cs, const DestCharset& dest_cs );
};

template <std::size_t SrcCharSize, std::size_t DestCharSize>
template <typename Preview, typename WCalc, typename SrcCharset, typename DestCharset>
void STRF_HD fmt_cv_string_printer<SrcCharSize, DestCharSize>::init_
    ( Preview& preview, const WCalc& wcalc
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
    strf::width_t fmt_width = afmt_.width;
    strf::width_t limit =
        ( Preview::width_required && preview.remaining_width() > fmt_width
        ? preview.remaining_width()
        : fmt_width );
    auto strw = wcalc.str_width(src_cs, limit, str_, len_, surr_poli_);
    if (fmt_width > strw) {
        fillcount = (fmt_width - strw).round();
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
        preview.subtract_width(strw + fillcount);
    } else {
        right_fillcount_ = 0;
        left_fillcount_ = 0;
        preview.subtract_width(strw);
    }
    STRF_IF_CONSTEXPR (Preview::size_required) {
        std::size_t s = 0;
        strf::transcode_size_f<SrcCharSize> transcode_size
                = transcoder.transcode_size_func();
        if (transcode_size != nullptr) {
            s = transcode_size(str_, str_ + len_, surr_poli_);
        } else {
            s = strf::decode_encode_size<SrcCharSize>
                ( src_cs.to_u32().transcode
                , dest_cs.from_u32().transcode_size_func()
                , str_, str_ + len_, inv_seq_poli_, surr_poli_ );
        }
        if (fillcount > 0) {
            s += dest_cs.encoded_char_size(afmt_.fill) * fillcount;
        }
        preview.add_size(s);
    }
}

template<std::size_t SrcCharSize, std::size_t DestCharSize>
void STRF_HD fmt_cv_string_printer<SrcCharSize, DestCharSize>::print_to
    ( strf::underlying_outbuf<DestCharSize>& ob ) const
{
    if (left_fillcount_ > 0) {
        encode_fill_(ob, left_fillcount_, afmt_.fill, inv_seq_poli_, surr_poli_);
    }
    if (can_transcode_directly()) {
        transcode_(ob, str_, str_ + len_, inv_seq_poli_, surr_poli_);
    } else {
        strf::decode_encode<SrcCharSize, DestCharSize>
            ( ob, src_to_u32_, u32_to_dest_, str_
            , str_ + len_, inv_seq_poli_, surr_poli_ );
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
class cv_printer_maker_without_charset;

template <typename CharT>
class cv_printer_maker_without_charset<CharT, CharT, true>
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
class cv_printer_maker_without_charset<SrcChar, DestChar, true>
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
        using enc_cat_in = strf::charset_c<SrcChar>;
        using input_tag_in = strf::string_input_tag<SrcChar>;
        const auto& charset_in = strf::get_facet<enc_cat_in, input_tag_in>(fp);

        using enc_cat_out = strf::charset_c<DestChar>;
        using input_tag_out = strf::string_input_tag<DestChar>;
        const auto& charset_out = strf::get_facet<enc_cat_out, input_tag_out>(fp);

        if (charset_in.id() == charset_out.id()) {
            return { strf::tag<strf::detail::string_printer<char_size>>()
                   , fp
                   , preview
                   , strf::detail::simple_string_view<DestChar>
                       { reinterpret_cast<const DestChar*>(input.value().begin())
                       , input.value().size() }
                   , strf::tag<DestChar>() };
        } else {
            return { strf::tag<strf::detail::cv_string_printer<char_size, char_size>>()
                   , fp, preview, input.value(), charset_in, strf::tag<DestChar>() };
        }
    }

    template <typename FPack, typename Preview, typename SrcCharset>
    static inline STRF_HD strf::detail::printer_variant
        < strf::detail::fmt_string_printer<char_size>
        , strf::detail::fmt_cv_string_printer<char_size, char_size> >
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format
            < strf::detail::simple_string_view<SrcChar>
            , strf::alignment_format_q<true>
            , strf::cv_format_with_charset<SrcChar, SrcCharset > > input )
    {
        using enc_cat_in = strf::charset_c<SrcChar>;
        using input_tag_in = strf::string_input_tag<SrcChar>;
        const auto& charset_in = strf::get_facet<enc_cat_in, input_tag_in>(fp);

        using enc_cat_out = strf::charset_c<DestChar>;
        using input_tag_out = strf::string_input_tag<DestChar>;
        const auto& charset_out = strf::get_facet<enc_cat_out, input_tag_out>(fp);

        if (charset_in.id() == charset_out.id()) {
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
                   , charset_in };
        }
    }
};

template <typename SrcChar, typename DestChar>
class cv_printer_maker_without_charset<SrcChar, DestChar, false>
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
class cv_printer_maker_with_charset;

template <typename SrcChar, typename DestChar>
class cv_printer_maker_with_charset<SrcChar, DestChar, true>
{
public:
    static_assert(sizeof(SrcChar) == sizeof(DestChar), "");

    template <typename FPack, typename Preview, typename SrcCharset>
    static inline STRF_HD strf::detail::printer_variant
        < strf::detail::string_printer<sizeof(DestChar)>
        , strf::detail::cv_string_printer<sizeof(SrcChar), sizeof(DestChar)> >
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format
            < strf::detail::simple_string_view<SrcChar>
            , strf::alignment_format_q<false>
            , strf::cv_format_with_charset<SrcChar, SrcCharset> > input )
    {
        using enc_cat = strf::charset_c<DestChar>;
        using input_tag = strf::string_input_tag<DestChar>;
        const auto& charset_from_facets
            = strf::get_facet<enc_cat, input_tag>(fp);
        if (input.get_charset().id() == charset_from_facets.id()) {
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
                   , fp, preview, input.value(), input.get_charset()
                   , strf::tag<DestChar>() };
        }
    }

    template <typename FPack, typename Preview, typename SrcCharset>
    static inline STRF_HD strf::detail::printer_variant
        < strf::detail::fmt_string_printer<sizeof(DestChar)>
        , strf::detail::fmt_cv_string_printer<sizeof(SrcChar), sizeof(DestChar)> >
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format
            < strf::detail::simple_string_view<SrcChar>
            , strf::alignment_format_q<true>
            , strf::cv_format_with_charset<SrcChar, SrcCharset> > input )
    {
        using enc_cat = strf::charset_c<DestChar>;
        using input_tag = strf::string_input_tag<DestChar>;
        const auto& charset_from_facets
            = strf::get_facet<enc_cat, input_tag>(fp);
        if (input.get_charset().id() == charset_from_facets.id()) {
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
                   , input.get_charset()
                   , strf::tag<DestChar>() };
        }
    }
};

template <typename SrcChar, typename DestChar>
class cv_printer_maker_with_charset<SrcChar, DestChar, false>
{
public:
    template <typename FPack, typename Preview, typename SrcCharset>
    static inline STRF_HD
    strf::detail::cv_string_printer<sizeof(SrcChar), sizeof(DestChar)>
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format
            < strf::detail::simple_string_view<SrcChar>
            , strf::alignment_format_q<false>
            , strf::cv_format_with_charset<SrcChar, SrcCharset> > input )
    {
        return { fp, preview, input.value(), input.get_charset()
               , strf::tag<DestChar>()};
    }

    template <typename FPack, typename Preview, typename SrcCharset>
    static inline STRF_HD
    strf::detail::fmt_cv_string_printer<sizeof(SrcChar), sizeof(DestChar)>
    make_printer
        ( const FPack& fp
        , Preview& preview
        , strf::value_with_format
            < strf::detail::simple_string_view<SrcChar>
            , strf::alignment_format_q<true>
            , strf::cv_format_with_charset<SrcChar, SrcCharset> > input )
    {
        return { fp
               , preview
               , input.value()
               , input.get_alignment_format_data()
               , input.get_charset()
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
    return strf::detail::cv_printer_maker_without_charset<SrcChar, DestChar, ss>
        :: make_printer(fp, preview, input);
}

template < typename DestChar, typename FPack, typename Preview
         , typename SrcChar, typename SrcCharset, bool WithAlignment >
inline STRF_HD auto make_printer
    ( strf::rank<1>
    , const FPack& fp
    , Preview& preview
    , strf::value_with_format
        < strf::detail::simple_string_view<SrcChar>
        , strf::alignment_format_q<WithAlignment>
        , strf::cv_format_with_charset<SrcChar, SrcCharset> > input )
{
    constexpr bool ss = sizeof(SrcChar) == sizeof(DestChar);
    return strf::detail::cv_printer_maker_with_charset<SrcChar, DestChar, ss>
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
         , typename SrcChar, typename SrcCharset >
inline STRF_HD strf::detail::cv_string_printer<sizeof(SrcChar), sizeof(DestChar)>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , strf::value_with_format
                < strf::detail::simple_string_view<SrcChar>
                , strf::alignment_format_q<false>
                , strf::sani_format_with_charset<SrcChar, SrcCharset> > input )
{
    return { fp, preview, input.value(), input.get_charset()
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
         , typename SrcChar, typename SrcCharset >
inline STRF_HD strf::detail::fmt_cv_string_printer<sizeof(SrcChar), sizeof(DestChar)>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , strf::value_with_format
                < strf::detail::simple_string_view<SrcChar>
                , strf::alignment_format_q<true>
                , strf::sani_format_with_charset<SrcChar, SrcCharset> > input )
{
    return { fp
           , preview
           , input.value()
           , input.get_alignment_format_data()
           , input.get_charset()
           , strf::tag<DestChar>() };
}

} // namespace strf

#endif  // STRF_DETAIL_INPUT_TYPES_CV_STRING_HPP

