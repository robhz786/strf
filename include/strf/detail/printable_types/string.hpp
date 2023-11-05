#ifndef STRF_DETAIL_PRINTABLE_TYPES_STRING_HPP
#define STRF_DETAIL_PRINTABLE_TYPES_STRING_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/facets/width_calculator.hpp>
#include <strf/detail/facets/charset.hpp>
#include <strf/detail/format_functions.hpp>
#include <strf/detail/printer.hpp>
#include <strf/detail/simple_string_view.hpp>

namespace strf {

namespace detail {

struct no_charset_tag {};

enum class transcoding_policy {
    no_transcode, // default: don't transcode nor sanitize. Char types must be equal
    unsafe,
    sani_only_if_different,
    sani_unconditionally
} ;

template <typename T, typename CharT, typename Charset, transcoding_policy>
class transcoding_formatter_fn;

template < typename CharT
         , typename Charset = no_charset_tag
         , transcoding_policy TranscPoli = transcoding_policy::no_transcode >
struct transcoding_formatter_q
{
    template <typename T>
    using fn = transcoding_formatter_fn<T, CharT, Charset, TranscPoli>;

    constexpr static auto transcoding_policy = TranscPoli;
    constexpr static bool has_charset = detail::is_charset<Charset>::value;
};

template <typename CharT>
using transcoding_formatter =
    transcoding_formatter_q<CharT, detail::no_charset_tag, transcoding_policy::no_transcode>;

template <typename T, typename CharT, typename Charset, transcoding_policy TranscPoli>
class transcoding_formatter_fn
{
    STRF_HD STRF_CONSTEXPR_IN_CXX14 const T& self_downcast_() const
    {
        // if we directly return *static_cast<const T*>(this), then
        // ubsan of gcc emits "reference binding to misaligned address"
        const T* d = static_cast<const T*>(this);
        return *d;
    }

    Charset charset_;

    template <typename OtherCharset, transcoding_policy NewPoli>
    using return_type_ = strf::fmt_replace
        < T
        , transcoding_formatter_q<CharT, Charset, TranscPoli>
        , transcoding_formatter_q<CharT, OtherCharset, NewPoli> >;

    template <typename OtherCharset>
    using return_type_unsafe_ =
        return_type_<OtherCharset, transcoding_policy::unsafe>;

    template <typename OtherCharset>
    using return_type_transcode_ =
        return_type_<OtherCharset, transcoding_policy::sani_only_if_different>;

    template <typename OtherCharset>
    using return_type_sani_ =
        return_type_<OtherCharset, transcoding_policy::sani_unconditionally>;

    template <transcoding_policy NewPoli
             , typename NewCharset
             , strf::detail::enable_if_t<detail::is_charset<NewCharset>::value, int> = 0 >
    constexpr STRF_HD return_type_<NewCharset, NewPoli>
    set_poli_and_charset_(NewCharset charset) const
    {
        static_assert( std::is_same<typename NewCharset::code_unit, CharT>::value
                     , "This charset is associated with another character type." );

        return return_type_<NewCharset, NewPoli>
            { self_downcast_()
            , strf::tag<transcoding_formatter_q<CharT, NewCharset, NewPoli>>()
            , charset };
    }

  public:

    constexpr transcoding_formatter_fn() noexcept = default;

    template <typename U, transcoding_policy OtherPoli, typename ThisCharset = Charset>
    constexpr STRF_HD explicit transcoding_formatter_fn
        ( const transcoding_formatter_fn<U, CharT, detail::no_charset_tag, OtherPoli>& ) noexcept
    {
    }

    template < typename U, transcoding_policy OtherPoli, typename OtherCharset
             , strf::detail::enable_if_t<is_charset<OtherCharset>::value, int> = 0 >
    constexpr STRF_HD explicit transcoding_formatter_fn
        ( const transcoding_formatter_fn<U, CharT, OtherCharset, OtherPoli>& other ) noexcept
        : charset_(other.get_charset())
    {
    }

    constexpr STRF_HD transcoding_formatter_fn(Charset cs) noexcept
        : charset_(cs)
    {
    }

    // format functions
    constexpr STRF_HD return_type_unsafe_<Charset> unsafe_transcode() const
    {
        return return_type_unsafe_<Charset>{ self_downcast_() };
    }
    constexpr STRF_HD return_type_transcode_<Charset> transcode() const
    {
        return return_type_transcode_<Charset>{ self_downcast_() };
    }
    constexpr STRF_HD return_type_sani_<Charset> sani() const
    {
        return return_type_sani_<Charset>{ self_downcast_() };
    }
    constexpr STRF_HD return_type_sani_<Charset> sanitize() const
    {
        return return_type_sani_<Charset>{ self_downcast_() };
    }

    template <typename OtherCharset>
    constexpr STRF_HD auto unsafe_transcode(OtherCharset charset) const
    {
        constexpr auto poli = transcoding_policy::unsafe;
        return set_poli_and_charset_<poli>(charset);
    }
    template <typename OtherCharset>
    constexpr STRF_HD auto transcode(OtherCharset charset) const
    {
        constexpr auto poli = transcoding_policy::sani_only_if_different;
        return set_poli_and_charset_<poli>(charset);
    }
    template <typename OtherCharset>
    constexpr STRF_HD auto sani(OtherCharset charset) const
    {
        constexpr auto poli = transcoding_policy::sani_unconditionally;
        return set_poli_and_charset_<poli>(charset);
    }
    template <typename OtherCharset>
    constexpr STRF_HD auto sanitize(OtherCharset charset) const
    {
        constexpr auto poli = transcoding_policy::sani_unconditionally;
        return set_poli_and_charset_<poli>(charset);
    }

    // aliases

    template <typename OtherCharset>
    constexpr STRF_HD auto transcode_from(OtherCharset charset) const
    {
        constexpr auto poli = transcoding_policy::sani_only_if_different;
        return set_poli_and_charset_<poli>(charset);
    }
    template <typename OtherCharset>
    constexpr STRF_HD auto sanitize_from(OtherCharset charset) const
    {
        constexpr auto poli = transcoding_policy::sani_unconditionally;
        return set_poli_and_charset_<poli>(charset);
    }

    // deprecated aliases

    STRF_DEPRECATED_MSG("conv was renamed to transcode")
    constexpr STRF_HD return_type_transcode_<Charset> conv() const
    {
        return return_type_transcode_<Charset>{ self_downcast_() };
    }
    STRF_DEPRECATED_MSG("convert_charset was renamed to transcode")
    constexpr STRF_HD return_type_transcode_<Charset> convert_charset() const
    {
        return return_type_transcode_<Charset>{ self_downcast_() };
    }
    STRF_DEPRECATED_MSG("sanitize_charset was renamed to sani")
    constexpr STRF_HD return_type_sani_<Charset> sanitize_charset() const
    {
        return return_type_sani_<Charset>{ self_downcast_() };
    }
    template <typename OtherCharset>
    STRF_DEPRECATED_MSG("convert_from_charset was renamed to transcode")
    constexpr STRF_HD auto convert_from_charset(OtherCharset charset) const
    {
        constexpr auto poli = transcoding_policy::sani_only_if_different;
        return set_poli_and_charset_<poli>(charset);
    }
    template <typename OtherCharset>
    STRF_DEPRECATED_MSG("sanitize_from_charset was renamed to sanitize")
    constexpr STRF_HD auto sanitize_from_charset(OtherCharset charset) const
    {
        constexpr auto poli = transcoding_policy::sani_unconditionally;
        return set_poli_and_charset_<poli>(charset);
    }

    // observers
    static constexpr STRF_HD transcoding_policy get_transcondig_policy() noexcept
    {
        return TranscPoli;
    }

    static constexpr STRF_HD bool has_charset() noexcept
    {
        return ! std::is_same<Charset, no_charset_tag>::value;
    }
    static constexpr STRF_HD bool has_static_charset() noexcept
    {
        return detail::is_static_charset<Charset>::value;
    }
    static constexpr STRF_HD bool has_dynamic_charset() noexcept
    {
        return detail::is_dynamic_charset<Charset>::value;
    }
    constexpr STRF_HD Charset get_charset() const noexcept
    {
        return charset_;
    }
};

template <typename T, bool Active>
class string_precision_formatter_fn;

template <bool Active>
struct string_precision_formatter
{
    template <typename T>
    using fn = strf::detail::string_precision_formatter_fn<T, Active>;
};

template <bool Active>
struct string_precision
{
};

template <>
struct string_precision<true>
{
    strf::width_t precision;
};

template <typename T>
class string_precision_formatter_fn<T, true>
{
public:
    constexpr STRF_HD explicit string_precision_formatter_fn(strf::width_t p) noexcept
        : precision_(p)
    {
    }
    template <typename U>
    constexpr STRF_HD explicit string_precision_formatter_fn
        ( strf::detail::string_precision_formatter_fn<U, true> other ) noexcept
        : precision_(other.precision())
    {
    }
    STRF_CONSTEXPR_IN_CXX14 STRF_HD T&& p(strf::width_t _) && noexcept
    {
        precision_ = _;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD strf::width_t precision() const noexcept
    {
        return precision_;
    }
    constexpr STRF_HD strf::detail::string_precision<true> get_string_precision() const noexcept
    {
        return strf::detail::string_precision<true>{precision_};
    }
    constexpr static STRF_HD bool has_string_precision() noexcept
    {
        return true;
    }

private:

    strf::width_t precision_;
};


template <typename T>
class string_precision_formatter_fn<T, false>
{
    using adapted_derived_type_
        = strf::fmt_replace< T
                           , strf::detail::string_precision_formatter<false>
                           , strf::detail::string_precision_formatter<true> >;

    STRF_HD constexpr const T& self_downcast_() const
    {
        return *static_cast<const T*>(this);
    }

public:

    constexpr string_precision_formatter_fn() noexcept = default;
    template <typename U>
    constexpr STRF_HD explicit string_precision_formatter_fn
        ( string_precision_formatter_fn<U, false> ) noexcept
    {
    }
    constexpr STRF_HD adapted_derived_type_ p(strf::width_t precision) const noexcept
    {
        return { self_downcast_()
               , strf::tag<strf::detail::string_precision_formatter<true> >{}
               , precision };
    }
    constexpr STRF_HD strf::detail::string_precision<false> get_string_precision() const noexcept
    {
        return strf::detail::string_precision<false>{};
    }
    constexpr static STRF_HD bool has_string_precision() noexcept
    {
        return false;
    }
};

template <typename SrcCharT> struct string_printing;

template <typename SrcCharT, typename DstCharT> class strcpy_printer;
template <typename SrcCharT, typename DstCharT> class aligned_strcpy_printer;
template <typename SrcCharT, typename DstCharT> class transcode_printer;
template <typename SrcCharT, typename DstCharT> class aligned_transcode_printer;
template <typename SrcCharT, typename DstCharT> class unsafe_transcode_printer;
template <typename SrcCharT, typename DstCharT> class aligned_unsafe_transcode_printer;
template <typename DstCharT> class transcode_printer_variant;
template <typename DstCharT> class aligned_transcode_printer_variant;

template <typename SrcCharT, typename PrePrinting, typename FPack>
struct string_printer_input
{
    using DstCharT = SrcCharT;
    using printer_type = strcpy_printer<SrcCharT, DstCharT>;

    strf::detail::simple_string_view<SrcCharT> arg;
    PrePrinting* pre;
    FPack facets;
};

enum class charsets_comparison { statically_equal, statically_different, dynamic };

template <typename CharsetA, typename CharsetB>
struct compare_charsets_2
{
    using CodeUnitA = typename CharsetA::code_unit;
    using CodeUnitB = typename CharsetB::code_unit;

    static constexpr charsets_comparison value =
        ( sizeof(CodeUnitA) == sizeof(CodeUnitB)
        ? charsets_comparison::dynamic
        : charsets_comparison::statically_different );
};

template <typename CharTA, strf::charset_id IdA, typename CharTB, strf::charset_id IdB>
struct compare_charsets_2<strf::static_charset<CharTA, IdA>, strf::static_charset<CharTB, IdB> >
{
    static constexpr charsets_comparison value =
        ( IdA == IdB
        ? charsets_comparison::statically_equal
        : charsets_comparison::statically_different );

    static_assert( ! (IdA == IdB && sizeof(CharTA) != sizeof(CharTB))
                 , "Same charset_id, but different Code Unit size" );
};


template < typename SrcCharT, typename DstCharT, typename Facets, typename SrcCharset >
struct compare_charsets;

template < typename SrcCharT, typename DstCharT, typename Facets >
struct compare_charsets<SrcCharT, DstCharT, Facets, no_charset_tag>
{
    using ftag = strf::string_input_tag<SrcCharT>;
    using src_charset  = strf::facet_type_in_pack<strf::charset_c< SrcCharT>, ftag, Facets>;
    using dst_charset = strf::facet_type_in_pack<strf::charset_c<DstCharT>, ftag, Facets>;

    static constexpr charsets_comparison value =
        compare_charsets_2<src_charset, dst_charset>::value;
};

template < typename SrcCharT, typename DstCharT, typename Facets, strf::charset_id CsId>
struct compare_charsets
    <SrcCharT, DstCharT, Facets, strf::static_charset<SrcCharT, CsId> >
{
    using ftag = strf::string_input_tag<SrcCharT>;
    using src_charset  = strf::static_charset<SrcCharT, CsId>;
    using dst_charset = strf::facet_type_in_pack<strf::charset_c<DstCharT>, ftag, Facets>;

    static constexpr charsets_comparison value =
        compare_charsets_2<src_charset, dst_charset>::value;
};

template < typename SrcCharT, typename DstCharT, typename Facets>
struct compare_charsets <SrcCharT, DstCharT, Facets, strf::dynamic_charset<SrcCharT> >
{
    static constexpr charsets_comparison value =
        ( sizeof(DstCharT) == sizeof(SrcCharT)
        ? charsets_comparison::dynamic
        : charsets_comparison::statically_different );
};

template <typename SrcCharT, typename DstCharT>
struct get_no_transcode_printer
{
    static_assert( std::is_same<SrcCharT, DstCharT>::value, "Character types mismatch");
    using type = strcpy_printer<SrcCharT, DstCharT>;
};
template <typename SrcCharT, typename DstCharT>
struct get_no_transcode_aligned_printer
{
    static_assert( std::is_same<SrcCharT, DstCharT>::value, "Character types mismatch");
    using type = aligned_strcpy_printer<SrcCharT, DstCharT>;
};

template <bool HasAlignment, charsets_comparison, transcoding_policy>
struct string_printer_type;

template <charsets_comparison CharsetCmp>
struct string_printer_type<true, CharsetCmp, transcoding_policy::no_transcode>
{
    template <typename SrcCharT, typename DstCharT>
    using type = typename get_no_transcode_aligned_printer<SrcCharT, DstCharT>::type;
};

template <charsets_comparison CharsetCmp>
struct string_printer_type<false, CharsetCmp, transcoding_policy::no_transcode>
{
    template <typename SrcCharT, typename DstCharT>
    using type = typename get_no_transcode_printer<SrcCharT, DstCharT>::type;
};

template <charsets_comparison CharsetCmp>
struct string_printer_type
    < true, CharsetCmp, transcoding_policy::sani_unconditionally>
{
    template <typename SrcCharT, typename DstCharT>
    using type = aligned_transcode_printer<SrcCharT, DstCharT>;
};

template <charsets_comparison CharsetCmp>
struct string_printer_type
    < false, CharsetCmp, transcoding_policy::sani_unconditionally>
{
    template <typename SrcCharT, typename DstCharT>
    using type = transcode_printer<SrcCharT, DstCharT>;
};

template <>
struct string_printer_type
    < true, charsets_comparison::dynamic, transcoding_policy::unsafe>
{
    template <typename SrcCharT, typename DstCharT>
    using type = aligned_transcode_printer_variant<DstCharT>;
};
template <>
struct string_printer_type
    < false, charsets_comparison::dynamic, transcoding_policy::unsafe>
{
    template <typename SrcCharT, typename DstCharT>
    using type = transcode_printer_variant<DstCharT>;
};
template <>
struct string_printer_type
    < true, charsets_comparison::dynamic, transcoding_policy::sani_only_if_different>
{
    template <typename SrcCharT, typename DstCharT>
    using type = aligned_transcode_printer_variant<DstCharT>;
};
template <>
struct string_printer_type
    < false, charsets_comparison::dynamic, transcoding_policy::sani_only_if_different>
{
    template <typename SrcCharT, typename DstCharT>
    using type = transcode_printer_variant<DstCharT>;
};

template <charsets_comparison CharsetCmp>
struct string_printer_type
    < true, CharsetCmp, transcoding_policy::unsafe>
{
    template <typename SrcCharT, typename DstCharT>
    using type = aligned_unsafe_transcode_printer<SrcCharT, DstCharT>;
};
template <charsets_comparison CharsetCmp>
struct string_printer_type
    < false, CharsetCmp, transcoding_policy::unsafe>
{
    template <typename SrcCharT, typename DstCharT>
    using type = unsafe_transcode_printer<SrcCharT, DstCharT>;
};
template <>
struct string_printer_type
    < true, charsets_comparison::statically_equal, transcoding_policy::sani_only_if_different>
{
    template <typename SrcCharT, typename DstCharT>
    using type = aligned_unsafe_transcode_printer<SrcCharT, DstCharT>;
};
template <>
struct string_printer_type
    < false, charsets_comparison::statically_equal, transcoding_policy::sani_only_if_different>
{
    template <typename SrcCharT, typename DstCharT>
    using type = unsafe_transcode_printer<SrcCharT, DstCharT>;
};

template <>
struct string_printer_type
    < true, charsets_comparison::statically_different, transcoding_policy::sani_only_if_different>
{
    template <typename SrcCharT, typename DstCharT>
    using type = aligned_transcode_printer<SrcCharT, DstCharT>;
};
template <>
struct string_printer_type
    < false, charsets_comparison::statically_different, transcoding_policy::sani_only_if_different>
{
    template <typename SrcCharT, typename DstCharT>
    using type = transcode_printer<SrcCharT, DstCharT>;
};

template <typename SrcCharT>
struct string_printing;

template < typename SrcCharT, typename DstCharT
         , bool HasPrecision, bool HasAlignment
         , typename Charset, transcoding_policy TranscPoli
         , typename PrePrinting, typename FPack >
struct fmt_string_printer_input
{
    static constexpr auto charset_cmp_ =
        compare_charsets<SrcCharT, DstCharT, FPack, Charset>::value;

    using printer_type = typename string_printer_type
        < HasAlignment, charset_cmp_, TranscPoli >
        :: template type <SrcCharT, DstCharT>;

    strf::printable_with_fmt
        < string_printing<SrcCharT>
        , strf::detail::string_precision_formatter<HasPrecision>
        , strf::alignment_formatter_q<HasAlignment>
        , transcoding_formatter_q<SrcCharT, Charset, TranscPoli> > arg;
    PrePrinting* pre;
    FPack facets;
};

template <typename SrcCharT>
struct string_printing
{
    using representative_type = strf::string_input_tag<SrcCharT>;
    using forwarded_type = strf::detail::simple_string_view<SrcCharT>;
    using formatters = strf::tag
        < strf::detail::string_precision_formatter<false>
        , strf::alignment_formatter
        , strf::detail::transcoding_formatter<SrcCharT> >;
    using is_overridable = std::false_type;

    template <typename DstCharT, typename PrePrinting, typename FPack>
    constexpr STRF_HD static auto make_input
        ( strf::tag<DstCharT>
        , PrePrinting* pre
        , const FPack& facets
        , forwarded_type x ) noexcept
        -> strf::detail::string_printer_input<SrcCharT, PrePrinting, FPack>
    {
        static_assert
            ( std::is_same<SrcCharT, DstCharT>::value
            , "Character type mismatch. Use `transcode` or `sani` format function." );

        return {x, pre, facets};
    }

    template < typename DstCharT, typename PrePrinting, typename FPack
             , bool HasPrecision, bool HasAlignment
             , typename Charset, transcoding_policy TranscPoli >
    constexpr STRF_HD static auto make_input
        ( strf::tag<DstCharT>
        , PrePrinting* pre
        , const FPack& facets
        , const strf::printable_with_fmt
            < string_printing<SrcCharT>
            , string_precision_formatter<HasPrecision>
            , strf::alignment_formatter_q<HasAlignment>
            , transcoding_formatter_q<SrcCharT, Charset, TranscPoli> >& x ) noexcept
        -> strf::detail::fmt_string_printer_input
            < SrcCharT, DstCharT, HasPrecision, HasAlignment
            , Charset, TranscPoli, PrePrinting, FPack >
    {
        return {x, pre, facets};
    }
};

} // namespace detail

template <typename CharIn>
constexpr STRF_HD auto tag_invoke
    (strf::printable_tag, strf::detail::simple_string_view<CharIn>) noexcept
    -> strf::detail::string_printing<CharIn>
    { return {}; }

#if defined(STRF_HAS_STD_STRING_DECLARATION)

template <typename CharIn, typename Traits, typename Allocator>
constexpr STRF_HD auto tag_invoke
    (strf::printable_tag, const std::basic_string<CharIn, Traits, Allocator>&) noexcept
    -> strf::detail::string_printing<CharIn>
    { return {}; }

#endif // defined(STRF_HAS_STD_STRING_DECLARATION)

#if defined(STRF_HAS_STD_STRING_VIEW)

template <typename CharIn, typename Traits>
constexpr STRF_HD auto tag_invoke
    (strf::printable_tag, std::basic_string_view<CharIn, Traits>) noexcept
    -> strf::detail::string_printing<CharIn>
    { return {}; }

#if defined(__cpp_char8_t)

constexpr STRF_HD auto tag_invoke
    (strf::printable_tag, std::basic_string_view<char8_t>) noexcept
    -> strf::detail::string_printing<char8_t>
    { return {}; }

#endif // defined(__cpp_char8_t)

constexpr STRF_HD auto tag_invoke(strf::printable_tag, std::basic_string_view<char>) noexcept
    -> strf::detail::string_printing<char>
    { return {}; }


constexpr STRF_HD auto tag_invoke(strf::printable_tag, std::basic_string_view<char16_t>) noexcept
    -> strf::detail::string_printing<char16_t>
    { return {}; }


constexpr STRF_HD auto tag_invoke(strf::printable_tag, std::basic_string_view<char32_t>) noexcept
    -> strf::detail::string_printing<char32_t>
    { return {}; }


constexpr STRF_HD auto tag_invoke(strf::printable_tag, std::basic_string_view<wchar_t>) noexcept
    -> strf::detail::string_printing<wchar_t>
    { return {}; }

#endif // defined(STRF_HAS_STD_STRING_VIEW)

#if defined(__cpp_char8_t)


constexpr STRF_HD auto tag_invoke(strf::printable_tag, const char8_t*) noexcept
    -> strf::detail::string_printing<char8_t>
    { return {}; }

#endif

constexpr STRF_HD auto tag_invoke(strf::printable_tag, const char*) noexcept
    -> strf::detail::string_printing<char>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::printable_tag, const char16_t*) noexcept
    -> strf::detail::string_printing<char16_t>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::printable_tag, const char32_t*) noexcept
    -> strf::detail::string_printing<char32_t>
    { return {}; }

constexpr STRF_HD auto tag_invoke(strf::printable_tag, const wchar_t*) noexcept
    -> strf::detail::string_printing<wchar_t>
    { return {}; }

namespace detail {

template <typename SrcCharT, typename DstCharT>
class strcpy_printer: public strf::printer<DstCharT>
{
public:
    static_assert(sizeof(SrcCharT) == sizeof(DstCharT), "");

    template <typename PrePrinting, typename FPack>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD explicit strcpy_printer
        ( const strf::detail::string_printer_input<SrcCharT, PrePrinting, FPack>& input )
        : str_(input.arg.data())
        , str_end_(input.arg.end())
    {
        STRF_IF_CONSTEXPR(PrePrinting::width_required) {
            auto&& wcalc = use_facet_<strf::width_calculator_c>(input.facets);
            auto w = wcalc.str_width
                ( use_facet_<strf::charset_c<SrcCharT>>(input.facets)
                , input.pre->remaining_width(), str_, str_end_ );
           input.pre->subtract_width(w);
        }
        input.pre->add_size(input.arg.length());
    }

    template < typename PrePrinting, typename FPack
             , typename Charset, transcoding_policy TranscPoli >
    STRF_CONSTEXPR_IN_CXX14 STRF_HD explicit strcpy_printer
        ( const strf::detail::fmt_string_printer_input
            < SrcCharT, DstCharT, false, false, Charset, TranscPoli, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().data())
        , str_end_(input.arg.value().end())
    {
        STRF_IF_CONSTEXPR(PrePrinting::width_required) {
            auto&& wcalc = use_facet_<strf::width_calculator_c>(input.facets);
            auto w = wcalc.str_width
                ( use_facet_<strf::charset_c<SrcCharT>>(input.facets)
                , input.pre->remaining_width()
                , str_
                , str_end_ );
           input.pre->subtract_width(w);
        }
        input.pre->add_size(input.arg.value().size());
    }

    template < typename PrePrinting, typename FPack
             , typename Charset, transcoding_policy TranscPoli >
    STRF_CONSTEXPR_IN_CXX14 STRF_HD explicit strcpy_printer
        ( const strf::detail::fmt_string_printer_input
            < SrcCharT, DstCharT, true, false, Charset, TranscPoli, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().data())
    {
        auto&& wcalc = use_facet_<strf::width_calculator_c>(input.facets);
        auto res = wcalc.str_width_and_pos
            ( use_facet_<strf::charset_c<SrcCharT>>(input.facets)
            , input.arg.precision()
            , str_
            , input.arg.value().size() );
        str_end_ = res.ptr;
        input.pre->subtract_width(res.width);
        input.pre->add_size(res.pos);
    }

    STRF_HD void print_to(strf::destination<DstCharT>& dst) const override;

private:

    const SrcCharT* str_;
    const SrcCharT* str_end_;

    template < typename Category
             , typename FPack
             , typename input_tag = strf::string_input_tag<SrcCharT> >
    static STRF_HD
    STRF_DECLTYPE_AUTO((strf::use_facet<Category, input_tag>(std::declval<FPack>())))
    use_facet_(const FPack& facets)
    {
        return facets.template use_facet<Category, input_tag>();
    }
};

template<typename SrcCharT, typename DstCharT>
STRF_HD void strcpy_printer<SrcCharT, DstCharT>::print_to
    ( strf::destination<DstCharT>& dst ) const
{
    strf::detail::output_buffer_interchar_copy(dst, str_, str_end_);
}

template <typename SrcCharT, typename DstCharT>
class aligned_strcpy_printer: public strf::printer<DstCharT>
{
public:
    static_assert(sizeof(SrcCharT) == sizeof(DstCharT), "");

    template < typename PrePrinting, typename FPack
             , typename Charset, transcoding_policy TranscPoli >
    STRF_HD explicit aligned_strcpy_printer
        ( const strf::detail::fmt_string_printer_input
            < SrcCharT, DstCharT, false, true, Charset, TranscPoli, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().data())
        , str_end_(input.arg.value().end())
        , afmt_(input.arg.get_alignment_format())
    {

        auto&& wcalc = use_facet_<strf::width_calculator_c>(input.facets);
        auto src_charset = use_facet_<strf::charset_c<SrcCharT>>(input.facets);
        auto dst_charset = use_facet_<strf::charset_c<DstCharT>>(input.facets);
        const strf::width_t limit =
            ( PrePrinting::width_required && input.pre->remaining_width() > afmt_.width
            ? input.pre->remaining_width()
            : afmt_.width );
        auto strw = wcalc.str_width(src_charset, limit, str_, str_end_);
        encode_fill_ = dst_charset.encode_fill_func();
        auto fillcount = init_(input.pre, strw);
        precalc_size_(input.pre, dst_charset, fillcount);
    }

    template < typename PrePrinting, typename FPack
             , typename Charset, transcoding_policy TranscPoli >
    STRF_HD explicit aligned_strcpy_printer
        ( const strf::detail::fmt_string_printer_input
            < SrcCharT, DstCharT, true, true, Charset, TranscPoli, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().begin())
        , afmt_(input.arg.get_alignment_format())
    {
        auto&& wcalc = use_facet_<strf::width_calculator_c>(input.facets);
        auto src_charset = use_facet_<strf::charset_c<SrcCharT>>(input.facets);
        auto dst_charset = use_facet_<strf::charset_c<DstCharT>>(input.facets);
        auto res = wcalc.str_width_and_pos
            ( src_charset, input.arg.precision(), str_, input.arg.value().end() );
        str_end_ = res.ptr;
        encode_fill_ = dst_charset.encode_fill_func();
        auto fillcount = init_(input.pre, res.width);
        precalc_size_(input.pre, dst_charset, fillcount);
    }

    STRF_HD void print_to(strf::destination<DstCharT>& dst) const override;

private:

    const SrcCharT* str_;
    const SrcCharT* str_end_;
    strf::encode_fill_f<DstCharT> encode_fill_;
    strf::alignment_format afmt_;
    int left_fillcount_{};
    int right_fillcount_{};

    template < typename Category
             , typename FPack
             , typename Tag = strf::string_input_tag<SrcCharT> >
    static STRF_HD
    STRF_DECLTYPE_AUTO((strf::use_facet<Category, Tag>(std::declval<FPack>())))
    use_facet_(const FPack& facets)
    {
        return facets.template use_facet<Category, Tag>();
    }

    template <typename PrePrinting>
    STRF_HD int init_(PrePrinting*, strf::width_t strw);

    template <typename Charset>
    STRF_HD void precalc_size_
        ( strf::size_accumulator<true>* pre, Charset charset, int fillcount )
    {
        pre->add_size(str_end_ - str_);
        if (fillcount > 0) {
            pre->add_size(fillcount * charset.encoded_char_size(afmt_.fill));
        }
    }

    template <typename Charset>
    STRF_HD void precalc_size_(strf::size_accumulator<false>*, Charset, int)
    {
    }
};

template<typename SrcCharT, typename DstCharT>
template <typename PrePrinting>
inline STRF_HD int aligned_strcpy_printer<SrcCharT, DstCharT>::init_
    ( PrePrinting* pre, strf::width_t strw )
{
    if (afmt_.width > strw) {
        const int fillcount = (afmt_.width - strw).round();
        switch(afmt_.alignment) {
            case strf::text_alignment::left:
                left_fillcount_ = 0;
                right_fillcount_ = fillcount;
                break;
            case strf::text_alignment::center: {
                int const halfcount = fillcount >> 1;
                left_fillcount_ = halfcount;
                right_fillcount_ = fillcount - halfcount;
                break;
            }
            default:
                left_fillcount_ = fillcount;
                right_fillcount_ = 0;
        }
        pre->subtract_width(strw);
        pre->subtract_width(static_cast<width_t>(fillcount));
        return fillcount;
    }
    right_fillcount_ = 0;
    left_fillcount_ = 0;
    pre->subtract_width(strw);
    return 0;
}

template<typename SrcCharT, typename DstCharT>
void STRF_HD aligned_strcpy_printer<SrcCharT, DstCharT>::print_to
    ( strf::destination<DstCharT>& dst ) const
{
    if (left_fillcount_ > 0) {
        encode_fill_(dst, left_fillcount_, afmt_.fill);
    }
    strf::detail::output_buffer_interchar_copy(dst, str_, str_end_);
    if (right_fillcount_ > 0) {
        encode_fill_(dst, right_fillcount_, afmt_.fill);
    }
}


template <typename SrcCharT, typename InputT>
constexpr STRF_HD auto do_get_src_charset(std::false_type, const InputT& input)
{
    return strf::use_facet
        <strf::charset_c<SrcCharT>, strf::string_input_tag<SrcCharT>>
        ( input.facets );
}

template <typename SrcCharT, typename InputT>
constexpr STRF_HD auto do_get_src_charset(std::true_type, const InputT& input)
{
    return input.arg.get_charset();
}

template < typename SrcCharT, typename DstCharT, bool HasP, bool HasA
         , typename Charset, transcoding_policy TranscPoli
         , typename PrePrinting, typename FPack >
constexpr STRF_HD auto get_src_charset
    ( const strf::detail::fmt_string_printer_input
        < SrcCharT, DstCharT, HasP, HasA, Charset, TranscPoli, PrePrinting, FPack>&
      input )
{
    return strf::detail::do_get_src_charset<SrcCharT>(is_charset<Charset>(), input);
}

template <typename SrcCharT, typename DstCharT>
class transcode_printer: public strf::printer<DstCharT>
{
public:

    template < typename PrePrinting, typename FPack
             , typename Charset, transcoding_policy TranscPoli >
    STRF_HD explicit transcode_printer
        ( const strf::detail::fmt_string_printer_input
            < SrcCharT, DstCharT, false, false, Charset, TranscPoli, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().data())
        , str_end_(input.arg.value().end())
        , err_notifier_
              ( use_facet_<strf::transcoding_error_notifier_c, SrcCharT>(input.facets).get() )
    {
        auto src_charset  = strf::detail::get_src_charset(input);
        auto dst_charset = use_facet_<strf::charset_c<DstCharT>, SrcCharT>(input.facets);
        if (input.pre->has_remaining_width()) {
            auto&& wcalc = use_facet_<strf::width_calculator_c, SrcCharT>(input.facets);
            auto w = wcalc.str_width( src_charset, input.pre->remaining_width(), str_, str_end_);
            input.pre->subtract_width(w);
        }
        init_(input.pre, src_charset, dst_charset);
    }

    template < typename PrePrinting, typename FPack
             , typename Charset, transcoding_policy TranscPoli >
    STRF_HD explicit transcode_printer
        ( const strf::detail::fmt_string_printer_input
            < SrcCharT, DstCharT, true, false, Charset, TranscPoli, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().data())
        , err_notifier_
            ( use_facet_<strf::transcoding_error_notifier_c, SrcCharT>(input.facets).get() )
    {
        auto src_charset  = strf::detail::get_src_charset(input);
        auto dst_charset = use_facet_<strf::charset_c<DstCharT>, SrcCharT>(input.facets);
        auto&& wcalc = use_facet_<strf::width_calculator_c, SrcCharT>(input.facets);
        auto res = wcalc.str_width_and_pos
            ( src_charset, input.arg.precision(), str_, input.arg.value().size() );
        str_end_ = res.ptr;
        input.pre->subtract_width(res.width);
        init_( input.pre, src_charset
             , use_facet_<strf::charset_c<DstCharT>, SrcCharT>(input.facets));
    }

    STRF_HD void print_to(strf::destination<DstCharT>& dst) const override;

private:

    template < typename PrePrinting, typename SrcCharset, typename DstCharset >
    STRF_HD void init_(PrePrinting* pre, SrcCharset src_charset, DstCharset dst_charset)
    {
        auto transcoder = find_transcoder(src_charset, dst_charset);
        STRF_MAYBE_UNUSED(transcoder);
        transcode_ = transcoder.transcode_func();
        if (transcode_ == nullptr) {
            src_to_u32_ = src_charset.to_u32().transcode_func();
            u32size_ = src_charset.to_u32().transcode_size_func();
            u32_to_dst_ = dst_charset.from_u32().transcode_func();
        }
        STRF_IF_CONSTEXPR (PrePrinting::size_required) {
            strf::transcode_size_f<SrcCharT> transcode_size
                = transcoder.transcode_size_func();
            std::ptrdiff_t s = 0;
            constexpr auto flags = strf::transcode_flags::none;
            if (transcode_size != nullptr) {
                s = transcode_size(str_, str_end_, strf::ssize_max, flags).ssize;
            } else {
                s = strf::decode_encode_size<SrcCharT>
                    ( src_charset.to_u32().transcode_func()
                    , dst_charset.from_u32().transcode_size_func()
                    , str_, str_end_, strf::ssize_max, flags )
                    .ssize;
            }
            pre->add_size(s);
        }
    }

    STRF_HD bool can_transcode_directly() const
    {
        return u32_to_dst_ == nullptr;
    }

    const SrcCharT* const str_;
    const SrcCharT* str_end_;
    union {
        strf::transcode_f<SrcCharT, DstCharT>  transcode_;
        strf::transcode_f<SrcCharT, char32_t>  src_to_u32_;
    };
    strf::transcode_size_f<SrcCharT> u32size_ = nullptr;
    strf::transcode_f<char32_t, DstCharT> u32_to_dst_ = nullptr;
    strf::transcoding_error_notifier* err_notifier_;

    template < typename Category, typename SrcChar, typename FPack
             , typename Tag = strf::string_input_tag<SrcChar> >
    static STRF_HD
    STRF_DECLTYPE_AUTO((strf::use_facet<Category, Tag>(std::declval<FPack>())))
    use_facet_(const FPack& facets)
    {
        return facets.template use_facet<Category, Tag>();
    }
};

template<typename SrcCharT, typename DstCharT>
STRF_HD void transcode_printer<SrcCharT, DstCharT>::print_to
    ( strf::destination<DstCharT>& dst ) const
{
    const auto flags = strf::transcode_flags::none;
    auto src_it = str_;
    auto dst_it = dst.buffer_ptr();
    auto dst_end = dst.buffer_end();
    if (can_transcode_directly()) {
        while (1) {
            auto r = transcode_(src_it, str_end_, dst_it, dst_end, err_notifier_, flags);
            dst.advance_to(r.dst_ptr);
            if (r.stop_reason != strf::transcode_stop_reason::insufficient_output_space) {
                break;
            }
            src_it = r.src_ptr;
            dst.recycle();
            dst_it = dst.buffer_ptr();
            dst_end = dst.buffer_end();
        }
    } else {
        while (1) {
            auto r = strf::decode_encode<SrcCharT, DstCharT>
                ( src_to_u32_, u32_to_dst_, src_it , str_end_
                , dst_it, dst_end, err_notifier_, flags );
            dst.advance_to(r.dst_ptr);
            if (r.stop_reason != strf::transcode_stop_reason::insufficient_output_space) {
                break;
            }
            dst.recycle();
            dst_it = dst.buffer_ptr();
            dst_end = dst.buffer_end();
            if (r.u32dist == 0) {
                src_it = r.stale_src_ptr;
            } else {
                src_it = u32size_(src_it, str_end_, r.u32dist, flags).src_ptr;
            }
        }
    }
}

template<typename SrcCharT, typename DstCharT>
class aligned_transcode_printer: public printer<DstCharT>
{
public:

    template < typename PrePrinting, typename FPack
             , typename Charset, transcoding_policy TranscPoli >
    STRF_HD explicit aligned_transcode_printer
        ( const strf::detail::fmt_string_printer_input
            < SrcCharT, DstCharT, false, true, Charset, TranscPoli, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().data())
        , str_end_(input.arg.value().end())
        , afmt_(input.arg.get_alignment_format())
        , err_notifier_
            ( use_facet_<strf::transcoding_error_notifier_c, SrcCharT>(input.facets).get() )
    {
        auto src_charset = strf::detail::get_src_charset(input);
        auto&& wcalc = use_facet_<strf::width_calculator_c, SrcCharT>(input.facets);
        const strf::width_t limit =
            ( PrePrinting::width_required && input.pre->remaining_width() > afmt_.width
            ? input.pre->remaining_width()
            : afmt_.width );
        auto str_width = wcalc.str_width(src_charset, limit, str_, str_end_);
        init_( input.pre, str_width, src_charset
             , use_facet_<strf::charset_c<DstCharT>, SrcCharT>(input.facets) );
    }

    template < typename PrePrinting, typename FPack
             , typename Charset, transcoding_policy TranscPoli>
    STRF_HD explicit aligned_transcode_printer
        ( const strf::detail::fmt_string_printer_input
            < SrcCharT, DstCharT, true, true, Charset, TranscPoli, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().data())
        , str_end_(input.arg.value().end())
        , afmt_(input.arg.get_alignment_format())
        , err_notifier_
            ( use_facet_<strf::transcoding_error_notifier_c, SrcCharT>(input.facets).get() )
    {
        auto src_charset = strf::detail::get_src_charset(input);
        auto&& wcalc = use_facet_<strf::width_calculator_c, SrcCharT>(input.facets);
        auto res = wcalc.str_width_and_pos
            ( src_charset, input.arg.precision(), str_
            , input.arg.value().size() );
        str_end_ = res.ptr;
        init_( input.pre, res.width, src_charset
             , use_facet_<strf::charset_c<DstCharT>, SrcCharT>(input.facets) );
    }

    STRF_HD void print_to(strf::destination<DstCharT>& dst) const override;

private:

    STRF_HD bool can_transcode_directly() const
    {
        return u32_to_dst_ == nullptr;
    }

    const SrcCharT* str_;
    const SrcCharT* str_end_;
    strf::alignment_format afmt_;
    union {
        strf::transcode_f<SrcCharT, DstCharT>  transcode_;
        strf::transcode_f<SrcCharT, char32_t>  src_to_u32_;
    };
    strf::transcode_size_f<SrcCharT> u32size_ = nullptr;
    strf::transcode_f<char32_t, DstCharT> u32_to_dst_ = nullptr;
    strf::encode_fill_f<DstCharT> encode_fill_ = nullptr;
    strf::transcoding_error_notifier* err_notifier_;
    int left_fillcount_ = 0;
    int right_fillcount_ = 0;

    template < typename Category, typename SrcChar, typename FPack
             , typename Tag = strf::string_input_tag<SrcChar> >
    static STRF_HD
    STRF_DECLTYPE_AUTO((strf::use_facet<Category, Tag>(std::declval<FPack>())))
    use_facet_(const FPack& facets)
    {
        return facets.template use_facet<Category, Tag>();
    }

    template < typename PrePrinting, typename SrcCharset, typename DstCharset>
    STRF_HD void init_
        ( PrePrinting* pre, strf::width_t str_width
        , SrcCharset src_charset, DstCharset dst_charset );
};

template <typename SrcCharT, typename DstCharT>
template <typename PrePrinting, typename SrcCharset, typename DstCharset>
void STRF_HD aligned_transcode_printer<SrcCharT, DstCharT>::init_
    ( PrePrinting* pre, strf::width_t str_width
    , SrcCharset src_charset, DstCharset dst_charset )
{
    encode_fill_ = dst_charset.encode_fill_func();
    auto transcoder = find_transcoder(src_charset, dst_charset);
    STRF_MAYBE_UNUSED(transcoder);
    transcode_ = transcoder.transcode_func();
    if (transcode_ == nullptr) {
        src_to_u32_ = src_charset.to_u32().transcode_func();
        u32size_ = src_charset.to_u32().transcode_size_func();
        u32_to_dst_ = dst_charset.from_u32().transcode_func();
    }
    int fillcount = 0;
    if (afmt_.width > str_width) {
        fillcount = (afmt_.width - str_width).round();
        switch(afmt_.alignment) {
            case strf::text_alignment::left:
                left_fillcount_ = 0;
                right_fillcount_ = fillcount;
                break;
            case strf::text_alignment::center: {
                const auto halfcount = fillcount / 2;
                left_fillcount_ = halfcount;
                right_fillcount_ = fillcount - halfcount;
                break;
            }
            default:
                left_fillcount_ = fillcount;
                right_fillcount_ = 0;
        }
        pre->subtract_width(str_width);
        pre->subtract_width(static_cast<width_t>(fillcount));
    } else {
        right_fillcount_ = 0;
        left_fillcount_ = 0;
        pre->subtract_width(str_width);
    }
    STRF_IF_CONSTEXPR (PrePrinting::size_required) {
        std::ptrdiff_t s = 0;
        strf::transcode_size_f<SrcCharT> transcode_size
                = transcoder.transcode_size_func();
        constexpr auto flags = strf::transcode_flags::none;
        if (transcode_size != nullptr) {
            s = transcode_size(str_, str_end_, strf::ssize_max, flags).ssize;
        } else {
            s = strf::decode_encode_size<SrcCharT>
                ( src_charset.to_u32().transcode_func()
                , dst_charset.from_u32().transcode_size_func()
                , str_, str_end_, strf::ssize_max, flags )
                .ssize;
        }
        if (fillcount > 0) {
            s += dst_charset.encoded_char_size(afmt_.fill) * fillcount;
        }
        pre->add_size(s);
    }
}

template<typename SrcCharT, typename DstCharT>
void STRF_HD aligned_transcode_printer<SrcCharT, DstCharT>::print_to
    ( strf::destination<DstCharT>& dst ) const
{
    if (left_fillcount_ > 0) {
        encode_fill_(dst, left_fillcount_, afmt_.fill);
    }
    constexpr auto flags = strf::transcode_flags::none;
    auto src_it = str_;
    auto dst_it = dst.buffer_ptr();
    auto dst_end = dst.buffer_end();
    if (can_transcode_directly()) {
        while (1) {
            auto r = transcode_(src_it, str_end_, dst_it, dst_end, err_notifier_, flags);
            dst.advance_to(r.dst_ptr);
            if (r.stop_reason != strf::transcode_stop_reason::insufficient_output_space) {
                break;
            }
            src_it = r.src_ptr;
            dst.recycle();
            dst_it = dst.buffer_ptr();
            dst_end = dst.buffer_end();
        }
    } else {
        while (1) {
            auto r = strf::decode_encode<SrcCharT, DstCharT>
                ( src_to_u32_, u32_to_dst_, src_it , str_end_
                , dst_it, dst_end, err_notifier_, flags );
            dst.advance_to(r.dst_ptr);
            if (r.stop_reason != strf::transcode_stop_reason::insufficient_output_space) {
                break;
            }
            dst.recycle();
            dst_it = dst.buffer_ptr();
            dst_end = dst.buffer_end();
            if (r.u32dist == 0) {
                src_it = r.stale_src_ptr;
            } else {
                src_it = u32size_(src_it, str_end_, r.u32dist, flags).src_ptr;
            }
        }
    }
    if (right_fillcount_ > 0) {
        encode_fill_(dst, right_fillcount_, afmt_.fill);
    }
}

template<typename SrcCharT, typename DstCharT>
class unsafe_transcode_printer: public strf::printer<DstCharT>
{
public:

    template < typename PrePrinting, typename FPack
             , typename Charset, transcoding_policy TranscPoli>
    STRF_HD explicit unsafe_transcode_printer
        ( const strf::detail::fmt_string_printer_input
            < SrcCharT, DstCharT, false, false, Charset, TranscPoli, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().data())
        , str_end_(input.arg.value().end())
        , err_notifier_
              ( use_facet_<strf::transcoding_error_notifier_c, SrcCharT>(input.facets).get() )
    {
        auto src_charset  = strf::detail::get_src_charset(input);
        auto dst_charset = use_facet_<strf::charset_c<DstCharT>, SrcCharT>(input.facets);
        if (input.pre->has_remaining_width()) {
            auto&& wcalc = use_facet_<strf::width_calculator_c, SrcCharT>(input.facets);
            auto w = wcalc.str_width( src_charset, input.pre->remaining_width()
                                    , str_, str_end_);
            input.pre->subtract_width(w);
        }
        init_(input.pre, src_charset, dst_charset);
    }

    template < typename PrePrinting, typename FPack
             , typename Charset, transcoding_policy TranscPoli>
    STRF_HD explicit unsafe_transcode_printer
        ( const strf::detail::fmt_string_printer_input
            < SrcCharT, DstCharT, true, false, Charset, TranscPoli, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().data())
        , err_notifier_
            ( use_facet_<strf::transcoding_error_notifier_c, SrcCharT>(input.facets).get() )
    {
        auto src_charset  = strf::detail::get_src_charset(input);
        auto dst_charset = use_facet_<strf::charset_c<DstCharT>, SrcCharT>(input.facets);
        auto&& wcalc = use_facet_<strf::width_calculator_c, SrcCharT>(input.facets);
        auto res = wcalc.str_width_and_pos
            ( src_charset, input.arg.precision(), str_
            , input.arg.value().size() );
        str_end_ = res.ptr;
        input.pre->subtract_width(res.width);
        init_( input.pre, src_charset
             , use_facet_<strf::charset_c<DstCharT>, SrcCharT>(input.facets));
    }
    STRF_HD void print_to(strf::destination<DstCharT>& dst) const override;

private:

    template < typename PrePrinting, typename SrcCharset, typename DstCharset >
    STRF_HD void init_(PrePrinting* pre, SrcCharset src_charset, DstCharset dst_charset)
    {
        auto transcoder = find_transcoder(src_charset, dst_charset);
        STRF_MAYBE_UNUSED(transcoder);
        transcode_ = transcoder.unsafe_transcode_func();
        if (transcode_ == nullptr) {
            src_to_u32_ = src_charset.to_u32().unsafe_transcode_func();
            u32size_ = src_charset.to_u32().transcode_size_func();
            u32_to_dst_ = dst_charset.from_u32().unsafe_transcode_func();
        }
        STRF_IF_CONSTEXPR (PrePrinting::size_required) {
            const auto flags = strf::transcode_flags::none;
            strf::unsafe_transcode_size_f<SrcCharT>  transcode_size
                = transcoder.unsafe_transcode_size_func();
            std::ptrdiff_t s = 0;
            if (transcode_size != nullptr) {
                s = transcode_size(str_, str_end_, strf::ssize_max, flags).ssize;
            } else {
                s = strf::unsafe_decode_encode_size<SrcCharT>
                    ( src_charset.to_u32().unsafe_transcode_func()
                    , dst_charset.from_u32().unsafe_transcode_size_func()
                    , str_, str_end_, strf::ssize_max )
                    .ssize;
            }
            pre->add_size(s);
        }
    }

    STRF_HD bool can_transcode_directly() const
    {
        return u32_to_dst_ == nullptr;
    }


    const SrcCharT* const str_;
    const SrcCharT* str_end_;
    union {
        strf::unsafe_transcode_f<SrcCharT, DstCharT>  transcode_;
        strf::unsafe_transcode_f<SrcCharT, char32_t>  src_to_u32_;
    };
    strf::transcode_size_f<SrcCharT> u32size_ = nullptr;
    strf::unsafe_transcode_f<char32_t, DstCharT> u32_to_dst_ = nullptr;
    strf::transcoding_error_notifier* err_notifier_;

    template < typename Category, typename SrcChar, typename FPack
             , typename Tag = strf::string_input_tag<SrcChar> >
    static STRF_HD
    STRF_DECLTYPE_AUTO((strf::use_facet<Category, Tag>(std::declval<FPack>())))
    use_facet_(const FPack& facets)
    {
        return facets.template use_facet<Category, Tag>();
    }
};

template<typename SrcCharT, typename DstCharT>
STRF_HD void unsafe_transcode_printer<SrcCharT, DstCharT>::print_to
    ( strf::destination<DstCharT>& dst ) const
{
    constexpr auto flags = strf::transcode_flags::none;

    auto src_it = str_;
    auto dst_it = dst.buffer_ptr();
    auto dst_end = dst.buffer_end();
    if (can_transcode_directly()) {
        while (1) {
            auto r = transcode_(src_it, str_end_, dst_it, dst_end, err_notifier_, flags);
            dst.advance_to(r.dst_ptr);
            if (r.stop_reason != strf::transcode_stop_reason::insufficient_output_space) {
                break;
            }
            src_it = r.src_ptr;
            dst.recycle();
            dst_it = dst.buffer_ptr();
            dst_end = dst.buffer_end();
        }
    } else {
        while (1) {
            auto r = strf::unsafe_decode_encode<SrcCharT, DstCharT>
                ( src_to_u32_, u32_to_dst_, src_it , str_end_
                , dst_it, dst_end, err_notifier_, flags );
            dst.advance_to(r.dst_ptr);
            if (r.stop_reason != strf::transcode_stop_reason::insufficient_output_space) {
                break;
            }
            dst.recycle();
            dst_it = dst.buffer_ptr();
            dst_end = dst.buffer_end();
            if (r.u32dist == 0) {
                src_it = r.stale_src_ptr;
            } else {
                src_it = u32size_(src_it, str_end_, r.u32dist, flags).src_ptr;
            }
        }
    }
}


template<typename SrcCharT, typename DstCharT>
class aligned_unsafe_transcode_printer: public printer<DstCharT>
{
public:

    template < typename PrePrinting, typename FPack
             , typename Charset, transcoding_policy TranscPoli >
    STRF_HD explicit aligned_unsafe_transcode_printer
        ( const strf::detail::fmt_string_printer_input
            < SrcCharT, DstCharT, false, true, Charset, TranscPoli, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().data())
        , str_end_(input.arg.value().end())
        , afmt_(input.arg.get_alignment_format())
        , err_notifier_
            ( use_facet_<strf::transcoding_error_notifier_c, SrcCharT>(input.facets).get() )
    {
        auto src_charset = strf::detail::get_src_charset(input);
        auto&& wcalc = use_facet_<strf::width_calculator_c, SrcCharT>(input.facets);
        const strf::width_t limit =
            ( PrePrinting::width_required && input.pre->remaining_width() > afmt_.width
            ? input.pre->remaining_width()
            : afmt_.width );
        auto str_width = wcalc.str_width(src_charset, limit, str_, str_end_);
        init_( input.pre, str_width, src_charset
             , use_facet_<strf::charset_c<DstCharT>, SrcCharT>(input.facets) );
    }

    template < typename PrePrinting, typename FPack
             , typename Charset, transcoding_policy TranscPoli>
    STRF_HD explicit aligned_unsafe_transcode_printer
        ( const strf::detail::fmt_string_printer_input
            < SrcCharT, DstCharT, true, true, Charset, TranscPoli, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().data())
        , str_end_(input.arg.value().end())
        , afmt_(input.arg.get_alignment_format())
        , err_notifier_
            ( use_facet_<strf::transcoding_error_notifier_c, SrcCharT>(input.facets).get() )
    {
        auto src_charset = strf::detail::get_src_charset(input);
        auto&& wcalc = use_facet_<strf::width_calculator_c, SrcCharT>(input.facets);
        auto res = wcalc.str_width_and_pos
            ( src_charset, input.arg.precision(), str_
            , input.arg.value().size() );
        str_end_ = res.ptr;
        init_( input.pre, res.width, src_charset
             , use_facet_<strf::charset_c<DstCharT>, SrcCharT>(input.facets) );
    }

    STRF_HD void print_to(strf::destination<DstCharT>& dst) const override;

private:

    STRF_HD bool can_transcode_directly() const
    {
        return u32_to_dst_ == nullptr;
    }

    const SrcCharT* str_;
    const SrcCharT* str_end_;
    strf::alignment_format afmt_;
    union {
        strf::unsafe_transcode_f<SrcCharT, DstCharT>  transcode_;
        strf::unsafe_transcode_f<SrcCharT, char32_t>  src_to_u32_;
    };
    strf::transcode_size_f<SrcCharT> u32size_ = nullptr;
    strf::unsafe_transcode_f<char32_t, DstCharT> u32_to_dst_ = nullptr;
    strf::encode_fill_f<DstCharT> encode_fill_ = nullptr;
    strf::transcoding_error_notifier* err_notifier_;
    int left_fillcount_ = 0;
    int right_fillcount_ = 0;

    template < typename Category, typename SrcChar, typename FPack
             , typename Tag = strf::string_input_tag<SrcChar> >
    static STRF_HD
    STRF_DECLTYPE_AUTO((strf::use_facet<Category, Tag>(std::declval<FPack>())))
    use_facet_(const FPack& facets)
    {
        return facets.template use_facet<Category, Tag>();
    }

    template < typename PrePrinting, typename SrcCharset, typename DstCharset>
    STRF_HD void init_
        ( PrePrinting* pre, strf::width_t str_width
        , SrcCharset src_charset, DstCharset dst_charset );
};

template <typename SrcCharT, typename DstCharT>
template <typename PrePrinting, typename SrcCharset, typename DstCharset>
void STRF_HD aligned_unsafe_transcode_printer<SrcCharT, DstCharT>::init_
    ( PrePrinting* pre, strf::width_t str_width
    , SrcCharset src_charset, DstCharset dst_charset )
{
    encode_fill_ = dst_charset.encode_fill_func();
    auto transcoder = find_transcoder(src_charset, dst_charset);
    STRF_MAYBE_UNUSED(transcoder);
    transcode_ = transcoder.unsafe_transcode_func();
    if (transcode_ == nullptr) {
        src_to_u32_ = src_charset.to_u32().unsafe_transcode_func();
        u32size_ = src_charset.to_u32().transcode_size_func();
        u32_to_dst_ = dst_charset.from_u32().unsafe_transcode_func();
    }
    int fillcount = 0;
    if (afmt_.width > str_width) {
        fillcount = (afmt_.width - str_width).round();
        switch(afmt_.alignment) {
            case strf::text_alignment::left:
                left_fillcount_ = 0;
                right_fillcount_ = fillcount;
                break;
            case strf::text_alignment::center: {
                const auto halfcount = fillcount / 2;
                left_fillcount_ = halfcount;
                right_fillcount_ = fillcount - halfcount;
                break;
            }
            default:
                left_fillcount_ = fillcount;
                right_fillcount_ = 0;
        }
        pre->subtract_width(str_width);
        pre->subtract_width(static_cast<width_t>(fillcount));
    } else {
        right_fillcount_ = 0;
        left_fillcount_ = 0;
        pre->subtract_width(str_width);
    }
    STRF_IF_CONSTEXPR (PrePrinting::size_required) {
        std::ptrdiff_t s = 0;
        strf::unsafe_transcode_size_f<SrcCharT> transcode_size
                = transcoder.unsafe_transcode_size_func();
        constexpr auto flags = strf::transcode_flags::none;
        if (transcode_size != nullptr) {
            s = transcode_size(str_, str_end_, strf::ssize_max, flags).ssize;
        } else {
            s = strf::unsafe_decode_encode_size<SrcCharT>
                ( src_charset.to_u32().unsafe_transcode_func()
                , dst_charset.from_u32().unsafe_transcode_size_func()
                , str_, str_end_, strf::ssize_max, flags )
                .ssize;
        }
        if (fillcount > 0) {
            s += dst_charset.encoded_char_size(afmt_.fill) * fillcount;
        }
        pre->add_size(s);
    }
}

template<typename SrcCharT, typename DstCharT>
void STRF_HD aligned_unsafe_transcode_printer<SrcCharT, DstCharT>::print_to
    ( strf::destination<DstCharT>& dst ) const
{
    if (left_fillcount_ > 0) {
        encode_fill_(dst, left_fillcount_, afmt_.fill);
    }
    constexpr auto flags = strf::transcode_flags::none;

    auto src_it = str_;
    auto dst_it = dst.buffer_ptr();
    auto dst_end = dst.buffer_end();
    if (can_transcode_directly()) {
        while (1) {
            auto r = transcode_(src_it, str_end_, dst_it, dst_end, err_notifier_, flags);
            dst.advance_to(r.dst_ptr);
            if (r.stop_reason != strf::transcode_stop_reason::insufficient_output_space) {
                break;
            }
            src_it = r.src_ptr;
            dst.recycle();
            dst_it = dst.buffer_ptr();
            dst_end = dst.buffer_end();
        }
    } else {
        while (1) {
            auto r = strf::unsafe_decode_encode<SrcCharT, DstCharT>
                ( src_to_u32_, u32_to_dst_, src_it , str_end_
                , dst_it, dst_end, err_notifier_, flags );
            dst.advance_to(r.dst_ptr);
            if (r.stop_reason != strf::transcode_stop_reason::insufficient_output_space) {
                break;
            }
            dst.recycle();
            dst_it = dst.buffer_ptr();
            dst_end = dst.buffer_end();
            if (r.u32dist == 0) {
                src_it = r.stale_src_ptr;
            } else {
                src_it = u32size_(src_it, str_end_, r.u32dist, flags).src_ptr;
            }
        }
    }
    if (right_fillcount_ > 0) {
        encode_fill_(dst, right_fillcount_, afmt_.fill);
    }
}


#if defined(STRF_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
//STRF_EXPLICIT_TEMPLATE class strcpy_printer<char8_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class strcpy_printer<char8_t, char>;
STRF_EXPLICIT_TEMPLATE class strcpy_printer<char, char8_t>;
#endif


//STRF_EXPLICIT_TEMPLATE class strcpy_printer<char, char>;
STRF_EXPLICIT_TEMPLATE class strcpy_printer<char16_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class strcpy_printer<char32_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class strcpy_printer<wchar_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class strcpy_printer<wchar_t, strf::detail::wchar_equiv>;
STRF_EXPLICIT_TEMPLATE class strcpy_printer<strf::detail::wchar_equiv, wchar_t>;

#if defined(__cpp_char8_t)
//STRF_EXPLICIT_TEMPLATE class aligned_strcpy_printer<char8_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_strcpy_printer<char8_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_strcpy_printer<char, char8_t>;
#endif

//STRF_EXPLICIT_TEMPLATE class aligned_strcpy_printer<char, char>;
//STRF_EXPLICIT_TEMPLATE class aligned_strcpy_printer<char16_t, char16_t>;
//STRF_EXPLICIT_TEMPLATE class aligned_strcpy_printer<char32_t, char32_t>;
//STRF_EXPLICIT_TEMPLATE class aligned_strcpy_printer<wchar_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_strcpy_printer<wchar_t, strf::detail::wchar_equiv>;
STRF_EXPLICIT_TEMPLATE class aligned_strcpy_printer<strf::detail::wchar_equiv, wchar_t>;

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class transcode_printer<char8_t, char>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<char8_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<char8_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<char8_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<char8_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<char, char8_t>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<char16_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<char32_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<wchar_t, char8_t>;
#endif

STRF_EXPLICIT_TEMPLATE class transcode_printer<char, char>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<char, char16_t>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<char, char32_t>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<char, wchar_t>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<char16_t, char>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<char16_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<char16_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<char16_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<char32_t, char>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<char32_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<char32_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<char32_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<wchar_t, char>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<wchar_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<wchar_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class transcode_printer<wchar_t, wchar_t>;

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char8_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char8_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char8_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char8_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char8_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char16_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char32_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<wchar_t, char8_t>;
#endif

STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char, char>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char16_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char16_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char16_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char16_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char32_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char32_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char32_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<char32_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<wchar_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<wchar_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<wchar_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_printer<wchar_t, wchar_t>;


#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char8_t, char>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char8_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char8_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char8_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char8_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char, char8_t>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char16_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char32_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<wchar_t, char8_t>;
#endif

STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char, char>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char, char16_t>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char, char32_t>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char, wchar_t>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char16_t, char>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char16_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char16_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char16_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char32_t, char>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char32_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char32_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<char32_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<wchar_t, char>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<wchar_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<wchar_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class unsafe_transcode_printer<wchar_t, wchar_t>;

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char8_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char8_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char8_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char8_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char8_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char16_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char32_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<wchar_t, char8_t>;
#endif

STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char, char>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char16_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char16_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char16_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char16_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char32_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char32_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char32_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<char32_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<wchar_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<wchar_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<wchar_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_unsafe_transcode_printer<wchar_t, wchar_t>;


#endif // defined(STRF_SEPARATE_COMPILATION)

template <typename DstCharT>
class transcode_printer_variant
{
public:

    template < typename PrePrinting, typename FPack, typename SrcCharT
             , bool HasPrecision, typename Charset, transcoding_policy TranscPoli >
    STRF_HD explicit transcode_printer_variant
        ( const strf::detail::fmt_string_printer_input
            < SrcCharT, DstCharT, HasPrecision, false, Charset, TranscPoli, PrePrinting, FPack >&
            input )
    {
        auto src_charset  = strf::detail::get_src_charset(input);
        using facet_tag = strf::string_input_tag<SrcCharT>;
        using dst_charset_cat = strf::charset_c<DstCharT>;
        auto dst_charset = strf::use_facet<dst_charset_cat, facet_tag>(input.facets);
        if (src_charset.id() == dst_charset.id()) {
            using printer_type = strcpy_printer<SrcCharT, DstCharT>;
            static_assert(sizeof(printer_type) <= pool_size_, "");

            new ((void*)&pool_) printer_type(input);
        } else {
            constexpr auto different = charsets_comparison::statically_different;
            using printer_type = typename string_printer_type<false, different, TranscPoli>
                :: template type<SrcCharT, DstCharT>;
            static_assert(sizeof(printer_type) <= pool_size_, "");

            new ((void*)&pool_) printer_type(input);
        }
    }

    transcode_printer_variant(const transcode_printer_variant &) = delete;
    transcode_printer_variant(transcode_printer_variant&&) = delete;
    transcode_printer_variant& operator=(const transcode_printer_variant&) = delete;
    transcode_printer_variant& operator=(transcode_printer_variant&&) = delete;

    STRF_HD ~transcode_printer_variant()
    {
        const auto& p = static_cast<const strf::printer<DstCharT>&>(*this) ;
        p.~printer();
    }

#if defined(__GNUC__) && (__GNUC__ <= 6)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

    STRF_HD explicit operator const strf::printer<DstCharT>& () const
    {
        return * reinterpret_cast<const strf::printer<DstCharT>*>(&pool_);
    }

#if defined(__GNUC__) && (__GNUC__ <= 6)
#  pragma GCC diagnostic pop
#endif

private:

    static constexpr std::size_t pool_size_ =
        sizeof(strf::detail::transcode_printer<DstCharT, DstCharT>);
    using storage_type_ = typename std::aligned_storage
        < pool_size_, alignof(strf::printer<DstCharT>)>
        :: type;

    storage_type_ pool_;
};

template <typename DstCharT>
class aligned_transcode_printer_variant
{
public:

    template < typename PrePrinting, typename FPack, typename SrcCharT
             , bool HasPrecision, typename Charset, transcoding_policy TranscPoli >
    STRF_HD explicit aligned_transcode_printer_variant
        ( const strf::detail::fmt_string_printer_input
            < SrcCharT, DstCharT, HasPrecision, true, Charset, TranscPoli, PrePrinting, FPack >&
            input )
    {
        auto src_charset  = strf::detail::get_src_charset(input);
        using facet_tag = strf::string_input_tag<SrcCharT>;
        using dst_charset_cat = strf::charset_c<DstCharT>;
        auto dst_charset = strf::use_facet<dst_charset_cat, facet_tag>(input.facets);

        if (src_charset.id() == dst_charset.id()) {
            using printer_type = aligned_strcpy_printer<SrcCharT, DstCharT>;
            static_assert(sizeof(printer_type) <= pool_size_, "");

            new ((void*)&pool_) printer_type(input);

        } else {
            constexpr auto different = charsets_comparison::statically_different;
            using printer_type = typename string_printer_type<true, different, TranscPoli>
                :: template type<SrcCharT, DstCharT>;
            static_assert(sizeof(printer_type) <= pool_size_, "");

            new ((void*)&pool_) printer_type(input);
        }
    }

    aligned_transcode_printer_variant(const aligned_transcode_printer_variant&) = delete;
    aligned_transcode_printer_variant(aligned_transcode_printer_variant&&) = delete;

    aligned_transcode_printer_variant&
    operator=(const aligned_transcode_printer_variant&) = delete;

    aligned_transcode_printer_variant&
    operator=(aligned_transcode_printer_variant&&) = delete;

    STRF_HD ~aligned_transcode_printer_variant()
    {
        const auto& p = static_cast<const strf::printer<DstCharT>&>(*this);
        p.~printer();
    }

#if defined(__GNUC__) && (__GNUC__ <= 6)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

    STRF_HD explicit operator const strf::printer<DstCharT>& () const
    {
        return * reinterpret_cast<const strf::printer<DstCharT>*>(&pool_);
    }

#if defined(__GNUC__) && (__GNUC__ <= 6)
#  pragma GCC diagnostic pop
#endif

private:

    static constexpr std::size_t pool_size_ =
        sizeof(strf::detail::aligned_transcode_printer<DstCharT, DstCharT>);
    using storage_type_ = typename std::aligned_storage
        < pool_size_, alignof(strf::printer<DstCharT>)>
        :: type;

    storage_type_ pool_;
};

} // namespace detail
} // namespace strf

#endif // STRF_DETAIL_PRINTABLE_TYPES_STRING_HPP

