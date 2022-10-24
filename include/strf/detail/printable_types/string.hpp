#ifndef STRF_DETAIL_INPUT_TYPES_STRING
#define STRF_DETAIL_INPUT_TYPES_STRING

#include <strf/detail/facets/width_calculator.hpp>
#include <strf/detail/facets/charset.hpp>
#include <strf/detail/format_functions.hpp>
#include <strf/detail/printer.hpp>
#include <strf/detail/simple_string_view.hpp>

namespace strf {

template <typename T, typename CharT>
class transcoding_formatter_fn;

template <typename T, typename CharT>
class transcoding_formatter_transcode_fn;

template <typename T, typename Charset>
class transcoding_formatter_transcode_with_charset_fn;

template <typename T, typename CharT>
class transcoding_formatter_sani_fn;

template <typename T, typename Charset>
class transcoding_formatter_sani_with_charset_fn;

template <typename CharT>
struct transcoding_formatter
{
    template <typename T>
    using fn = strf::transcoding_formatter_fn<T, CharT>;

    constexpr static bool shall_not_transcode_nor_sanitize = true;
    constexpr static bool shall_sanitize = false;
    constexpr static bool has_charset = false;
};

template <typename CharT>
struct transcoding_formatter_transcode
{
    template <typename T>
    using fn = strf::transcoding_formatter_transcode_fn<T, CharT>;

    constexpr static bool shall_not_transcode_nor_sanitize = false;
    constexpr static bool shall_sanitize = false;
    constexpr static bool has_charset = false;
};

template <typename Charset>
struct transcoding_formatter_transcode_with_charset
{
    template <typename T>
    using fn = strf::transcoding_formatter_transcode_with_charset_fn<T, Charset>;

    constexpr static bool shall_not_transcode_nor_sanitize = false;
    constexpr static bool shall_sanitize = false;
    constexpr static bool has_charset = true;
};

template <typename CharT>
struct transcoding_formatter_sani
{
    template <typename T>
    using fn = strf::transcoding_formatter_sani_fn<T, CharT>;

    constexpr static bool shall_not_transcode_nor_sanitize = false;
    constexpr static bool shall_sanitize = true;
    constexpr static bool has_charset = false;
};

template <typename Charset>
struct transcoding_formatter_sani_with_charset
{
    template <typename T>
    using fn = strf::transcoding_formatter_sani_with_charset_fn<T, Charset>;

    constexpr static bool shall_not_transcode_nor_sanitize = false;
    constexpr static bool shall_sanitize = true;
    constexpr static bool has_charset = true;
};

template <typename T, typename CharT>
class transcoding_formatter_fn
{
    STRF_HD constexpr const T& self_downcast_() const
    {
        return *static_cast<const T*>(this);
    }

    using return_type_transcode_ = strf::fmt_replace
        < T
        , strf::transcoding_formatter<CharT>
        , strf::transcoding_formatter_transcode<CharT> >;

    template <typename Charset>
    using return_type_transcode_from_ = strf::fmt_replace
        < T
        , strf::transcoding_formatter<CharT>
        , strf::transcoding_formatter_transcode_with_charset<Charset> >;

    using return_type_sani_ = strf::fmt_replace
        < T
        , strf::transcoding_formatter<CharT>
        , strf::transcoding_formatter_sani<CharT> >;

    template <typename Charset>
    using return_type_sani_from_ = strf::fmt_replace
        < T
        , strf::transcoding_formatter<CharT>
        , strf::transcoding_formatter_sani_with_charset<Charset> >;

public:

    constexpr STRF_HD transcoding_formatter_fn() noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit transcoding_formatter_fn
        ( const transcoding_formatter_fn<U, CharT>& ) noexcept
    {
    }

    STRF_DEPRECATED_MSG("convert_charset was renamed to transcode")
    constexpr STRF_HD return_type_transcode_ convert_charset() const
    {
        return return_type_transcode_{ self_downcast_() };
    }

    constexpr STRF_HD return_type_transcode_ transcode() const
    {
        return return_type_transcode_{ self_downcast_() };
    }

    template <typename Charset>
    constexpr STRF_HD return_type_transcode_from_<Charset> transcode_from(Charset charset) const
    {
        static_assert( std::is_same<typename Charset::code_unit, CharT>::value
                     , "This charset does not match the string's character type." );

        return return_type_transcode_from_<Charset>
            { self_downcast_()
            , strf::tag<strf::transcoding_formatter_transcode_with_charset<Charset>>{}
            , charset };
    }

    template <typename Charset>
    STRF_DEPRECATED_MSG("convert_from_charset was renamed to transcode_from")
    constexpr STRF_HD return_type_transcode_from_<Charset> convert_from_charset(Charset charset) const
    {
        return transocode_from(charset);
    }

    STRF_DEPRECATED_MSG("conv was renamed to transcode")
    constexpr STRF_HD return_type_transcode_ conv() const
    {
        return return_type_transcode_{ self_downcast_() };
    }

    template <typename Charset>
    constexpr STRF_HD return_type_transcode_from_<Charset> transcode(Charset charset) const
    {
        return return_type_transcode_from_<Charset>
            { self_downcast_()
            , strf::tag<strf::transcoding_formatter_transcode_with_charset<Charset>>{}
            , charset };
    }

    template <typename Charset>
    STRF_DEPRECATED_MSG("conv was renamed to transcode")
    constexpr STRF_HD return_type_transcode_from_<Charset> conv(Charset charset) const
    {
        return transcode(charset);
    }

    constexpr STRF_HD return_type_sani_ sanitize_charset() const
    {
        return return_type_sani_{ self_downcast_() };
    }
    template <typename Charset>
    constexpr STRF_HD return_type_sani_from_<Charset> sanitize_from_charset(Charset charset) const
    {
        static_assert( std::is_same<typename Charset::code_unit, CharT>::value
                     , "This charset does not match the string's character type." );
        return
            { self_downcast_()
            , strf::tag<strf::transcoding_formatter_sani_with_charset<Charset>>{}
            , charset };
    }
    constexpr STRF_HD return_type_sani_ sani() const
    {
        return sanitize_charset();
    }
    template <typename Charset>
    constexpr STRF_HD return_type_sani_from_<Charset> sani(Charset charset) const
    {
        return sanitize_from_charset(charset);
    }

    // observers
    constexpr static STRF_HD bool shall_not_transcode_nor_sanitize()  noexcept { return false; }
    constexpr static STRF_HD bool shall_sanitize() noexcept { return false; }
    constexpr static STRF_HD bool has_charset()   noexcept { return false; }

    // backwards compatibility
    STRF_DEPRECATED constexpr STRF_HD auto convert_encoding() const
        -> return_type_transcode_
    {
        return convert_charset();
    }
    template <typename Charset>
    STRF_DEPRECATED constexpr STRF_HD auto convert_from_encoding(Charset charset) const
        -> return_type_transcode_from_<Charset>
    {
        return convert_from_charset(charset);
    }
    STRF_DEPRECATED constexpr STRF_HD auto sanitize_encoding() const
        -> return_type_sani_
    {
        return sanitize_charset();
    }
    template <typename Charset>
    STRF_DEPRECATED constexpr STRF_HD auto sanitize_from_encoding(Charset charset) const
        -> return_type_sani_from_<Charset>
    {
        return sanitize_from_charset(charset);
    }
};

template <typename T, typename CharT>
class transcoding_formatter_transcode_fn
{
public:
    constexpr STRF_HD transcoding_formatter_transcode_fn() noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit transcoding_formatter_transcode_fn
        ( const transcoding_formatter_transcode_fn<U, CharT>& ) noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit transcoding_formatter_transcode_fn
        ( const strf::transcoding_formatter_fn<U, CharT>& ) noexcept
    {
    }

    // observers
    constexpr static STRF_HD bool shall_not_transcode_nor_sanitize()  noexcept { return true; }
    constexpr static STRF_HD bool shall_sanitize() noexcept { return false; }
    constexpr static STRF_HD bool has_charset()   noexcept { return false; }
};

template <typename T, typename CharT>
class transcoding_formatter_sani_fn
{
public:
    constexpr STRF_HD transcoding_formatter_sani_fn() noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit transcoding_formatter_sani_fn
        ( const transcoding_formatter_sani_fn<U, CharT>& ) noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit transcoding_formatter_sani_fn
        ( const strf::transcoding_formatter_fn<U, CharT>& ) noexcept
    {
    }

    // observers
    constexpr static STRF_HD bool shall_not_transcode_nor_sanitize()  noexcept { return false; }
    constexpr static STRF_HD bool shall_sanitize() noexcept { return true; }
    constexpr static STRF_HD bool has_charset()   noexcept { return false; }
};

template <typename T, typename Charset>
class transcoding_formatter_transcode_with_charset_fn
{
public:

    STRF_HD explicit transcoding_formatter_transcode_with_charset_fn(Charset e)
        : charset_(e)
    {
    }

    template <typename U>
    STRF_HD explicit transcoding_formatter_transcode_with_charset_fn
        ( const strf::transcoding_formatter_transcode_with_charset_fn<U, Charset>& other ) noexcept
        : charset_(other.get_charset())
    {
    }
    template <typename U>
    STRF_HD explicit transcoding_formatter_transcode_with_charset_fn
        ( const strf::transcoding_formatter_sani_with_charset_fn<U, Charset>& other ) noexcept
        : charset_(other.get_charset())
    {
    }

    STRF_HD Charset get_charset() const noexcept
    {
        return charset_;
    }

    // observers
    constexpr static STRF_HD bool shall_not_transcode_nor_sanitize()  noexcept { return true; }
    constexpr static STRF_HD bool shall_sanitize() noexcept { return false; }
    constexpr static STRF_HD bool has_charset()   noexcept { return true; }

private:

    Charset charset_;
};


template <typename T, typename Charset>
class transcoding_formatter_sani_with_charset_fn
{
public:

    STRF_HD explicit transcoding_formatter_sani_with_charset_fn(Charset e)
        : charset_(e)
    {
    }

    template <typename U>
    STRF_HD explicit transcoding_formatter_sani_with_charset_fn
        ( const strf::transcoding_formatter_transcode_with_charset_fn<U, Charset>& other ) noexcept
        : charset_(other.get_charset())
    {
    }

    template <typename U>
    STRF_HD explicit transcoding_formatter_sani_with_charset_fn
        ( const strf::transcoding_formatter_sani_with_charset_fn<U, Charset>& other ) noexcept
        : charset_(other.get_charset())
    {
    }

    STRF_HD Charset get_charset() const noexcept
    {
        return charset_;
    }

    // observers
    constexpr static STRF_HD bool shall_not_transcode_nor_sanitize()  noexcept { return true; }
    constexpr static STRF_HD bool shall_sanitize() noexcept { return true; }
    constexpr static STRF_HD bool has_charset()   noexcept { return true; }

private:

    Charset charset_;
};

template <typename T, bool Active>
class string_precision_formatter_fn;

template <bool Active>
struct string_precision_formatter
{
    template <typename T>
    using fn = strf::string_precision_formatter_fn<T, Active>;
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
        ( strf::string_precision_formatter_fn<U, true> other ) noexcept
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
    constexpr STRF_HD strf::string_precision<true> get_string_precision() const noexcept
    {
        return strf::string_precision<true>{precision_};
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
                           , strf::string_precision_formatter<false>
                           , strf::string_precision_formatter<true> >;

    STRF_HD constexpr const T& self_downcast_() const
    {
        return *static_cast<const T*>(this);
    }

public:

    constexpr STRF_HD string_precision_formatter_fn() noexcept
    {
    }
    template <typename U>
    constexpr STRF_HD explicit string_precision_formatter_fn
        ( strf::string_precision_formatter_fn<U, false> ) noexcept
    {
    }
    constexpr STRF_HD adapted_derived_type_ p(strf::width_t precision) const noexcept
    {
        return { self_downcast_()
               , strf::tag<string_precision_formatter<true> >{}
               , precision };
    }
    constexpr STRF_HD strf::string_precision<false> get_string_precision() const noexcept
    {
        return strf::string_precision<false>{};
    }
    constexpr static STRF_HD bool has_string_precision() noexcept
    {
        return false;
    }
};

namespace detail {

template <typename SrcCharT> struct string_printing;

} // namespace detail

namespace detail {

template <typename SrcCharT, typename DestCharT> class string_printer;
template <typename SrcCharT, typename DestCharT> class aligned_string_printer;
template <typename SrcCharT, typename DestCharT> class transcode_string_printer;
template <typename SrcCharT, typename DestCharT> class aligned_transcode_string_printer;
template <typename DestCharT> class transcode_string_printer_variant;
template <typename DestCharT> class aligned_transcode_string_printer_variant;

template <typename CharT, typename PrePrinting, typename FPack>
struct string_printer_input
{
    using printer_type = strf::detail::string_printer<CharT, CharT>;

    strf::detail::simple_string_view<CharT> arg;
    PrePrinting& pre;
    FPack facets;
};

template < typename DestCharT, typename SrcCharT, bool HasAlignment
         , bool NeverConvert, bool ShallSanitize, bool HasCharset >
struct string_printer_type
{
    static_assert( ! NeverConvert, "");

    using type = strf::detail::conditional_t
        < ShallSanitize || sizeof(SrcCharT) != sizeof(DestCharT)
        , strf::detail::conditional_t
            < HasAlignment
            , strf::detail::aligned_transcode_string_printer<SrcCharT, DestCharT>
            , strf::detail::transcode_string_printer<SrcCharT, DestCharT> >
        , strf::detail::conditional_t
            < std::is_same<SrcCharT,DestCharT>::value && ! HasCharset
            , strf::detail::conditional_t
                < HasAlignment
                , strf::detail::aligned_string_printer<SrcCharT, DestCharT>
                , strf::detail::string_printer<SrcCharT, DestCharT> >
            , strf::detail::conditional_t
                < HasAlignment
                , strf::detail::aligned_transcode_string_printer_variant<DestCharT>
                , strf::detail::transcode_string_printer_variant<DestCharT> > > >;
};

template < typename DestCharT, typename SrcCharT, bool HasAlignment
         , bool ShallSanitize, bool HasCharset >
struct string_printer_type<DestCharT, SrcCharT, HasAlignment, true, ShallSanitize, HasCharset>
{
    static_assert( ! ShallSanitize, "");
    static_assert( ! HasCharset, "");
    static_assert( std::is_same<SrcCharT, DestCharT>::value
                 , "Character type mismatch. Use `transcode` or `sani` format function." );

    using type = strf::detail::conditional_t
        < HasAlignment
        , strf::detail::aligned_string_printer<SrcCharT, DestCharT>
        , strf::detail::string_printer<SrcCharT, DestCharT> >;
};

template <typename SrcCharT>
struct string_printing;

template < typename DestCharT, typename SrcCharT
         , bool HasPrecision, bool HasAlignment, typename TranscodeFormatter
         , typename PrePrinting, typename FPack >
struct fmt_string_printer_input
{
    using printer_type = typename strf::detail::string_printer_type
        < DestCharT, SrcCharT, HasAlignment
        , TranscodeFormatter::shall_not_transcode_nor_sanitize
        , TranscodeFormatter::shall_sanitize
        , TranscodeFormatter::has_charset >
        :: type;

    strf::printable_with_fmt
        < string_printing<SrcCharT>
        , strf::string_precision_formatter<HasPrecision>
        , strf::alignment_formatter_q<HasAlignment>
        , TranscodeFormatter > arg;
    PrePrinting& pre;
    FPack facets;
};

template <typename SrcCharT>
struct string_printing
{
    using representative_type = strf::string_input_tag<SrcCharT>;
    using forwarded_type = strf::detail::simple_string_view<SrcCharT>;
    using formatters = strf::tag
        < strf::string_precision_formatter<false>
        , strf::alignment_formatter
        , strf::transcoding_formatter<SrcCharT> >;
    using is_overridable = std::false_type;

    template <typename DestCharT, typename PrePrinting, typename FPack>
    constexpr STRF_HD static auto make_input
        ( strf::tag<DestCharT>
        , PrePrinting& pre
        , const FPack& facets
        , forwarded_type x ) noexcept
        -> strf::detail::string_printer_input<DestCharT, PrePrinting, FPack>
    {
        static_assert
            ( std::is_same<SrcCharT, DestCharT>::value
            , "Character type mismatch. Use `transcode` or `sani` format function." );

        return {x, pre, facets};
    }

    template < typename DestCharT, typename PrePrinting, typename FPack
             , bool HasPrecision, bool HasAlignment, typename TranscodeFormatter>
    constexpr STRF_HD static auto make_input
        ( strf::tag<DestCharT>
        , PrePrinting& pre
        , const FPack& facets
        , const strf::printable_with_fmt
            < string_printing<SrcCharT>
            , strf::string_precision_formatter<HasPrecision>
            , strf::alignment_formatter_q<HasAlignment>
            , TranscodeFormatter >& x ) noexcept
        -> strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, HasPrecision, HasAlignment
            , TranscodeFormatter, PrePrinting, FPack >
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

template <typename SrcCharT, typename DestCharT>
class string_printer: public strf::printer<DestCharT>
{
public:
    static_assert(sizeof(SrcCharT) == sizeof(DestCharT), "");

    template <typename PrePrinting, typename FPack>
    STRF_CONSTEXPR_IN_CXX14 STRF_HD explicit string_printer
        ( const strf::detail::string_printer_input<SrcCharT, PrePrinting, FPack>& input )
        : str_(input.arg.data())
        , len_(input.arg.length())
    {
        STRF_IF_CONSTEXPR(PrePrinting::width_required) {
            auto&& wcalc = use_facet_<strf::width_calculator_c>(input.facets);
            auto w = wcalc.str_width
                ( use_facet_<strf::charset_c<SrcCharT>>(input.facets)
                , input.pre.remaining_width(), str_, len_
                , use_facet_<strf::surrogate_policy_c>(input.facets) );
           input.pre.subtract_width(w);
        }
        input.pre.add_size(input.arg.length());
    }

    template < typename PrePrinting, typename FPack, typename CvFormat >
    STRF_CONSTEXPR_IN_CXX14 STRF_HD explicit string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, false, false, CvFormat, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().data())
        , len_(input.arg.value().size())
    {
        STRF_IF_CONSTEXPR(PrePrinting::width_required) {
            auto&& wcalc = use_facet_<strf::width_calculator_c>(input.facets);
            auto w = wcalc.str_width
                ( use_facet_<strf::charset_c<SrcCharT>>(input.facets)
                , input.pre.remaining_width()
                , str_
                , input.arg.value().size()
                , use_facet_<strf::surrogate_policy_c>(input.facets) );
           input.pre.subtract_width(w);
        }
        input.pre.add_size(input.arg.value().size());
    }

    template < typename PrePrinting, typename FPack, typename CvFormat >
    STRF_CONSTEXPR_IN_CXX14 STRF_HD explicit string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, true, false, CvFormat, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().data())
    {
        auto&& wcalc = use_facet_<strf::width_calculator_c>(input.facets);
        auto res = wcalc.str_width_and_pos
            ( use_facet_<strf::charset_c<SrcCharT>>(input.facets)
            , input.arg.precision()
            , str_
            , input.arg.value().size()
            , use_facet_<strf::surrogate_policy_c>(input.facets) );
        len_ = res.pos;
        input.pre.subtract_width(res.width);
        input.pre.add_size(res.pos);
    }

    STRF_HD void print_to(strf::destination<DestCharT>& dest) const override;

private:

    const SrcCharT* str_;
    std::size_t len_;

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

template<typename SrcCharT, typename DestCharT>
STRF_HD void string_printer<SrcCharT, DestCharT>::print_to
    ( strf::destination<DestCharT>& dest ) const
{
    strf::detail::output_buffer_interchar_copy(dest, str_, len_);
}

template <typename SrcCharT, typename DestCharT>
class aligned_string_printer: public strf::printer<DestCharT>
{
public:
    static_assert(sizeof(SrcCharT) == sizeof(DestCharT), "");

    template < typename PrePrinting, typename FPack, typename CvFormat >
    STRF_HD explicit aligned_string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, false, true, CvFormat, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().data())
        , len_(input.arg.value().size())
        , afmt_(input.arg.get_alignment_format())
    {

        auto&& wcalc = use_facet_<strf::width_calculator_c>(input.facets);
        auto src_charset = use_facet_<strf::charset_c<SrcCharT>>(input.facets);
        auto dest_charset = use_facet_<strf::charset_c<DestCharT>>(input.facets);
        strf::width_t limit =
            ( PrePrinting::width_required && input.pre.remaining_width() > afmt_.width
            ? input.pre.remaining_width()
            : afmt_.width );
        auto surr_poli = use_facet_<strf::surrogate_policy_c>(input.facets);
        auto strw = wcalc.str_width(src_charset, limit, str_, len_, surr_poli);
        encode_fill_ = dest_charset.encode_fill_func();
        auto fillcount = init_(input.pre, strw);
        precalc_size_(input.pre, dest_charset, fillcount);
    }

    template < typename PrePrinting, typename FPack, typename CvFormat >
    STRF_HD explicit aligned_string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, true, true, CvFormat, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().begin())
        , afmt_(input.arg.get_alignment_format())
    {
        auto&& wcalc = use_facet_<strf::width_calculator_c>(input.facets);
        auto src_charset = use_facet_<strf::charset_c<SrcCharT>>(input.facets);
        auto dest_charset = use_facet_<strf::charset_c<DestCharT>>(input.facets);
        auto surr_poli = use_facet_<strf::surrogate_policy_c>(input.facets);
        auto res = wcalc.str_width_and_pos
            ( src_charset, input.arg.precision(), str_, input.arg.value().size(), surr_poli );
        len_ = res.pos;
        encode_fill_ = dest_charset.encode_fill_func();
        auto fillcount = init_(input.pre, res.width);
        precalc_size_(input.pre, dest_charset, fillcount);
    }

    STRF_HD ~aligned_string_printer();

    STRF_HD void print_to(strf::destination<DestCharT>& dest) const override;

private:

    const SrcCharT* str_;
    std::size_t len_;
    strf::encode_fill_f<DestCharT> encode_fill_;
    strf::alignment_format afmt_;
    std::uint16_t left_fillcount_;
    std::uint16_t right_fillcount_;

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
    STRF_HD std::uint16_t init_(PrePrinting&, strf::width_t strw);

    template <typename Charset>
    STRF_HD void precalc_size_( strf::size_accumulator<true>& pre
                              , Charset charset, std::uint16_t fillcount )
    {
        pre.add_size(len_);
        if (fillcount > 0) {
            pre.add_size(fillcount * charset.encoded_char_size(afmt_.fill));
        }
    }

    template <typename Charset>
    STRF_HD void precalc_size_(strf::size_accumulator<false>&, Charset, std::uint16_t)
    {
    }
};

template<typename SrcCharT, typename DestCharT>
STRF_HD aligned_string_printer<SrcCharT, DestCharT>::~aligned_string_printer()
{
}

template<typename SrcCharT, typename DestCharT>
template <typename PrePrinting>
inline STRF_HD std::uint16_t aligned_string_printer<SrcCharT, DestCharT>::init_
    ( PrePrinting& pre, strf::width_t strw )
{
    if (afmt_.width > strw) {
        std::uint16_t fillcount = (afmt_.width - strw).round();
        switch(afmt_.alignment) {
            case strf::text_alignment::left:
                left_fillcount_ = 0;
                right_fillcount_ = fillcount;
                break;
            case strf::text_alignment::center: {
                std::uint16_t halfcount = fillcount >> 1;
                left_fillcount_ = halfcount;
                right_fillcount_ = fillcount - halfcount;
                break;
            }
            default:
                left_fillcount_ = fillcount;
                right_fillcount_ = 0;
        }
        pre.subtract_width(strw + fillcount);
        return fillcount;
    } else {
        right_fillcount_ = 0;
        left_fillcount_ = 0;
        pre.subtract_width(strw);
        return 0;
    }
}

template<typename SrcCharT, typename DestCharT>
void STRF_HD aligned_string_printer<SrcCharT, DestCharT>::print_to
    ( strf::destination<DestCharT>& dest ) const
{
    if (left_fillcount_ > 0) {
        encode_fill_(dest, left_fillcount_, afmt_.fill);
    }
    strf::detail::output_buffer_interchar_copy(dest, str_, len_);
    if (right_fillcount_ > 0) {
        encode_fill_(dest, right_fillcount_, afmt_.fill);
    }
}

template < typename DestCharT, typename PrePrinting, typename FPack
         , typename SrcCharT, bool HasP, bool HasA, typename SrcCharset >
constexpr STRF_HD auto get_src_charset
    ( const strf::detail::fmt_string_printer_input
        < DestCharT, SrcCharT, HasP, HasA
        , strf::transcoding_formatter_transcode_with_charset<SrcCharset>, PrePrinting, FPack>&
      input )
    -> decltype(input.arg.get_charset())
{
    static_assert( std::is_same<typename SrcCharset::code_unit, SrcCharT>::value
                 , "This charset is associated with another character type." );
    return input.arg.get_charset();
}

template < typename DestCharT, typename PrePrinting, typename FPack
         , typename SrcCharT, bool HasP, bool HasA, typename SrcCharset >
constexpr STRF_HD auto get_src_charset
    ( const strf::detail::fmt_string_printer_input
        < DestCharT, SrcCharT, HasP, HasA
        , strf::transcoding_formatter_sani_with_charset<SrcCharset>, PrePrinting, FPack >&
      input )
    -> decltype(input.arg.get_charset())
{
    static_assert( std::is_same<typename SrcCharset::code_unit, SrcCharT>::value
                 , "This charset is associated with another character type." );
    return input.arg.get_charset();
}

template < typename DestCharT, typename PrePrinting, typename FPack
         , typename SrcCharT, bool HasP, bool HasA >
constexpr STRF_HD auto get_src_charset
    ( const strf::detail::fmt_string_printer_input
        < DestCharT, SrcCharT, HasP, HasA
        , strf::transcoding_formatter_transcode<SrcCharT>, PrePrinting, FPack >&
      input )
    -> decltype(strf::use_facet
                   <strf::charset_c<SrcCharT>, strf::string_input_tag<SrcCharT>>
                   (input.facets))
{
    return strf::use_facet
        <strf::charset_c<SrcCharT>, strf::string_input_tag<SrcCharT>>
        ( input.facets );
}

template < typename DestCharT, typename PrePrinting, typename FPack
         , typename SrcCharT, bool HasP, bool HasA >
constexpr STRF_HD auto get_src_charset
    ( const strf::detail::fmt_string_printer_input
        < DestCharT, SrcCharT, HasP, HasA
        , strf::transcoding_formatter_sani<SrcCharT>, PrePrinting, FPack >&
      input )
    -> decltype(strf::use_facet
                   <strf::charset_c<SrcCharT>, strf::string_input_tag<SrcCharT>>
                   (input.facets))
{
    return strf::use_facet
        <strf::charset_c<SrcCharT>, strf::string_input_tag<SrcCharT>>
        ( input.facets );
}

template<typename SrcCharT, typename DestCharT>
class transcode_string_printer: public strf::printer<DestCharT>
{
public:

    template <typename PrePrinting, typename FPack, typename CvFormat>
    STRF_HD explicit transcode_string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, false, false, CvFormat, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().data())
        , len_(input.arg.value().size())
        , err_notifier_
              ( use_facet_<strf::transcoding_error_notifier_c, SrcCharT>(input.facets).get() )
        , surr_poli_(use_facet_<strf::surrogate_policy_c, SrcCharT>(input.facets))
    {
        auto src_charset  = strf::detail::get_src_charset(input);
        auto dest_charset = use_facet_<strf::charset_c<DestCharT>, SrcCharT>(input.facets);
        STRF_IF_CONSTEXPR (PrePrinting::width_required) {
            auto&& wcalc = use_facet_<strf::width_calculator_c, SrcCharT>(input.facets);
            auto w = wcalc.str_width( src_charset, input.pre.remaining_width()
                                    , str_, len_, surr_poli_);
            input.pre.subtract_width(w);
        }
        init_(input.pre, src_charset, dest_charset);
    }

    template <typename PrePrinting, typename FPack, typename CvFormat>
    STRF_HD explicit transcode_string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, true, false, CvFormat, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().data())
        , err_notifier_
            ( use_facet_<strf::transcoding_error_notifier_c, SrcCharT>(input.facets).get() )
        , surr_poli_(use_facet_<strf::surrogate_policy_c, SrcCharT>(input.facets))
    {
        auto src_charset  = strf::detail::get_src_charset(input);
        auto dest_charset = use_facet_<strf::charset_c<DestCharT>, SrcCharT>(input.facets);
        auto&& wcalc = use_facet_<strf::width_calculator_c, SrcCharT>(input.facets);
        auto res = wcalc.str_width_and_pos
            ( src_charset, input.arg.precision(), str_
            , input.arg.value().size(), surr_poli_ );
        len_ = res.pos;
        input.pre.subtract_width(res.width);
        init_( input.pre, src_charset
             , use_facet_<strf::charset_c<DestCharT>, SrcCharT>(input.facets));
    }

    STRF_HD ~transcode_string_printer() { }

    STRF_HD void print_to(strf::destination<DestCharT>& dest) const override;

private:

    template < typename PrePrinting, typename SrcCharset, typename DestCharset >
    STRF_HD void init_(PrePrinting& pre, SrcCharset src_charset, DestCharset dest_charset)
    {
        auto transcoder = find_transcoder(src_charset, dest_charset);
        STRF_MAYBE_UNUSED(transcoder);
        transcode_ = transcoder.transcode_func();
        if (transcode_ == nullptr) {
            src_to_u32_ = src_charset.to_u32().transcode_func();
            u32_to_dest_ = dest_charset.from_u32().transcode_func();
        }
        STRF_IF_CONSTEXPR (PrePrinting::size_required) {
            strf::transcode_size_f<SrcCharT>  transcode_size
                = transcoder.transcode_size_func();
            std::size_t s = 0;
            if (transcode_size != nullptr) {
                s = transcode_size(str_, len_, surr_poli_);
            } else {
                s = strf::decode_encode_size<SrcCharT>
                    ( src_charset.to_u32().transcode_func()
                    , dest_charset.from_u32().transcode_size_func()
                    , str_, len_, err_notifier_, surr_poli_ );
            }
            pre.add_size(s);
        }
    }

    STRF_HD bool can_transcode_directly() const
    {
        return u32_to_dest_ == nullptr;
    }

    const SrcCharT* const str_;
    std::size_t len_;
    union {
        strf::transcode_f<SrcCharT, DestCharT>  transcode_;
        strf::transcode_f<SrcCharT, char32_t>  src_to_u32_;
    };
    strf::transcode_f<char32_t, DestCharT>  u32_to_dest_ = nullptr;
    strf::transcoding_error_notifier* err_notifier_;
    const strf::surrogate_policy surr_poli_;

    template < typename Category, typename SrcChar, typename FPack
             , typename Tag = strf::string_input_tag<SrcChar> >
    static STRF_HD
    STRF_DECLTYPE_AUTO((strf::use_facet<Category, Tag>(std::declval<FPack>())))
    use_facet_(const FPack& facets)
    {
        return facets.template use_facet<Category, Tag>();
    }
};

template<typename SrcCharT, typename DestCharT>
STRF_HD void transcode_string_printer<SrcCharT, DestCharT>::print_to
    ( strf::destination<DestCharT>& dest ) const
{
    if (can_transcode_directly()) {
        transcode_(dest, str_, len_, err_notifier_, surr_poli_);
    } else {
        strf::decode_encode<SrcCharT, DestCharT>
            ( dest, src_to_u32_, u32_to_dest_, str_
            , len_, err_notifier_, surr_poli_ );
    }
}

template<typename SrcCharT, typename DestCharT>
class aligned_transcode_string_printer: public printer<DestCharT>
{
public:

    template <typename PrePrinting, typename FPack, typename CvFormat>
    STRF_HD explicit aligned_transcode_string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, false, true, CvFormat, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().data())
        , len_(input.arg.value().size())
        , afmt_(input.arg.get_alignment_format())
        , err_notifier_
            ( use_facet_<strf::transcoding_error_notifier_c, SrcCharT>(input.facets).get() )
        , surr_poli_(use_facet_<strf::surrogate_policy_c, SrcCharT>(input.facets))
    {
        auto src_charset = strf::detail::get_src_charset(input);
        auto&& wcalc = use_facet_<strf::width_calculator_c, SrcCharT>(input.facets);
        strf::width_t limit =
            ( PrePrinting::width_required && input.pre.remaining_width() > afmt_.width
            ? input.pre.remaining_width()
            : afmt_.width );
        auto str_width = wcalc.str_width(src_charset, limit, str_, len_, surr_poli_);
        init_( input.pre, str_width, src_charset
             , use_facet_<strf::charset_c<DestCharT>, SrcCharT>(input.facets) );
    }

    template <typename PrePrinting, typename FPack, typename CvFormat>
    STRF_HD explicit aligned_transcode_string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, true, true, CvFormat, PrePrinting, FPack >&
            input )
        : str_(input.arg.value().data())
        , len_(input.arg.value().size())
        , afmt_(input.arg.get_alignment_format())
        , err_notifier_
            ( use_facet_<strf::transcoding_error_notifier_c, SrcCharT>(input.facets).get() )
        , surr_poli_(use_facet_<strf::surrogate_policy_c, SrcCharT>(input.facets))
    {
        auto src_charset = strf::detail::get_src_charset(input);
        auto&& wcalc = use_facet_<strf::width_calculator_c, SrcCharT>(input.facets);
        auto res = wcalc.str_width_and_pos
            ( src_charset, input.arg.precision(), str_
            , input.arg.value().size(), surr_poli_ );
        len_ = res.pos;
        init_( input.pre, res.width, src_charset
             , use_facet_<strf::charset_c<DestCharT>, SrcCharT>(input.facets) );
    }

    STRF_HD void print_to(strf::destination<DestCharT>& dest) const override;

private:

    STRF_HD bool can_transcode_directly() const
    {
        return u32_to_dest_ == nullptr;
    }

    const SrcCharT* str_;
    std::size_t len_;
    strf::alignment_format afmt_;
    union {
        strf::transcode_f<SrcCharT, DestCharT>  transcode_;
        strf::transcode_f<SrcCharT, char32_t>  src_to_u32_;
    };
    strf::transcode_f<char32_t, DestCharT>  u32_to_dest_ = nullptr;
    strf::encode_fill_f<DestCharT> encode_fill_ = nullptr;
    strf::transcoding_error_notifier* err_notifier_;
    const strf::surrogate_policy  surr_poli_;
    std::uint16_t left_fillcount_ = 0;
    std::uint16_t right_fillcount_ = 0;

    template < typename Category, typename SrcChar, typename FPack
             , typename Tag = strf::string_input_tag<SrcChar> >
    static STRF_HD
    STRF_DECLTYPE_AUTO((strf::use_facet<Category, Tag>(std::declval<FPack>())))
    use_facet_(const FPack& facets)
    {
        return facets.template use_facet<Category, Tag>();
    }

    template < typename PrePrinting, typename SrcCharset, typename DestCharset>
    STRF_HD void init_
        ( PrePrinting& pre, strf::width_t str_width
        , SrcCharset src_charset, DestCharset dest_charset );
};

template <typename SrcCharT, typename DestCharT>
template <typename PrePrinting, typename SrcCharset, typename DestCharset>
void STRF_HD aligned_transcode_string_printer<SrcCharT, DestCharT>::init_
    ( PrePrinting& pre, strf::width_t str_width
    , SrcCharset src_charset, DestCharset dest_charset )
{
    encode_fill_ = dest_charset.encode_fill_func();
    auto transcoder = find_transcoder(src_charset, dest_charset);
    STRF_MAYBE_UNUSED(transcoder);
    transcode_ = transcoder.transcode_func();
    if (transcode_ == nullptr) {
        src_to_u32_ = src_charset.to_u32().transcode_func();
        u32_to_dest_ = dest_charset.from_u32().transcode_func();
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
        pre.subtract_width(str_width + fillcount);
    } else {
        right_fillcount_ = 0;
        left_fillcount_ = 0;
        pre.subtract_width(str_width);
    }
    STRF_IF_CONSTEXPR (PrePrinting::size_required) {
        std::size_t s = 0;
        strf::transcode_size_f<SrcCharT> transcode_size
                = transcoder.transcode_size_func();
        if (transcode_size != nullptr) {
            s = transcode_size(str_, len_, surr_poli_);
        } else {
            s = strf::decode_encode_size<SrcCharT>
                ( src_charset.to_u32().transcode_func()
                , dest_charset.from_u32().transcode_size_func()
                , str_, len_, err_notifier_, surr_poli_ );
        }
        if (fillcount > 0) {
            s += dest_charset.encoded_char_size(afmt_.fill) * fillcount;
        }
        pre.add_size(s);
    }
}

template<typename SrcCharT, typename DestCharT>
void STRF_HD aligned_transcode_string_printer<SrcCharT, DestCharT>::print_to
    ( strf::destination<DestCharT>& dest ) const
{
    if (left_fillcount_ > 0) {
        encode_fill_(dest, left_fillcount_, afmt_.fill);
    }
    if (can_transcode_directly()) {
        transcode_(dest, str_, len_, err_notifier_, surr_poli_);
    } else {
        strf::decode_encode<SrcCharT, DestCharT>
            ( dest, src_to_u32_, u32_to_dest_, str_
            , len_, err_notifier_, surr_poli_ );
    }
    if (right_fillcount_ > 0) {
        encode_fill_(dest, right_fillcount_, afmt_.fill);
    }
}

#if defined(STRF_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
//STRF_EXPLICIT_TEMPLATE class string_printer<char8_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class string_printer<char8_t, char>;
STRF_EXPLICIT_TEMPLATE class string_printer<char, char8_t>;
#endif


//STRF_EXPLICIT_TEMPLATE class string_printer<char, char>;
STRF_EXPLICIT_TEMPLATE class string_printer<char16_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class string_printer<char32_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class string_printer<wchar_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class string_printer<wchar_t, strf::detail::wchar_equiv>;
STRF_EXPLICIT_TEMPLATE class string_printer<strf::detail::wchar_equiv, wchar_t>;

#if defined(__cpp_char8_t)
//STRF_EXPLICIT_TEMPLATE class aligned_string_printer<char8_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_string_printer<char8_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_string_printer<char, char8_t>;
#endif

//STRF_EXPLICIT_TEMPLATE class aligned_string_printer<char, char>;
//STRF_EXPLICIT_TEMPLATE class aligned_string_printer<char16_t, char16_t>;
//STRF_EXPLICIT_TEMPLATE class aligned_string_printer<char32_t, char32_t>;
//STRF_EXPLICIT_TEMPLATE class aligned_string_printer<wchar_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_string_printer<wchar_t, strf::detail::wchar_equiv>;
STRF_EXPLICIT_TEMPLATE class aligned_string_printer<strf::detail::wchar_equiv, wchar_t>;

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char8_t, char>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char8_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char8_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char8_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char8_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char, char8_t>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char16_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char32_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<wchar_t, char8_t>;
#endif

STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char, char>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char, char16_t>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char, char32_t>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char, wchar_t>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char16_t, char>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char16_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char16_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char16_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char32_t, char>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char32_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char32_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<char32_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<wchar_t, char>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<wchar_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<wchar_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class transcode_string_printer<wchar_t, wchar_t>;

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char8_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char8_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char8_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char8_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char8_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char16_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char32_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<wchar_t, char8_t>;
#endif

STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char, char>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char16_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char16_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char16_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char16_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char32_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char32_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char32_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<char32_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<wchar_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<wchar_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<wchar_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_transcode_string_printer<wchar_t, wchar_t>;

#endif // defined(STRF_SEPARATE_COMPILATION)

template <typename DestCharT>
class transcode_string_printer_variant
{
public:

    template < typename PrePrinting, typename FPack, typename SrcCharT
             , bool HasPrecision, typename CvFormat >
    STRF_HD explicit transcode_string_printer_variant
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, HasPrecision, false, CvFormat, PrePrinting, FPack >&
            input )
    {
        auto src_charset  = strf::detail::get_src_charset(input);
        using facet_tag = strf::string_input_tag<SrcCharT>;
        using dest_charset_cat = strf::charset_c<DestCharT>;
        auto dest_charset = strf::use_facet<dest_charset_cat, facet_tag>(input.facets);
        if (src_charset.id() == dest_charset.id()) {
            new ((void*)&pool_) strf::detail::string_printer<SrcCharT, DestCharT>(input);
        } else {
            new ((void*)&pool_) strf::detail::transcode_string_printer<SrcCharT, DestCharT>(input);
        }
    }

    STRF_HD ~transcode_string_printer_variant()
    {
        const auto& p = static_cast<const strf::printer<DestCharT>&>(*this) ;
        p.~printer();
    }

#if defined(__GNUC__) && (__GNUC__ <= 6)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

    STRF_HD explicit operator const strf::printer<DestCharT>& () const
    {
        return * reinterpret_cast<const strf::printer<DestCharT>*>(&pool_);
    }

#if defined(__GNUC__) && (__GNUC__ <= 6)
#  pragma GCC diagnostic pop
#endif

private:

    static constexpr std::size_t pool_size_ =
        sizeof(strf::detail::transcode_string_printer<DestCharT, DestCharT>);
    using storage_type_ = typename std::aligned_storage
        < pool_size_, alignof(strf::printer<DestCharT>)>
        :: type;

    storage_type_ pool_;
};

template<typename DestCharT>
class aligned_transcode_string_printer_variant
{
public:

    template < typename PrePrinting, typename FPack, typename SrcCharT
             , bool HasPrecision, typename CvFormat >
    STRF_HD explicit aligned_transcode_string_printer_variant
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, HasPrecision, true, CvFormat, PrePrinting, FPack >&
            input )
    {
        auto src_charset  = strf::detail::get_src_charset(input);
        using facet_tag = strf::string_input_tag<SrcCharT>;
        using dest_charset_cat = strf::charset_c<DestCharT>;
        auto dest_charset = strf::use_facet<dest_charset_cat, facet_tag>(input.facets);

        if (src_charset.id() == dest_charset.id()) {
            new ((void*)&pool_) strf::detail::aligned_string_printer<SrcCharT, DestCharT> (input);
        } else {
            new ((void*)&pool_)
                strf::detail::aligned_transcode_string_printer<SrcCharT, DestCharT>(input);
        }
    }

    STRF_HD ~aligned_transcode_string_printer_variant()
    {
        const auto& p = static_cast<const strf::printer<DestCharT>&>(*this);
        p.~printer();
    }

#if defined(__GNUC__) && (__GNUC__ <= 6)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

    STRF_HD explicit operator const strf::printer<DestCharT>& () const
    {
        return * reinterpret_cast<const strf::printer<DestCharT>*>(&pool_);
    }

#if defined(__GNUC__) && (__GNUC__ <= 6)
#  pragma GCC diagnostic pop
#endif

private:

    static constexpr std::size_t pool_size_ =
        sizeof(strf::detail::aligned_transcode_string_printer<DestCharT, DestCharT>);
    using storage_type_ = typename std::aligned_storage
        < pool_size_, alignof(strf::printer<DestCharT>)>
        :: type;

    storage_type_ pool_;
};

} // namespace detail
} // namespace strf

#endif  /* STRF_DETAIL_INPUT_TYPES_CHAR_PTR */

