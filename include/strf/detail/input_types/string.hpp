#ifndef STRF_DETAIL_INPUT_TYPES_STRING
#define STRF_DETAIL_INPUT_TYPES_STRING

#include <limits>
#include <strf/detail/facets/width_calculator.hpp>
#include <strf/detail/format_functions.hpp>
#include <strf/facets_pack.hpp>

namespace strf {
namespace detail {

template <typename CharIn>
class simple_string_view
{
public:

    using iterator = const CharIn*;
    using const_iterator = const CharIn*;

#if defined(STRF_HAS_STD_STRING_VIEW)

    template <typename Traits>
    constexpr STRF_HD simple_string_view(std::basic_string_view<CharIn, Traits> sv)
        : begin_(sv.data())
        , len_(sv.size())
    {
    }

#endif //defined(STRF_HAS_STD_STRING_VIEW)

    template <typename Traits, typename Allocator>
    STRF_HD simple_string_view(const std::basic_string<CharIn, Traits, Allocator>& s)
        : begin_(s.data())
        , len_(s.size())
    {
    }

    constexpr STRF_HD simple_string_view(const CharIn* begin, const CharIn* end) noexcept
        : begin_(begin)
        , len_(end - begin)
    {
    }
    constexpr STRF_HD simple_string_view(const CharIn* str, std::size_t len) noexcept
        : begin_(str)
        , len_(len)
    {
    }

    STRF_CONSTEXPR_CHAR_TRAITS
    STRF_HD simple_string_view(const CharIn* str) noexcept
        : begin_(str)
        , len_(strf::detail::str_length<CharIn>(str))
    {
    }
    constexpr STRF_HD const CharIn* begin() const
    {
        return begin_;
    }
    constexpr STRF_HD const CharIn* data() const
    {
        return begin_;
    }
    constexpr STRF_HD const CharIn* end() const
    {
        return begin_ + len_;
    }
    constexpr STRF_HD std::size_t size() const
    {
        return len_;
    }
    constexpr STRF_HD std::size_t length() const
    {
        return len_;
    }

private:

    const CharIn* begin_;
    const std::size_t len_;
};

} // namespace detail

template <typename CharT>
struct no_conv_format;

template <typename CharT>
struct conv_format;

template <typename CharT, typename Encoding>
struct conv_format_with_encoding;

template <typename CharT>
struct sani_format;

template <typename CharT, typename Encoding>
struct sani_format_with_encoding;

template <typename CharT, typename T>
class no_conv_format_fn
{
public:

    constexpr STRF_HD no_conv_format_fn() noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit no_conv_format_fn
        ( const no_conv_format_fn<CharT, U>& ) noexcept
    {
    }

    constexpr STRF_HD auto convert_encoding() const
    {
        using return_type = strf::fmt_replace< T
                                             , strf::no_conv_format<CharT>
                                             , strf::conv_format<CharT> >;
        return return_type{ static_cast<const T&>(*this) };
    }

    template <typename Encoding>
    constexpr STRF_HD auto convert_encoding(Encoding enc) const
    {
        using return_type = strf::fmt_replace
            < T
            , strf::no_conv_format<CharT>
            , strf::conv_format_with_encoding<CharT, Encoding> >;

        return return_type
            { static_cast<const T&>(*this)
            , strf::tag<strf::conv_format_with_encoding<CharT, Encoding>>{}
            , enc };
    }
    constexpr STRF_HD auto conv() const
    {
        return convert_encoding();
    }
    template <typename Encoding>
    constexpr STRF_HD auto conv(Encoding enc) const
    {
        return convert_encoding(enc);
    }

    constexpr STRF_HD auto sanitize_encoding() const
    {
        using return_type = strf::fmt_replace< T
                                             , strf::no_conv_format<CharT>
                                             , strf::sani_format<CharT> >;
        return return_type{ static_cast<const T&>(*this) };
    }
    template <typename Encoding>
    constexpr STRF_HD auto sanitize_encoding(Encoding enc) const
    {
        using return_type = strf::fmt_replace
            < T
            , strf::no_conv_format<CharT>
            , strf::sani_format_with_encoding<CharT, Encoding> >;

        return return_type
            { static_cast<const T&>(*this)
            , strf::tag<strf::sani_format_with_encoding<CharT, Encoding>>{}
            , enc };
    }
    constexpr auto sani() const
    {
        return sanitize_encoding();
    }
    template <typename Encoding>
    constexpr auto sani(Encoding enc) const
    {
        return sanitize_encoding(enc);
    }
};

template <typename CharT, typename T>
struct conv_format_fn
{
    constexpr STRF_HD conv_format_fn() noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit conv_format_fn
        ( const conv_format_fn<CharT, U>& ) noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit conv_format_fn
        ( const strf::no_conv_format_fn<CharT, U>& ) noexcept
    {
    }
};

template <typename CharT, typename Encoding, typename T>
class conv_format_with_encoding_fn
{
public:

    conv_format_with_encoding_fn(Encoding e)
        : encoding_(e)
    {
    }

    conv_format_with_encoding_fn
        ( const conv_format_with_encoding_fn& other ) noexcept = default;

    template <typename U>
    explicit conv_format_with_encoding_fn
        ( const strf::conv_format_with_encoding_fn<CharT, Encoding, U>& other ) noexcept
        : encoding_(other.get_encoding())
    {
    }

    template <typename U>
    explicit conv_format_with_encoding_fn
        ( const strf::no_conv_format_fn<CharT, U>& other ) noexcept
        : encoding_(other.get_encoding())
    {
    }

    Encoding get_encoding() const
    {
        return encoding_;
    }

private:

    Encoding encoding_;
};

template <typename CharT>
struct no_conv_format
{
    template <typename T>
    using fn = strf::no_conv_format_fn<CharT, T>;
};

template <typename CharT>
struct conv_format
{
    template <typename T>
    using fn = strf::conv_format_fn<CharT, T>;
};

template <typename CharT, typename Encoding>
struct conv_format_with_encoding
{
    template <typename T>
    using fn = strf::conv_format_with_encoding_fn<CharT, Encoding, T>;
};

template <typename CharT>
struct sani_format
{
    template <typename T>
    using fn = strf::conv_format_fn<CharT, T>;
};

template <typename CharT, typename Encoding>
struct sani_format_with_encoding
{
    template <typename T>
    using fn = strf::conv_format_with_encoding_fn<CharT, Encoding, T>;
};

template <typename T, bool Active>
class string_precision_format_fn;

template <bool Active>
struct string_precision_format
{
    template <typename T>
    using fn = strf::string_precision_format_fn<T, Active>;
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
class string_precision_format_fn<T, true>
{
public:
    constexpr STRF_HD string_precision_format_fn(strf::width_t p) noexcept
        : precision_(p)
    {
    }
    template <typename U>
    constexpr STRF_HD string_precision_format_fn
        ( strf::string_precision_format_fn<U, true> other ) noexcept
        : precision_(other.precision())
    {
    }
    constexpr STRF_HD T&& p(strf::width_t _) && noexcept
    {
        precision_ = _;
        return static_cast<T&&>(*this);
    }
    constexpr STRF_HD strf::width_t precision() const noexcept
    {
        return precision_;
    }
    constexpr STRF_HD auto get_string_precision() const noexcept
    {
        return strf::string_precision<true>{precision_};
    }

private:

    strf::width_t precision_;
};


template <typename T>
class string_precision_format_fn<T, false>
{
    using adapted_derived_type_
        = strf::fmt_replace< T
                           , strf::string_precision_format<false>
                           , strf::string_precision_format<true> >;
public:

    constexpr STRF_HD string_precision_format_fn() noexcept
    {
    }
    template <typename U>
    constexpr STRF_HD string_precision_format_fn
        ( strf::string_precision_format_fn<U, false> ) noexcept
    {
    }
    constexpr STRF_HD adapted_derived_type_ p(strf::width_t precision) const noexcept
    {
        return { static_cast<const T&>(*this)
               , strf::tag<string_precision_format<true> >{}
               , precision };
    }
    constexpr STRF_HD auto get_string_precision() const noexcept
    {
        return strf::string_precision<false>{};
    }
};

template <typename CharIn, bool HasPrecision = false, bool HasAlignment = false>
using string_with_format = strf::value_with_format
    < strf::detail::simple_string_view<CharIn>
    , strf::string_precision_format<HasPrecision>
    , strf::alignment_format_q<HasAlignment>
    , strf::no_conv_format<CharIn> >;

namespace detail {

template <typename CharIn>
struct string_fmt_traits
{
    using fmt_type = strf::string_with_format<CharIn>;
};

} // namespace detail


// template <typename CharIn, typename Traits, typename Allocator>
// struct fmt_traits<std::basic_string<CharIn, Traits, Allocator>>
//     : strf::detail::string_fmt_traits<CharIn>
// {
// };

template <typename CharIn, typename Traits, typename Allocator>
constexpr STRF_HD strf::detail::string_fmt_traits<CharIn>
get_fmt_traits(strf::tag<>, std::basic_string<CharIn, Traits, Allocator>)
{ return {}; }

template <typename CharIn>
constexpr STRF_HD strf::detail::string_fmt_traits<CharIn>
get_fmt_traits(strf::tag<>, strf::detail::simple_string_view<CharIn>)
{ return {}; }

#if defined(STRF_HAS_STD_STRING_VIEW)

template <typename CharIn, typename Traits>
constexpr STRF_HD strf::detail::string_fmt_traits<CharIn>
get_fmt_traits(strf::tag<>, std::basic_string_view<CharIn, Traits>)
{ return {}; }

#if defined(__cpp_char8_t)

constexpr STRF_HD strf::detail::string_fmt_traits<char8_t>
get_fmt_traits(strf::tag<>, std::basic_string_view<char8_t>)
{ return {}; }

#endif // defined(__cpp_char8_t)

constexpr STRF_HD strf::detail::string_fmt_traits<char>
get_fmt_traits(strf::tag<>, std::basic_string_view<char>)
{ return {}; }

constexpr STRF_HD strf::detail::string_fmt_traits<char16_t>
get_fmt_traits(strf::tag<>, std::basic_string_view<char16_t>)
{ return {}; }

constexpr STRF_HD strf::detail::string_fmt_traits<char32_t>
get_fmt_traits(strf::tag<>, std::basic_string_view<char32_t>)
{ return {}; }

constexpr STRF_HD strf::detail::string_fmt_traits<wchar_t>
get_fmt_traits(strf::tag<>, std::basic_string_view<wchar_t>)
{ return {}; }

#endif // defined(STRF_HAS_STD_STRING_VIEW)

#if defined(__cpp_char8_t)

constexpr STRF_HD strf::detail::string_fmt_traits<char8_t>
get_fmt_traits(strf::tag<>, const char8_t*)
{ return {}; }

#endif

constexpr STRF_HD strf::detail::string_fmt_traits<char>
get_fmt_traits(strf::tag<>, const char*)
{ return {}; }

constexpr STRF_HD strf::detail::string_fmt_traits<char16_t>
get_fmt_traits(strf::tag<>, const char16_t*)
{ return {}; }

constexpr STRF_HD strf::detail::string_fmt_traits<char32_t>
get_fmt_traits(strf::tag<>, const char32_t*)
{ return {}; }

constexpr STRF_HD strf::detail::string_fmt_traits<wchar_t>
get_fmt_traits(strf::tag<>, const wchar_t*)
{ return {}; }

namespace detail {

template <std::size_t> class string_printer;

template <std::size_t> class aligned_string_printer;

template<std::size_t SrcCharSize, std::size_t DestCharSize>
class conv_string_printer;

template<std::size_t SrcCharSize, std::size_t DestCharSize>
class aligned_conv_string_printer;

template<std::size_t DestCharSize> class conv_string_printer_variant;

template<std::size_t DestCharSize> class aligned_conv_string_printer_variant;

template <typename CharT, typename FPack, typename Preview>
struct string_printer_input
{
    using printer_type = strf::detail::string_printer<sizeof(CharT)>;

    FPack fp;
    Preview& preview;
    const CharT* str;
    std::size_t len;
};

template < typename DestCharT, typename SrcCharT, bool HasPrecision, bool HasAlignment
         , typename CvFormat >
struct mp_string_printer
{
    using type = std::conditional_t
        < HasAlignment
        , strf::detail::aligned_conv_string_printer<sizeof(SrcCharT), sizeof(DestCharT)>
        , strf::detail::conv_string_printer<sizeof(SrcCharT), sizeof(DestCharT)> >;
};

template <typename DestCharT, typename SrcCharT, bool HasPrecision, bool HasAlignment >
struct mp_string_printer
    < DestCharT, SrcCharT, HasPrecision, HasAlignment
    , strf::no_conv_format<SrcCharT> >
{
    static_assert( std::is_same<SrcCharT, DestCharT>::value
                 , "Character type mismatch. Use `conv` function." );

    using type = std::conditional_t
        < HasAlignment
        , strf::detail::aligned_string_printer<sizeof(DestCharT)>
        , strf::detail::string_printer<sizeof(DestCharT)> >;
};

template < typename DestCharT, typename SrcCharT, bool HasPrecision, bool HasAlignment >
struct mp_string_printer
    < DestCharT, SrcCharT, HasPrecision, HasAlignment
    , strf::conv_format<SrcCharT> >
{
    using type = std::conditional_t
        < std::is_same<SrcCharT,DestCharT>::value
        , std::conditional_t
            < HasAlignment
            , strf::detail::aligned_string_printer<sizeof(DestCharT)>
            , strf::detail::string_printer<sizeof(DestCharT)> >
        , std::conditional_t
            < sizeof(SrcCharT) == sizeof(DestCharT)
            , std::conditional_t
                < HasAlignment
                , strf::detail::aligned_conv_string_printer_variant<sizeof(DestCharT)>
                , strf::detail::conv_string_printer_variant<sizeof(DestCharT)> >
            , std::conditional_t
                < HasAlignment
                , strf::detail::aligned_conv_string_printer
                    < sizeof(SrcCharT), sizeof(DestCharT) >
                , strf::detail::conv_string_printer
                    < sizeof(SrcCharT), sizeof(DestCharT) > > > >;
};

template < typename DestCharT, typename SrcCharT, bool HasPrecision
         , bool HasAlignment, typename Encoding >
struct mp_string_printer
    < DestCharT, SrcCharT, HasPrecision, HasAlignment
    , strf::conv_format_with_encoding<SrcCharT, Encoding> >
{
    using type = std::conditional_t
        < sizeof(SrcCharT) == sizeof(DestCharT)
        , std::conditional_t
            < HasAlignment
            , strf::detail::aligned_conv_string_printer_variant<sizeof(DestCharT)>
            , strf::detail::conv_string_printer_variant<sizeof(DestCharT)> >
        , std::conditional_t
            < HasAlignment
            , strf::detail::aligned_conv_string_printer
                < sizeof(SrcCharT), sizeof(DestCharT) >
            , strf::detail::conv_string_printer
                < sizeof(SrcCharT), sizeof(DestCharT) > > >;
};

template < typename DestCharT, typename FPack, typename Preview
         , typename SrcCharT, bool HasPrecision, bool HasAlignment
         , typename CvFormat >
struct fmt_string_printer_input
{
    using printer_type = typename strf::detail::mp_string_printer
        < DestCharT, SrcCharT, HasPrecision, HasAlignment, CvFormat>
        :: type;

    using value_with_format_type = strf::value_with_format
        < strf::detail::simple_string_view<SrcCharT>
        , strf::string_precision_format<HasPrecision>
        , strf::alignment_format_q<HasAlignment>
        , CvFormat >;

    FPack fp;
    Preview& preview;
    value_with_format_type vwf;
};

template <typename DestCharT, typename FPack, typename Preview, typename SrcCharT>
struct string_printable_traits
{
    static_assert( std::is_same<SrcCharT, DestCharT>::value
                 , "Character type mismatch. Use `conv` or `sani` format function." );

    constexpr static STRF_HD
    strf::detail::string_printer_input<DestCharT, FPack, Preview>
    make_input ( const FPack& fp, Preview& preview
               , strf::detail::simple_string_view<SrcCharT> str )
    {
        return {fp, preview, str.data(), str.size()};
    }
};

} // namespace detail

template <typename DestCharT, typename FPack, typename Preview>
constexpr STRF_HD
strf::detail::string_printable_traits
    < DestCharT, FPack, Preview, char >
get_printable_traits(Preview&, const char*)
{ return {}; }

#if defined(__cpp_char8_t)

template <typename DestCharT, typename FPack, typename Preview>
constexpr STRF_HD strf::detail::string_printable_traits
    < DestCharT, FPack, Preview, char8_t >
get_printable_traits(Preview&, const char8_t*)
{ return {}; }

#endif // defined(__cpp_char8_t)

template <typename DestCharT, typename FPack, typename Preview>
constexpr STRF_HD strf::detail::string_printable_traits
    < DestCharT, FPack, Preview, char16_t >
get_printable_traits(Preview&, const char16_t*)
{ return {}; }

template <typename DestCharT, typename FPack, typename Preview>
constexpr STRF_HD strf::detail::string_printable_traits
    < DestCharT, FPack, Preview, char32_t >
get_printable_traits(Preview&, const char32_t*)
{ return {}; }

template <typename DestCharT, typename FPack, typename Preview>
constexpr STRF_HD strf::detail::string_printable_traits
    < DestCharT, FPack, Preview, wchar_t >
get_printable_traits(Preview&, const wchar_t*)
{ return {}; }

template < typename DestCharT, typename FPack, typename Preview
         , typename SrcCharT, typename Traits, typename Allocator >
constexpr STRF_HD strf::detail::string_printable_traits
    < DestCharT, FPack, Preview, SrcCharT >
get_printable_traits(Preview&, const std::basic_string<SrcCharT, Traits, Allocator>&)
{ return {}; }

#if defined(STRF_HAS_STD_STRING_VIEW)

template < typename DestCharT, typename FPack, typename Preview
         , typename SrcCharT, typename Traits >
constexpr STRF_HD strf::detail::string_printable_traits
    < DestCharT, FPack, Preview, SrcCharT >
get_printable_traits(Preview&, const std::basic_string_view<SrcCharT, Traits>&)
{ return {}; }

#endif //defined(STRF_HAS_STD_STRING_VIEW)

template < typename DestCharT, typename FPack, typename Preview, typename SrcCharT >
constexpr STRF_HD strf::detail::string_printable_traits
    < DestCharT, FPack, Preview, SrcCharT >
get_printable_traits(Preview&, const strf::detail::simple_string_view<SrcCharT>&)
{ return {}; }

template < typename DestCharT, typename FPack, typename Preview, typename SrcCharT
         , bool HasPrecision, bool HasAlignment, typename CvFormat >
struct printable_traits
    < DestCharT, FPack, Preview
    , strf::value_with_format
          < strf::detail::simple_string_view<SrcCharT>
          , strf::string_precision_format<HasPrecision>
          , strf::alignment_format_q<HasAlignment>
          , CvFormat > >
{
    template <typename Arg>
    constexpr static STRF_HD strf::detail::fmt_string_printer_input
        < DestCharT, FPack, Preview, SrcCharT
        , HasPrecision, HasAlignment, CvFormat>
    make_input(const FPack& fp, Preview& preview, const Arg& arg)
    {
        return {fp, preview, arg};
    }
};

namespace detail {

template <std::size_t CharSize>
class string_printer: public strf::printer<CharSize>
{
public:
    using char_type = strf::underlying_char_type<CharSize>;

    template <typename CharT, typename FPack, typename Preview>
    constexpr STRF_HD string_printer
        ( const strf::detail::string_printer_input<CharT, FPack, Preview>& input )
        : str_(reinterpret_cast<const char_type*>(input.str))
        , len_(input.len)
    {
        static_assert(CharSize == sizeof(CharT), "");

        STRF_IF_CONSTEXPR(Preview::width_required) {
            decltype(auto) wcalc = get_facet_<strf::width_calculator_c, CharT>(input.fp);
            auto w = wcalc.str_width
                ( get_facet_<strf::char_encoding_c<CharT>, CharT>(input.fp)
                , input.preview.remaining_width(), str_, len_
                , get_facet_<strf::surrogate_policy_c, CharT>(input.fp) );
           input.preview.subtract_width(w);
        }
        input.preview.add_size(input.len);
    }

    template < typename DestCharT, typename FPack, typename Preview
             , typename SrcCharT, typename CvFormat >
    constexpr STRF_HD string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, FPack, Preview, SrcCharT, false, false, CvFormat >&
            input )
        : str_(reinterpret_cast<const char_type*>(input.vwf.value().data()))
        , len_(input.vwf.value().size())
    {
        static_assert(CharSize == sizeof(SrcCharT), "");
        static_assert(CharSize == sizeof(DestCharT), "");

        STRF_IF_CONSTEXPR(Preview::width_required) {
            decltype(auto) wcalc = get_facet_<strf::width_calculator_c, SrcCharT>(input.fp);
            auto w = wcalc.str_width
                ( get_facet_<strf::char_encoding_c<SrcCharT>, SrcCharT>(input.fp)
                , input.preview.remaining_width()
                , str_
                , input.vwf.value().size()
                , get_facet_<strf::surrogate_policy_c, SrcCharT>(input.fp) );
           input.preview.subtract_width(w);
        }
        input.preview.add_size(input.vwf.value().size());
    }

    template < typename DestCharT, typename FPack, typename Preview
             , typename SrcCharT, typename CvFormat >
    constexpr STRF_HD string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, FPack, Preview, SrcCharT, true, false, CvFormat >&
            input )
        : str_(reinterpret_cast<const char_type*>(input.vwf.value().data()))
    {
        static_assert(CharSize == sizeof(SrcCharT), "");
        static_assert(CharSize == sizeof(DestCharT), "");

        decltype(auto) wcalc = get_facet_<strf::width_calculator_c, SrcCharT>(input.fp);
        auto res = wcalc.str_width_and_pos
            ( get_facet_<strf::char_encoding_c<SrcCharT>, SrcCharT>(input.fp)
            , input.vwf.precision()
            , str_
            , input.vwf.value().size()
            , get_facet_<strf::surrogate_policy_c, SrcCharT>(input.fp) );
        len_ = res.pos;
        input.preview.subtract_width(res.width);
        input.preview.add_size(res.pos);
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    const char_type* str_;
    std::size_t len_;

    template <typename Category, typename CharT, typename FPack>
    static STRF_HD decltype(auto) get_facet_(const FPack& fp)
    {
        using input_tag = strf::string_input_tag<CharT>;
        return fp.template get_facet<Category, input_tag>();
    }
};

template<std::size_t CharSize>
STRF_HD void string_printer<CharSize>::print_to(strf::underlying_outbuf<CharSize>& ob) const
{
    strf::write(ob, str_, len_);
}

template <std::size_t CharSize>
class aligned_string_printer: public strf::printer<CharSize>
{
public:
    using char_type = strf::underlying_char_type<CharSize>;

    template < typename DestCharT, typename FPack, typename Preview
             , typename SrcCharT, typename CvFormat >
    STRF_HD aligned_string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, FPack, Preview, SrcCharT, false, true, CvFormat >&
            input )
        : str_(reinterpret_cast<const char_type*>(input.vwf.value().data()))
        , len_(input.vwf.value().size())
        , afmt_(input.vwf.get_alignment_format_data())
    {
        static_assert(CharSize == sizeof(SrcCharT), "");
        static_assert(CharSize == sizeof(DestCharT), "");

        decltype(auto) wcalc = get_facet_<strf::width_calculator_c, SrcCharT>(input.fp);
        auto enc = get_facet_<strf::char_encoding_c<SrcCharT>, SrcCharT>(input.fp);
        strf::width_t limit =
            ( Preview::width_required && input.preview.remaining_width() > afmt_.width
            ? input.preview.remaining_width()
            : afmt_.width );
        auto surr_poli = get_facet_<strf::surrogate_policy_c, SrcCharT>(input.fp);
        auto strw = wcalc.str_width(enc, limit, str_, len_, surr_poli);
        encode_fill_ = enc.encode_fill_func();
        auto fillcount = init_(input.preview, strw);
        preview_size_(input.preview, enc, fillcount);
    }

    template < typename DestCharT, typename FPack, typename Preview
             , typename SrcCharT, typename CvFormat >
    STRF_HD aligned_string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, FPack, Preview, SrcCharT, true, true, CvFormat >&
            input )
        : str_(reinterpret_cast<const char_type*>(input.vwf.value().begin()))
        , afmt_(input.vwf.get_alignment_format_data())
    {
        static_assert(CharSize == sizeof(SrcCharT), "");
        static_assert(CharSize == sizeof(DestCharT), "");

        decltype(auto) wcalc = get_facet_<strf::width_calculator_c, SrcCharT>(input.fp);
        auto enc = get_facet_<strf::char_encoding_c<SrcCharT>, SrcCharT>(input.fp);
        auto surr_poli = get_facet_<strf::surrogate_policy_c, SrcCharT>(input.fp);
        auto res = wcalc.str_width_and_pos
            ( enc, input.vwf.precision(), str_, input.vwf.value().size(), surr_poli );
        len_ = res.pos;
        encode_fill_ = enc.encode_fill_func();
        auto fillcount = init_(input.preview, res.width);
        preview_size_(input.preview, enc, fillcount);
    }

    STRF_HD ~aligned_string_printer();

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    const char_type* str_;
    std::size_t len_;
    strf::encode_fill_f<CharSize> encode_fill_;
    strf::alignment_format_data afmt_;
    std::int16_t left_fillcount_;
    std::int16_t right_fillcount_;

    template <typename Category, typename CharT, typename FPack>
    static STRF_HD decltype(auto) get_facet_(const FPack& fp)
    {
        using input_tag = strf::string_input_tag<CharT>;
        return fp.template get_facet<Category, input_tag>();
    }

    template <typename Preview>
    STRF_HD std::uint16_t init_(Preview&, strf::width_t strw);

    template <typename Encoding>
    STRF_HD void preview_size_( strf::size_preview<true>& preview
                              , Encoding enc, std::uint16_t fillcount )
    {
        preview.add_size(len_);
        if (fillcount > 0) {
            preview.add_size(fillcount * enc.encoded_char_size(afmt_.fill));
        }
    }

    template <typename Encoding>
    STRF_HD void preview_size_(strf::size_preview<false>&, Encoding, std::uint16_t)
    {
    }
};

template<std::size_t CharSize>
STRF_HD aligned_string_printer<CharSize>::~aligned_string_printer()
{
}

template<std::size_t CharSize>
template <typename Preview>
inline STRF_HD std::uint16_t aligned_string_printer<CharSize>::init_
    ( Preview& preview, strf::width_t strw )
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
        preview.subtract_width(strw + fillcount);
        return fillcount;
    } else {
        right_fillcount_ = 0;
        left_fillcount_ = 0;
        preview.subtract_width(strw);
        return 0;
    }
}

template<std::size_t CharSize>
void STRF_HD aligned_string_printer<CharSize>::print_to
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (left_fillcount_ > 0) {
        encode_fill_(ob, left_fillcount_, afmt_.fill);
    }
    strf::write(ob, str_, len_);
    if (right_fillcount_ > 0) {
        encode_fill_(ob, right_fillcount_, afmt_.fill);
    }
}

#if defined(STRF_SEPARATE_COMPILATION)

STRF_EXPLICIT_TEMPLATE class string_printer<1>;
STRF_EXPLICIT_TEMPLATE class string_printer<2>;
STRF_EXPLICIT_TEMPLATE class string_printer<4>;

STRF_EXPLICIT_TEMPLATE class aligned_string_printer<1>;
STRF_EXPLICIT_TEMPLATE class aligned_string_printer<2>;
STRF_EXPLICIT_TEMPLATE class aligned_string_printer<4>;

#endif // defined(STRF_SEPARATE_COMPILATION)

} // namespace detail

} // namespace strf

#endif  /* STRF_DETAIL_INPUT_TYPES_CHAR_PTR */

