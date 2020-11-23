#ifndef STRF_DETAIL_INPUT_TYPES_STRING
#define STRF_DETAIL_INPUT_TYPES_STRING

#include <strf/detail/facets/width_calculator.hpp>
#include <strf/facets_pack.hpp>

namespace strf {
namespace detail {

template <typename CharIn>
class simple_string_view
{
public:

    using iterator = const CharIn*;
    using const_iterator = const CharIn*;

    template < typename StringType
             , typename = decltype(std::declval<const StringType&>().data())
             , typename = decltype(std::declval<const StringType&>().size())  >
    constexpr STRF_HD simple_string_view(const StringType& s)
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
struct no_conv_formatter;

template <typename CharT>
struct conv_formatter;

template <typename Encoding>
struct conv_formatter_with_encoding;

template <typename CharT>
struct sani_formatter;

template <typename Encoding>
struct sani_formatter_with_encoding;

template <typename T, typename CharT>
class no_conv_formatter_fn
{
public:

    constexpr STRF_HD no_conv_formatter_fn() noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit no_conv_formatter_fn
        ( const no_conv_formatter_fn<U, CharT>& ) noexcept
    {
    }

    constexpr STRF_HD auto convert_encoding() const
    {
        using return_type = strf::fmt_replace< T
                                             , strf::no_conv_formatter<CharT>
                                             , strf::conv_formatter<CharT> >;
        return return_type{ static_cast<const T&>(*this) };
    }

    template <typename Encoding>
    constexpr STRF_HD auto convert_from_encoding(Encoding enc) const
    {
        static_assert( std::is_same<typename Encoding::char_type, CharT>::value
                     , "This encoding is associated with another character type." );
        using return_type = strf::fmt_replace
            < T
            , strf::no_conv_formatter<CharT>
            , strf::conv_formatter_with_encoding<Encoding> >;

        return return_type
            { static_cast<const T&>(*this)
            , strf::tag<strf::conv_formatter_with_encoding<Encoding>>{}
            , enc };
    }
    constexpr STRF_HD auto conv() const
    {
        return convert_encoding();
    }
    template <typename Encoding>
    constexpr STRF_HD auto conv(Encoding enc) const
    {
        return convert_from_encoding(enc);
    }

    constexpr STRF_HD auto sanitize_encoding() const
    {
        using return_type = strf::fmt_replace< T
                                             , strf::no_conv_formatter<CharT>
                                             , strf::sani_formatter<CharT> >;
        return return_type{ static_cast<const T&>(*this) };
    }
    template <typename Encoding>
    constexpr STRF_HD auto sanitize_from_encoding(Encoding enc) const
    {
        static_assert( std::is_same<typename Encoding::char_type, CharT>::value
                     , "This encoding is associated with another character type." );
        using return_type = strf::fmt_replace
            < T
            , strf::no_conv_formatter<CharT>
            , strf::sani_formatter_with_encoding<Encoding> >;

        return return_type
            { static_cast<const T&>(*this)
            , strf::tag<strf::sani_formatter_with_encoding<Encoding>>{}
            , enc };
    }
    constexpr STRF_HD auto sani() const
    {
        return sanitize_encoding();
    }
    template <typename Encoding>
    constexpr STRF_HD auto sani(Encoding enc) const
    {
        return sanitize_from_encoding(enc);
    }
};

template <typename T, typename CharT>
struct conv_formatter_fn
{
    constexpr STRF_HD conv_formatter_fn() noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit conv_formatter_fn
        ( const conv_formatter_fn<U, CharT>& ) noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit conv_formatter_fn
        ( const strf::no_conv_formatter_fn<U, CharT>& ) noexcept
    {
    }
};

template <typename T, typename Encoding>
class conv_formatter_with_encoding_fn
{
public:

    STRF_HD conv_formatter_with_encoding_fn(Encoding e)
        : encoding_(e)
    {
    }

    conv_formatter_with_encoding_fn
        ( const conv_formatter_with_encoding_fn& other ) noexcept = default;

    template <typename U>
    STRF_HD explicit conv_formatter_with_encoding_fn
        ( const strf::conv_formatter_with_encoding_fn<U, Encoding>& other ) noexcept
        : encoding_(other.get_encoding())
    {
    }

    STRF_HD Encoding get_encoding() const
    {
        return encoding_;
    }

private:

    Encoding encoding_;
};

template <typename CharT>
struct no_conv_formatter
{
    template <typename T>
    using fn = strf::no_conv_formatter_fn<T, CharT>;
};

template <typename CharT>
struct conv_formatter
{
    template <typename T>
    using fn = strf::conv_formatter_fn<T, CharT>;
};

template <typename Encoding>
struct conv_formatter_with_encoding
{
    template <typename T>
    using fn = strf::conv_formatter_with_encoding_fn<T, Encoding>;
};

template <typename CharT>
struct sani_formatter
{
    template <typename T>
    using fn = strf::conv_formatter_fn<T, CharT>;
};

template <typename Encoding>
struct sani_formatter_with_encoding
{
    template <typename T>
    using fn = strf::conv_formatter_with_encoding_fn<T, Encoding>;
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
    constexpr STRF_HD string_precision_formatter_fn(strf::width_t p) noexcept
        : precision_(p)
    {
    }
    template <typename U>
    constexpr STRF_HD string_precision_formatter_fn
        ( strf::string_precision_formatter_fn<U, true> other ) noexcept
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
class string_precision_formatter_fn<T, false>
{
    using adapted_derived_type_
        = strf::fmt_replace< T
                           , strf::string_precision_formatter<false>
                           , strf::string_precision_formatter<true> >;
public:

    constexpr STRF_HD string_precision_formatter_fn() noexcept
    {
    }
    template <typename U>
    constexpr STRF_HD string_precision_formatter_fn
        ( strf::string_precision_formatter_fn<U, false> ) noexcept
    {
    }
    constexpr STRF_HD adapted_derived_type_ p(strf::width_t precision) const noexcept
    {
        return { static_cast<const T&>(*this)
               , strf::tag<string_precision_formatter<true> >{}
               , precision };
    }
    constexpr STRF_HD auto get_string_precision() const noexcept
    {
        return strf::string_precision<false>{};
    }
};

namespace detail {

template <typename SrcCharT> struct string_printing;

} // namespace detail

namespace detail {

template <typename SrcCharT, typename DestCharT> class string_printer;
template <typename SrcCharT, typename DestCharT> class aligned_string_printer;
template <typename SrcCharT, typename DestCharT> class conv_string_printer;
template <typename SrcCharT, typename DestCharT> class aligned_conv_string_printer;
template <typename DestCharT> class conv_string_printer_variant;
template <typename DestCharT> class aligned_conv_string_printer_variant;

template <typename CharT, typename Preview, typename FPack>
struct string_printer_input
{
    using printer_type = strf::detail::string_printer<CharT, CharT>;

    const CharT* str;
    std::size_t len;
    Preview& preview;
    FPack facets;
};

template < typename DestCharT, typename SrcCharT, bool HasPrecision, bool HasAlignment
         , typename CvFormat >
struct mp_string_printer
{
    using type = std::conditional_t
        < HasAlignment
        , strf::detail::aligned_conv_string_printer<SrcCharT, DestCharT>
        , strf::detail::conv_string_printer<SrcCharT, DestCharT> >;
};

template <typename DestCharT, typename SrcCharT, bool HasPrecision, bool HasAlignment >
struct mp_string_printer
    < DestCharT, SrcCharT, HasPrecision, HasAlignment
    , strf::no_conv_formatter<SrcCharT> >
{
    static_assert( std::is_same<SrcCharT, DestCharT>::value
                 , "Character type mismatch. Use `conv` function." );

    using type = std::conditional_t
        < HasAlignment
        , strf::detail::aligned_string_printer<SrcCharT, DestCharT>
        , strf::detail::string_printer<SrcCharT, DestCharT> >;
};

template < typename DestCharT, typename SrcCharT, bool HasPrecision, bool HasAlignment >
struct mp_string_printer
    < DestCharT, SrcCharT, HasPrecision, HasAlignment
    , strf::conv_formatter<SrcCharT> >
{
    using type = std::conditional_t
        < std::is_same<SrcCharT,DestCharT>::value
        , std::conditional_t
            < HasAlignment
            , strf::detail::aligned_string_printer<SrcCharT, DestCharT>
            , strf::detail::string_printer<SrcCharT, DestCharT> >
        , std::conditional_t
            < sizeof(SrcCharT) == sizeof(DestCharT)
            , std::conditional_t
                < HasAlignment
                , strf::detail::aligned_conv_string_printer_variant<DestCharT>
                , strf::detail::conv_string_printer_variant<DestCharT> >
            , std::conditional_t
                < HasAlignment
                , strf::detail::aligned_conv_string_printer
                    < SrcCharT, DestCharT >
                , strf::detail::conv_string_printer
                    < SrcCharT, DestCharT > > > >;
};

template < typename DestCharT, typename SrcCharT, bool HasPrecision
         , bool HasAlignment, typename Encoding >
struct mp_string_printer
    < DestCharT, SrcCharT, HasPrecision, HasAlignment
    , strf::conv_formatter_with_encoding<Encoding> >
{
    static_assert( std::is_same<typename Encoding::char_type, SrcCharT>::value
                 , "This encoding is associated with another character type." );

    using type = std::conditional_t
        < sizeof(SrcCharT) == sizeof(DestCharT)
        , std::conditional_t
            < HasAlignment
            , strf::detail::aligned_conv_string_printer_variant<DestCharT>
            , strf::detail::conv_string_printer_variant<DestCharT> >
        , std::conditional_t
            < HasAlignment
            , strf::detail::aligned_conv_string_printer
                < SrcCharT, DestCharT >
            , strf::detail::conv_string_printer
                < SrcCharT, DestCharT > > >;
};

template <typename SrcCharT>
struct string_printing;

template <typename SrcCharT, bool HasPrecision, bool HasAlignment, typename CvFormat>
using string_with_formatters = strf::value_with_formatters
    < string_printing<SrcCharT>
    , strf::string_precision_formatter<HasPrecision>
    , strf::alignment_formatter_q<HasAlignment>
    , CvFormat >;

template < typename DestCharT, typename SrcCharT
         , bool HasPrecision, bool HasAlignment, typename CvFormat
         , typename Preview, typename FPack >
struct fmt_string_printer_input
{
    using printer_type = typename strf::detail::mp_string_printer
        < DestCharT, SrcCharT, HasPrecision, HasAlignment, CvFormat>
        :: type;

    strf::detail::string_with_formatters<SrcCharT, HasPrecision, HasAlignment, CvFormat> arg;
    Preview& preview;
    FPack facets;
};

template <typename SrcCharT>
struct string_printing
{
#if defined(__CUDA_ARCH__)
    using facet_tag = void; // to-do try to fix that
#else
    using facet_tag = strf::string_input_tag<SrcCharT>;
#endif
    using forwarded_type = strf::detail::simple_string_view<SrcCharT>;
    using formatters = strf::tag
        < strf::string_precision_formatter<false>
        , strf::alignment_formatter
        , strf::no_conv_formatter<SrcCharT> >;

    template <typename DestCharT, typename Preview, typename FPack>
    constexpr STRF_HD static auto make_printer_input
        (Preview& preview, const FPack& facets,  forwarded_type x ) noexcept
        -> strf::detail::string_printer_input<DestCharT, Preview, FPack>
    {
        static_assert
            ( std::is_same<SrcCharT, DestCharT>::value
            , "Character type mismatch. Use `conv` or `sani` format function." );

        return {x.data(), x.length(), preview, facets};
    }

    template < typename DestCharT, typename Preview, typename FPack
             , bool HasPrecision, bool HasAlignment, typename CvFormat >
    constexpr STRF_HD static auto make_printer_input
        ( Preview& preview
        , const FPack& facets
        , const strf::detail::string_with_formatters
            < SrcCharT, HasPrecision, HasAlignment, CvFormat >& x ) noexcept
        -> strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, HasPrecision, HasAlignment, CvFormat, Preview, FPack>
    {
        return {x, preview, facets};
    }
};

} // namespace detail

template <typename CharIn>
constexpr STRF_HD auto tag_invoke
    (strf::print_traits_tag, strf::detail::simple_string_view<CharIn>) noexcept
    -> strf::detail::string_printing<CharIn>
    { return {}; }

#if defined(STRF_HAS_STD_STRING_DECLARATION)

template <typename CharIn, typename Traits, typename Allocator>
constexpr STRF_HD auto tag_invoke
    (strf::print_traits_tag, const std::basic_string<CharIn, Traits, Allocator>&) noexcept
    -> strf::detail::string_printing<CharIn>
    { return {}; }

#endif // defined(STRF_HAS_STD_STRING_DECLARATION)

#if defined(STRF_HAS_STD_STRING_VIEW)

template <typename CharIn, typename Traits>
constexpr STRF_HD auto tag_invoke
    (strf::print_traits_tag, std::basic_string_view<CharIn, Traits>) noexcept
    -> strf::detail::string_printing<CharIn>
    { return {}; }

#if defined(__cpp_char8_t)

constexpr STRF_HD auto tag_invoke
    (strf::print_traits_tag, std::basic_string_view<char8_t>) noexcept
    -> strf::detail::string_printing<char8_t>
    { return {}; }

#endif // defined(__cpp_char8_t)

constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, std::basic_string_view<char>) noexcept
    -> strf::detail::string_printing<char>
    { return {}; }


constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, std::basic_string_view<char16_t>) noexcept
    -> strf::detail::string_printing<char16_t>
    { return {}; }


constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, std::basic_string_view<char32_t>) noexcept
    -> strf::detail::string_printing<char32_t>
    { return {}; }


constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, std::basic_string_view<wchar_t>) noexcept
    -> strf::detail::string_printing<wchar_t>
    { return {}; }

#endif // defined(STRF_HAS_STD_STRING_VIEW)

#if defined(__cpp_char8_t)


constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, const char8_t*) noexcept
    -> strf::detail::string_printing<char8_t>
    { return {}; }

#endif


constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, const char*) noexcept
    -> strf::detail::string_printing<char>
    { return {}; }


constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, const char16_t*) noexcept
    -> strf::detail::string_printing<char16_t>
    { return {}; }


constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, const char32_t*) noexcept
    -> strf::detail::string_printing<char32_t>
    { return {}; }


constexpr STRF_HD auto tag_invoke(strf::print_traits_tag, const wchar_t*) noexcept
    -> strf::detail::string_printing<wchar_t>
    { return {}; }

namespace detail {

template <typename SrcCharT, typename DestCharT>
class string_printer: public strf::printer<DestCharT>
{
public:
    static_assert(sizeof(SrcCharT) == sizeof(DestCharT), "");

    template <typename Preview, typename FPack>
    constexpr STRF_HD string_printer
        ( const strf::detail::string_printer_input<SrcCharT, Preview, FPack>& input )
        : str_(input.str)
        , len_(input.len)
    {
        STRF_IF_CONSTEXPR(Preview::width_required) {
            decltype(auto) wcalc = get_facet_<strf::width_calculator_c>(input.facets);
            auto w = wcalc.str_width
                ( get_facet_<strf::char_encoding_c<SrcCharT>>(input.facets)
                , input.preview.remaining_width(), str_, len_
                , get_facet_<strf::surrogate_policy_c>(input.facets) );
           input.preview.subtract_width(w);
        }
        input.preview.add_size(input.len);
    }

    template < typename Preview, typename FPack, typename CvFormat >
    constexpr STRF_HD string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, false, false, CvFormat, Preview, FPack >&
            input )
        : str_(input.arg.value().data())
        , len_(input.arg.value().size())
    {
        STRF_IF_CONSTEXPR(Preview::width_required) {
            decltype(auto) wcalc = get_facet_<strf::width_calculator_c>(input.facets);
            auto w = wcalc.str_width
                ( get_facet_<strf::char_encoding_c<SrcCharT>>(input.facets)
                , input.preview.remaining_width()
                , str_
                , input.arg.value().size()
                , get_facet_<strf::surrogate_policy_c>(input.facets) );
           input.preview.subtract_width(w);
        }
        input.preview.add_size(input.arg.value().size());
    }

    template < typename Preview, typename FPack, typename CvFormat >
    constexpr STRF_HD string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, true, false, CvFormat, Preview, FPack >&
            input )
        : str_(input.arg.value().data())
    {
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c>(input.facets);
        auto res = wcalc.str_width_and_pos
            ( get_facet_<strf::char_encoding_c<SrcCharT>>(input.facets)
            , input.arg.precision()
            , str_
            , input.arg.value().size()
            , get_facet_<strf::surrogate_policy_c>(input.facets) );
        len_ = res.pos;
        input.preview.subtract_width(res.width);
        input.preview.add_size(res.pos);
    }

    STRF_HD void print_to(strf::basic_outbuff<DestCharT>& ob) const override;

private:

    const SrcCharT* str_;
    std::size_t len_;

    template <typename Category, typename FPack>
    static STRF_HD decltype(auto) get_facet_(const FPack& facets)
    {
        using input_tag = strf::string_input_tag<SrcCharT>;
        return facets.template get_facet<Category, input_tag>();
    }
};

template<typename SrcCharT, typename DestCharT>
STRF_HD void string_printer<SrcCharT, DestCharT>::print_to
    ( strf::basic_outbuff<DestCharT>& ob ) const
{
    strf::detail::outbuff_interchar_copy(ob, str_, len_);
}

template <typename SrcCharT, typename DestCharT>
class aligned_string_printer: public strf::printer<DestCharT>
{
public:
    static_assert(sizeof(SrcCharT) == sizeof(DestCharT), "");

    template < typename Preview, typename FPack, typename CvFormat >
    STRF_HD aligned_string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, false, true, CvFormat, Preview, FPack >&
            input )
        : str_(input.arg.value().data())
        , len_(input.arg.value().size())
        , afmt_(input.arg.get_alignment_format())
    {

        decltype(auto) wcalc = get_facet_<strf::width_calculator_c>(input.facets);
        auto src_enc = get_facet_<strf::char_encoding_c<SrcCharT>>(input.facets);
        auto dest_enc = get_facet_<strf::char_encoding_c<DestCharT>>(input.facets);
        strf::width_t limit =
            ( Preview::width_required && input.preview.remaining_width() > afmt_.width
            ? input.preview.remaining_width()
            : afmt_.width );
        auto surr_poli = get_facet_<strf::surrogate_policy_c>(input.facets);
        auto strw = wcalc.str_width(src_enc, limit, str_, len_, surr_poli);
        encode_fill_ = dest_enc.encode_fill_func();
        auto fillcount = init_(input.preview, strw);
        preview_size_(input.preview, dest_enc, fillcount);
    }

    template < typename Preview, typename FPack, typename CvFormat >
    STRF_HD aligned_string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, true, true, CvFormat, Preview, FPack >&
            input )
        : str_(input.arg.value().begin())
        , afmt_(input.arg.get_alignment_format())
    {
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c>(input.facets);
        auto src_enc = get_facet_<strf::char_encoding_c<SrcCharT>>(input.facets);
        auto dest_enc = get_facet_<strf::char_encoding_c<DestCharT>>(input.facets);
        auto surr_poli = get_facet_<strf::surrogate_policy_c>(input.facets);
        auto res = wcalc.str_width_and_pos
            ( src_enc, input.arg.precision(), str_, input.arg.value().size(), surr_poli );
        len_ = res.pos;
        encode_fill_ = dest_enc.encode_fill_func();
        auto fillcount = init_(input.preview, res.width);
        preview_size_(input.preview, dest_enc, fillcount);
    }

    STRF_HD ~aligned_string_printer();

    STRF_HD void print_to(strf::basic_outbuff<DestCharT>& ob) const override;

private:

    const SrcCharT* str_;
    std::size_t len_;
    strf::encode_fill_f<DestCharT> encode_fill_;
    strf::alignment_format afmt_;
    std::int16_t left_fillcount_;
    std::int16_t right_fillcount_;

    template <typename Category, typename FPack>
    static STRF_HD decltype(auto) get_facet_(const FPack& facets)
    {
        using input_tag = strf::string_input_tag<SrcCharT>;
        return facets.template get_facet<Category, input_tag>();
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

template<typename SrcCharT, typename DestCharT>
STRF_HD aligned_string_printer<SrcCharT, DestCharT>::~aligned_string_printer()
{
}

template<typename SrcCharT, typename DestCharT>
template <typename Preview>
inline STRF_HD std::uint16_t aligned_string_printer<SrcCharT, DestCharT>::init_
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

template<typename SrcCharT, typename DestCharT>
void STRF_HD aligned_string_printer<SrcCharT, DestCharT>::print_to
    ( strf::basic_outbuff<DestCharT>& ob ) const
{
    if (left_fillcount_ > 0) {
        encode_fill_(ob, left_fillcount_, afmt_.fill);
    }
    strf::detail::outbuff_interchar_copy(ob, str_, len_);
    if (right_fillcount_ > 0) {
        encode_fill_(ob, right_fillcount_, afmt_.fill);
    }
}

template < typename DestCharT, typename Preview, typename FPack
         , typename SrcCharT, bool HasP, bool HasA, typename SrcEncoding >
constexpr STRF_HD decltype(auto) get_src_encoding
    ( const strf::detail::fmt_string_printer_input
        < DestCharT, SrcCharT, HasP, HasA
        , strf::conv_formatter_with_encoding<SrcEncoding>, Preview, FPack>&
      input )
{
    static_assert( std::is_same<typename SrcEncoding::char_type, SrcCharT>::value
                 , "This encoding is associated with another character type." );
    return input.arg.get_encoding();
}

template < typename DestCharT, typename Preview, typename FPack
         , typename SrcCharT, bool HasP, bool HasA, typename SrcEncoding >
constexpr STRF_HD decltype(auto) get_src_encoding
    ( const strf::detail::fmt_string_printer_input
        < DestCharT, SrcCharT, HasP, HasA
        , strf::sani_formatter_with_encoding<SrcEncoding>, Preview, FPack >&
      input )
{
    static_assert( std::is_same<typename SrcEncoding::char_type, SrcCharT>::value
                 , "This encoding is associated with another character type." );
    return input.arg.get_encoding();
}

template < typename DestCharT, typename Preview, typename FPack
         , typename SrcCharT, bool HasP, bool HasA >
constexpr STRF_HD decltype(auto) get_src_encoding
    ( const strf::detail::fmt_string_printer_input
        < DestCharT, SrcCharT, HasP, HasA, strf::conv_formatter<SrcCharT>, Preview, FPack >&
      input )
{
    return strf::get_facet
        <strf::char_encoding_c<SrcCharT>, strf::string_input_tag<SrcCharT>>
        ( input.facets );
}

template < typename DestCharT, typename Preview, typename FPack
         , typename SrcCharT, bool HasP, bool HasA >
constexpr STRF_HD decltype(auto) get_src_encoding
    ( const strf::detail::fmt_string_printer_input
        < DestCharT, SrcCharT, HasP, HasA, strf::sani_formatter<SrcCharT>, Preview, FPack >&
      input )
{
    return strf::get_facet
        <strf::char_encoding_c<SrcCharT>, strf::string_input_tag<SrcCharT>>
        ( input.facets );
}

template<typename SrcCharT, typename DestCharT>
class conv_string_printer: public strf::printer<DestCharT>
{
public:

    template <typename Preview, typename FPack, typename CvFormat>
    STRF_HD conv_string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, false, false, CvFormat, Preview, FPack >&
            input )
        : str_(input.arg.value().data())
        , len_(input.arg.value().size())
        , inv_seq_notifier_(get_facet_<strf::invalid_seq_notifier_c, SrcCharT>(input.facets))
        , surr_poli_(get_facet_<strf::surrogate_policy_c, SrcCharT>(input.facets))
    {
        auto src_enc  = strf::detail::get_src_encoding(input);
        auto dest_enc = get_facet_<strf::char_encoding_c<DestCharT>, SrcCharT>(input.facets);
        STRF_IF_CONSTEXPR (Preview::width_required) {
            decltype(auto) wcalc = get_facet_<strf::width_calculator_c, SrcCharT>(input.facets);
            auto w = wcalc.str_width( src_enc, input.preview.remaining_width()
                                    , str_, len_, surr_poli_);
            input.preview.subtract_width(w);
        }
        init_(input.preview, src_enc, dest_enc);
    }

    template <typename Preview, typename FPack, typename CvFormat>
    STRF_HD conv_string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, true, false, CvFormat, Preview, FPack >&
            input )
        : str_(input.arg.value().data())
        , inv_seq_notifier_(get_facet_<strf::invalid_seq_notifier_c, SrcCharT>(input.facets))
        , surr_poli_(get_facet_<strf::surrogate_policy_c, SrcCharT>(input.facets))
    {
        auto src_enc  = strf::detail::get_src_encoding(input);
        auto dest_enc = get_facet_<strf::char_encoding_c<DestCharT>, SrcCharT>(input.facets);
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c, SrcCharT>(input.facets);
        auto res = wcalc.str_width_and_pos
            ( src_enc, input.arg.precision(), str_
            , input.arg.value().size(), surr_poli_ );
        len_ = res.pos;
        input.preview.subtract_width(res.width);
        init_( input.preview, src_enc
             , get_facet_<strf::char_encoding_c<DestCharT>, SrcCharT>(input.facets));
    }

    STRF_HD ~conv_string_printer() { }

    STRF_HD void print_to(strf::basic_outbuff<DestCharT>& ob) const override;

private:

    template < typename Preview, typename SrcEncoding, typename DestEncoding >
    STRF_HD void init_(Preview& preview, SrcEncoding src_enc, DestEncoding dest_enc)
    {
        decltype(auto) transcoder = find_transcoder(src_enc, dest_enc);
        transcode_ = transcoder.transcode_func();
        if (transcode_ == nullptr) {
            src_to_u32_ = src_enc.to_u32().transcode_func();
            u32_to_dest_ = dest_enc.from_u32().transcode_func();
        }
        STRF_IF_CONSTEXPR (Preview::size_required) {
            strf::transcode_size_f<SrcCharT>  transcode_size
                = transcoder.transcode_size_func();
            std::size_t s = 0;
            if (transcode_size != nullptr) {
                s = transcode_size(str_, len_, surr_poli_);
            } else {
                s = strf::decode_encode_size<SrcCharT>
                    ( src_enc.to_u32().transcode_func()
                    , dest_enc.from_u32().transcode_size_func()
                    , str_, len_, inv_seq_notifier_, surr_poli_ );
            }
            preview.add_size(s);
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
    const strf::invalid_seq_notifier inv_seq_notifier_;
    const strf::surrogate_policy surr_poli_;
    template <typename Category, typename SrcChar, typename FPack>
    static STRF_HD decltype(auto) get_facet_(const FPack& facets)
    {
        using input_tag = strf::string_input_tag<SrcChar>;
        return facets.template get_facet<Category, input_tag>();
    }
};

template<typename SrcCharT, typename DestCharT>
STRF_HD void conv_string_printer<SrcCharT, DestCharT>::print_to
    ( strf::basic_outbuff<DestCharT>& ob ) const
{
    if (can_transcode_directly()) {
        transcode_(ob, str_, len_, inv_seq_notifier_, surr_poli_);
    } else {
        strf::decode_encode<SrcCharT, DestCharT>
            ( ob, src_to_u32_, u32_to_dest_, str_
            , len_, inv_seq_notifier_, surr_poli_ );
    }
}

template<typename SrcCharT, typename DestCharT>
class aligned_conv_string_printer: public printer<DestCharT>
{
public:

    template <typename Preview, typename FPack, typename CvFormat>
    STRF_HD aligned_conv_string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, false, true, CvFormat, Preview, FPack >&
            input )
        : str_(input.arg.value().data())
        , len_(input.arg.value().size())
        , afmt_(input.arg.get_alignment_format())
        , inv_seq_notifier_(get_facet_<strf::invalid_seq_notifier_c, SrcCharT>(input.facets))
        , surr_poli_(get_facet_<strf::surrogate_policy_c, SrcCharT>(input.facets))
    {
        auto src_enc = strf::detail::get_src_encoding(input);
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c, SrcCharT>(input.facets);
        strf::width_t limit =
            ( Preview::width_required && input.preview.remaining_width() > afmt_.width
            ? input.preview.remaining_width()
            : afmt_.width );
        auto str_width = wcalc.str_width(src_enc, limit, str_, len_, surr_poli_);
        init_( input.preview, str_width, src_enc
             , get_facet_<strf::char_encoding_c<DestCharT>, SrcCharT>(input.facets) );
    }

    template <typename Preview, typename FPack, typename CvFormat>
    STRF_HD aligned_conv_string_printer
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, true, true, CvFormat, Preview, FPack >&
            input )
        : str_(input.arg.value().data())
        , len_(input.arg.value().size())
        , afmt_(input.arg.get_alignment_format())
        , inv_seq_notifier_(get_facet_<strf::invalid_seq_notifier_c, SrcCharT>(input.facets))
        , surr_poli_(get_facet_<strf::surrogate_policy_c, SrcCharT>(input.facets))
    {
        auto src_enc = strf::detail::get_src_encoding(input);
        decltype(auto) wcalc = get_facet_<strf::width_calculator_c, SrcCharT>(input.facets);
        auto res = wcalc.str_width_and_pos
            ( src_enc, input.arg.precision(), str_
            , input.arg.value().size(), surr_poli_ );
        len_ = res.pos;
        init_( input.preview, res.width, src_enc
             , get_facet_<strf::char_encoding_c<DestCharT>, SrcCharT>(input.facets) );
    }

    STRF_HD void print_to(strf::basic_outbuff<DestCharT>& ob) const override;

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
    const strf::invalid_seq_notifier inv_seq_notifier_;
    const strf::surrogate_policy  surr_poli_;
    std::uint16_t left_fillcount_ = 0;
    std::uint16_t right_fillcount_ = 0;

    template <typename Category, typename SrcChar, typename FPack>
    static STRF_HD decltype(auto) get_facet_(const FPack& facets)
    {
        using input_tag = strf::string_input_tag<SrcChar>;
        return facets.template get_facet<Category, input_tag>();
    }

    template < typename Preview, typename SrcEncoding, typename DestEncoding>
    STRF_HD void init_
        ( Preview& preview, strf::width_t str_width
        , SrcEncoding src_enc, DestEncoding dest_enc );
};

template <typename SrcCharT, typename DestCharT>
template <typename Preview, typename SrcEncoding, typename DestEncoding>
void STRF_HD aligned_conv_string_printer<SrcCharT, DestCharT>::init_
    ( Preview& preview, strf::width_t str_width
    , SrcEncoding src_enc, DestEncoding dest_enc )
{
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
        strf::transcode_size_f<SrcCharT> transcode_size
                = transcoder.transcode_size_func();
        if (transcode_size != nullptr) {
            s = transcode_size(str_, len_, surr_poli_);
        } else {
            s = strf::decode_encode_size<SrcCharT>
                ( src_enc.to_u32().transcode_func()
                , dest_enc.from_u32().transcode_size_func()
                , str_, len_, inv_seq_notifier_, surr_poli_ );
        }
        if (fillcount > 0) {
            s += dest_enc.encoded_char_size(afmt_.fill) * fillcount;
        }
        preview.add_size(s);
    }
}

template<typename SrcCharT, typename DestCharT>
void STRF_HD aligned_conv_string_printer<SrcCharT, DestCharT>::print_to
    ( strf::basic_outbuff<DestCharT>& ob ) const
{
    if (left_fillcount_ > 0) {
        encode_fill_(ob, left_fillcount_, afmt_.fill);
    }
    if (can_transcode_directly()) {
        transcode_(ob, str_, len_, inv_seq_notifier_, surr_poli_);
    } else {
        strf::decode_encode<SrcCharT, DestCharT>
            ( ob, src_to_u32_, u32_to_dest_, str_
            , len_, inv_seq_notifier_, surr_poli_ );
    }
    if (right_fillcount_ > 0) {
        encode_fill_(ob, right_fillcount_, afmt_.fill);
    }
}

#if defined(STRF_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class string_printer<char8_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class string_printer<char8_t, char>;
STRF_EXPLICIT_TEMPLATE class string_printer<char, char8_t>;
#endif


STRF_EXPLICIT_TEMPLATE class string_printer<char, char>;
STRF_EXPLICIT_TEMPLATE class string_printer<char16_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class string_printer<char32_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class string_printer<wchar_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class string_printer<wchar_t, strf::detail::wchar_equiv>;
STRF_EXPLICIT_TEMPLATE class string_printer<strf::detail::wchar_equiv, wchar_t>;

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class aligned_string_printer<char8_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_string_printer<char8_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_string_printer<char, char8_t>;
#endif

STRF_EXPLICIT_TEMPLATE class aligned_string_printer<char, char>;
STRF_EXPLICIT_TEMPLATE class aligned_string_printer<char16_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_string_printer<char32_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_string_printer<wchar_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_string_printer<wchar_t, strf::detail::wchar_equiv>;
STRF_EXPLICIT_TEMPLATE class aligned_string_printer<strf::detail::wchar_equiv, wchar_t>;

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char8_t, char>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char8_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char8_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char8_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char8_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char, char8_t>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char16_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char32_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<wchar_t, char8_t>;
#endif

STRF_EXPLICIT_TEMPLATE class conv_string_printer<char, char>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char, char16_t>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char, char32_t>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char, wchar_t>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char16_t, char>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char16_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char16_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char16_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char32_t, char>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char32_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char32_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<char32_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<wchar_t, char>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<wchar_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<wchar_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class conv_string_printer<wchar_t, wchar_t>;

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char8_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char8_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char8_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char8_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char8_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char16_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char32_t, char8_t>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<wchar_t, char8_t>;
#endif

STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char, char>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char16_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char16_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char16_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char16_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char32_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char32_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char32_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<char32_t, wchar_t>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<wchar_t, char>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<wchar_t, char16_t>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<wchar_t, char32_t>;
STRF_EXPLICIT_TEMPLATE class aligned_conv_string_printer<wchar_t, wchar_t>;

#endif // defined(STRF_SEPARATE_COMPILATION)

template <typename DestCharT>
class conv_string_printer_variant
{
public:

    template < typename Preview, typename FPack, typename SrcCharT
             , bool HasPrecision, typename CvFormat >
    STRF_HD conv_string_printer_variant
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, HasPrecision, false, CvFormat, Preview, FPack >&
            input )
    {
        auto src_encoding  = strf::detail::get_src_encoding(input);
        using facet_tag = strf::string_input_tag<SrcCharT>;
        using dest_enc_cat = strf::char_encoding_c<DestCharT>;
        auto dest_encoding = strf::get_facet<dest_enc_cat, facet_tag>(input.facets);
        if (src_encoding.id() == dest_encoding.id()) {
            new ((void*)&pool_) strf::detail::string_printer<SrcCharT, DestCharT>(input);
        } else {
            new ((void*)&pool_) strf::detail::conv_string_printer<SrcCharT, DestCharT>(input);
        }
    }

    STRF_HD ~conv_string_printer_variant()
    {
        const strf::printer<DestCharT>& p = *this;
        p.~printer();
    }

#if defined(__GNUC__) && (__GNUC__ == 6)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

    STRF_HD operator const strf::printer<DestCharT>& () const
    {
        return * reinterpret_cast<const strf::printer<DestCharT>*>(&pool_);
    }

#if defined(__GNUC__) && (__GNUC__ == 6)
#  pragma GCC diagnostic pop
#endif

private:

    static constexpr std::size_t pool_size_ =
        sizeof(strf::detail::conv_string_printer<DestCharT, DestCharT>);
    using storage_type_ = typename std::aligned_storage_t
        < pool_size_, alignof(strf::printer<DestCharT>)>;

    storage_type_ pool_;
};

template<typename DestCharT>
class aligned_conv_string_printer_variant
{
public:

    template < typename Preview, typename FPack, typename SrcCharT
             , bool HasPrecision, typename CvFormat >
    STRF_HD aligned_conv_string_printer_variant
        ( const strf::detail::fmt_string_printer_input
            < DestCharT, SrcCharT, HasPrecision, true, CvFormat, Preview, FPack >&
            input )
    {
        auto src_encoding  = strf::detail::get_src_encoding(input);
        using facet_tag = strf::string_input_tag<SrcCharT>;
        using dest_enc_cat = strf::char_encoding_c<DestCharT>;
        auto dest_encoding = strf::get_facet<dest_enc_cat, facet_tag>(input.facets);

        if (src_encoding.id() == dest_encoding.id()) {
            new ((void*)&pool_) strf::detail::aligned_string_printer<SrcCharT, DestCharT> (input);
        } else {
            new ((void*)&pool_)
                strf::detail::aligned_conv_string_printer<SrcCharT, DestCharT>(input);
        }
    }

    STRF_HD ~aligned_conv_string_printer_variant()
    {
        const strf::printer<DestCharT>& p = *this;
        p.~printer();
    }

#if defined(__GNUC__) && (__GNUC__ == 6)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

    STRF_HD operator const strf::printer<DestCharT>& () const
    {
        return * reinterpret_cast<const strf::printer<DestCharT>*>(&pool_);
    }

#if defined(__GNUC__) && (__GNUC__ == 6)
#  pragma GCC diagnostic pop
#endif

private:

    static constexpr std::size_t pool_size_ =
        sizeof(strf::detail::aligned_conv_string_printer<DestCharT, DestCharT>);
    using storage_type_ = typename std::aligned_storage_t
        < pool_size_, alignof(strf::printer<DestCharT>)>;

    storage_type_ pool_;
};

} // namespace detail
} // namespace strf

#endif  /* STRF_DETAIL_INPUT_TYPES_CHAR_PTR */

