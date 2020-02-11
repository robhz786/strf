#ifndef STRF_DETAIL_INPUT_TYPES_STRING
#define STRF_DETAIL_INPUT_TYPES_STRING

#include <limits>
#include <string>
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
struct no_cv_format;

template <typename CharT>
struct cv_format;

template <typename CharT, typename Charset>
struct cv_format_with_charset;

template <typename CharT>
struct sani_format;

template <typename CharT, typename Charset>
struct sani_format_with_charset;

template <typename CharT, typename T>
class no_cv_format_fn
{
public:

    constexpr STRF_HD no_cv_format_fn() noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit no_cv_format_fn
        ( const no_cv_format_fn<CharT, U>& ) noexcept
    {
    }

    constexpr STRF_HD auto convert_charset() const
    {
        using return_type = strf::fmt_replace< T
                                             , strf::no_cv_format<CharT>
                                             , strf::cv_format<CharT> >;
        return return_type{ static_cast<const T&>(*this) };
    }

    template <typename Charset>
    constexpr STRF_HD auto convert_charset(const Charset& cs) const
    {
        using return_type = strf::fmt_replace
            < T
            , strf::no_cv_format<CharT>
            , strf::cv_format_with_charset<CharT, Charset> >;

        return return_type
            { static_cast<const T&>(*this)
            , strf::tag<strf::cv_format_with_charset<CharT, Charset>>{}
            , cs };
    }
    constexpr STRF_HD auto cv() const
    {
        return convert_charset();
    }
    template <typename Charset>
    constexpr STRF_HD auto cv(const Charset& cs) const
    {
        return convert_charset(cs);
    }

    constexpr STRF_HD auto sanitize_charset() const
    {
        using return_type = strf::fmt_replace< T
                                             , strf::no_cv_format<CharT>
                                             , strf::sani_format<CharT> >;
        return return_type{ static_cast<const T&>(*this) };
    }
    template <typename Charset>
    constexpr STRF_HD auto sanitize_charset(const Charset& cs) const
    {
        using return_type = strf::fmt_replace
            < T
            , strf::no_cv_format<CharT>
            , strf::sani_format_with_charset<CharT, Charset> >;

        return return_type
            { static_cast<const T&>(*this)
             , strf::tag<strf::sani_format_with_charset<CharT, Charset>>{}
            , cs };
    }
    constexpr auto sani() const
    {
        return sanitize_charset();
    }
    template <typename Charset>
    constexpr auto sani(const Charset& cs) const
    {
        return sanitize_charset(cs);
    }
};

template <typename CharT, typename T>
struct cv_format_fn
{
    constexpr STRF_HD cv_format_fn() noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit cv_format_fn
        ( const cv_format_fn<CharT, U>& ) noexcept
    {
    }

    template <typename U>
    constexpr STRF_HD explicit cv_format_fn
        ( const strf::no_cv_format_fn<CharT, U>& ) noexcept
    {
    }
};

template <typename CharT, typename Charset, typename T>
class cv_format_with_charset_fn
{
public:

    cv_format_with_charset_fn(const Charset& e)
        : charset_(e)
    {
    }

    cv_format_with_charset_fn
        ( const cv_format_with_charset_fn& other ) noexcept = default;

    template <typename U>
    explicit cv_format_with_charset_fn
        ( const strf::cv_format_with_charset_fn<CharT, Charset, U>& other ) noexcept
        : charset_(other.get_charset())
    {
    }

    template <typename U>
    explicit cv_format_with_charset_fn
        ( const strf::no_cv_format_fn<CharT, U>& other ) noexcept
        : charset_(other.get_charset())
    {
    }

    const Charset& get_charset() const
    {
        return charset_;
    }

private:

    const Charset& charset_;
};

template <typename CharT>
struct no_cv_format
{
    template <typename T>
    using fn = strf::no_cv_format_fn<CharT, T>;
};

template <typename CharT>
struct cv_format
{
    template <typename T>
    using fn = strf::cv_format_fn<CharT, T>;
};

template <typename CharT, typename Charset>
struct cv_format_with_charset
{
    template <typename T>
    using fn = strf::cv_format_with_charset_fn<CharT, Charset, T>;
};

template <typename CharT>
struct sani_format
{
    template <typename T>
    using fn = strf::cv_format_fn<CharT, T>;
};

template <typename CharT, typename Charset>
struct sani_format_with_charset
{
    template <typename T>
    using fn = strf::cv_format_with_charset_fn<CharT, Charset, T>;
};

template <typename CharIn>
using string_with_format = strf::value_with_format
    < strf::detail::simple_string_view<CharIn>
    , strf::alignment_format_q<false>
    , strf::no_cv_format<CharIn> >;

template <typename CharIn>
constexpr STRF_HD auto make_fmt
    ( strf::rank<1>
    , const strf::detail::simple_string_view<CharIn>& str) noexcept
{
    return strf::string_with_format<CharIn>{str};
}

template <typename CharIn, typename Traits, typename Allocator>
STRF_HD auto make_fmt
    ( strf::rank<1>
    , const std::basic_string<CharIn, Traits, Allocator>& str) noexcept
{
    return strf::string_with_format<CharIn>{{str.data(), str.size()}};
}

#if defined(STRF_HAS_STD_STRING_VIEW)

template <typename CharIn, typename Traits>
constexpr STRF_HD auto make_fmt
    ( strf::rank<1>
    , const std::basic_string_view<CharIn, Traits>& str) noexcept
{
    return strf::string_with_format<CharIn>{{str.data(), str.size()}};
}

#endif // defined(STRF_HAS_STD_STRING_VIEW)

#if defined(__cpp_char8_t)

STRF_CONSTEXPR_CHAR_TRAITS STRF_HD
auto make_fmt(strf::rank<1>, const char8_t* str)
{
    auto len = strf::detail::str_length<char8_t>(str);
    return strf::string_with_format<char8_t>{{str, len}};
}

#endif

STRF_CONSTEXPR_CHAR_TRAITS STRF_HD
auto  make_fmt(strf::rank<1>, const char* str)
{
    auto len = strf::detail::str_length<char>(str);
    return strf::string_with_format<char>{{str, len}};
}

STRF_CONSTEXPR_CHAR_TRAITS STRF_HD
auto  make_fmt(strf::rank<1>, const wchar_t* str)
{
    auto len = strf::detail::str_length<wchar_t>(str);
    return strf::string_with_format<wchar_t>{{str, len}};
}

STRF_CONSTEXPR_CHAR_TRAITS STRF_HD
auto  make_fmt(strf::rank<1>, const char16_t* str)
{
    auto len = strf::detail::str_length<char16_t>(str);
    return strf::string_with_format<char16_t>{{str, len}};
}

STRF_CONSTEXPR_CHAR_TRAITS STRF_HD
auto  make_fmt(strf::rank<1>, const char32_t* str)
{
    auto len = strf::detail::str_length<char32_t>(str);
    return strf::string_with_format<char32_t>{{str, len}};
}

namespace detail {

template <std::size_t CharSize>
class string_printer: public strf::printer<CharSize>
{
public:
    using char_type = strf::underlying_char_type<CharSize>;

    template <typename FPack, typename Preview, typename CharT>
    STRF_HD string_printer
        ( const FPack& fp
        , Preview& preview
        , simple_string_view<CharT> str
        , strf::tag<CharT> ) noexcept
        : str_(reinterpret_cast<const char_type*>(str.begin()))
        , len_(str.size())
    {
        (void)fp;
        STRF_IF_CONSTEXPR(Preview::width_required) {
            decltype(auto) wcalc = get_facet_<strf::width_calculator_c, CharT>(fp);
            auto w = wcalc.width( get_facet_<strf::charset_c<CharT>, CharT>(fp)
                                , preview.remaining_width(), str_, len_
                                , get_facet_<strf::invalid_seq_policy_c, CharT>(fp)
                                , get_facet_<strf::surrogate_policy_c, CharT>(fp) );
            preview.subtract_width(w);
        }
        preview.add_size(len_);
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    const char_type* str_;
    const std::size_t len_;

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
class fmt_string_printer: public strf::printer<CharSize>
{
public:
    using char_type = strf::underlying_char_type<CharSize>;

    template <typename FPack, typename Preview, typename CharT>
    STRF_HD fmt_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<CharT> str
        , strf::alignment_format_data text_alignment
        , strf::tag<CharT> )
        : str_(reinterpret_cast<const char_type*>(str.begin()))
        , len_(str.size())
        , afmt_(text_alignment)
        , inv_seq_poli_(get_facet_<strf::invalid_seq_policy_c, CharT>(fp))
        , surr_poli_(get_facet_<strf::surrogate_policy_c, CharT>(fp))
    {
         init_( preview
              , get_facet_<strf::width_calculator_c, CharT>(fp)
              , get_facet_<strf::charset_c<CharT>, CharT>(fp) );
    }

    STRF_HD ~fmt_string_printer();

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    const char_type* str_;
    std::size_t len_;
    strf::encode_fill_f<CharSize> encode_fill_;
    strf::alignment_format_data afmt_;
    std::int16_t left_fillcount_;
    std::int16_t right_fillcount_;
    const strf::invalid_seq_policy inv_seq_poli_;
    const strf::surrogate_policy surr_poli_;

    template <typename Category, typename CharT, typename FPack>
    static STRF_HD decltype(auto) get_facet_(const FPack& fp)
    {
        using input_tag = strf::string_input_tag<CharT>;
        return fp.template get_facet<Category, input_tag>();
    }

    template <typename Preview, typename WCalc, typename Charset>
    STRF_HD void init_(Preview&, const WCalc&, const Charset&);
};

template<std::size_t CharSize>
STRF_HD fmt_string_printer<CharSize>::~fmt_string_printer()
{
}

template<std::size_t CharSize>
template <typename Preview, typename WCalc, typename Charset>
inline STRF_HD void fmt_string_printer<CharSize>::init_
    ( Preview& preview, const WCalc& wcalc, const Charset& cs )
{
    encode_fill_ = cs.encode_fill;
    std::uint16_t fillcount = 0;
    strf::width_t fmt_width = afmt_.width;
    strf::width_t limit =
        ( Preview::width_required && preview.remaining_width() > fmt_width
        ? preview.remaining_width()
        : fmt_width );
    auto strw = wcalc.width(cs, limit, str_, len_ , inv_seq_poli_, surr_poli_);
    if (fmt_width > strw) {
        fillcount = (fmt_width - strw).round();
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
    } else {
        right_fillcount_ = 0;
        left_fillcount_ = 0;
        preview.subtract_width(strw);
    }

    STRF_IF_CONSTEXPR (Preview::size_required) {
        preview.add_size(len_);
        if (fillcount > 0) {
             preview.add_size(fillcount * cs.encoded_char_size(afmt_.fill));
        }
    }
}

template<std::size_t CharSize>
void STRF_HD fmt_string_printer<CharSize>::print_to
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (left_fillcount_ > 0) {
        encode_fill_( ob, left_fillcount_, afmt_.fill, inv_seq_poli_, surr_poli_ );
    }
    strf::write(ob, str_, len_);
    if (right_fillcount_ > 0) {
        encode_fill_( ob, right_fillcount_, afmt_.fill, inv_seq_poli_, surr_poli_ );
    }
}

#if defined(STRF_SEPARATE_COMPILATION)

STRF_EXPLICIT_TEMPLATE class string_printer<1>;
STRF_EXPLICIT_TEMPLATE class string_printer<2>;
STRF_EXPLICIT_TEMPLATE class string_printer<4>;

STRF_EXPLICIT_TEMPLATE class fmt_string_printer<1>;
STRF_EXPLICIT_TEMPLATE class fmt_string_printer<2>;
STRF_EXPLICIT_TEMPLATE class fmt_string_printer<4>;

#endif // defined(STRF_SEPARATE_COMPILATION)

} // namespace detail

template <typename CharT, typename FPack, typename Preview>
inline STRF_HD strf::detail::string_printer<sizeof(CharT)>
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, const CharT* str)
{
    return {fp, preview, str, strf::tag<CharT>()};
}

#if defined(__cpp_char8_t)

template <typename CharOut, typename FPack, typename Preview>
inline STRF_HD strf::detail::string_printer<sizeof(CharOut)>
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, const char8_t* str)
{
    static_assert( std::is_same<char8_t, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    strf::detail::simple_string_view<CharOut> strv = str;
    return {fp, preview, strv, strf::tag<CharOut>()};
}

#endif

template <typename CharOut, typename FPack, typename Preview>
inline STRF_HD strf::detail::string_printer<sizeof(CharOut)>
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, const char* str)
{
    static_assert( std::is_same<char, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    strf::detail::simple_string_view<CharOut> strv = str;
    return {fp, preview, strv, strf::tag<CharOut>()};
}

template <typename CharOut, typename FPack, typename Preview>
inline STRF_HD strf::detail::string_printer<sizeof(CharOut)>
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, const char16_t* str)
{
    static_assert( std::is_same<char16_t, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    strf::detail::simple_string_view<CharOut> strv = str;
    return {fp, preview, strv, strf::tag<CharOut>()};
}

template <typename CharOut, typename FPack, typename Preview>
inline STRF_HD strf::detail::string_printer<sizeof(CharOut)>
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, const char32_t* str)
{
    static_assert( std::is_same<char32_t, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    strf::detail::simple_string_view<CharOut> strv = str;
    return {fp, preview, strv, strf::tag<CharOut>()};
}

template <typename CharOut, typename FPack, typename Preview>
inline STRF_HD strf::detail::string_printer<sizeof(CharOut)>
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, const wchar_t* str)
{
    static_assert( std::is_same<wchar_t, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    strf::detail::simple_string_view<CharOut> strv = str;
    return {fp, preview, strv, strf::tag<CharOut>()};
}

template
    < typename CharOut
    , typename FPack
    , typename Preview
    , typename CharIn
    , typename Traits
    , typename Allocator >
inline STRF_HD strf::detail::string_printer<sizeof(CharOut)>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , const std::basic_string<CharIn, Traits, Allocator>& str )
{
    static_assert( std::is_same<CharIn, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, preview, {str.data(), str.size()}, strf::tag<CharOut>()};
}

template
    < typename CharOut
    , typename FPack
    , typename Preview
    , typename CharIn >
inline STRF_HD strf::detail::string_printer<sizeof(CharOut)>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , const strf::detail::simple_string_view<CharIn>& str )
{
    static_assert( std::is_same<CharIn, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, preview, str, strf::tag<CharOut>()};
}

#if defined(STRF_HAS_STD_STRING_VIEW)

template
    < typename CharOut
    , typename FPack
    , typename Preview
    , typename CharIn
    , typename Traits >
inline STRF_HD strf::detail::string_printer<sizeof(CharOut)>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , const std::basic_string_view<CharIn, Traits>& str )
{
    static_assert( std::is_same<CharIn, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, preview, {str.data(), str.size()}, strf::tag<CharOut>()};
}

#endif //defined(STRF_HAS_STD_STRING_VIEW)

template <typename CharOut, typename FPack, typename Preview, typename CharIn>
inline STRF_HD strf::detail::fmt_string_printer<sizeof(CharOut)>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , const strf::value_with_format
                < strf::detail::simple_string_view<CharIn>
                , strf::alignment_format_q<true>
                , strf::no_cv_format<CharIn> > input )
{
    static_assert( std::is_same<CharIn, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return { fp, preview, input.value(), input.get_alignment_format_data()
           , strf::tag<CharOut>()};
}

} // namespace strf

#endif  /* STRF_DETAIL_INPUT_TYPES_CHAR_PTR */

