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
        : _begin(begin)
        , _len(end - begin)
    {
    }
    constexpr STRF_HD simple_string_view(const CharIn* str, std::size_t len) noexcept
        : _begin(str)
        , _len(len)
    {
    }

    STRF_CONSTEXPR_CHAR_TRAITS
    STRF_HD simple_string_view(const CharIn* str) noexcept
        : _begin(str)
        , _len(strf::detail::str_length<CharIn>(str))
    {
    }
    constexpr STRF_HD const CharIn* begin() const
    {
        return _begin;
    }
    constexpr STRF_HD const CharIn* end() const
    {
        return _begin + _len;
    }
    constexpr STRF_HD std::size_t size() const
    {
        return _len;
    }
    constexpr STRF_HD std::size_t length() const
    {
        return _len;
    }

private:

    const CharIn* _begin;
    const std::size_t _len;
};

} // namespace detail

template <typename CharT>
struct no_cv_format;

template <typename CharT>
struct cv_format;

template <typename CharT, typename Encoding>
struct cv_format_with_encoding;

template <typename CharT>
struct sani_format;

template <typename CharT, typename Encoding>
struct sani_format_with_encoding;

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

    template <typename Encoding>
    constexpr STRF_HD auto convert_charset(const Encoding& enc) const
    {
        using return_type = strf::fmt_replace
            < T
            , strf::no_cv_format<CharT>
            , strf::cv_format_with_encoding<CharT, Encoding> >;

        return return_type
            { static_cast<const T&>(*this)
            , strf::tag<strf::cv_format_with_encoding<CharT, Encoding>>{}
            , enc };
    }
    constexpr STRF_HD auto cv() const
    {
        return convert_charset();
    }
    template <typename Encoding>
    constexpr STRF_HD auto cv(const Encoding& enc) const
    {
        return convert_charset(enc);
    }

    constexpr STRF_HD auto sanitize_charset() const
    {
        using return_type = strf::fmt_replace< T
                                             , strf::no_cv_format<CharT>
                                             , strf::sani_format<CharT> >;
        return return_type{ static_cast<const T&>(*this) };
    }
    template <typename Encoding>
    constexpr STRF_HD auto sanitize_charset(const Encoding& enc) const
    {
        using return_type = strf::fmt_replace
            < T
            , strf::no_cv_format<CharT>
            , strf::sani_format_with_encoding<CharT, Encoding> >;

        return return_type
            { static_cast<const T&>(*this)
             , strf::tag<strf::sani_format_with_encoding<CharT, Encoding>>{}
            , enc };
    }
    constexpr auto sani() const
    {
        return sanitize_charset();
    }
    template <typename Encoding>
    constexpr auto sani(const Encoding& enc) const
    {
        return sanitize_charset(enc);
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

template <typename CharT, typename Encoding, typename T>
class cv_format_with_encoding_fn
{
public:

    cv_format_with_encoding_fn(const Encoding& e)
        : _encoding(e)
    {
    }

    cv_format_with_encoding_fn
        ( const cv_format_with_encoding_fn& other ) noexcept = default;

    template <typename U>
    explicit cv_format_with_encoding_fn
        ( const strf::cv_format_with_encoding_fn<CharT, Encoding, U>& other ) noexcept
        : _encoding(other.get_encoding())
    {
    }

    template <typename U>
    explicit cv_format_with_encoding_fn
        ( const strf::no_cv_format_fn<CharT, U>& other ) noexcept
        : _encoding(other.get_encoding())
    {
    }

    const Encoding& get_encoding() const
    {
        return _encoding;
    }

private:

    const Encoding& _encoding;
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

template <typename CharT, typename Encoding>
struct cv_format_with_encoding
{
    template <typename T>
    using fn = strf::cv_format_with_encoding_fn<CharT, Encoding, T>;
};

template <typename CharT>
struct sani_format
{
    template <typename T>
    using fn = strf::cv_format_fn<CharT, T>;
};

template <typename CharT, typename Encoding>
struct sani_format_with_encoding
{
    template <typename T>
    using fn = strf::cv_format_with_encoding_fn<CharT, Encoding, T>;
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
    using char_type = strf::underlying_outbuf_char_type<CharSize>;

    template <typename FPack, typename Preview, typename CharT>
    STRF_HD string_printer
        ( const FPack& fp
        , Preview& preview
        , simple_string_view<CharT> str
        , strf::tag<CharT> ) noexcept
        : _str(reinterpret_cast<const char_type*>(str.begin()))
        , _len(str.size())
    {
        (void)fp;
        STRF_IF_CONSTEXPR(Preview::width_required) {
            decltype(auto) wcalc = _get_facet<strf::width_calculator_c, CharT>(fp);
            auto w = wcalc.width( _get_facet<strf::encoding_c<CharT>, CharT>(fp)
                                , preview.remaining_width(), _str, _len
                                , _get_facet<strf::encoding_error_c, CharT>(fp)
                                , _get_facet<strf::surrogate_policy_c, CharT>(fp) );
            preview.subtract_width(w);
        }
        preview.add_size(_len);
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    const char_type* _str;
    const std::size_t _len;

    template <typename Category, typename CharT, typename FPack>
    static STRF_HD decltype(auto) _get_facet(const FPack& fp)
    {
        using input_tag = strf::string_input_tag<CharT>;
        return fp.template get_facet<Category, input_tag>();
    }
};

template<std::size_t CharSize>
STRF_HD void string_printer<CharSize>::print_to(strf::underlying_outbuf<CharSize>& ob) const
{
    strf::write(ob, _str, _len);
}

template <std::size_t CharSize>
class fmt_string_printer: public strf::printer<CharSize>
{
public:
    using char_type = strf::underlying_outbuf_char_type<CharSize>;

    template <typename FPack, typename Preview, typename CharT>
    STRF_HD fmt_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<CharT> str
        , strf::alignment_format_data text_alignment
        , strf::tag<CharT> )
        : _str(reinterpret_cast<const char_type*>(str.begin()))
        , _len(str.size())
        , _afmt(text_alignment)
        , _enc_err(_get_facet<strf::encoding_error_c, CharT>(fp))
        , _allow_surr(_get_facet<strf::surrogate_policy_c, CharT>(fp))
    {
         _init( preview
              , _get_facet<strf::width_calculator_c, CharT>(fp)
              , _get_facet<strf::encoding_c<CharT>, CharT>(fp) );
    }

    STRF_HD ~fmt_string_printer();

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    const char_type* _str;
    std::size_t _len;
    strf::encode_fill_func<CharSize> _encode_fill;
    strf::alignment_format_data _afmt;
    std::int16_t _left_fillcount;
    std::int16_t _right_fillcount;
    const strf::encoding_error _enc_err;
    const strf::surrogate_policy _allow_surr;

    template <typename Category, typename CharT, typename FPack>
    static STRF_HD decltype(auto) _get_facet(const FPack& fp)
    {
        using input_tag = strf::string_input_tag<CharT>;
        return fp.template get_facet<Category, input_tag>();
    }

    template <typename Preview, typename WCalc, typename Encoding>
    STRF_HD void _init(Preview&, const WCalc&, const Encoding&);
};

template<std::size_t CharSize>
STRF_HD fmt_string_printer<CharSize>::~fmt_string_printer()
{
}

template<std::size_t CharSize>
template <typename Preview, typename WCalc, typename Encoding>
inline STRF_HD void fmt_string_printer<CharSize>::_init
    ( Preview& preview, const WCalc& wcalc, const Encoding& enc )
{
    _encode_fill = enc.encode_fill;
    std::uint16_t fillcount = 0;
    strf::width_t fmt_width = _afmt.width;
    strf::width_t limit =
        ( Preview::width_required && preview.remaining_width() > fmt_width
        ? preview.remaining_width()
        : fmt_width );
    auto strw = wcalc.width(enc, limit, _str, _len , _enc_err, _allow_surr);
    if (fmt_width > strw) {
        fillcount = (fmt_width - strw).round();
        switch(_afmt.alignment) {
            case strf::text_alignment::left:
                _left_fillcount = 0;
                _right_fillcount = fillcount;
                break;
            case strf::text_alignment::center: {
                std::uint16_t halfcount = fillcount >> 1;
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
        preview.add_size(_len);
        if (fillcount > 0) {
             preview.add_size(fillcount * enc.encoded_char_size(_afmt.fill));
        }
    }
}

template<std::size_t CharSize>
void STRF_HD fmt_string_printer<CharSize>::print_to
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (_left_fillcount > 0) {
        _encode_fill( ob, _left_fillcount, _afmt.fill, _enc_err, _allow_surr );
    }
    strf::write(ob, _str, _len);
    if (_right_fillcount > 0) {
        _encode_fill( ob, _right_fillcount, _afmt.fill, _enc_err, _allow_surr );
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

