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

template <typename CharT>
struct cv_format_with_encoding;

template <typename CharT>
struct sani_format;

template <typename CharT>
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

    constexpr STRF_HD auto convert_charset(strf::encoding<CharT> enc) const
    {
        using return_type = strf::fmt_replace
            < T
            , strf::no_cv_format<CharT>
            , strf::cv_format_with_encoding<CharT> >;

        return return_type
            { static_cast<const T&>(*this)
            , strf::tag<strf::cv_format_with_encoding<CharT>>{}
            , enc };
    }
    constexpr STRF_HD auto cv() const
    {
        return convert_charset();
    }
    constexpr STRF_HD auto cv(strf::encoding<CharT> enc) const
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

    constexpr STRF_HD auto sanitize_charset(strf::encoding<CharT> enc) const
    {
        using return_type = strf::fmt_replace
            < T
            , strf::no_cv_format<CharT>
            , strf::sani_format_with_encoding<CharT> >;

        return return_type
            { static_cast<const T&>(*this)
            , strf::tag<strf::sani_format_with_encoding<CharT>>{}
            , enc };
    }
    constexpr auto sani() const
    {
        return sanitize_charset();
    }
    constexpr auto sani(strf::encoding<CharT> enc) const
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

template <typename CharT, typename T>
class cv_format_with_encoding_fn
{
public:

    cv_format_with_encoding_fn(strf::encoding<CharT> e)
        : _encoding(e)
    {
    }

    cv_format_with_encoding_fn
        ( const cv_format_with_encoding_fn& other ) noexcept = default;

    template <typename U>
    explicit cv_format_with_encoding_fn
        ( const strf::cv_format_with_encoding_fn<CharT, U>& other ) noexcept
        : _encoding(other.get_encoding())
    {
    }

    template <typename U>
    explicit cv_format_with_encoding_fn
        ( const strf::no_cv_format_fn<CharT, U>& other ) noexcept
        : _encoding(other.get_encoding())
    {
    }

    strf::encoding<CharT> get_encoding() const
    {
        return _encoding;
    }

private:

    strf::encoding<CharT> _encoding;
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

template <typename CharT>
struct cv_format_with_encoding
{
    template <typename T>
    using fn = strf::cv_format_with_encoding_fn<CharT, T>;
};

template <typename CharT>
struct sani_format
{
    template <typename T>
    using fn = strf::cv_format_fn<CharT, T>;
};

template <typename CharT>
struct sani_format_with_encoding
{
    template <typename T>
    using fn = strf::cv_format_with_encoding_fn<CharT, T>;
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

    template <typename FPack, bool RequireSize, typename CharT>
    STRF_HD string_printer
        ( const FPack&
        , strf::print_preview<RequireSize, false>& preview
        , simple_string_view<CharT> str
        , strf::tag<CharT> ) noexcept
        : _str(reinterpret_cast<const char_type*>(str.begin()))
        , _len(str.size())
    {
        preview.add_size(_len);
    }

    template <typename FPack, bool RequireSize, typename CharT>
    STRF_HD string_printer
        ( const FPack& fp
        , strf::print_preview<RequireSize, true>& preview
        , simple_string_view<CharT> str
        , strf::tag<CharT> ) noexcept
        : _str(reinterpret_cast<const char_type*>(str.begin()))
        , _len(str.size())
    {
        preview.add_size(_len);
        _calc_width( preview
                   , _get_facet<strf::width_calculator_c<CharSize>, CharT>(fp)
                   , _get_facet<strf::encoding_c<CharT>, CharT>(fp).as_underlying()
                   , _get_facet<strf::encoding_error_c, CharT>(fp)
                   , _get_facet<strf::surrogate_policy_c, CharT>(fp) );
    }

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    constexpr STRF_HD void _calc_width
        ( strf::width_preview<true>& wpreview
        , const strf::fast_width<CharSize>&
        , const strf::underlying_encoding<CharSize>&
        , strf::encoding_error
        , strf::surrogate_policy ) noexcept
    {
        auto remaining_width = wpreview.remaining_width().floor();
        if (static_cast<std::ptrdiff_t>(_len) <= remaining_width) {
            wpreview.subtract_width(static_cast<std::int16_t>(_len));
        } else {
            wpreview.clear_remaining_width();
        }
    }

    constexpr STRF_HD void _calc_width
        ( strf::width_preview<true>& wpreview
        , const strf::width_as_u32len<CharSize>&
        , const strf::underlying_encoding<CharSize>& enc
        , strf::encoding_error
        , strf::surrogate_policy )
    {
        auto limit = wpreview.remaining_width();
        if (limit > 0) {
            auto count = enc.codepoints_count(_str, _str + _len, limit.ceil());
            if (static_cast<std::ptrdiff_t>(count) < limit.ceil()) {
                wpreview.subtract_width(static_cast<std::int16_t>(count));
            } else {
                wpreview.clear_remaining_width();
            }
        }
    }

    constexpr STRF_HD void _calc_width
        ( strf::width_preview<true>& wpreview
        , const strf::width_calculator<CharSize>& wcalc
        , const strf::underlying_encoding<CharSize>& enc
        , strf::encoding_error  enc_err
        , strf::surrogate_policy  allow_surr )
    {
        auto limit = wpreview.remaining_width();
        if (limit > 0) {
            auto w = wcalc.width(limit, _str, _len, enc, enc_err, allow_surr);
            if (w < limit) {
                wpreview.subtract_width(w);
            } else {
                wpreview.clear_remaining_width();
            }
            // wcalc.subtract_width( wpreview, _str, _len
            //                      , encoding, enc_err, allow_surr );
        }
    }

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
        : _str(reinterpret_cast<const char_type*>(str.begin()), str.size())
        , _afmt(text_alignment)
        , _encoding(_get_facet<strf::encoding_c<CharT>, CharT>(fp).as_underlying())
        , _enc_err(_get_facet<strf::encoding_error_c, CharT>(fp))
        , _allow_surr(_get_facet<strf::surrogate_policy_c, CharT>(fp))
    {
        _init(preview, _get_facet<strf::width_calculator_c<CharSize>, CharT>(fp));
        _calc_size(preview);
    }

    STRF_HD ~fmt_string_printer();

    STRF_HD void print_to(strf::underlying_outbuf<CharSize>& ob) const override;

private:

    strf::detail::simple_string_view<char_type> _str;
    strf::alignment_format_data _afmt;
    const strf::underlying_encoding<CharSize>& _encoding;
    std::int16_t _fillcount = 0;
    const strf::encoding_error _enc_err;
    const strf::surrogate_policy _allow_surr;

    template <typename Category, typename CharT, typename FPack>
    static STRF_HD decltype(auto) _get_facet(const FPack& fp)
    {
        using input_tag = strf::string_input_tag<CharT>;
        return fp.template get_facet<Category, input_tag>();
    }

    template <bool RequiringWidth>
    STRF_HD void _init( strf::width_preview<RequiringWidth>& preview
                      , const strf::fast_width<CharSize>&);

    template <bool RequiringWidth>
    STRF_HD void _init( strf::width_preview<RequiringWidth>& preview
                      , const strf::width_as_u32len<CharSize>&);

    template <bool RequiringWidth>
    STRF_HD void _init( strf::width_preview<RequiringWidth>& preview
                      , const strf::width_calculator<CharSize>&);

    constexpr STRF_HD void _calc_size(strf::size_preview<false>&) const
    {
    }

    STRF_HD void _calc_size(strf::size_preview<true>& preview) const
    {
        preview.add_size(_str.length());
        if (_fillcount > 0) {
             preview.add_size( _fillcount
                             * _encoding.char_size(_afmt.fill) );
        }
    }

    STRF_HD void _write_str(strf::underlying_outbuf<CharSize>& ob) const;
    STRF_HD void _write_fill( strf::underlying_outbuf<CharSize>& ob
                    , unsigned count ) const;
};

template<std::size_t CharSize>
STRF_HD fmt_string_printer<CharSize>::~fmt_string_printer()
{
}

template<std::size_t CharSize>
template <bool RequiringWidth>
inline STRF_HD void fmt_string_printer<CharSize>::_init
    ( strf::width_preview<RequiringWidth>& preview
    , const strf::fast_width<CharSize>&)
{
    auto len = _str.length();
    if (_afmt.width > static_cast<std::ptrdiff_t>(len)) {
        _fillcount = _afmt.width - static_cast<std::int16_t>(len);
        preview.subtract_width(_afmt.width);
    } else {
        preview.checked_subtract_width(len);
    }
}

template<std::size_t CharSize>
template <bool RequiringWidth>
inline STRF_HD void fmt_string_printer<CharSize>::_init
    ( strf::width_preview<RequiringWidth>& preview
    , const strf::width_as_u32len<CharSize>&)
{
    auto cp_count = _encoding.codepoints_count( _str.begin()
                                              , _str.end()
                                              , _afmt.width );
    if (_afmt.width > static_cast<std::ptrdiff_t>(cp_count)) {
        _fillcount = _afmt.width - static_cast<std::int16_t>(cp_count);
        preview.subtract_width(_afmt.width);
    } else {
        preview.checked_subtract_width(cp_count);
    }

}

template <std::size_t CharSize>
template <bool RequiringWidth>
inline STRF_HD void fmt_string_printer<CharSize>::_init
    ( strf::width_preview<RequiringWidth>& preview
    , const strf::width_calculator<CharSize>& wc)
{
    strf::width_t wmax = _afmt.width;
    strf::width_t wdiff = 0;
    if (preview.remaining_width() > _afmt.width) {
        wmax = preview.remaining_width();
        wdiff = preview.remaining_width() - _afmt.width;
    }
    auto str_width = wc.width( wmax
                             , _str.begin(), _str.length()
                             , _encoding, _enc_err, _allow_surr );

    strf::width_t fmt_width{_afmt.width};
    if (fmt_width > str_width) {
        auto wfill = (fmt_width - str_width);
        _fillcount = wfill.round();
        preview.subtract_width(wfill + _fillcount);
    } else {
        preview.subtract_width(str_width);
    }
}

template<std::size_t CharSize>
void STRF_HD fmt_string_printer<CharSize>::print_to
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    if (_fillcount > 0) {
        switch (_afmt.alignment) {
            case strf::text_alignment::left: {
                _write_str(ob);
                _write_fill(ob, _fillcount);
                break;
            }
            case strf::text_alignment::center: {
                auto halfcount = _fillcount >> 1;
                _write_fill(ob, halfcount);
                _write_str(ob);
                _write_fill(ob, _fillcount - halfcount);
                break;
            }
            default: {
                _write_fill(ob, _fillcount);
                _write_str(ob);
            }
        }
    } else {
        _write_str(ob);
    }
}

template <std::size_t CharSize>
void STRF_HD fmt_string_printer<CharSize>::_write_str
    ( strf::underlying_outbuf<CharSize>& ob ) const
{
    strf::write(ob, _str.begin(), _str.length());
}

template <std::size_t CharSize>
void STRF_HD fmt_string_printer<CharSize>::_write_fill
    ( strf::underlying_outbuf<CharSize>& ob
    , unsigned count ) const
{
    _encoding.encode_fill( ob, count, _afmt.fill, _enc_err, _allow_surr );
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
    return {fp, preview, str, strf::tag<CharOut>()};
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

