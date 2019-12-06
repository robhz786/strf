#ifndef STRF_DETAIL_INPUT_TYPES_STRING
#define STRF_DETAIL_INPUT_TYPES_STRING

#include <algorithm>
#include <limits>
#include <strf/detail/facets/width_calculator.hpp>
#include <strf/detail/format_functions.hpp>
#include <strf/facets_pack.hpp>

STRF_NAMESPACE_BEGIN

namespace detail {

template <typename CharIn>
class simple_string_view
{
public:

    constexpr simple_string_view(const CharIn* str, std::size_t len) noexcept
        : _begin(str)
        , _len(len)
    {
    }

    constexpr simple_string_view(const simple_string_view&) noexcept = default;

    STRF_CONSTEXPR_CHAR_TRAITS
    simple_string_view(const CharIn* str) noexcept
        : _begin(str)
        , _len(std::char_traits<CharIn>::length(str))
    {
    }
    constexpr const CharIn* begin() const
    {
        return _begin;
    }
    constexpr const CharIn* end() const
    {
        return _begin + _len;
    }
    constexpr std::size_t size() const
    {
        return _len;
    }
    constexpr std::size_t length() const
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

    constexpr no_cv_format_fn() noexcept = default;
    constexpr no_cv_format_fn(const no_cv_format_fn& other) noexcept = default;

    template <typename U>
    constexpr explicit no_cv_format_fn
        ( const no_cv_format_fn<CharT, U>& ) noexcept
    {
    }

    constexpr auto cv() const
    {
        using return_type = strf::fmt_replace< T
                                             , strf::no_cv_format<CharT>
                                             , strf::cv_format<CharT> >;
        return return_type{ static_cast<const T&>(*this) };
    }

    constexpr auto cv(strf::encoding<CharT> enc) const
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

    constexpr auto sani() const
    {
        using return_type = strf::fmt_replace< T
                                             , strf::no_cv_format<CharT>
                                             , strf::sani_format<CharT> >;
        return return_type{ static_cast<const T&>(*this) };
    }

    constexpr auto sani(strf::encoding<CharT> enc) const
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
};

template <typename CharT, typename T>
struct cv_format_fn
{
    constexpr cv_format_fn() noexcept = default;
    constexpr cv_format_fn(const cv_format_fn& other) noexcept = default;

    template <typename U>
    constexpr explicit cv_format_fn
        ( const cv_format_fn<CharT, U>& ) noexcept
    {
    }

    template <typename U>
    constexpr explicit cv_format_fn
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

template <typename CharIn, typename Traits, typename Allocator>
auto make_fmt( strf::tag<>
             , const std::basic_string<CharIn, Traits, Allocator>& str) noexcept
{
    return strf::string_with_format<CharIn>{{str.data(), str.size()}};
}

#if defined(STRF_HAS_STD_STRING_VIEW)

template <typename CharIn, typename Traits>
constexpr auto
make_fmt( strf::tag<>
        , const std::basic_string_view<CharIn, Traits>& str) noexcept
{
    return strf::string_with_format<CharIn>{{str.data(), str.size()}};
}

#endif // defined(STRF_HAS_STD_STRING_VIEW)

#if defined(__cpp_char8_t)

STRF_CONSTEXPR_CHAR_TRAITS
auto make_fmt(strf::tag<>, const char8_t* str)
{
    auto len = std::char_traits<char8_t>::length(str);
    return strf::string_with_format<char8_t>{{str, len}};
}

#endif

STRF_CONSTEXPR_CHAR_TRAITS
auto make_fmt(strf::tag<>, const char* str)
{
    auto len = std::char_traits<char>::length(str);
    return strf::string_with_format<char>{{str, len}};
}

STRF_CONSTEXPR_CHAR_TRAITS
auto make_fmt(strf::tag<>, const wchar_t* str)
{
    auto len = std::char_traits<wchar_t>::length(str);
    return strf::string_with_format<wchar_t>{{str, len}};
}

STRF_CONSTEXPR_CHAR_TRAITS
auto make_fmt(strf::tag<>, const char16_t* str)
{
    auto len = std::char_traits<char16_t>::length(str);
    return strf::string_with_format<char16_t>{{str, len}};
}

STRF_CONSTEXPR_CHAR_TRAITS
auto make_fmt(strf::tag<>, const char32_t* str)
{
    auto len = std::char_traits<char32_t>::length(str);
    return strf::string_with_format<char32_t>{{str, len}};
}

namespace detail {

template <typename CharT>
class string_printer: public strf::printer<CharT>
{
public:

    template <typename FPack, bool RequireSize>
    string_printer
        ( const FPack&
        , strf::print_preview<RequireSize, false>& preview
        , simple_string_view<CharT> str) noexcept
        : _str(str.begin())
        , _len(str.size())
    {
        preview.add_size(_len);
    }

    template <typename FPack, bool RequireSize>
    string_printer
        ( const FPack& fp
        , strf::print_preview<RequireSize, true>& preview
        , simple_string_view<CharT> str ) noexcept
        : _str(str.begin())
        , _len(str.size())
    {
        preview.add_size(_len);
        _calc_width( preview
                   , _get_facet<strf::width_calculator_c<CharT>>(fp)
                   , _get_facet<strf::encoding_c<CharT>>(fp)
                   , _get_facet<strf::encoding_error_c>(fp)
                   , _get_facet<strf::surrogate_policy_c>(fp) );
    }

    void print_to(strf::basic_outbuf<CharT>& ob) const override;

private:

    constexpr void _calc_width
        ( strf::width_preview<true>& wpreview
        , const strf::fast_width<CharT>&
        , strf::encoding<CharT>
        , strf::encoding_error
        , strf::surrogate_policy ) noexcept
    {
        auto remaining_width = wpreview.remaining_width().floor();
        if (static_cast<std::ptrdiff_t>(_len) <= remaining_width)
        {
            wpreview.subtract_width(static_cast<std::int16_t>(_len));
        }
        else
        {
            wpreview.clear_remaining_width();
        }
    }

    constexpr void _calc_width
        ( strf::width_preview<true>& wpreview
        , const strf::width_as_u32len<CharT>&
        , strf::encoding<CharT> enc
        , strf::encoding_error
        , strf::surrogate_policy )
    {
        auto limit = wpreview.remaining_width();
        if (limit > 0)
        {
            auto count = enc.codepoints_count(_str, _str + _len, limit.ceil());
            if (static_cast<std::ptrdiff_t>(count) < limit.ceil())
            {
                wpreview.subtract_width(static_cast<std::int16_t>(count));
            }
            else
            {
                wpreview.clear_remaining_width();
            }
        }
    }

    constexpr void _calc_width
        ( strf::width_preview<true>& wpreview
        , const strf::width_calculator<CharT>& wcalc
        , strf::encoding<CharT> enc
        , strf::encoding_error  enc_err
        , strf::surrogate_policy  allow_surr )
    {
        auto limit = wpreview.remaining_width();
        if (limit > 0)
        {
            auto w = wcalc.width(limit, _str, _len, enc, enc_err, allow_surr);
            if (w < limit)
            {
                wpreview.subtract_width(w);
            }
            else
            {
                wpreview.clear_remaining_width();
            }
            // wcalc.subtract_width( wpreview, _str, _len
            //                      , encoding, enc_err, allow_surr );
        }
    }

    const CharT* _str;
    const std::size_t _len;

    template <typename Category, typename FPack>
    static decltype(auto) _get_facet(const FPack& fp)
    {
        using input_tag = strf::string_input_tag<CharT>;
        return fp.template get_facet<Category, input_tag>();
    }
};

template<typename CharT>
void string_printer<CharT>::print_to(strf::basic_outbuf<CharT>& ob) const
{
    strf::write(ob, _str, _len);
}

template <typename CharT>
class fmt_string_printer: public strf::printer<CharT>
{
public:

    template <typename FPack, typename Preview>
    fmt_string_printer
        ( const FPack& fp
        , Preview& preview
        , strf::detail::simple_string_view<CharT> str
        , strf::alignment_format_data text_alignment )
        : _str(str)
        , _afmt(text_alignment)
        , _encoding(_get_facet<strf::encoding_c<CharT>>(fp))
        , _enc_err(_get_facet<strf::encoding_error_c>(fp))
        , _allow_surr(_get_facet<strf::surrogate_policy_c>(fp))
    {
        _init(preview, _get_facet<strf::width_calculator_c<CharT>>(fp));
        _calc_size(preview);
    }

    ~fmt_string_printer();

    void print_to(strf::basic_outbuf<CharT>& ob) const override;

private:

    strf::detail::simple_string_view<CharT> _str;
    strf::alignment_format_data _afmt;
    const strf::encoding<CharT> _encoding;
    std::int16_t _fillcount = 0;
    const strf::encoding_error _enc_err;
    const strf::surrogate_policy _allow_surr;

    template <typename Category, typename FPack>
    static decltype(auto) _get_facet(const FPack& fp)
    {
        using input_tag = strf::string_input_tag<CharT>;
        return fp.template get_facet<Category, input_tag>();
    }

    template <bool RequiringWidth>
    void _init( strf::width_preview<RequiringWidth>& preview
              , const strf::fast_width<CharT>&);

    template <bool RequiringWidth>
    void _init( strf::width_preview<RequiringWidth>& preview
              , const strf::width_as_u32len<CharT>&);

    template <bool RequiringWidth>
    void _init( strf::width_preview<RequiringWidth>& preview
              , const strf::width_calculator<CharT>&);

    constexpr void _calc_size(strf::size_preview<false>&) const
    {
    }

    void _calc_size(strf::size_preview<true>& preview) const
    {
        preview.add_size(_str.length());
        if (_fillcount > 0)
        {
             preview.add_size( _fillcount
                             * _encoding.char_size(_afmt.fill) );
        }
    }

    void _write_str(strf::basic_outbuf<CharT>& ob) const;
    void _write_fill( strf::basic_outbuf<CharT>& ob
                    , unsigned count ) const;
};

template<typename CharT>
fmt_string_printer<CharT>::~fmt_string_printer()
{
}

template<typename CharT>
template <bool RequiringWidth>
inline void fmt_string_printer<CharT>::_init
    ( strf::width_preview<RequiringWidth>& preview
    , const strf::fast_width<CharT>&)
{
    auto len = _str.length();
    if (_afmt.width > static_cast<std::ptrdiff_t>(len))
    {
        _fillcount = _afmt.width - static_cast<std::int16_t>(len);
        preview.subtract_width(_afmt.width);
    }
    else
    {
        preview.checked_subtract_width(len);
    }
}

template<typename CharT>
template <bool RequiringWidth>
inline void fmt_string_printer<CharT>::_init
    ( strf::width_preview<RequiringWidth>& preview
    , const strf::width_as_u32len<CharT>&)
{
    auto cp_count = _encoding.codepoints_count( _str.begin()
                                              , _str.end()
                                              , _afmt.width );
    if (_afmt.width > static_cast<std::ptrdiff_t>(cp_count))
    {
        _fillcount = _afmt.width - static_cast<std::int16_t>(cp_count);
        preview.subtract_width(_afmt.width);
    }
    else
    {
        preview.checked_subtract_width(cp_count);
    }

}

template <typename CharT>
template <bool RequiringWidth>
inline void fmt_string_printer<CharT>::_init
    ( strf::width_preview<RequiringWidth>& preview
    , const strf::width_calculator<CharT>& wc)
{
    strf::width_t wmax = _afmt.width;
    strf::width_t wdiff = 0;
    if (preview.remaining_width() > _afmt.width)
    {
        wmax = preview.remaining_width();
        wdiff = preview.remaining_width() - _afmt.width;
    }
    auto str_width = wc.width( wmax
                             , _str.begin(), _str.length()
                             , _encoding, _enc_err, _allow_surr );

    strf::width_t fmt_width{_afmt.width};
    if (fmt_width > str_width)
    {
        auto wfill = (fmt_width - str_width);
        _fillcount = wfill.round();
        preview.subtract_width(wfill + _fillcount);
    }
    else
    {
        preview.subtract_width(str_width);
    }
}

template<typename CharT>
void fmt_string_printer<CharT>::print_to
    ( strf::basic_outbuf<CharT>& ob ) const
{
    if (_fillcount > 0)
    {
        switch (_afmt.alignment)
        {
            case strf::text_alignment::left:
            {
                _write_str(ob);
                _write_fill(ob, _fillcount);
                break;
            }
            case strf::text_alignment::center:
            {
                auto halfcount = _fillcount >> 1;
                _write_fill(ob, halfcount);
                _write_str(ob);
                _write_fill(ob, _fillcount - halfcount);
                break;
            }
            default:
            {
                _write_fill(ob, _fillcount);
                _write_str(ob);
            }
        }
    }
    else
    {
        _write_str(ob);
    }
}

template <typename CharT>
void fmt_string_printer<CharT>::_write_str
    ( strf::basic_outbuf<CharT>& ob ) const
{
    strf::write(ob, _str.begin(), _str.length());
}

template <typename CharT>
void fmt_string_printer<CharT>::_write_fill
    ( strf::basic_outbuf<CharT>& ob
    , unsigned count ) const
{
    _encoding.encode_fill( ob, count, _afmt.fill, _enc_err, _allow_surr );
}

#if defined(STRF_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
STRF_EXPLICIT_TEMPLATE class string_printer<char8_t>;
STRF_EXPLICIT_TEMPLATE class fmt_string_printer<char8_t>;
#endif

STRF_EXPLICIT_TEMPLATE class string_printer<char>;
STRF_EXPLICIT_TEMPLATE class string_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE class string_printer<char32_t>;
STRF_EXPLICIT_TEMPLATE class string_printer<wchar_t>;

STRF_EXPLICIT_TEMPLATE class fmt_string_printer<char>;
STRF_EXPLICIT_TEMPLATE class fmt_string_printer<char16_t>;
STRF_EXPLICIT_TEMPLATE class fmt_string_printer<char32_t>;
STRF_EXPLICIT_TEMPLATE class fmt_string_printer<wchar_t>;

#endif // defined(STRF_SEPARATE_COMPILATION)

} // namespace detail

template <typename CharT, typename FPack, typename Preview>
inline strf::detail::string_printer<CharT>
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, const CharT* str)
{
    return {fp, preview, str};
}

#if defined(__cpp_char8_t)

template <typename CharOut, typename FPack, typename Preview>
inline strf::detail::string_printer<CharOut>
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, const char8_t* str)
{
    static_assert( std::is_same<char8_t, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, preview, str};
}

#endif

template <typename CharOut, typename FPack, typename Preview>
inline strf::detail::string_printer<CharOut>
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, const char* str)
{
    static_assert( std::is_same<char, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, preview, str};
}

template <typename CharOut, typename FPack, typename Preview>
inline strf::detail::string_printer<CharOut>
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, const char16_t* str)
{
    static_assert( std::is_same<char16_t, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, preview, str};
}

template <typename CharOut, typename FPack, typename Preview>
inline strf::detail::string_printer<CharOut>
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, const char32_t* str)
{
    static_assert( std::is_same<char32_t, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, preview, str};
}

template <typename CharOut, typename FPack, typename Preview>
inline strf::detail::string_printer<CharOut>
make_printer(strf::rank<1>, const FPack& fp, Preview& preview, const wchar_t* str)
{
    static_assert( std::is_same<wchar_t, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, preview, str};
}

template
    < typename CharOut
    , typename FPack
    , typename Preview
    , typename CharIn
    , typename Traits
    , typename Allocator >
inline strf::detail::string_printer<CharOut>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , const std::basic_string<CharIn, Traits, Allocator>& str )
{
    static_assert( std::is_same<CharIn, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, preview, {str.data(), str.size()}};
}

#if defined(STRF_HAS_STD_STRING_VIEW)

template
    < typename CharOut
    , typename FPack
    , typename Preview
    , typename CharIn
    , typename Traits >
inline strf::detail::string_printer<CharOut>
make_printer( strf::rank<1>
            , const FPack& fp
            , Preview& preview
            , const std::basic_string_view<CharIn, Traits>& str )
{
    static_assert( std::is_same<CharIn, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, preview, {str.data(), str.size()}};
}

#endif //defined(STRF_HAS_STD_STRING_VIEW)

template <typename CharOut, typename FPack, typename Preview, typename CharIn>
inline strf::detail::fmt_string_printer<CharOut>
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
    return {fp, preview, input.value(), input.get_alignment_format_data()};
}

STRF_NAMESPACE_END

#endif  /* STRF_DETAIL_INPUT_TYPES_CHAR_PTR */

