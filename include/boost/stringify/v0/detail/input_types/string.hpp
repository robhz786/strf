#ifndef STRF_V0_DETAIL_INPUT_TYPES_STRING
#define STRF_V0_DETAIL_INPUT_TYPES_STRING

#include <algorithm>
#include <limits>
#include <boost/stringify/v0/detail/facets/width_calculator.hpp>
#include <boost/stringify/v0/detail/format_functions.hpp>
#include <boost/stringify/v0/facets_pack.hpp>

STRF_V0_NAMESPACE_BEGIN

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

template <typename CharIn>
using string_with_format = stringify::v0::value_with_format
    < stringify::v0::detail::simple_string_view<CharIn>
    , stringify::v0::alignment_format >;

template <typename CharIn, typename Traits, typename Allocator>
auto make_fmt( stringify::v0::tag
             , const std::basic_string<CharIn, Traits, Allocator>& str) noexcept
{
    return stringify::v0::string_with_format<CharIn>{{str.data(), str.size()}};
}

#if defined(STRF_HAS_STD_STRING_VIEW)

template <typename CharIn, typename Traits>
constexpr auto
make_fmt( stringify::v0::tag
        , const std::basic_string_view<CharIn, Traits>& str) noexcept
{
    return stringify::v0::string_with_format<CharIn>{{str.data(), str.size()}};
}

#endif // defined(STRF_HAS_STD_STRING_VIEW)

#if defined(__cpp_char8_t)

STRF_CONSTEXPR_CHAR_TRAITS
auto make_fmt(stringify::v0::tag, const char8_t* str)
{
    auto len = std::char_traits<char8_t>::length(str);
    return stringify::v0::string_with_format<char8_t>{{str, len}};
}

#endif

STRF_CONSTEXPR_CHAR_TRAITS
auto make_fmt(stringify::v0::tag, const char* str)
{
    auto len = std::char_traits<char>::length(str);
    return stringify::v0::string_with_format<char>{{str, len}};
}

STRF_CONSTEXPR_CHAR_TRAITS
auto make_fmt(stringify::v0::tag, const wchar_t* str)
{
    auto len = std::char_traits<wchar_t>::length(str);
    return stringify::v0::string_with_format<wchar_t>{{str, len}};
}

STRF_CONSTEXPR_CHAR_TRAITS
auto make_fmt(stringify::v0::tag, const char16_t* str)
{
    auto len = std::char_traits<char16_t>::length(str);
    return stringify::v0::string_with_format<char16_t>{{str, len}};
}

STRF_CONSTEXPR_CHAR_TRAITS
auto make_fmt(stringify::v0::tag, const char32_t* str)
{
    auto len = std::char_traits<char32_t>::length(str);
    return stringify::v0::string_with_format<char32_t>{{str, len}};
}

namespace detail {

template <typename CharT>
class string_printer: public stringify::v0::printer<CharT>
{
public:

    template <typename FPack, bool RequireSize>
    string_printer
        ( const FPack&
        , stringify::v0::print_preview<RequireSize, false>& preview
        , const CharT* str
        , std::size_t len ) noexcept
        : _str(str)
        , _len(len)
    {
        preview.add_size(len);
    }

    template <typename FPack, bool RequireSize>
    string_printer
        ( const FPack& fp
        , stringify::v0::print_preview<RequireSize, true>& preview
        , const CharT* str
        , std::size_t len ) noexcept
        : _str(str)
        , _len(len)
    {
        preview.add_size(len);
        _calc_width( preview
                   , _get_facet<stringify::v0::width_calculator_c<CharT>>(fp)
                   , _get_facet<stringify::v0::encoding_c<CharT>>(fp)
                   , _get_facet<stringify::v0::encoding_error_c>(fp)
                   , _get_facet<stringify::v0::surrogate_policy_c>(fp) );
    }

    void print_to(stringify::v0::basic_outbuf<CharT>& ob) const override;

private:

    constexpr void _calc_width
        ( stringify::v0::width_preview<true>& wpreview
        , const stringify::v0::width_as_len<CharT>&
        , stringify::v0::encoding<CharT>
        , stringify::v0::encoding_error
        , stringify::v0::surrogate_policy ) noexcept
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
        ( stringify::v0::width_preview<true>& wpreview
        , const stringify::v0::width_as_u32len<CharT>&
        , stringify::v0::encoding<CharT> enc
        , stringify::v0::encoding_error
        , stringify::v0::surrogate_policy )
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
        ( stringify::v0::width_preview<true>& wpreview
        , const stringify::v0::width_calculator<CharT>& wcalc
        , stringify::v0::encoding<CharT> enc
        , stringify::v0::encoding_error  enc_err
        , stringify::v0::surrogate_policy  allow_surr )
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
        using input_tag = stringify::v0::string_input_tag<CharT>;
        return fp.template get_facet<Category, input_tag>();
    }
};

template<typename CharT>
void string_printer<CharT>::print_to(stringify::v0::basic_outbuf<CharT>& ob) const
{
    stringify::v0::write(ob, _str, _len);
}

template <typename CharT>
class fmt_string_printer: public stringify::v0::printer<CharT>
{
public:

    template <typename FPack, typename Preview>
    fmt_string_printer
        ( const FPack& fp
        , Preview& preview
        , const stringify::v0::string_with_format<CharT>& input )
        : _fmt(input)
        , _encoding(_get_facet<stringify::v0::encoding_c<CharT>>(fp))
        , _enc_err(_get_facet<stringify::v0::encoding_error_c>(fp))
        , _allow_surr(_get_facet<stringify::v0::surrogate_policy_c>(fp))
    {
        _init(preview, _get_facet<stringify::v0::width_calculator_c<CharT>>(fp));
        _calc_size(preview);
    }

    ~fmt_string_printer();

    void print_to(stringify::v0::basic_outbuf<CharT>& ob) const override;

private:

    const stringify::v0::string_with_format<CharT> _fmt;
    const stringify::v0::encoding<CharT> _encoding;
    std::int16_t _fillcount = 0;
    const stringify::v0::encoding_error _enc_err;
    const stringify::v0::surrogate_policy _allow_surr;

    template <typename Category, typename FPack>
    static decltype(auto) _get_facet(const FPack& fp)
    {
        using input_tag = stringify::v0::string_input_tag<CharT>;
        return fp.template get_facet<Category, input_tag>();
    }

    template <bool RequiringWidth>
    void _init( stringify::v0::width_preview<RequiringWidth>& preview
              , const stringify::v0::width_as_len<CharT>&);

    template <bool RequiringWidth>
    void _init( stringify::v0::width_preview<RequiringWidth>& preview
              , const stringify::v0::width_as_u32len<CharT>&);

    template <bool RequiringWidth>
    void _init( stringify::v0::width_preview<RequiringWidth>& preview
              , const stringify::v0::width_calculator<CharT>&);

    constexpr void _calc_size(stringify::size_preview<false>&) const
    {
    }

    void _calc_size(stringify::size_preview<true>& preview) const
    {
        preview.add_size(_fmt.value().length());
        if (_fillcount > 0)
        {
             preview.add_size( _fillcount
                             * _encoding.char_size(_fmt.fill(), _enc_err) );
        }
    }

    void _write_str(stringify::v0::basic_outbuf<CharT>& ob) const;
    void _write_fill( stringify::v0::basic_outbuf<CharT>& ob
                    , unsigned count ) const;
};

template<typename CharT>
fmt_string_printer<CharT>::~fmt_string_printer()
{
}

template<typename CharT>
template <bool RequiringWidth>
inline void fmt_string_printer<CharT>::_init
    ( stringify::v0::width_preview<RequiringWidth>& preview
    , const stringify::v0::width_as_len<CharT>&)
{
    auto len = _fmt.value().length();
    if (_fmt.width() > static_cast<std::ptrdiff_t>(len))
    {
        _fillcount = _fmt.width() - static_cast<std::int16_t>(len);
        preview.subtract_width(_fmt.width());
    }
    else
    {
        preview.checked_subtract_width(len);
    }
}

template<typename CharT>
template <bool RequiringWidth>
inline void fmt_string_printer<CharT>::_init
    ( stringify::v0::width_preview<RequiringWidth>& preview
    , const stringify::v0::width_as_u32len<CharT>&)
{
    auto cp_count = _encoding.codepoints_count( _fmt.value().begin()
                                              , _fmt.value().end()
                                              , _fmt.width() );
    if (_fmt.width() > static_cast<std::ptrdiff_t>(cp_count))
    {
        _fillcount = _fmt.width() - static_cast<std::int16_t>(cp_count);
        preview.subtract_width(_fmt.width());
    }
    else
    {
        preview.checked_subtract_width(cp_count);
    }

}

template <typename CharT>
template <bool RequiringWidth>
inline void fmt_string_printer<CharT>::_init
    ( stringify::v0::width_preview<RequiringWidth>& preview
    , const stringify::v0::width_calculator<CharT>& wc)
{
    stringify::v0::width_t wmax = _fmt.width();
    stringify::v0::width_t wdiff = 0;
    if (preview.remaining_width() > _fmt.width())
    {
        wmax = preview.remaining_width();
        wdiff = preview.remaining_width() - _fmt.width();
    }
    auto str_width = wc.width( wmax
                             , _fmt.value().begin(), _fmt.value().length()
                             , _encoding, _enc_err, _allow_surr );

    stringify::v0::width_t fmt_width{_fmt.width()};
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
    ( stringify::v0::basic_outbuf<CharT>& ob ) const
{
    if (_fillcount > 0)
    {
        switch (_fmt.alignment())
        {
            case stringify::v0::text_alignment::left:
            {
                _write_str(ob);
                _write_fill(ob, _fillcount);
                break;
            }
            case stringify::v0::text_alignment::center:
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
    ( stringify::v0::basic_outbuf<CharT>& ob ) const
{
    stringify::v0::write(ob, _fmt.value().begin(), _fmt.value().length());
}

template <typename CharT>
void fmt_string_printer<CharT>::_write_fill
    ( stringify::v0::basic_outbuf<CharT>& ob
    , unsigned count ) const
{
    _encoding.encode_fill( ob, count, _fmt.fill(), _enc_err, _allow_surr );
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
inline stringify::v0::detail::string_printer<CharT>
make_printer(const FPack& fp, Preview& preview, const CharT* str)
{
    return {fp, preview, str, std::char_traits<CharT>::length(str)};
}

#if defined(__cpp_char8_t)

template <typename CharOut, typename FPack, typename Preview>
inline stringify::v0::detail::string_printer<CharOut>
make_printer(const FPack& fp, Preview& preview, const char8_t* str)
{
    static_assert( std::is_same<char8_t, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, preview, str, std::char_traits<char8_t>::length(str)};
}

#endif

template <typename CharOut, typename FPack, typename Preview>
inline stringify::v0::detail::string_printer<CharOut>
make_printer(const FPack& fp, Preview& preview, const char* str)
{
    static_assert( std::is_same<char, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, preview, str, std::char_traits<char>::length(str)};
}

template <typename CharOut, typename FPack, typename Preview>
inline stringify::v0::detail::string_printer<CharOut>
make_printer(const FPack& fp, Preview& preview, const char16_t* str)
{
    static_assert( std::is_same<char16_t, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, preview, str, std::char_traits<char16_t>::length(str)};
}

template <typename CharOut, typename FPack, typename Preview>
inline stringify::v0::detail::string_printer<CharOut>
make_printer(const FPack& fp, Preview& preview, const char32_t* str)
{
    static_assert( std::is_same<char32_t, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, preview, str, std::char_traits<char32_t>::length(str)};
}

template <typename CharOut, typename FPack, typename Preview>
inline stringify::v0::detail::string_printer<CharOut>
make_printer(const FPack& fp, Preview& preview, const wchar_t* str)
{
    static_assert( std::is_same<wchar_t, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, preview, str, std::char_traits<wchar_t>::length(str)};
}

template
    < typename CharOut
    , typename FPack
    , typename Preview
    , typename CharIn
    , typename Traits
    , typename Allocator >
inline stringify::v0::detail::string_printer<CharOut>
make_printer(const FPack& fp, Preview& preview, const std::basic_string<CharIn, Traits, Allocator>& str)
{
    static_assert( std::is_same<CharIn, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, preview, str.data(), str.size()};
}

#if defined(STRF_HAS_STD_STRING_VIEW)

template
    < typename CharOut
    , typename FPack
    , typename Preview
    , typename CharIn
    , typename Traits >
inline stringify::v0::detail::string_printer<CharOut>
make_printer(const FPack& fp, Preview& preview, const std::basic_string_view<CharIn, Traits>& str)
{
    static_assert( std::is_same<CharIn, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, preview, str.data(), str.size()};
}

#endif //defined(STRF_HAS_STD_STRING_VIEW)

template <typename CharOut, typename FPack, typename Preview, typename CharIn>
inline stringify::v0::detail::fmt_string_printer<CharOut>
make_printer(const FPack& fp, Preview& preview, const stringify::v0::string_with_format<CharIn>& input)
{
    static_assert( std::is_same<CharIn, CharOut>::value
                 , "Character type mismatch. Use fmt_cv function." );
    return {fp, preview, input};
}

STRF_V0_NAMESPACE_END

#endif  /* STRF_V0_DETAIL_INPUT_TYPES_CHAR_PTR */

