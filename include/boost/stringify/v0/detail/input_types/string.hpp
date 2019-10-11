#ifndef BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_STRING
#define BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_STRING

#include <algorithm>
#include <limits>
#include <boost/stringify/v0/detail/facets/width_calculator.hpp>
#include <boost/stringify/v0/detail/format_functions.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

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

    BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
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

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

template <typename CharIn, typename Traits>
constexpr auto
make_fmt( stringify::v0::tag
        , const std::basic_string_view<CharIn, Traits>& str) noexcept
{
    return stringify::v0::string_with_format<CharIn>{{str.data(), str.size()}};
}

#endif // defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

#if defined(__cpp_char8_t)

BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
auto make_fmt(stringify::v0::tag, const char8_t* str)
{
    auto len = std::char_traits<char8_t>::length(str);
    return stringify::v0::string_with_format<char8_t>{{str, len}};
}

#endif

BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
auto make_fmt(stringify::v0::tag, const char* str)
{
    auto len = std::char_traits<char>::length(str);
    return stringify::v0::string_with_format<char>{{str, len}};
}

BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
auto make_fmt(stringify::v0::tag, const wchar_t* str)
{
    auto len = std::char_traits<wchar_t>::length(str);
    return stringify::v0::string_with_format<wchar_t>{{str, len}};
}

BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
auto make_fmt(stringify::v0::tag, const char16_t* str)
{
    auto len = std::char_traits<char16_t>::length(str);
    return stringify::v0::string_with_format<char16_t>{{str, len}};
}

BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
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

    template <typename FPack>
    string_printer
        ( const FPack& fp
        , const CharT* str
        , std::size_t len ) noexcept
        : _str(str)
        , _len(len)
        , _wcalc(_get_facet<stringify::v0::width_calculator_c>(fp))
        , _encoding(_get_facet<stringify::v0::encoding_c<CharT>>(fp))
        , _enc_err(_get_facet<stringify::v0::encoding_error_c>(fp))
        , _allow_surr(_get_facet<stringify::v0::surrogate_policy_c>(fp))
    {
    }

    std::size_t necessary_size() const override;

    void write(boost::basic_outbuf<CharT>& ob) const override;

    int width(int limit) const override;

private:

    const CharT* _str;
    const std::size_t _len;
    const stringify::v0::width_calculator _wcalc;
    const stringify::v0::encoding<CharT> _encoding;
    const stringify::v0::encoding_error  _enc_err;
    const stringify::v0::surrogate_policy  _allow_surr;

    template <typename Category, typename FPack>
    static decltype(auto) _get_facet(const FPack& fp)
    {
        using input_tag = stringify::v0::string_input_tag<CharT>;
        return fp.template get_facet<Category, input_tag>();
    }
};

template<typename CharT>
std::size_t string_printer<CharT>::necessary_size() const
{
    return _len;
}

template<typename CharT>
void string_printer<CharT>::write(boost::basic_outbuf<CharT>& ob) const
{
    boost::write(ob, _str, _len);
}

template<typename CharT>
int string_printer<CharT>::width(int limit) const
{
    return _wcalc.width(limit, _str, _len, _encoding, _enc_err, _allow_surr);
}

template <typename CharT>
class fmt_string_printer: public stringify::v0::printer<CharT>
{
public:

    template <typename FPack>
    fmt_string_printer
        ( const FPack& fp
        , const stringify::v0::string_with_format<CharT>& input )
        : _fmt(input)
        , _wcalc(_get_facet<stringify::v0::width_calculator_c>(fp))
        , _encoding(_get_facet<stringify::v0::encoding_c<CharT>>(fp))
        , _enc_err(_get_facet<stringify::v0::encoding_error_c>(fp))
        , _allow_surr(_get_facet<stringify::v0::surrogate_policy_c>(fp))
    {
        _init();
    }

    ~fmt_string_printer();

    std::size_t necessary_size() const override;

    void write(boost::basic_outbuf<CharT>& ob) const override;

    int width(int limit) const override;

private:

    const stringify::v0::string_with_format<CharT> _fmt;
    const stringify::v0::width_calculator _wcalc;
    const stringify::v0::encoding<CharT> _encoding;
    unsigned _fillcount = 0;
    const stringify::v0::encoding_error _enc_err;
    const stringify::v0::surrogate_policy _allow_surr;

    template <typename Category, typename FPack>
    static decltype(auto) _get_facet(const FPack& fp)
    {
        using input_tag = stringify::v0::string_input_tag<CharT>;
        return fp.template get_facet<Category, input_tag>();
    }

    void _init();

    void _write_str(boost::basic_outbuf<CharT>& ob) const;

    void _write_fill( boost::basic_outbuf<CharT>& ob
                    , unsigned count ) const;
};

template<typename CharT>
fmt_string_printer<CharT>::~fmt_string_printer()
{
}

template<typename CharT>
void fmt_string_printer<CharT>::_init()
{
    if (_fmt.width() > 0)
    {
        auto wstr = _wcalc.width( _fmt.width()
                                , _fmt.value().begin(), _fmt.value().length()
                                , _encoding, _enc_err, _allow_surr );
        _fillcount = _fmt.width() > wstr ? _fmt.width() - wstr : 0;
    }
}

template<typename CharT>
std::size_t fmt_string_printer<CharT>::necessary_size() const
{
    if (_fillcount > 0)
    {
        return _fillcount * _encoding.char_size(_fmt.fill(), _enc_err)
            + _fmt.value().length();
    }
    return _fmt.value().length();
}

template<typename CharT>
int fmt_string_printer<CharT>::width(int limit) const
{
    if (_fillcount > 0)
    {
        return _fmt.width();
    }
    return _wcalc.width( limit, _fmt.value().begin(), _fmt.value().length()
                       , _encoding, _enc_err, _allow_surr );
}

template<typename CharT>
void fmt_string_printer<CharT>::write
    ( boost::basic_outbuf<CharT>& ob ) const
{
    if (_fillcount > 0)
    {
        switch (_fmt.alignment())
        {
            case stringify::v0::alignment_e::left:
            {
                _write_str(ob);
                _write_fill(ob, _fillcount);
                break;
            }
            case stringify::v0::alignment_e::center:
            {
                auto halfcount = _fillcount / 2;
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
    ( boost::basic_outbuf<CharT>& ob ) const
{
    boost::write(ob, _fmt.value().begin(), _fmt.value().length());
}

template <typename CharT>
void fmt_string_printer<CharT>::_write_fill
    ( boost::basic_outbuf<CharT>& ob
    , unsigned count ) const
{
    _encoding.encode_fill( ob, count, _fmt.fill(), _enc_err, _allow_surr );
}

#if defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char8_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_string_printer<char8_t>;
#endif

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_string_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_string_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_string_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class fmt_string_printer<wchar_t>;

#endif // defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

} // namespace detail

#if defined(__cpp_char8_t)

template <typename CharOut, typename FPack>
inline stringify::v0::detail::string_printer<CharOut>
make_printer(const FPack& fp, const char8_t* str)
{
    static_assert( std::is_same<char8_t, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, str, std::char_traits<char8_t>::length(str)};
}

#endif

template <typename CharOut, typename FPack>
inline stringify::v0::detail::string_printer<CharOut>
make_printer(const FPack& fp, const char* str)
{
    static_assert( std::is_same<char, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, str, std::char_traits<char>::length(str)};
}

template <typename CharOut, typename FPack>
inline stringify::v0::detail::string_printer<CharOut>
make_printer(const FPack& fp, const char16_t* str)
{
    static_assert( std::is_same<char16_t, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, str, std::char_traits<char16_t>::length(str)};
}

template <typename CharOut, typename FPack>
inline stringify::v0::detail::string_printer<CharOut>
make_printer(const FPack& fp, const char32_t* str)
{
    static_assert( std::is_same<char32_t, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, str, std::char_traits<char32_t>::length(str)};
}

template <typename CharOut, typename FPack>
inline stringify::v0::detail::string_printer<CharOut>
make_printer(const FPack& fp, const wchar_t* str)
{
    static_assert( std::is_same<wchar_t, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, str, std::char_traits<wchar_t>::length(str)};
}

template
    < typename CharOut
    , typename FPack
    , typename CharIn
    , typename Traits
    , typename Allocator >
inline stringify::v0::detail::string_printer<CharOut>
make_printer(const FPack& fp, const std::basic_string<CharIn, Traits, Allocator>& str)
{
    static_assert( std::is_same<CharIn, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, str.data(), str.size()};
}

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

template
    < typename CharOut
    , typename FPack
    , typename CharIn
    , typename Traits >
inline stringify::v0::detail::string_printer<CharOut>
make_printer(const FPack& fp, const std::basic_string_view<CharIn, Traits>& str)
{
    static_assert( std::is_same<CharIn, CharOut>::value
                 , "Character type mismatch. Use cv function." );
    return {fp, str.data(), str.size()};
}

#endif //defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

template <typename CharOut, typename FPack, typename CharIn>
inline stringify::v0::detail::fmt_string_printer<CharOut>
make_printer(const FPack& fp, const stringify::v0::string_with_format<CharIn>& input)
{
    static_assert( std::is_same<CharIn, CharOut>::value
                 , "Character type mismatch. Use fmt_cv function." );
    return {fp, input};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_DETAIL_INPUT_TYPES_CHAR_PTR */

