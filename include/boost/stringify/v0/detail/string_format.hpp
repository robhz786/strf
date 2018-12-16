#ifndef BOOST_STRINGIFY_V0_DETAIL_STRING_FORMAT_HPP
#define BOOST_STRINGIFY_V0_DETAIL_STRING_FORMAT_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/basic_types.hpp>
#include <boost/stringify/v0/facets/encoding.hpp>
#include <boost/utility/string_view.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharIn>
struct string_format
{
    template <typename T>
    class fn
    {
    public:

        template <typename>
        friend class fn;

        constexpr fn() = default;

        template <typename U>
        fn(const fn<U>& u)
            : _encoding(u._encoding)
            , _sani(u._sani)
        {
        }

        constexpr T& sani(bool s = true) &
        {
            _sani = s;
            return *this;
        }
        constexpr T&& sani(bool s = true) &&
        {
            _sani = s;
            return static_cast<T&&>(*this);
        }
        constexpr T& encoding(const stringify::v0::encoding<CharIn>& e) &
        {
            _encoding = & e;
            return *this;
        }
        constexpr T&& encoding(const stringify::v0::encoding<CharIn>& e) &&
        {
            _encoding = & e;
            return static_cast<T&&>(*this);
        }
        constexpr bool get_sani() const
        {
            return _sani;
        }
        bool has_encoding() const
        {
            return  _encoding != nullptr;
        }
        const stringify::v0::encoding<CharIn>& encoding() const
        {
            BOOST_ASSERT(has_encoding());
            return *_encoding;
        }

    private:

        const stringify::v0::encoding<CharIn>* _encoding = nullptr;
        bool _sani = false;
    };
};


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
    , stringify::v0::string_format<CharIn>
    , stringify::v0::alignment_format >;

template <typename CharIn, typename Traits>
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
auto make_fmt(stringify::v0::tag, const std::basic_string<CharIn, Traits>& str)
{
    return stringify::v0::string_with_format<CharIn>{{str.data(), str.size()}};
}

template <typename CharIn, typename Traits>
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
auto make_fmt(stringify::v0::tag, const boost::basic_string_view<CharIn, Traits>& str)
{
    return stringify::v0::string_with_format<CharIn>{{str.data(), str.size()}};
}

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

template <typename CharIn, typename Traits>
constexpr auto
make_fmt(stringify::v0::tag, const std::basic_string_view<CharIn, Traits>& str)
{
    return stringify::v0::string_with_format<CharIn>{{str.data(), str.size()}};
}

#endif // defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

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

template <typename CharIn>
class cv_string
{
public:

    constexpr cv_string(const CharIn* str, std::size_t len) noexcept
       : _str(str, len)
    {
    }

    constexpr cv_string
        ( const CharIn* str
        , std::size_t len
        , const stringify::v0::encoding<CharIn>& enc ) noexcept
        : _str(str, len)
        , _enc(&enc)
    {
    }

    constexpr const CharIn* begin() const
    {
        return _str.begin();
    }
    constexpr const CharIn* end() const
    {
        return _str.end();
    }
    constexpr std::size_t length() const
    {
        return _str.size();
    }
    constexpr std::size_t size() const
    {
        return _str.size();
    }
    constexpr bool has_encoding() const
    {
        return _enc != nullptr;
    }
    constexpr const stringify::v0::encoding<CharIn>& encoding() const
    {
        return *_enc;
    }
    constexpr void set_encoding(const stringify::v0::encoding<CharIn>& enc)
    {
        _enc = &enc;
    }
    
    stringify::v0::detail::simple_string_view<CharIn> _str;
    const stringify::v0::encoding<CharIn>* _enc = nullptr;
};

} // namespace detail


BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string<char> cv(const char* str)
{
    return {str, std::char_traits<char>::length(str)};
}
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string<char16_t> cv(const char16_t* str)
{
    return {str, std::char_traits<char16_t>::length(str)};
}
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string<char32_t> cv(const char32_t* str)
{
    return {str, std::char_traits<char32_t>::length(str)};
}
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string<wchar_t> cv(const wchar_t* str)
{
    return {str, std::char_traits<wchar_t>::length(str)};
}

template <typename CharIn>
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string<CharIn> cv
    ( const CharIn* str
    , const stringify::v0::encoding<CharIn>& enc )
{
    return {str, std::char_traits<CharIn>::length(str), enc};
}

template <typename CharIn, typename Traits>
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string<CharIn> cv
    ( const std::basic_string<CharIn, Traits>& str )
{
    return {str.data(), str.size()};
}

template <typename CharIn, typename Traits>
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string<CharIn> cv
    ( const std::basic_string<CharIn, Traits>& str
    , const stringify::v0::encoding<CharIn>& enc )
{
    return {str.data(), str.size(), enc};
}

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

template <typename CharIn, typename Traits>
constexpr stringify::v0::detail::cv_string<CharIn> cv
    ( const std::basic_string_view<CharIn, Traits>& str )
{
    return {str.data(), str.size()};
}

template <typename CharIn, typename Traits>
constexpr stringify::v0::detail::cv_string<CharIn> cv
    ( const std::basic_string_view<CharIn, Traits>& str
    , const stringify::v0::encoding<CharIn>& enc )
{
    return { str.data(), str.size(), &enc };
}

#endif

namespace detail {

template <typename CharIn>
using cv_string_with_format = stringify::v0::value_with_format
    < stringify::v0::detail::cv_string<CharIn>
    , stringify::v0::string_format<CharIn>
    , stringify::v0::alignment_format >;

template <typename CharIn>
constexpr auto make_fmt
    ( stringify::v0::tag
    , const stringify::v0::detail::cv_string<CharIn>& cv_str )
{
    return stringify::v0::detail::cv_string_with_format<char>{cv_str};
}

} // namespace detail

BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string_with_format<char> fmt_cv(const char* str)
{
    stringify::v0::detail::cv_string<char> cv_str
        { str, std::char_traits<char>::length(str) };
    return stringify::v0::detail::cv_string_with_format<char>{cv_str};
}
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string_with_format<char16_t> fmt_cv(const char16_t* str)
{
    stringify::v0::detail::cv_string<char16_t> cv_str
        { str, std::char_traits<char16_t>::length(str) };
    return stringify::v0::detail::cv_string_with_format<char16_t>{cv_str};
}
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string_with_format<char32_t> fmt_cv(const char32_t* str)
{
    stringify::v0::detail::cv_string<char32_t> cv_str
        { str, std::char_traits<char32_t>::length(str) };
    return stringify::v0::detail::cv_string_with_format<char32_t>{cv_str};
}
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string_with_format<wchar_t> fmt_cv(const wchar_t* str)
{
    stringify::v0::detail::cv_string<wchar_t> cv_str
        { str, std::char_traits<wchar_t>::length(str) };
    return stringify::v0::detail::cv_string_with_format<wchar_t>{cv_str};

}

template <typename CharIn>
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string_with_format<CharIn> fmt_cv
    ( const CharIn* str
    , const stringify::v0::encoding<CharIn>& enc )
{
    stringify::v0::detail::cv_string<CharIn> cv_str{str, enc};
    return stringify::v0::detail::cv_string_with_format<CharIn>{cv_str};
}

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

template <typename CharIn, typename Traits>
constexpr stringify::v0::detail::cv_string_with_format<CharIn> fmt_cv
    ( const std::basic_string_view<CharIn, Traits>& str )
{
    stringify::v0::detail::cv_string<CharIn> cv_str{str.data(), str.size()};
    return stringify::v0::detail::cv_string_with_format<CharIn>{cv_str};
}

template <typename CharIn, typename Traits>
constexpr stringify::v0::detail::cv_string_with_format<CharIn> fmt_cv
    ( const std::basic_string_view<CharIn, Traits>& str
    , const stringify::v0::encoding<CharIn>& enc )
{
    stringify::v0::detail::cv_string<CharIn> cv_str
        { str.data(), str.size(), &enc };
    return stringify::v0::detail::cv_string_with_format<CharIn>{cv_str};
}

#endif

template <typename CharIn, typename Traits>
BOOST_STRINGIFY_CONSTEXPR_CHAR_TRAITS
stringify::v0::detail::cv_string_with_format<CharIn> fmt_cv
    ( const CharIn* str
    , const stringify::v0::encoding<CharIn>& enc )
{
    stringify::v0::detail::cv_string<CharIn> cv_str
        { str, std::char_traits<CharIn>::length(str), &enc};
    return stringify::v0::detail::cv_string_with_format<CharIn>{cv_str};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_STRING_FORMAT_HPP

