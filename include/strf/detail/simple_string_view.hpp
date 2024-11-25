#ifndef STRF_DETAIL_SIMPLE_STRING_VIEW_HPP
#define STRF_DETAIL_SIMPLE_STRING_VIEW_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/detail/strf_def.hpp>

namespace strf {
namespace detail {

template <typename CharIn>
class simple_string_view
{
public:

    using char_type = CharIn;
    using iterator = const CharIn*;
    using const_iterator = const CharIn*;

    simple_string_view() = default;
    ~simple_string_view() = default;
    simple_string_view(const simple_string_view&) = default;
    simple_string_view(simple_string_view&&) = default;


#if defined(STRF_HAS_STD_STRING_VIEW)

    //template <typename CharTraits>
    struct string_view_tag {};

    template <typename Traits>
    constexpr STRF_HD simple_string_view
        ( string_view_tag, std::basic_string_view<CharIn, Traits> s )
        : begin_(s.data())
        , len_(s.size())
    {
    }

    template < typename StringT
             , typename Traits = typename StringT::traits_type
             , std::enable_if_t
               < std::is_convertible<StringT, std::basic_string_view<CharIn, Traits>>::value
               , int > = 0>
    constexpr STRF_HD simple_string_view(const StringT& s)
        : simple_string_view
            ( string_view_tag{}
            , static_cast<std::basic_string_view<CharIn, Traits>>(s) )
    {
    }

#elif defined(STRF_HAS_STD_STRING_DECLARATION)

    template < typename CharTraits, typename Allocator >
    constexpr STRF_HD simple_string_view
        ( const std::basic_string<CharIn, CharTraits, Allocator>& s )
        : begin_(s.data())
        , len_(s.size())
    {
    }

#endif // defined(STRF_HAS_STD_STRING_VIEW)

    constexpr STRF_HD simple_string_view(const CharIn* begin, const CharIn* end) noexcept
        : begin_(begin)
        , len_(detail::safe_cast_size_t(end - begin))
    {
    }
    constexpr STRF_HD simple_string_view(const CharIn* str, std::size_t len) noexcept
        : begin_(str)
        , len_(len)
    {
    }

    STRF_CONSTEXPR_IN_CXX14
    simple_string_view& operator=(const simple_string_view& s) noexcept = default;

    STRF_CONSTEXPR_IN_CXX14
    simple_string_view& operator=(simple_string_view&& s) noexcept = default;

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
    constexpr STRF_HD std::ptrdiff_t ssize() const
    {
        return static_cast<std::ptrdiff_t>(len_);
    }
    constexpr STRF_HD std::size_t length() const
    {
        return len_;
    }
    constexpr STRF_HD CharIn operator[](std::size_t pos) const
    {
        return begin_[pos];
    }

private:

    const CharIn* begin_ = nullptr;
    std::size_t len_ = 0;
};

template <typename CharT>
STRF_HD bool operator==
    ( strf::detail::simple_string_view<CharT> str1
    , strf::detail::simple_string_view<CharT> str2 )
{
    if (str1.size() != str2.size())
        return false;

    return strf::detail::str_equal(str1.data(), str2.data(), str1.size());
}

template <typename CharT>
STRF_HD bool operator==
    ( const CharT* str1, strf::detail::simple_string_view<CharT> str2 )
{
    return strf::detail::simple_string_view<CharT>(str1) == str2;
}

template <typename CharT>
STRF_HD bool operator==
    ( strf::detail::simple_string_view<CharT> str1, const CharT* str2 )
{
    return str1 == strf::detail::simple_string_view<CharT>(str2);
}

template <typename CharT>
STRF_HD bool operator!=
    ( strf::detail::simple_string_view<CharT> str1
    , strf::detail::simple_string_view<CharT> str2 )
{
    if (str1.size() != str2.size())
        return true;

    return !strf::detail::str_equal(str1.data(), str2.data(), str1.size());
}

template <typename CharT>
STRF_HD bool operator!=
    ( const CharT* str1, strf::detail::simple_string_view<CharT> str2 )
{
    return strf::detail::simple_string_view<CharT>(str1) != str2;
}

template <typename CharT>
STRF_HD bool operator!=
    ( strf::detail::simple_string_view<CharT> str1, const CharT* str2 )
{
    return str1 != strf::detail::simple_string_view<CharT>(str2);
}

template <typename CharT>
constexpr STRF_HD simple_string_view<CharT> make_simple_string_view
    ( const CharT* str, std::size_t len) noexcept
{
    return {str, len};
}

template <typename CharT>
constexpr STRF_HD simple_string_view<CharT> make_simple_string_view
    ( const CharT* str, const CharT* str_end) noexcept
{
    return {str, str_end};
}

template <typename CharT>
constexpr STRF_HD simple_string_view<CharT> to_simple_string_view(simple_string_view<CharT> s)
{
    return simple_string_view<CharT>{s};
}

template <typename CharT>
constexpr STRF_HD simple_string_view<CharT> to_simple_string_view(const CharT* cstr)
{
    return simple_string_view<CharT>{cstr};
}

#if defined(STRF_HAS_STD_STRING_VIEW)

template < typename StringT
         , typename CharT = typename StringT::value_type
         , typename Traits = typename StringT::traits_type
         , std::enable_if_t
             < std::is_convertible<StringT, std::basic_string_view<CharT, Traits>>::value
             , int > = 0>
constexpr STRF_HD strf::detail::simple_string_view<CharT> to_simple_string_view
    ( const StringT& s ) noexcept
{
    return { typename simple_string_view<CharT>::string_view_tag{}
           , static_cast< std::basic_string_view<CharT, Traits> >(s) };
}

#if defined(__cpp_char8_t)

constexpr STRF_HD strf::detail::simple_string_view<char8_t> to_simple_string_view
    ( std::basic_string_view<char8_t> s) noexcept
    { return {s.data(), s.size()}; }

#endif // defined(__cpp_char8_t)

constexpr STRF_HD strf::detail::simple_string_view<char> to_simple_string_view
    (std::basic_string_view<char> s) noexcept
    { return {s.data(), s.size()}; }

constexpr STRF_HD strf::detail::simple_string_view<char16_t> to_simple_string_view
    (std::basic_string_view<char16_t> s) noexcept
    { return {s.data(), s.size()}; }

constexpr STRF_HD strf::detail::simple_string_view<char32_t> to_simple_string_view
    (std::basic_string_view<char32_t> s) noexcept
    { return {s.data(), s.size()}; }

constexpr STRF_HD strf::detail::simple_string_view<wchar_t> to_simple_string_view
    (std::basic_string_view<wchar_t> s) noexcept
    { return {s.data(), s.size()}; }

#elif defined(STRF_HAS_STD_STRING_DECLARATION)

template <typename CharT, typename Traits, typename Allocator>
constexpr STRF_HD strf::detail::simple_string_view<CharT> to_simple_string_view
    (const std::basic_string<CharT, Traits, Allocator>& s) noexcept
    { return {s.data(), s.size()}; }

#endif // defined(STRF_HAS_STD_STRING_VIEW)

} // namespace detail
} // namespace strf

#endif  // STRF_DETAIL_SIMPLE_STRING_VIEW_HPP

