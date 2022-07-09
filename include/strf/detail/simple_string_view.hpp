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

    using iterator = const CharIn*;
    using const_iterator = const CharIn*;

#if defined(STRF_HAS_STD_STRING_DECLARATION)

    template < typename CharTraits, typename Allocator >
    constexpr STRF_HD simple_string_view
        ( const std::basic_string<CharIn, CharTraits, Allocator>& s )
        : begin_(s.data())
        , len_(s.size())
    {
    }

#endif // defined(STRF_HAS_STD_STRING_DECLARATION)

#if defined(STRF_HAS_STD_STRING_VIEW)

    constexpr STRF_HD simple_string_view
        ( std::basic_string_view<CharIn, std::char_traits<CharIn> > s )
        : begin_(s.data())
        , len_(s.size())
    {
    }

    template
        < typename Traits
        , strf::detail::enable_if_t
            < std::is_same<Traits, std::char_traits<CharIn>>::value, int> = 0 >
    constexpr STRF_HD simple_string_view
        ( std::basic_string_view<CharIn, Traits> s ) noexcept
        : begin_(s.data())
        , len_(s.size())
    {
    }

#endif // defined(STRF_HAS_STD_STRING_VIEW)

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
    constexpr STRF_HD CharIn operator[](std::size_t pos) const
    {
        return begin_[pos];
    }

private:

    const CharIn* begin_;
    const std::size_t len_;
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

} // namespace detail
} // namespace strf

#endif  // STRF_DETAIL_SIMPLE_STRING_VIEW_HPP

