#ifndef STRF_DETAIL_OUTPUT_TYPES_CHAR_PTR_HPP
#define STRF_DETAIL_OUTPUT_TYPES_CHAR_PTR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <strf/destination.hpp>
#include <strf/outbuf.hpp>

STRF_NAMESPACE_BEGIN

namespace detail {

template <typename CharT>
class basic_cstr_writer_creator
{
public:

    using char_type = CharT;
    using finish_type = typename basic_cstr_writer<CharT>::result;

    constexpr STRF_HD
    basic_cstr_writer_creator(CharT* dest, CharT* dest_end) noexcept
        : _dest(dest)
        , _dest_end(dest_end)
    {
        STRF_ASSERT(dest < dest_end);
    }

    STRF_HD basic_cstr_writer<CharT> create() const
    {
        return basic_cstr_writer<CharT>{_dest, _dest_end};
    }

private:

    CharT* _dest;
    CharT* _dest_end;
};

}

#if defined(__cpp_char8_t)

template<std::size_t N>
inline STRF_HD auto to(char8_t (&dest)[N])
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char8_t> >
        (dest, dest + N);
}

inline STRF_HD auto to(char8_t* dest, char8_t* end)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char8_t> >
        (dest, end);
}

inline STRF_HD auto to(char8_t* dest, std::size_t count)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char8_t> >
        (dest, dest + count);
}

#endif

template<std::size_t N>
inline STRF_HD auto to(char (&dest)[N])
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char> >
        (dest, dest + N);
}

inline STRF_HD auto to(char* dest, char* end)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char> >
        (dest, end);
}

inline STRF_HD auto to(char* dest, std::size_t count)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char> >
        (dest, dest + count);
}

template<std::size_t N>
inline STRF_HD auto to(char16_t (&dest)[N])
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char16_t> >
        (dest, dest + N);
}

inline STRF_HD auto to(char16_t* dest, char16_t* end)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char16_t> >
        (dest, end);
}

inline STRF_HD auto to(char16_t* dest, std::size_t count)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char16_t> >
        (dest, dest + count);
}

template<std::size_t N>
inline STRF_HD auto to(char32_t (&dest)[N])
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char32_t> >
        (dest, dest + N);
}

inline STRF_HD auto to(char32_t* dest, char32_t* end)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char32_t> >
        (dest, end);
}

inline STRF_HD auto to(char32_t* dest, std::size_t count)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<char32_t> >
        (dest, dest + count);
}

template<std::size_t N>
inline STRF_HD auto to(wchar_t (&dest)[N])
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<wchar_t> >
        (dest, dest + N);
}

inline STRF_HD auto to(wchar_t* dest, wchar_t* end)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<wchar_t> >
        (dest, end);
}

inline STRF_HD auto to(wchar_t* dest, std::size_t count)
{
    return strf::destination_no_reserve
        < strf::detail::basic_cstr_writer_creator<wchar_t> >
        (dest, dest + count);
}

STRF_NAMESPACE_END

#endif  /* STRF_DETAIL_OUTPUT_TYPES_CHAR_PTR_HPP */

