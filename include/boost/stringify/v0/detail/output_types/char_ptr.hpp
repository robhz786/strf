#ifndef BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_CHAR_PTR_HPP
#define BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_CHAR_PTR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <boost/stringify/v0/dispatcher.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

#if defined(__cpp_char8_t)

template<std::size_t N>
inline auto write(char8_t (&dest)[N])
{
    using writer = boost::basic_cstr_writer<char8_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, char8_t*, char8_t*>
        (dest, dest + N);
}

#endif

template<std::size_t N>
inline auto write(char (&dest)[N])
{
    using writer = boost::basic_cstr_writer<char>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, char*, char*>
        (dest, dest + N);
}

inline auto write(char* dest, char* end)
{
    using writer = boost::basic_cstr_writer<char>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, char*, char* >
        (dest, end);
}

inline auto write(char* dest, std::size_t count)
{
    using writer = boost::basic_cstr_writer<char>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, char*, char*>
        (dest, dest + count);
}


template<std::size_t N>
inline auto write(char16_t (&dest)[N])
{
    using writer = boost::basic_cstr_writer<char16_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, char16_t*, char16_t*>
        (dest, dest + N);
}

inline auto write(char16_t* dest, char16_t* end)
{
    using writer = boost::basic_cstr_writer<char16_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, char16_t*, char16_t*>
        (dest, end);
}

inline auto write(char16_t* dest, std::size_t count)
{
    using writer = boost::basic_cstr_writer<char16_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                     , writer, char16_t*, char16_t* >
        (dest, dest + count);
}

template<std::size_t N>
inline auto write(char32_t (&dest)[N])
{
    using writer = boost::basic_cstr_writer<char32_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, char32_t*, char32_t*>
        (dest, dest + N);
}

inline auto write(char32_t* dest, char32_t* end)
{
    using writer = boost::basic_cstr_writer<char32_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, char32_t*, char32_t*>
        (dest, end);
}

inline auto write(char32_t* dest, std::size_t count)
{
    using writer = boost::basic_cstr_writer<char32_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, char32_t*, char32_t* >
        (dest, dest + count);
}

template<std::size_t N>
inline auto write(wchar_t (&dest)[N])
{
    using writer = boost::basic_cstr_writer<wchar_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, wchar_t*, wchar_t*>
        (dest, dest + N);
}

inline auto write(wchar_t* dest, wchar_t* end)
{
    using writer = boost::basic_cstr_writer<wchar_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                     , writer, wchar_t*, wchar_t*>
        (dest, end);
}


inline auto write(wchar_t* dest, std::size_t count)
{
    using writer = boost::basic_cstr_writer<wchar_t>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, wchar_t*, wchar_t* >
        (dest, dest + count);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_CHAR_PTR_HPP */

