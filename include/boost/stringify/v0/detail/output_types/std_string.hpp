#ifndef BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STRING_HPP
#define BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//#include <system_error>
#include <boost/outbuf/string.hpp>
#include <boost/stringify/v0/dispatcher.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

#if !defined(BOOST_NO_EXCEPTIONS)

template <typename CharT, typename Traits, typename Allocator>
auto append(std::basic_string<CharT, Traits, Allocator>& str)
{
    using str_type = std::basic_string<CharT, Traits, Allocator>;
    using writer = boost::basic_string_appender
        < CharT, Traits, Allocator >;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer
                                    , str_type& >
        (str);
}


template <typename CharT, typename Traits, typename Allocator>
auto assign(std::basic_string<CharT, Traits, Allocator>& str)
{
    str.clear();
    return append(str);
}

template< typename CharT
        , typename Traits = std::char_traits<CharT>
        , typename Allocator = std::allocator<CharT> >
constexpr boost::stringify::v0::dispatcher
    < stringify::v0::facets_pack<>
    , boost::basic_string_maker<CharT, Traits, Allocator> >
    to_basic_string{};

#if defined(__cpp_char8_t)

constexpr boost::stringify::v0::dispatcher
    < stringify::v0::facets_pack<>
    , boost::basic_string_maker<char8_t> >
    to_u8string{};

#endif

constexpr boost::stringify::v0::dispatcher
    < stringify::v0::facets_pack<>
    , boost::basic_string_maker<char> >
    to_string{};

constexpr boost::stringify::v0::dispatcher
    < stringify::v0::facets_pack<>
    , boost::basic_string_maker<char16_t> >
    to_u16string{};

constexpr boost::stringify::v0::dispatcher
    < stringify::v0::facets_pack<>
    , boost::basic_string_maker<char32_t> >
    to_u32string{};

constexpr boost::stringify::v0::dispatcher
    < stringify::v0::facets_pack<>
    , boost::basic_string_maker<wchar_t> >
    to_wstring{};

#endif // !defined(BOOST_NO_EXCEPTIONS)

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STRING_HPP

