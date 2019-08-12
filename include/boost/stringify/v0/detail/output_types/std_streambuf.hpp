#ifndef BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STREAMBUF_HPP
#define BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STREAMBUF_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <streambuf>
#include <boost/stringify/v0/dispatcher.hpp>
#include <boost/outbuf/streambuf.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT, typename Traits = std::char_traits<CharT> >
inline auto write( std::basic_streambuf<CharT, Traits>& dest )
{
    using writer = boost::basic_streambuf_writer<CharT, Traits>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer
                                    , std::basic_streambuf<CharT, Traits>& >
        (dest);
}


template<typename CharT, typename Traits = std::char_traits<CharT> >
inline auto write( std::basic_streambuf<CharT, Traits>* dest )
{
    return stringify::v0::write(*dest);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STREAMBUF_HPP

