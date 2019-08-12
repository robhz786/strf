#ifndef BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_FILE_HPP
#define BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_FILE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <cstdio>
#include <cstring>
#include <boost/stringify/v0/dispatcher.hpp>
#include <boost/outbuf/cfile.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT = char>
inline auto write(std::FILE* destination)
{
    using writer = boost::narrow_cfile_writer<CharT>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, FILE* >
        (destination);
}

inline auto wwrite(std::FILE* destination)
{
    using writer = boost::wide_cfile_writer;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, FILE* >
        (destination);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_FILE_HPP

