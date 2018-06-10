//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//[ width_calculation_in_encoding_conversion_sample

#include <boost/stringify.hpp>
#include <boost/assert.hpp>

namespace strf = boost::stringify::v0;

int main()
{
    // from UTF-8 to UTF-16
    auto u16str = strf::to_u16string(strf::right(u8"\u0800", 4, U'.'));
    BOOST_ASSERT(u16str.value() == u".\u0800");

    // from UTF-16 to UTF-8
    auto u8str  = strf::to_string(strf::right(u"\u0800", 4, U'.'));
    BOOST_ASSERT(u8str.value() == u8"...\u0800");

    return 0;
}
//]
