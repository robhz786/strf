//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//[ width_calculation_in_encoding_convertion_sample

#include <boost/stringify.hpp>
#include <boost/assert.hpp>

namespace strf = boost::stringify::v0;

int main()
{
    // from UTF-8 to UTF-16
    auto u16str = strf::make_u16string() &= {{ "\u0800", {4, U'.'}}};
    BOOST_ASSERT(u16str == u".\u0800");

    // from UTF-16 to UTF-8
    auto u8str  = strf::make_string()    &= {{u"\u0800", {4, U'.'}}};
    BOOST_ASSERT(u8str == "...\u0800");

    return 0;
}
//]
