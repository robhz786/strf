//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//[ custom_width_calculation_sample

#include <boost/stringify.hpp>

namespace strf = boost::stringify::v0;

int my_width_calculator( int limit
                       , const char32_t* it
                       , const char32_t* end )
{
    int w = 0;
    for (; w < limit && it != end; ++it)
    {
        auto ch = *it;
        w += ( ch == U'\u2E3A' ? 4
             : ch == U'\u2014' ? 2
             : 1 );
    }
    return w;
}

int main()
{
    auto w_as_u32len = strf::width_as_u32len();
    auto w_as_len    = strf::width_as_len();
    auto w_func      = strf::width_as(my_width_calculator);

    // in UTF-8

    const char* str = u8"\u2E3A\u2E3A\u2014";

    auto s_len = strf::to_string.facets(w_as_len) (strf::right(str, 12, U'.'));
    auto s_cpc = strf::to_string.facets(w_as_u32len) (strf::right(str, 12, U'.'));
    auto s_my  = strf::to_string.facets(w_func)     (strf::right(str, 12, U'.'));

    BOOST_ASSERT(s_len == u8"...\u2E3A\u2E3A\u2014");       // width == strlen(str) == 9
    BOOST_ASSERT(s_cpc == u8".........\u2E3A\u2E3A\u2014"); // width == 3
    BOOST_ASSERT(s_my  == u8"..\u2E3A\u2E3A\u2014");        // width == 4 + 4 + 2 == 10

    // in UTF-16

    const char16_t* str16 = u"\u2E3A\u2E3A\u2014";

    auto s16_len = strf::to_u16string.facets(w_as_len) (strf::right(str16, 12, U'.'));
    auto s16_cpc = strf::to_u16string.facets(w_as_u32len) (strf::right(str16, 12, U'.'));
    auto s16_my  = strf::to_u16string.facets(w_func)     (strf::right(str16, 12, U'.'));

    BOOST_ASSERT(s16_len == u".........\u2E3A\u2E3A\u2014"); // width == 3
    BOOST_ASSERT(s16_cpc == u".........\u2E3A\u2E3A\u2014"); // width == 3
    BOOST_ASSERT(s16_my  == u"..\u2E3A\u2E3A\u2014");        // width == 4 + 4 + 2 == 10

    return 0;
}
//]

