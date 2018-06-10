//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

//[ custom_width_calculation_sample

#include <boost/stringify.hpp>

namespace strf = boost::stringify::v0;

int my_width_calculator(char32_t ch)
{
    if(ch == U'\u2014') return 2; // the em-dash
    if(ch == U'\u2E3A') return 4; // the two-em-dash
    return 1;
}

int main()
{
    auto width_as_cpc = strf::width_as_codepoints_count();
    auto width_as_len = strf::width_as_length();
    auto width_my     = strf::width_as(my_width_calculator);

    // in UTF-8

    const char* str = u8"\u2E3A\u2E3A\u2014";

    auto s_len = strf::to_string.facets(width_as_len) (strf::right(str, 12, U'.'));
    auto s_cpc = strf::to_string.facets(width_as_cpc) (strf::right(str, 12, U'.'));
    auto s_my  = strf::to_string.facets(width_my)     (strf::right(str, 12, U'.'));

    BOOST_ASSERT(s_len.value() == u8"...\u2E3A\u2E3A\u2014");       // width == strlen(str) == 9
    BOOST_ASSERT(s_cpc.value() == u8".........\u2E3A\u2E3A\u2014"); // width == 3
    BOOST_ASSERT(s_my.value()  == u8"..\u2E3A\u2E3A\u2014");        // width == 4 + 4 + 2 == 10

    // in UTF-16

    const char16_t* str16 = u"\u2E3A\u2E3A\u2014";

    auto s16_len = strf::to_u16string.facets(width_as_len) (strf::right(str16, 12, U'.'));
    auto s16_cpc = strf::to_u16string.facets(width_as_cpc) (strf::right(str16, 12, U'.'));
    auto s16_my  = strf::to_u16string.facets(width_my)     (strf::right(str16, 12, U'.'));

    BOOST_ASSERT(s16_len.value() == u".........\u2E3A\u2E3A\u2014"); // width == 3
    BOOST_ASSERT(s16_cpc.value() == u".........\u2E3A\u2E3A\u2014"); // width == 3
    BOOST_ASSERT(s16_my.value()  == u"..\u2E3A\u2E3A\u2014");        // width == 4 + 4 + 2 == 10

    return 0;
}
//]

