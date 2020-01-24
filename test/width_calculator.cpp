//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>
#include "test_utils.hpp"

int custom_width_calculator_function( int limit
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
    // auto wtable = strf::width_as(custom_width_calculator_function);

    // TEST(u8"  \u2E3A\u2E3A\u2014") .with(wtable)
    //     (strf::right(u8"\u2E3A\u2E3A\u2014", 12));

    // TEST( u"  \u2E3A\u2E3A\u2014") .with(wtable)
    //     (strf::right( u"\u2E3A\u2E3A\u2014", 12));

    // TEST( U"  \u2E3A\u2E3A\u2014") .with(wtable)
    //     (strf::right( U"\u2E3A\u2E3A\u2014", 12));

    // TEST( L"  \u2E3A\u2E3A\u2014") .with(wtable)
    //     (strf::right( L"\u2E3A\u2E3A\u2014", 12));

    // TEST(u8"  \u2E3A\u2E3A\u2014") .with(wtable)
    //     (strf::cv( u"\u2E3A\u2E3A\u2014") > 12);

    // TEST( u"  \u2E3A\u2E3A\u2014") .with(wtable)
    //     (strf::cv(u8"\u2E3A\u2E3A\u2014") > 12);


    TEST(u8"         \u2E3A\u2E3A\u2014")
        .with( strf::width_as_u32len{} )
        (strf::right(u8"\u2E3A\u2E3A\u2014", 12));

    TEST( u"         \u2E3A\u2E3A\u2014")
        .with(strf::width_as_u32len{})
        (strf::right( u"\u2E3A\u2E3A\u2014", 12));

    TEST( U"         \u2E3A\u2E3A\u2014")
        .with(strf::width_as_u32len{})
        (strf::right( U"\u2E3A\u2E3A\u2014", 12));

    TEST( L"         \u2E3A\u2E3A\u2014")
        .with(strf::width_as_u32len{})
        (strf::right( L"\u2E3A\u2E3A\u2014", 12));

    TEST(u8"         \u2E3A\u2E3A\u2014")
        .with(strf::width_as_u32len{})
        (strf::cv(u8"\u2E3A\u2E3A\u2014") > 12);

    TEST( u"         \u2E3A\u2E3A\u2014")
        .with(strf::width_as_u32len{})
        (strf::cv(u"\u2E3A\u2E3A\u2014") > 12);


    TEST(u8"   \u2E3A\u2E3A\u2014")
        (strf::fmt(u8"\u2E3A\u2E3A\u2014") > 12);

    TEST( u"         \u2E3A\u2E3A\u2014")
        (strf::fmt( u"\u2E3A\u2E3A\u2014") > 12);

    TEST( U"         \u2E3A\u2E3A\u2014")
        (strf::fmt( U"\u2E3A\u2E3A\u2014") > 12);

    TEST( u"   \u2E3A\u2E3A\u2014")
        (strf::cv(u8"\u2E3A\u2E3A\u2014") > 12);

    TEST(u8"         \u2E3A\u2E3A\u2014")
        (strf::cv( u"\u2E3A\u2E3A\u2014") > 12);

    TEST(u8"         \u2E3A\u2E3A\u2014")
        (strf::cv( U"\u2E3A\u2E3A\u2014") > 12);

    TEST( u"         \u2E3A\u2E3A\u2014")
        (strf::cv( U"\u2E3A\u2E3A\u2014") > 12);

    return test_finish();
}
