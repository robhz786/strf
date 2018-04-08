//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"

namespace strf = boost::stringify::v0;


int custom_width_calculator_function(char32_t ch)
{
    if(ch == U'\u2E3A') return 4;
    if(ch == U'\u2014') return 2;
    return 1;
}

int main()
{
    auto cp_count = strf::width_as_codepoints_count();
    auto wtable = strf::width_as(custom_width_calculator_function);

    TEST(u8"  \u2E3A\u2E3A\u2014")        .facets(wtable)   .exception(strf::right(u8"\u2E3A\u2E3A\u2014", 12));
    TEST( u"  \u2E3A\u2E3A\u2014")        .facets(wtable)   .exception(strf::right( u"\u2E3A\u2E3A\u2014", 12));
    TEST( U"  \u2E3A\u2E3A\u2014")        .facets(wtable)   .exception(strf::right( U"\u2E3A\u2E3A\u2014", 12));
    TEST( L"  \u2E3A\u2E3A\u2014")        .facets(wtable)   .exception(strf::right( L"\u2E3A\u2E3A\u2014", 12));
    TEST(u8"  \u2E3A\u2E3A\u2014")        .facets(wtable)   .exception(strf::right( u"\u2E3A\u2E3A\u2014", 12));
    TEST( u"  \u2E3A\u2E3A\u2014")        .facets(wtable)   .exception(strf::right(u8"\u2E3A\u2E3A\u2014", 12));

    TEST(u8"         \u2E3A\u2E3A\u2014") .facets(cp_count) .exception(strf::right(u8"\u2E3A\u2E3A\u2014", 12));
    TEST( u"         \u2E3A\u2E3A\u2014") .facets(cp_count) .exception(strf::right( u"\u2E3A\u2E3A\u2014", 12));
    TEST( U"         \u2E3A\u2E3A\u2014") .facets(cp_count) .exception(strf::right( U"\u2E3A\u2E3A\u2014", 12));
    TEST( L"         \u2E3A\u2E3A\u2014") .facets(cp_count) .exception(strf::right( L"\u2E3A\u2E3A\u2014", 12));
    TEST(u8"         \u2E3A\u2E3A\u2014") .facets(cp_count) .exception(strf::right(u8"\u2E3A\u2E3A\u2014", 12));
    TEST( u"         \u2E3A\u2E3A\u2014") .facets(cp_count) .exception(strf::right( u"\u2E3A\u2E3A\u2014", 12));

    TEST(u8"   \u2E3A\u2E3A\u2014")                    .exception(strf::right(u8"\u2E3A\u2E3A\u2014", 12));
    TEST( u"         \u2E3A\u2E3A\u2014")              .exception(strf::right( u"\u2E3A\u2E3A\u2014", 12));
    TEST( U"         \u2E3A\u2E3A\u2014")              .exception(strf::right( U"\u2E3A\u2E3A\u2014", 12));

    TEST( u"   \u2E3A\u2E3A\u2014")                    .exception(strf::right(u8"\u2E3A\u2E3A\u2014", 12));
    TEST(u8"         \u2E3A\u2E3A\u2014")              .exception(strf::right( u"\u2E3A\u2E3A\u2014", 12));
    TEST(u8"         \u2E3A\u2E3A\u2014")              .exception(strf::right( U"\u2E3A\u2E3A\u2014", 12));
    TEST( u"         \u2E3A\u2E3A\u2014")              .exception(strf::right( U"\u2E3A\u2E3A\u2014", 12));


    return report_errors() || boost::report_errors();
}
