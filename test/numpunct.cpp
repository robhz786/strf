//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>
#include "test_utils.hpp"

int main()
{
    {
        strf::monotonic_grouping<10> grouper(4);
        TEST_TRUE(grouper.thousands_sep_count(1) == 0);
        TEST_TRUE(grouper.thousands_sep_count(4) == 0);
        TEST_TRUE(grouper.thousands_sep_count(5) == 1);
        TEST_TRUE(grouper.thousands_sep_count(8) == 1);
        TEST_TRUE(grouper.thousands_sep_count(9) == 2);
        TEST_TRUE(grouper.thousands_sep_count(12) == 2);
    }

    auto big_value = 10000000000000000000ull;
    {
        auto punct = strf::str_grouping<10>{"\4\3\2"}.thousands_sep(U'.');
        TEST("1.00.00.000.0000").with(punct)(100000000000ll);
    }
    {
        strf::str_grouping<10> grouper{std::string()};
        TEST("10000000000000000000") .with(grouper) (big_value);
        TEST_TRUE(grouper.thousands_sep_count(1) == 0);
        TEST_TRUE(grouper.thousands_sep_count(99) == 0);
    }
    {
        strf::str_grouping<10> grouper{std::string("\0", 1)};
        TEST("10000000000000000000") .with(grouper) (big_value);
        TEST_TRUE(grouper.thousands_sep_count(1) == 0);
        TEST_TRUE(grouper.thousands_sep_count(99) == 0);
    }
    {
        strf::str_grouping<10> grouper{std::string("\001\002\003\xFF", 4)};
        TEST("10000000000000,000,00,0") .with(grouper) (big_value);
        TEST("0") .with(grouper) (0);

        TEST_TRUE(grouper.thousands_sep_count(1) == 0);
        TEST_TRUE(grouper.thousands_sep_count(2) == 1);
        TEST_TRUE(grouper.thousands_sep_count(3) == 1);
        TEST_TRUE(grouper.thousands_sep_count(4) == 2);
        TEST_TRUE(grouper.thousands_sep_count(5) == 2);
        TEST_TRUE(grouper.thousands_sep_count(6) == 2);
        TEST_TRUE(grouper.thousands_sep_count(7) == 3);
        TEST_TRUE(grouper.thousands_sep_count(8) == 3);
        TEST_TRUE(grouper.thousands_sep_count(9) == 3);
        TEST_TRUE(grouper.thousands_sep_count(10) == 3);
        TEST_TRUE(grouper.thousands_sep_count(11) == 3);
        TEST_TRUE(grouper.thousands_sep_count(99) == 3);
    }
    {
        strf::str_grouping<10> grouper{std::string("\001\002\003\000\002", 5)};
        TEST("10,000,000,000,000,000,00,0") .with(grouper) (big_value);
        TEST("0") .with(grouper) (0);

        TEST_TRUE(grouper.thousands_sep_count(1) == 0);
        TEST_TRUE(grouper.thousands_sep_count(2) == 1);
        TEST_TRUE(grouper.thousands_sep_count(3) == 1);
        TEST_TRUE(grouper.thousands_sep_count(4) == 2);
        TEST_TRUE(grouper.thousands_sep_count(5) == 2);
        TEST_TRUE(grouper.thousands_sep_count(6) == 2);
        TEST_TRUE(grouper.thousands_sep_count(7) == 3);
        TEST_TRUE(grouper.thousands_sep_count(8) == 3);
        TEST_TRUE(grouper.thousands_sep_count(9) == 3);
        TEST_TRUE(grouper.thousands_sep_count(10) == 4);
        TEST_TRUE(grouper.thousands_sep_count(11) == 4);
    }
    {
        strf::str_grouping<10> grouper{std::string("\xff")};
        TEST("10000000000000000000") .with(grouper) (big_value);
        TEST("0") .with(grouper) (0);

    }
    {
        auto grouper = strf::str_grouping<10>{std::string("\x0f\002")}.thousands_sep(',');
        TEST("1,00,00,000000000000000") .with(grouper) (big_value);
        TEST("100000000000000") .with(grouper) (100000000000000);
        TEST("0") .with(grouper) (0);

        TEST_TRUE(grouper.thousands_sep_count(1) == 0);
        TEST_TRUE(grouper.thousands_sep_count(15) == 0);
        TEST_TRUE(grouper.thousands_sep_count(16) == 1);
        TEST_TRUE(grouper.thousands_sep_count(17) == 1);
        TEST_TRUE(grouper.thousands_sep_count(18) == 2);
    }

    return test_finish();;
}
