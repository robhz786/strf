//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

void STRF_TEST_FUNC test_numpunct()
{
    {
        strf::numpunct<10> grouper(4);
        TEST_TRUE(grouper.thousands_sep_count(1) == 0);
        TEST_TRUE(grouper.thousands_sep_count(4) == 0);
        TEST_TRUE(grouper.thousands_sep_count(5) == 1);
        TEST_TRUE(grouper.thousands_sep_count(8) == 1);
        TEST_TRUE(grouper.thousands_sep_count(9) == 2);
        TEST_TRUE(grouper.thousands_sep_count(12) == 2);
    }

    auto big_value = 10000000000000000000ull;
    {
        auto punct = strf::numpunct<10>{4, 3, 2}.thousands_sep(U'.');
        TEST("1.00.00.000.0000").with(punct)(100000000000ll);
    }
    {
        auto punct = strf::numpunct<10>{3};
        TEST("100,000,000,000").with(punct)(100000000000ll);
    }
    {
        strf::numpunct<10> grouper{};
        TEST("10000000000000000000") .with(grouper) (big_value);
        TEST_TRUE(grouper.thousands_sep_count(1) == 0);
        TEST_TRUE(grouper.thousands_sep_count(99) == 0);
    }
    {
        strf::numpunct<10> grouper{1, 2, 3, -1};
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
        strf::numpunct<10> grouper{1, 2, 3};
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
        strf::numpunct<10> grouper{-1};
        TEST("10000000000000000000") .with(grouper) (big_value);
        TEST("0") .with(grouper) (0);

    }
    {
        auto grouper = strf::numpunct<10>{15, 2};
        TEST("1,00,00,000000000000000") .with(grouper) (big_value);
        TEST("100000000000000") .with(grouper) (100000000000000);
        TEST("0") .with(grouper) (0);

        TEST_TRUE(grouper.thousands_sep_count(1) == 0);
        TEST_TRUE(grouper.thousands_sep_count(15) == 0);
        TEST_TRUE(grouper.thousands_sep_count(16) == 1);
        TEST_TRUE(grouper.thousands_sep_count(17) == 1);
        TEST_TRUE(grouper.thousands_sep_count(18) == 2);
    }

    //
    //     digits_grouping from string
    //
    {
        strf::digits_grouping result("");
        strf::digits_grouping expected{};
        TEST_TRUE(result == expected);
    }
    {
        strf::digits_grouping result("\x03\x02");
        strf::digits_grouping expected{3, 2};
        TEST_TRUE(result == expected);
    }
    {
        strf::digits_grouping result("\x03\xFF");
        strf::digits_grouping expected{3, -1};
        TEST_TRUE(result == expected);
    }

    //
    //     digits_grouping_creator
    //
    {
        strf::digits_grouping_creator creator;
        strf::digits_grouping expected;
        TEST_TRUE(creator.finish() == expected);
        TEST_FALSE(creator.failed());
    }
    {
        strf::digits_grouping_creator creator;
        strf::digits_grouping expected;
        TEST_TRUE(creator.finish_no_more_sep() == expected);
        TEST_FALSE(creator.failed());
    }
    {
        strf::digits_grouping_creator creator;
        strf::digits_grouping expected{strf::digits_grouping::grp_max};
        creator.push_high(strf::digits_grouping::grp_max);
        TEST_TRUE(creator.finish() == expected);
        TEST_FALSE(creator.failed());
    }
    {
        strf::digits_grouping_creator creator;
        strf::digits_grouping expected{strf::digits_grouping::grp_max, -1};
        creator.push_high(strf::digits_grouping::grp_max);
        TEST_TRUE(creator.finish_no_more_sep() == expected);
        TEST_FALSE(creator.failed());
    }
    {
        strf::digits_grouping_creator creator;
        strf::digits_grouping expected{1, 2};
        creator.push_high(1);
        creator.push_high(2);
        TEST_TRUE(creator.finish() == expected);
        TEST_FALSE(creator.failed());
    }
    {
        strf::digits_grouping_creator creator;
        strf::digits_grouping expected{1, 2};
        creator.push_high(1);
        creator.push_high(2);
        creator.push_high(2);
        creator.push_high(2);
        TEST_TRUE(creator.finish() == expected);
        TEST_FALSE(creator.failed());
    }
    {
        strf::digits_grouping_creator creator;
        strf::digits_grouping expected{1, 2, 2, 2, -1};
        creator.push_high(1);
        creator.push_high(2);
        creator.push_high(2);
        creator.push_high(2);
        TEST_TRUE(creator.finish_no_more_sep() == expected);
        TEST_FALSE(creator.failed());
    }

    {
        // too many groups
        strf::digits_grouping_creator creator;
        strf::digits_grouping expected{};
        for (unsigned i = 0; i <= strf::digits_grouping::grps_count_max; ++i) {
            creator.push_high(i);
        }
        TEST_TRUE(creator.finish() == expected);
        TEST_TRUE(creator.failed())
    }
    {
        // too many groups
        strf::digits_grouping_creator creator;
        strf::digits_grouping expected{};
        for (unsigned i = 0; i < strf::digits_grouping::grps_count_max; ++i) {
            creator.push_high(i);
        }
        TEST_TRUE(creator.finish_no_more_sep() == expected);
        TEST_TRUE(creator.failed())
    }
    {
        // negative group
        strf::digits_grouping_creator creator;
        strf::digits_grouping expected{};
        creator.push_high(2);
        creator.push_high(-1);
        creator.push_high(3);
        TEST_TRUE(creator.finish() == expected);
        TEST_TRUE(creator.failed());
    }
    {
        // group too big
        strf::digits_grouping_creator creator;
        strf::digits_grouping expected{};
        creator.push_high(2);
        creator.push_high(strf::digits_grouping::grp_max + 1);
        creator.push_high(3);
        TEST_TRUE(creator.finish() == expected);
        TEST_TRUE(creator.failed())
    }
}
