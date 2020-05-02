//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>
#include "test_utils.hpp"

int main()
{
    unsigned char groups[100];
    auto * groups_end = groups + sizeof(groups);

    {
        strf::monotonic_grouping<10> grouper(4);
        TEST_TRUE(grouper.thousands_sep_count(1) == 0);
        TEST_TRUE(grouper.thousands_sep_count(4) == 0);
        TEST_TRUE(grouper.thousands_sep_count(5) == 1);
        TEST_TRUE(grouper.thousands_sep_count(8) == 1);
        TEST_TRUE(grouper.thousands_sep_count(9) == 2);
        TEST_TRUE(grouper.thousands_sep_count(12) == 2);

        std::fill(groups, groups_end, static_cast<std::uint8_t>(0xff));
        {
            auto num_groups = grouper.groups(3, groups);
            TEST_TRUE(num_groups == 1);
            TEST_TRUE(groups[0] == 3);
            TEST_TRUE(groups[1] == 0xff);
        }

        std::fill(groups, groups_end, static_cast<std::uint8_t>(0xff));
        {
            auto num_groups = grouper.groups(4, groups);
            TEST_TRUE(num_groups == 1);
            TEST_TRUE(groups[0] == 4);
            TEST_TRUE(groups[1] == 0xff);
        }

        std::fill(groups, groups_end, static_cast<std::uint8_t>(0xff));
        {
            auto num_groups = grouper.groups(5, groups);
            TEST_TRUE(num_groups == 2);
            TEST_TRUE(groups[0] == 4);
            TEST_TRUE(groups[1] == 1);
            TEST_TRUE(groups[2] == 0xff);
        }

        std::fill(groups, groups_end, static_cast<std::uint8_t>(0xff));
        {
            auto num_groups = grouper.groups(8, groups);
            TEST_TRUE(num_groups == 2);
            TEST_TRUE(groups[0] == 4);
            TEST_TRUE(groups[1] == 4);
            TEST_TRUE(groups[2] == 0xff);
        }

        std::fill(groups, groups_end, static_cast<std::uint8_t>(0xff));
        {
            auto num_groups = grouper.groups(9, groups);
            TEST_TRUE(num_groups == 1 + 2);
            TEST_TRUE(groups[0] == 4);
            TEST_TRUE(groups[1] == 4);
            TEST_TRUE(groups[2] == 1);
            TEST_TRUE(groups[3] == 0xff);
        }
    }

    auto big_value = 10000000000000000000ull;
    {
        strf::str_grouping<10> grouper{std::string("\001\002\003\000", 4)};
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

        std::uint8_t groups[100];
        {
            auto num_groups = grouper.groups(20, groups);
            TEST_TRUE(num_groups == 4);
            TEST_TRUE(groups[0] == 1);
            TEST_TRUE(groups[1] == 2);
            TEST_TRUE(groups[2] == 3);
            TEST_TRUE(groups[3] == 14);
        }
        {
            auto num_groups = grouper.groups(6, groups);
            TEST_TRUE(num_groups == 3);
            TEST_TRUE(groups[0] == 1);
            TEST_TRUE(groups[1] == 2);
            TEST_TRUE(groups[2] == 3);
        }
        {
            auto num_groups = grouper.groups(5, groups);
            TEST_TRUE(num_groups == 3);
            TEST_TRUE(groups[0] == 1);
            TEST_TRUE(groups[1] == 2);
            TEST_TRUE(groups[2] == 2);
        }
        {
            auto num_groups = grouper.groups(3, groups);
            TEST_TRUE(num_groups == 2);
            TEST_TRUE(groups[0] == 1);
            TEST_TRUE(groups[1] == 2);
        }
        {
            auto num_groups = grouper.groups(2, groups);
            TEST_TRUE(num_groups == 2);
            TEST_TRUE(groups[0] == 1);
            TEST_TRUE(groups[1] == 1);
        }
    }
    {
        strf::str_grouping<10> grouper{std::string("\001\002\003")};
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

        std::uint8_t groups[100];
        {
            auto num_groups = grouper.groups(12, groups);
            TEST_TRUE(num_groups == 5);
            TEST_TRUE(groups[0] == 1);
            TEST_TRUE(groups[1] == 2);
            TEST_TRUE(groups[2] == 3);
            TEST_TRUE(groups[3] == 3);
            TEST_TRUE(groups[4] == 3);
        }
        {
            auto num_groups = grouper.groups(11, groups);
            TEST_TRUE(num_groups == 5);
            TEST_TRUE(groups[0] == 1);
            TEST_TRUE(groups[1] == 2);
            TEST_TRUE(groups[2] == 3);
            TEST_TRUE(groups[3] == 3);
            TEST_TRUE(groups[4] == 2);
        }
        {
            auto num_groups = grouper.groups(6, groups);
            TEST_TRUE(num_groups == 3);
            TEST_TRUE(groups[0] == 1);
            TEST_TRUE(groups[1] == 2);
            TEST_TRUE(groups[2] == 3);
        }
        {
            auto num_groups = grouper.groups(4, groups);
            TEST_TRUE(num_groups == 3);
            TEST_TRUE(groups[0] == 1);
            TEST_TRUE(groups[1] == 2);
            TEST_TRUE(groups[2] == 1);
        }
        {
            auto num_groups = grouper.groups(3, groups);
            TEST_TRUE(num_groups == 2);
            TEST_TRUE(groups[0] == 1);
            TEST_TRUE(groups[1] == 2);
        }
        {
            auto num_groups = grouper.groups(2, groups);
            TEST_TRUE(num_groups == 2);
            TEST_TRUE(groups[0] == 1);
            TEST_TRUE(groups[1] == 1);
        }
        {
            auto num_groups = grouper.groups(1, groups);
            TEST_TRUE(num_groups == 1);
            TEST_TRUE(groups[0] == 1);
        }
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

        {
            std::fill(groups, groups_end, static_cast<std::uint8_t>(0xff));
            auto num_groups = grouper.groups(15, groups);
            TEST_TRUE(num_groups == 1);
            TEST_TRUE(groups[0] == 15);
            TEST_TRUE(groups[1] == 0xff);
        }
        {
            std::fill(groups, groups_end, static_cast<std::uint8_t>(0xff));
            auto num_groups = grouper.groups(16, groups);
            TEST_TRUE(num_groups == 2);
            TEST_TRUE(groups[0] == 15);
            TEST_TRUE(groups[1] == 1);
            TEST_TRUE(groups[2] == 0xff);
        }
    }

    return test_finish();;
}
