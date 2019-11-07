//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify.hpp>
#include "test_utils.hpp"

int main()
{
    unsigned char groups[100];
    auto * groups_end = groups + sizeof(groups);

    {
        strf::monotonic_grouping<10> grouper(4);
        BOOST_TEST(grouper.thousands_sep_count(0) == 0);
        BOOST_TEST(grouper.thousands_sep_count(1) == 0);
        BOOST_TEST(grouper.thousands_sep_count(4) == 0);
        BOOST_TEST(grouper.thousands_sep_count(5) == 1);
        BOOST_TEST(grouper.thousands_sep_count(8) == 1);
        BOOST_TEST(grouper.thousands_sep_count(9) == 2);
        BOOST_TEST(grouper.thousands_sep_count(12) == 2);

        std::fill(groups, groups_end, static_cast<std::uint8_t>(0xff));
        {
            auto num_groups = grouper.groups(3, groups);
            BOOST_TEST(num_groups == 1);
            BOOST_TEST(groups[0] == 3);
            BOOST_TEST(groups[1] == 0xff);
        }

        std::fill(groups, groups_end, static_cast<std::uint8_t>(0xff));
        {
            auto num_groups = grouper.groups(4, groups);
            BOOST_TEST(num_groups == 1);
            BOOST_TEST(groups[0] == 4);
            BOOST_TEST(groups[1] == 0xff);
        }

        std::fill(groups, groups_end, static_cast<std::uint8_t>(0xff));
        {
            auto num_groups = grouper.groups(5, groups);
            BOOST_TEST(num_groups == 2);
            BOOST_TEST(groups[0] == 4);
            BOOST_TEST(groups[1] == 1);
            BOOST_TEST(groups[2] == 0xff);
        }

        std::fill(groups, groups_end, static_cast<std::uint8_t>(0xff));
        {
            auto num_groups = grouper.groups(8, groups);
            BOOST_TEST(num_groups == 2);
            BOOST_TEST(groups[0] == 4);
            BOOST_TEST(groups[1] == 4);
            BOOST_TEST(groups[2] == 0xff);
        }

        std::fill(groups, groups_end, static_cast<std::uint8_t>(0xff));
        {
            auto num_groups = grouper.groups(9, groups);
            BOOST_TEST(num_groups == 1 + 2);
            BOOST_TEST(groups[0] == 4);
            BOOST_TEST(groups[1] == 4);
            BOOST_TEST(groups[2] == 1);
            BOOST_TEST(groups[3] == 0xff);
        }
    }

    auto big_value = 10000000000000000000ull;
    {
        strf::str_grouping<10> grouper{std::string("\0", 1)};
        TEST("1000") .facets(grouper) (1000);
        TEST("0") .facets(grouper) (0);

        BOOST_TEST(grouper.thousands_sep_count(0) == 0);
        BOOST_TEST(grouper.thousands_sep_count(1) == 0);
        BOOST_TEST(grouper.thousands_sep_count(2) == 0);
        BOOST_TEST(grouper.thousands_sep_count(3) == 0);

    }
    {
        strf::str_grouping<10> grouper{std::string("\001\002\003\000", 4)};
        TEST("10000000000000,000,00,0") .facets(grouper) (big_value);
        TEST("0") .facets(grouper) (0);

        BOOST_TEST(grouper.thousands_sep_count(0) == 0);
        BOOST_TEST(grouper.thousands_sep_count(1) == 0);
        BOOST_TEST(grouper.thousands_sep_count(2) == 1);
        BOOST_TEST(grouper.thousands_sep_count(3) == 1);
        BOOST_TEST(grouper.thousands_sep_count(4) == 2);
        BOOST_TEST(grouper.thousands_sep_count(5) == 2);
        BOOST_TEST(grouper.thousands_sep_count(6) == 2);
        BOOST_TEST(grouper.thousands_sep_count(7) == 3);
        BOOST_TEST(grouper.thousands_sep_count(8) == 3);
        BOOST_TEST(grouper.thousands_sep_count(9) == 3);
        BOOST_TEST(grouper.thousands_sep_count(10) == 3);
        BOOST_TEST(grouper.thousands_sep_count(11) == 3);
        BOOST_TEST(grouper.thousands_sep_count(99) == 3);
    }
    {
        strf::str_grouping<10> grouper{std::string("\001\002\003")};
        TEST("10,000,000,000,000,000,00,0") .facets(grouper) (big_value);
        TEST("0") .facets(grouper) (0);

        BOOST_TEST(grouper.thousands_sep_count(0) == 0);
        BOOST_TEST(grouper.thousands_sep_count(1) == 0);
        BOOST_TEST(grouper.thousands_sep_count(2) == 1);
        BOOST_TEST(grouper.thousands_sep_count(3) == 1);
        BOOST_TEST(grouper.thousands_sep_count(4) == 2);
        BOOST_TEST(grouper.thousands_sep_count(5) == 2);
        BOOST_TEST(grouper.thousands_sep_count(6) == 2);
        BOOST_TEST(grouper.thousands_sep_count(7) == 3);
        BOOST_TEST(grouper.thousands_sep_count(8) == 3);
        BOOST_TEST(grouper.thousands_sep_count(9) == 3);
        BOOST_TEST(grouper.thousands_sep_count(10) == 4);
        BOOST_TEST(grouper.thousands_sep_count(11) == 4);

    }
    {
        strf::str_grouping<10> grouper{std::string("\xff")};
        TEST("10000000000000000000") .facets(grouper) (big_value);
        TEST("0") .facets(grouper) (0);

    }
    {
        auto grouper = strf::str_grouping<10>{std::string("\x0f\002")}.thousands_sep(',');
        TEST("1,00,00,000000000000000") .facets(grouper) (big_value);
        TEST("100000000000000") .facets(grouper) (100000000000000);
        TEST("0") .facets(grouper) (0);


        BOOST_TEST(grouper.thousands_sep_count(0) == 0);
        BOOST_TEST(grouper.thousands_sep_count(1) == 0);
        BOOST_TEST(grouper.thousands_sep_count(15) == 0);
        BOOST_TEST(grouper.thousands_sep_count(16) == 1);
        BOOST_TEST(grouper.thousands_sep_count(17) == 1);
        BOOST_TEST(grouper.thousands_sep_count(18) == 2);

        {
            std::fill(groups, groups_end, static_cast<std::uint8_t>(0xff));
            auto num_groups = grouper.groups(15, groups);
            BOOST_TEST(num_groups == 1);
            BOOST_TEST(groups[0] == 15);
            BOOST_TEST(groups[1] == 0xff);
        }
        {
            std::fill(groups, groups_end, static_cast<std::uint8_t>(0xff));
            auto num_groups = grouper.groups(16, groups);
            BOOST_TEST(num_groups == 2);
            BOOST_TEST(groups[0] == 15);
            BOOST_TEST(groups[1] == 1);
            BOOST_TEST(groups[2] == 0xff);
        }
    }

    return boost::report_errors();;
}
