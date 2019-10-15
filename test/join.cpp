//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <boost/stringify.hpp>

int main()
{
    namespace strf = boost::stringify::v0;

    TEST("   abcdef123   ") (strf::join_center(15)("abc", "de", "f", 123));
    TEST("   abcdef123") (strf::join_split(12, -5)("abc", "de", "f", 123));
    TEST("   abcdef123") (strf::join_split(12, 0)("abc", "de", "f", 123));
    TEST("abcdef   123") (strf::join_split(12, 3)("abc", "de", "f", 123));
    TEST("abcdef123   ") (strf::join_split(12, 4)("abc", "de", "f", 123));
    TEST("abcdef123   ") (strf::join_split(12, 5)("abc", "de", "f", 123));
    TEST("abcdef123   ") (strf::join_left(12)("abc", "def", 123));
    TEST("   abcdef123") (strf::join_right(12)("abc", "def", 123));
    TEST("~~~abcdef123") (strf::join_right(12, '~')("abc", "def", 123));
    TEST("abcdef123") (strf::join_center(9)("abc", "def", 123));
    TEST("abcdef123") (strf::join_split(9, 1)("abc", "def", 1, 23));
    TEST("abcdef123") (strf::join_left(9)("abc", "def", 123));
    TEST("abcdef123") (strf::join_right(9)("abc", "def", 123));
    TEST("abcdef123") (strf::join_center(8)("abc", "def", 123));
    TEST("abcdef123") (strf::join_split(8, 1)("abc", "def", 1, 23));
    TEST("abcdef123") (strf::join_left(8)("abc", "def", 123));
    TEST("abcdef123") (strf::join_right(8)("abc", "def", 123));
    TEST("abcdef123") (strf::join("abc", "def", 123));

    return boost::report_errors();
}
