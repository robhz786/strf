//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

void STRF_TEST_FUNC test_join()
{
    TEST("   abcdef123   ") (strf::join("abc", "de", "f", 123) ^ 15);
    TEST("abcdef123   ") (strf::join("abc", "def", 123) < 12);
    TEST("   abcdef123") (strf::join("abc", "def", 123) > 12);
    TEST("~~~abcdef123") (strf::join("abc", "def", 123).fill('~') > 12);
    TEST("abcdef123") (strf::join("abc", "def", 123) ^ 9);
    TEST("abcdef123") (strf::join("abc", "def", 123) < 9);
    TEST("abcdef123") (strf::join("abc", "def", 123) > 9);
    TEST("abcdef123") (strf::join("abc", "def", 123) ^ 8);
    TEST("abcdef123") (strf::join("abc", "def", 123) < 8);
    TEST("abcdef123") (strf::join("abc", "def", 123) > 8);
    TEST("abcdef123") (strf::join("abc", "def", 123));


    TEST("   abcdef123   ") (strf::join_center(15)("abc", "de", "f", 123));
    TEST("abcdef123   ") (strf::join_left(12)("abc", "def", 123));
    TEST("   abcdef123") (strf::join_right(12)("abc", "def", 123));
    TEST("~~~abcdef123") (strf::join_right(12, '~')("abc", "def", 123));
    TEST("abcdef123") (strf::join_center(9)("abc", "def", 123));
    TEST("abcdef123") (strf::join_left(9)("abc", "def", 123));
    TEST("abcdef123") (strf::join_right(9)("abc", "def", 123));
    TEST("abcdef123") (strf::join_center(8)("abc", "def", 123));
    TEST("abcdef123") (strf::join_left(8)("abc", "def", 123));
    TEST("abcdef123") (strf::join_right(8)("abc", "def", 123));
    TEST("abcdef123") (strf::join("abc", "def", 123));

    // empty joins
    TEST("") (strf::join());

    TEST("        ") (strf::join() ^ 8);
    TEST("        ") (strf::join() < 8);
    TEST("        ") (strf::join() > 8);

    TEST("        ") (strf::join_center(8)());
    TEST("        ") (strf::join_left(8)());
    TEST("        ") (strf::join_right(8)());


    // join inside join

    TEST("    --{ abc }--") (strf::join_right(15)( "--{"
                                                 , strf::join_center(5)('a', 'b', 'c')
                                                 , "}--" ));
    TEST("      --{abc}--") (strf::join_right(15)( "--{"
                                                 , strf::join_center(2)('a', 'b', 'c')
                                                 , "}--" ));
    TEST("--{abc}--") (strf::join_right(8)( "--{"
                                          , strf::join_center(2)('a', 'b', 'c')
                                          , "}--" ));
    TEST("--{ abc }--") (strf::join_right(10)( "--{"
                                             , strf::join_center(5)('a', 'b', 'c')
                                             , "}--" ));
    TEST("      --{abc}--") (strf::join_right(15)( "--{"
                                                 , strf::join('a', 'b', 'c')
                                                 , "}--" ));
    TEST("--{abc}--") (strf::join_right(8)( "--{"
                                          , strf::join('a', 'b', 'c')
                                          , "}--" ));


    TEST("    --{ abc }--") (strf::join( "--{"
                                       , strf::join('a', 'b', 'c') ^ 5
                                       , "}--" ) > 15);
    TEST("      --{abc}--") (strf::join( "--{"
                                       , strf::join_center(2)('a', 'b', 'c') ^ 2
                                       , "}--" ) > 15);
    TEST("--{abc}--") (strf::join( "--{"
                                 , strf::join_center(2)('a', 'b', 'c') ^ 2
                                 , "}--" ) > 8);
    TEST("--{ abc }--") (strf::join( "--{"
                                   , strf::join_center(5)('a', 'b', 'c') ^ 5
                                   , "}--" ) > 10);
    TEST("      --{abc}--") (strf::join( "--{"
                                       , strf::join('a', 'b', 'c')
                                       , "}--" ) > 15);
    TEST("--{abc}--") (strf::join( "--{"
                                 , strf::join('a', 'b', 'c')
                                 , "}--" ) > 8);
}

REGISTER_STRF_TEST(test_join);

