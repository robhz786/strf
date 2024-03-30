//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

STRF_TEST_FUNC void test_input_char()
{
    TEST("a")    ( 'a' );
    TEST("a")    ( strf::fmt('a') );
    TEST("aaaa") ( strf::multi('a', 4) );
    TEST("  aa") ( strf::multi('a', 2) > 4 );

    TEST("    a") ( strf::right('a', 5) );
    TEST("a    ") ( strf::left('a', 5)  );
    TEST("aa   ") ( strf::multi('a', 2) < 5 );

    TEST("....a") ( strf::right('a', 5, '.')  );
    TEST("a....") ( strf::left('a', 5, '.')   );
    TEST("..a..") ( strf::center('a', 5, '.') );

    TEST("...aa") ( strf::right('a', 5, '.').multi(2)  );
    TEST("aa...") ( strf::left('a', 5, '.').multi(2)   );
    TEST(".aa..") ( strf::center('a', 5, '.').multi(2) );

    TEST(".....") ( strf::right('a', 5, '.').multi(0)  );
    TEST(".....") ( strf::left('a', 5, '.').multi(0)   );
    TEST(".....") ( strf::center('a', 5, '.').multi(0) );

    TEST("a")      ( strf::join_left(0, '.')('a') );
    TEST("a")      ( strf::join_left(0, '.')(strf::fmt('a')) );
    TEST("   a")   ( strf::join_left(1, '.')(strf::right('a', 4)) );
    TEST("   a..") ( strf::join_left(6, '.')(strf::right('a', 4)) );

    TEST("  aa")   ( strf::join_left(2, '.')(strf::multi('a', 2) > 4) );
    TEST("  aa")   ( strf::join_left(2, '.')(strf::multi('a', 2) > 4) );
    TEST("  aa")   ( strf::join_left(4, '.')(strf::multi('a', 2) > 4) );
    TEST("  aa..") ( strf::join_left(6, '.')(strf::multi('a', 2) > 4) );

    TEST("aaaa")   ( strf::join_left(2, '.')(strf::multi('a', 4) > 2) );
    TEST("aaaa")   ( strf::join_left(4, '.')(strf::multi('a', 4) > 2) );
    TEST("aaaa..") ( strf::join_left(6, '.')(strf::multi('a', 4) > 2) );

    TEST("aaaa")   ( strf::join_left(2, '.')(strf::multi('a', 4) > 4) );
    TEST("aaaa")   ( strf::join_left(4, '.')(strf::multi('a', 4) > 4) );
    TEST("aaaa..") ( strf::join_left(6, '.')(strf::multi('a', 4) > 4) );
}

REGISTER_STRF_TEST(test_input_char)
