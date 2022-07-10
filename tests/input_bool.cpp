//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

STRF_TEST_FUNC void test_input_bool()
{
    TEST ( "true") ( true );
    TEST (u"true") ( true );
    TEST (U"true") ( true );
    TEST (L"true") ( true );

    TEST ( "True") (strf::lettercase::mixed, true);
    TEST (u"True") (strf::lettercase::mixed, true);
    TEST (U"True") (strf::lettercase::mixed, true);
    TEST (L"True") (strf::lettercase::mixed, true);

    TEST ( "TRUE") (strf::lettercase::upper, true);
    TEST (u"TRUE") (strf::lettercase::upper, true);
    TEST (U"TRUE") (strf::lettercase::upper, true);
    TEST (L"TRUE") (strf::lettercase::upper, true);

    TEST ( "false") ( false );
    TEST (u"false") ( false );
    TEST (U"false") ( false );
    TEST (L"false") ( false );

    TEST ( "False") (strf::lettercase::mixed, false);
    TEST (u"False") (strf::lettercase::mixed, false);
    TEST (U"False") (strf::lettercase::mixed, false);
    TEST (L"False") (strf::lettercase::mixed, false);

    TEST ( "FALSE") (strf::lettercase::upper, false);
    TEST (u"FALSE") (strf::lettercase::upper, false);
    TEST (U"FALSE") (strf::lettercase::upper, false);
    TEST (L"FALSE") (strf::lettercase::upper, false);

    TEST ("true")        (  strf::left(true, 4) );
    TEST ("true")        (  strf::left(true, 3) );
    TEST ("true      ")  (  strf::left(true, 10) );
    TEST ("   true   ")  (  strf::center(true, 10) );
    TEST ("      true")  (  strf::right(true, 10) );
    TEST ("      True") (strf::lettercase::mixed, strf::right(true, 10));
    TEST ("      TRUE") (strf::lettercase::upper, strf::right(true, 10));

    TEST ("false")  (  strf::left(false, 5) );
    TEST ("false")  (  strf::left(false, 4) );
    TEST ("false     ")  (  strf::left(false, 10) );
    TEST ("  false   ")  (  strf::center(false, 10) );
    TEST ("     false")  (  strf::right(false, 10) );
    TEST ("     False") (strf::lettercase::mixed, strf::right(false, 10));
    TEST ("     FALSE") (strf::lettercase::upper, strf::right(false, 10));

    constexpr auto j = strf::join_right(20, U'_');

    TEST( "________________true") ( j(true) );
    TEST(u"________________true") ( j(true) );
    TEST(U"________________true") ( j(true) );
    TEST(L"________________true") ( j(true) );
    TEST( "_______________false") ( j(false) );
    TEST(u"_______________false") ( j(false) );
    TEST(U"_______________false") ( j(false) );
    TEST(L"_______________false") ( j(false) );

    TEST ("__________true      ") ( j(strf::left(true, 10)) );
    TEST ("__________   true   ") ( j(strf::center(true, 10)) );
    TEST ("__________      true") ( j(strf::right(true, 10)) );

    TEST ("__________false     ") ( j(strf::left(false, 10)) );
    TEST ("__________  false   ") ( j(strf::center(false, 10)) );
    TEST ("__________     false") ( j(strf::right(false, 10)) );
}

REGISTER_STRF_TEST(test_input_bool);

