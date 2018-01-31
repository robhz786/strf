//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"
#include <limits>

int main()
{
    namespace strf = boost::stringify::v0;

    TEST ( "0") &= { 0 };
    TEST (u"0") &= { 0 };
    TEST (U"0") &= { 0 };
    TEST (L"0") &= { 0 };

    TEST ( "0") &= { (unsigned)0 };
    TEST (u"0") &= { (unsigned)0 };
    TEST (U"0") &= { (unsigned)0 };
    TEST (L"0") &= { (unsigned)0 };

    TEST ( "123") &= { 123 };
    TEST (u"123") &= { 123 };
    TEST (U"123") &= { 123 };
    TEST (L"123") &= { 123 };

    TEST ( "-123") &= { -123 };
    TEST (u"-123") &= { -123 };
    TEST (U"-123") &= { -123 };
    TEST (L"-123") &= { -123 };

    TEST ( std::to_string(INT32_MAX).c_str()) &= { INT32_MAX };
    TEST (std::to_wstring(INT32_MAX).c_str()) &= { INT32_MAX };

    TEST ( std::to_string(INT32_MIN).c_str()) &= { INT32_MIN };
    TEST (std::to_wstring(INT32_MIN).c_str()) &= { INT32_MIN };

    TEST ( std::to_string(UINT32_MAX).c_str()) &= { UINT32_MAX };
    TEST (std::to_wstring(UINT32_MAX).c_str()) &= { UINT32_MAX };


    TEST("f")                        &= { {0xf, "x"} };
    TEST("ff")                       &= { {0xff, "x"} };
    TEST("ffff")                     &= { {0xffff, "x"} };
    TEST("fffff")                    &= { {0xfffffl, "x"} };
    TEST("fffffffff")                &= { {0xfffffffffLL, "x"} };
    TEST("ffffffffffffffff")         &= { {0xffffffffffffffffLL, "x"} };
    TEST("0")                        &= { {0, "x"} };
    TEST("1")                        &= { {0x1, "x"} };
    TEST("10")                       &= { {0x10, "x"} };
    TEST("100")                      &= { {0x100, "x"} };
    TEST("10000")                    &= { {0x10000l, "x"} };
    TEST("100000000")                &= { {0x100000000LL, "x"} };
    TEST("1000000000000000")         &= { {0x1000000000000000LL, "x"} };
    TEST("7")                        &= { {07, "o"} };
    TEST("77")                       &= { {077, "o"} };
    TEST("7777")                     &= { {07777, "o"} };
    TEST("777777777")                &= { {0777777777l, "o"} };
    TEST("7777777777777777")         &= { {07777777777777777LL, "o"} };
    TEST("777777777777777777777")    &= { {0777777777777777777777LL, "o"} };
    TEST("0")                        &= { {0, "o"} };
    TEST("1")                        &= { {01, "o"} };
    TEST("10")                       &= { {010, "o"} };
    TEST("100")                      &= { {0100, "o"} };
    TEST("10000")                    &= { {010000, "o"} };
    TEST("100000000")                &= { {0100000000l, "o"} };
    TEST("10000000000000000")        &= { {010000000000000000LL, "o"} };
    TEST("1000000000000000000000")   &= { {01000000000000000000000LL, "o"} };

    TEST("9")                    &= { 9 };
    TEST("99")                   &= { 99 };
    TEST("9999")                 &= { 9999 };
    TEST("99999999")             &= { 99999999l };
    TEST("999999999999999999")   &= { 999999999999999999LL };
    TEST("-9")                   &= { -9 };
    TEST("-99")                  &= { -99 };
    TEST("-9999")                &= { -9999 };
    TEST("-99999999")            &= { -99999999l };
    TEST("-999999999999999999")  &= { -999999999999999999LL };
    TEST("0")                    &= { 0 };
    TEST("1")                    &= { 1 };
    TEST("10")                   &= { 10 };
    TEST("100")                  &= { 100 };
    TEST("10000")                &= { 10000 };
    TEST("100000000")            &= { 100000000l };
    TEST("1000000000000000000")  &= { 1000000000000000000LL };
    TEST("10000000000000000000") &= { 10000000000000000000uLL };
    TEST("-1")                   &= { -1 };
    TEST("-10")                  &= { -10 };
    TEST("-100")                 &= { -100 };
    TEST("-10000")               &= { -10000 };
    TEST("-100000000")           &= { -100000000l };
    TEST("-1000000000000000000") &= { -1000000000000000000LL };

    // formatting characters:

    TEST ("_____1234567890") &= { {1234567890l, {15, U'_'}} };

    TEST ("       123")  &= { {123, 10} };
    TEST (".......123")  &= { {123,  {10, '.', "-"}} };
    TEST ("......+123")  &= { {123,  {10, '.', "+"}} };
    TEST ("......-123")  &= { {-123, {10, '.', "+"}} };
    TEST ("........+0")  &= { {0,    {10, '.', "+"}} };
    TEST (".......123")  &= { {123u, {10, '.', "+"}} };

    TEST ("......+123")  &= { {123,  {10, '.', "+"}} };
    TEST ("......-123")  &= { {-123, {10, '.', "+"}} };
    TEST ("........+0")  &= { {0,    {10, '.', "+"}} };
    TEST (".......123")  &= { {123u, {10, '.', "+"}} };

    TEST (".......123")  &= { {123,  {10, '.', "="}} };
    TEST ("+......123")  &= { {123,  {10, '.', "=+"}} };
    TEST ("-......123")  &= { {-123, {10, '.', "=+"}} };
    TEST ("+........0")  &= { {0,    {10, '.', "=+"}} };
    TEST (".........0")  &= { {0,    {10, '.', "="}} };
    TEST (".......123")  &= { {123u, {10, '.', "=+"}} };

    TEST ("123.......")  &= { {123,  {10, '.', "<"}} };
    TEST ("+123......")  &= { {123,  {10, '.', "<+"}} };
    TEST ("-123......")  &= { {-123, {10, '.', "<+"}} };
    TEST ("+0........")  &= { {0,    {10, '.', "<+"}} };
    TEST ("0.........")  &= { {0,    {10, '.', "<"}} };
    TEST ("123.......")  &= { {123u, {10, '.', "<+"}} };

    TEST ("...123....")  &= { {123,  {10, '.', "^"}} };
    TEST ("...+123...")  &= { {123,  {10, '.', "^+"}} };
    TEST ("...-123...")  &= { {-123, {10, '.', "^+"}} };
    TEST ("....+0....")  &= { {0,    {10, '.', "^+"}} };
    TEST ("....0.....")  &= { {0,    {10, '.', "^"}} };
    TEST ("...123....")  &= { {123u, {10, '.', "^+"}} };

    // hexadecimal case
    TEST("0X1234567890ABCDEF") &= { {0x1234567890abcdefLL, "#X"} };
    TEST("0x1234567890abcdef") &= { {0x1234567890abcdefLL, "#x"} };

    // hexadecimal aligment

    TEST("        aa")   &= { {0xAA, {10, "x"}} };
    TEST("      0xaa")   &= { {0xAA, {10, "#x"}} };
    TEST("        aa")   &= { {0xAA, {10, ">x"}} };
    TEST("      0xaa")   &= { {0xAA, {10, ">#x"}} };
    TEST("aa        ")   &= { {0xAA, {10, "<x"}} };
    TEST("0xaa      ")   &= { {0xAA, {10, "<#x"}} };
    TEST("        aa")   &= { {0xAA, {10, "=x"}} };
    TEST("0x      aa")   &= { {0xAA, {10, "=#x"}} };
    TEST("    aa    ")   &= { {0xAA, {10, "^x"}} };
    TEST("   0xaa   ")   &= { {0xAA, {10, "^#x"}} };

    // octadecimal aligment

    TEST("        77")   &= { {077, {10, "o"}} };
    TEST("       077")   &= { {077, {10, "#o"}} };
    TEST("        77")   &= { {077, {10, ">o"}} };
    TEST("       077")   &= { {077, {10, ">#o"}} };
    TEST("77        ")   &= { {077, {10, "<o"}} };
    TEST("077       ")   &= { {077, {10, "<#o"}} };
    TEST("        77")   &= { {077, {10, "=o"}} };
    TEST("0       77")   &= { {077, {10, "=#o"}} };
    TEST("    77    ")   &= { {077, {10, "^o"}} };
    TEST("   077    ")   &= { {077, {10, "^#o"}} };

    // showpos in octadecimal and hexadecimal must not have any effect

    TEST("aa") &= { {0xAA, "+x"} };
    TEST("77") &= { {077,  "+o"} };


    // inside joins

    TEST("     123")   &= { {strf::join_right(8), {123}} };
    TEST("...123~~")   &= { {strf::join_right(8, '.'), {{123, {5, U'~', "<"}}}} };
    TEST(".....123")   &= { {strf::join_right(8, '.'), {{123, {3, U'~', "<"}}}} };
    TEST(".....123")   &= { {strf::join_right(8, '.'), {{123, {2, U'~', "<"}}}} };

    TEST("123")    &= { {strf::join_right(3), {123}} };
    TEST("123~~")  &= { {strf::join_right(5, '.'), {{123, {5, U'~', "<"}}}} };
    TEST("123")    &= { {strf::join_right(3, '.'), {{123, {3, U'~', "<"}}}} };
    TEST("123")    &= { {strf::join_right(3, '.'), {{123, {2, U'~', "<"}}}} };
    TEST("123")    &= { {strf::join_right(2), {123}} };

    TEST("123~~")  &= { {strf::join_right(4, '.'), {{123, {5, U'~', "<"}}}} };
    TEST("123")    &= { {strf::join_right(2, '.'), {{123, {3, U'~', "<"}}}} };
    TEST("123")    &= { {strf::join_right(2, '.'), {{123, {2, U'~', "<"}}}} };

    {
        auto punct = strf::monotonic_grouping<10>{3};

        TEST("       0").with(punct) &= {{0, 8}};
        TEST("     100").with(punct) &= {{100, 8}};
        TEST("   1,000").with(punct) &= {{1000, 8}};
        TEST("    1000").with(punct) &= {{0x1000, {8, "x"}}};

        TEST("       0").with(punct) &= { {strf::join_right(8), {0}} };
        TEST("     100").with(punct) &= { {strf::join_right(8), {100}} };
        TEST("   1,000").with(punct) &= { {strf::join_right(8), {1000}} };
        TEST("    1000").with(punct) &= { {strf::join_right(8), {{0x1000, "x"}}} };
    }

    {
        auto punct = strf::monotonic_grouping<16>{3}.thousands_sep('\'');

        TEST("     0x0").with(punct) &= {{0x0, {8, "#x"}}};
        TEST("   0x100").with(punct) &= {{0x100, {8, "#x"}}};
        TEST(" 0x1'000").with(punct) &= {{0x1000, {8, "#x"}}};
        TEST("   1'000").with(punct) &= {{0x1000, {8, "x"}}};

        TEST("     0x0").with(punct) &= { {strf::join_right(8), {{0x0, "#x"}}} };
        TEST("   0x100").with(punct) &= { {strf::join_right(8), {{0x100, "#x"}}} };
        TEST(" 0x1'000").with(punct) &= { {strf::join_right(8), {{0x1000, "#x"}}} };

        TEST("     0x0").with(punct) &= { {strf::join_right(8), {{0x0, "#x"}}} };
        TEST("   0x100").with(punct) &= { {strf::join_right(8), {{0x100, "#x"}}} };
        TEST(" 0x1'000").with(punct) &= { {strf::join_right(8), {{0x1000, "#x"}}} };
    }


    int rc = report_errors() || boost::report_errors();
    return rc;
}






