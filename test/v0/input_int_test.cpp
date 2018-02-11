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

    TEST("f")                        &= { strf::hex(0xf) };
    TEST("ff")                       &= { strf::hex(0xff) };
    TEST("ffff")                     &= { strf::hex(0xffff) };
    TEST("fffff")                    &= { strf::hex(0xfffffl) };
    TEST("fffffffff")                &= { strf::hex(0xfffffffffLL) };
    TEST("ffffffffffffffff")         &= { strf::hex(0xffffffffffffffffLL) };
    TEST("0")                        &= { strf::hex(0) };
    TEST("1")                        &= { strf::hex(0x1) };
    TEST("10")                       &= { strf::hex(0x10) };
    TEST("100")                      &= { strf::hex(0x100) };
    TEST("10000")                    &= { strf::hex(0x10000l) };
    TEST("100000000")                &= { strf::hex(0x100000000LL) };
    TEST("1000000000000000")         &= { strf::hex(0x1000000000000000LL) };
    TEST("7")                        &= { strf::oct(07) };
    TEST("77")                       &= { strf::oct(077) };
    TEST("7777")                     &= { strf::oct(07777) };
    TEST("777777777")                &= { strf::oct(0777777777l) };
    TEST("7777777777777777")         &= { strf::oct(07777777777777777LL) };
    TEST("777777777777777777777")    &= { strf::oct(0777777777777777777777LL) };
    TEST("0")                        &= { strf::oct(0) };
    TEST("1")                        &= { strf::oct(01) };
    TEST("10")                       &= { strf::oct(010) };
    TEST("100")                      &= { strf::oct(0100) };
    TEST("10000")                    &= { strf::oct(010000) };
    TEST("100000000")                &= { strf::oct(0100000000l) };
    TEST("10000000000000000")        &= { strf::oct(010000000000000000LL) };
    TEST("1000000000000000000000")   &= { strf::oct(01000000000000000000000LL) };

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

    TEST ("_____1234567890") &= { strf::right(1234567890l, 15, U'_') };

    TEST ("       123")  &= {  strf::right(123 , 10) };
    TEST (".......123")  &= {  strf::right(123 , 10, '.') };
    TEST ("......+123")  &= { +strf::right(123 , 10, '.') };
    TEST ("......-123")  &= { +strf::right(-123, 10, '.') };
    TEST ("........+0")  &= { +strf::right(0   , 10, '.') };
    TEST (".......123")  &= { +strf::right(123u, 10, '.') };

    TEST ("......+123")  &= { +strf::right(123 , 10, '.') };
    TEST ("......-123")  &= { +strf::right(-123, 10, '.') };
    TEST ("........+0")  &= { +strf::right(0   , 10, '.') };
    TEST (".......123")  &= { +strf::right(123u, 10, '.') };

    TEST (".......123")  &= {  strf::internal(123,  10, '.') };
    TEST ("+......123")  &= { +strf::internal(123,  10, '.') };
    TEST ("-......123")  &= { +strf::internal(-123, 10, '.') };
    TEST ("+........0")  &= { +strf::internal(0,    10, '.') };
    TEST (".........0")  &= {  strf::internal(0,    10, '.') };
    TEST (".......123")  &= { +strf::internal(123u, 10, '.') };

    TEST ("123.......")  &= {  strf::left(123,  10, '.') };
    TEST ("+123......")  &= { +strf::left(123,  10, '.') };
    TEST ("-123......")  &= { +strf::left(-123, 10, '.') };
    TEST ("+0........")  &= { +strf::left(0,    10, '.') };
    TEST ("0.........")  &= {  strf::left(0,    10, '.') };
    TEST ("123.......")  &= { +strf::left(123u, 10, '.') };

    TEST ("...123....")  &= {  strf::center(123,  10, '.') };
    TEST ("...+123...")  &= { +strf::center(123,  10, '.') };
    TEST ("...-123...")  &= { +strf::center(-123, 10, '.') };
    TEST ("....+0....")  &= { +strf::center(0,    10, '.') };
    TEST ("....0.....")  &= {  strf::center(0,    10, '.') };
    TEST ("...123....")  &= { +strf::center(123u, 10, '.') };

    // hexadecimal case
    TEST("0X1234567890ABCDEF") &= { ~strf::uphex(0x1234567890abcdefLL) };
    TEST("0x1234567890abcdef") &= { ~strf::hex(0x1234567890abcdefLL) };

    // hexadecimal aligment

    TEST("        aa")   &= {  strf::hex(0xAA).width(10) };
    TEST("      0xaa")   &= { ~strf::hex(0xAA)>10 };
    TEST("        aa")   &= {  strf::hex(0xAA)>10 };
    TEST("      0xaa")   &= { ~strf::hex(0xAA)>10 };
    TEST("aa        ")   &= {  strf::hex(0xAA)<10 };
    TEST("0xaa      ")   &= { ~strf::hex(0xAA)<10 };
    TEST("        aa")   &= {  strf::hex(0xAA)%10 };
    TEST("0x      aa")   &= { ~strf::hex(0xAA)%10 };
    TEST("    aa    ")   &= {  strf::hex(0xAA)^10 };
    TEST("   0xaa   ")   &= { ~strf::hex(0xAA)^10 };

    // octadecimal aligment

    TEST("        77")   &= {  strf::oct(077).width(10) };
    TEST("       077")   &= { ~strf::oct(077)>10 };
    TEST("        77")   &= {  strf::oct(077)>10 };
    TEST("       077")   &= { ~strf::oct(077)>10 };
    TEST("77        ")   &= {  strf::oct(077)<10 };
    TEST("077       ")   &= { ~strf::oct(077)<10 };
    TEST("        77")   &= {  strf::oct(077)%10 };
    TEST("0       77")   &= { ~strf::oct(077)%10 };
    TEST("    77    ")   &= {  strf::oct(077)^10 };
    TEST("   077    ")   &= { ~strf::oct(077)^10 };

    // showpos in octadecimal and hexadecimal must not have any effect

    TEST("aa") &= { +strf::hex(0xAA) };
    TEST("77") &= { +strf::oct(077) };


    // inside joins

    TEST("     123")   &= { {strf::join_right(8), {123}} };
    TEST("...123~~")   &= { {strf::join_right(8, '.'), {strf::left(123, 5, U'~')}} };
    TEST(".....123")   &= { {strf::join_right(8, '.'), {strf::left(123, 3, U'~')}} };
    TEST(".....123")   &= { {strf::join_right(8, '.'), {strf::left(123, 2, U'~')}} };

    TEST("123")    &= { {strf::join_right(3), {123}} };
    TEST("123~~")  &= { {strf::join_right(5, '.'), {strf::left(123, 5, U'~')}} };
    TEST("123")    &= { {strf::join_right(3, '.'), {strf::left(123, 3, U'~')}} };
    TEST("123")    &= { {strf::join_right(3, '.'), {strf::left(123, 2, U'~')}} };
    TEST("123")    &= { {strf::join_right(2), {123}} };

    TEST("123~~")  &= { {strf::join_right(4, '.'), {strf::left(123, 5, U'~')}} };
    TEST("123")    &= { {strf::join_right(2, '.'), {strf::left(123, 3, U'~')}} };
    TEST("123")    &= { {strf::join_right(2, '.'), {strf::left(123, 2, U'~')}} };

    {
        auto punct = strf::monotonic_grouping<10>{3};

        TEST("       0").with(punct) &= {strf::right(0, 8)};
        TEST("     100").with(punct) &= {strf::right(100, 8)};
        TEST("   1,000").with(punct) &= {strf::right(1000, 8)};
        TEST("    1000").with(punct) &= {strf::hex(0x1000) > 8};

        TEST("       0").with(punct) &= { {strf::join_right(8), {0}} };
        TEST("     100").with(punct) &= { {strf::join_right(8), {100}} };
        TEST("   1,000").with(punct) &= { {strf::join_right(8), {1000}} };
        TEST("    1000").with(punct) &= { {strf::join_right(8), {strf::hex(0x1000)}} };
    }

    {
        auto punct = strf::monotonic_grouping<16>{3}.thousands_sep('\'');

        TEST("     0x0").with(punct) &= {~strf::hex(0x0) > 8};
        TEST("   0x100").with(punct) &= {~strf::hex(0x100) > 8};
        TEST(" 0x1'000").with(punct) &= {~strf::hex(0x1000) > 8};
        TEST("   1'000").with(punct) &= { strf::hex(0x1000) > 8};

        TEST("     0x0").with(punct) &= { {strf::join_right(8), {~strf::hex(0x0)}} };
        TEST("   0x100").with(punct) &= { {strf::join_right(8), {~strf::hex(0x100)}} };
        TEST(" 0x1'000").with(punct) &= { {strf::join_right(8), {~strf::hex(0x1000)}} };

        TEST("     0x0").with(punct) &= { {strf::join_right(8), {~strf::hex(0x0)}} };
        TEST("   0x100").with(punct) &= { {strf::join_right(8), {~strf::hex(0x100)}} };
        TEST(" 0x1'000").with(punct) &= { {strf::join_right(8), {~strf::hex(0x1000)}} };
    }


    int rc = report_errors() || boost::report_errors();
    return rc;
}






