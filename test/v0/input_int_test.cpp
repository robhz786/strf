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

    TEST ( "0") .exception( 0 );
    TEST (u"0") .exception( 0 );
    TEST (U"0") .exception( 0 );
    TEST (L"0") .exception( 0 );

    TEST ( "0") .exception( (unsigned)0 );
    TEST (u"0") .exception( (unsigned)0 );
    TEST (U"0") .exception( (unsigned)0 );
    TEST (L"0") .exception( (unsigned)0 );

    TEST ( "123") .exception( 123 );
    TEST (u"123") .exception( 123 );
    TEST (U"123") .exception( 123 );
    TEST (L"123") .exception( 123 );

    TEST ( "-123") .exception( -123 );
    TEST (u"-123") .exception( -123 );
    TEST (U"-123") .exception( -123 );
    TEST (L"-123") .exception( -123 );

    TEST ( std::to_string(INT32_MAX).c_str()) .exception( INT32_MAX );
    TEST (std::to_wstring(INT32_MAX).c_str()) .exception( INT32_MAX );

    TEST ( std::to_string(INT32_MIN).c_str()) .exception( INT32_MIN );
    TEST (std::to_wstring(INT32_MIN).c_str()) .exception( INT32_MIN );

    TEST ( std::to_string(UINT32_MAX).c_str()) .exception( UINT32_MAX );
    TEST (std::to_wstring(UINT32_MAX).c_str()) .exception( UINT32_MAX );

    TEST("f")                        .exception( strf::hex(0xf) );
    TEST("ff")                       .exception( strf::hex(0xff) );
    TEST("ffff")                     .exception( strf::hex(0xffff) );
    TEST("fffff")                    .exception( strf::hex(0xfffffl) );
    TEST("fffffffff")                .exception( strf::hex(0xfffffffffLL) );
    TEST("ffffffffffffffff")         .exception( strf::hex(0xffffffffffffffffLL) );
    TEST("0")                        .exception( strf::hex(0) );
    TEST("1")                        .exception( strf::hex(0x1) );
    TEST("10")                       .exception( strf::hex(0x10) );
    TEST("100")                      .exception( strf::hex(0x100) );
    TEST("10000")                    .exception( strf::hex(0x10000l) );
    TEST("100000000")                .exception( strf::hex(0x100000000LL) );
    TEST("1000000000000000")         .exception( strf::hex(0x1000000000000000LL) );
    TEST("7")                        .exception( strf::oct(07) );
    TEST("77")                       .exception( strf::oct(077) );
    TEST("7777")                     .exception( strf::oct(07777) );
    TEST("777777777")                .exception( strf::oct(0777777777l) );
    TEST("7777777777777777")         .exception( strf::oct(07777777777777777LL) );
    TEST("777777777777777777777")    .exception( strf::oct(0777777777777777777777LL) );
    TEST("0")                        .exception( strf::oct(0) );
    TEST("1")                        .exception( strf::oct(01) );
    TEST("10")                       .exception( strf::oct(010) );
    TEST("100")                      .exception( strf::oct(0100) );
    TEST("10000")                    .exception( strf::oct(010000) );
    TEST("100000000")                .exception( strf::oct(0100000000l) );
    TEST("10000000000000000")        .exception( strf::oct(010000000000000000LL) );
    TEST("1000000000000000000000")   .exception( strf::oct(01000000000000000000000LL) );

    TEST("9")                    .exception( 9 );
    TEST("99")                   .exception( 99 );
    TEST("9999")                 .exception( 9999 );
    TEST("99999999")             .exception( 99999999l );
    TEST("999999999999999999")   .exception( 999999999999999999LL );
    TEST("-9")                   .exception( -9 );
    TEST("-99")                  .exception( -99 );
    TEST("-9999")                .exception( -9999 );
    TEST("-99999999")            .exception( -99999999l );
    TEST("-999999999999999999")  .exception( -999999999999999999LL );
    TEST("0")                    .exception( 0 );
    TEST("1")                    .exception( 1 );
    TEST("10")                   .exception( 10 );
    TEST("100")                  .exception( 100 );
    TEST("10000")                .exception( 10000 );
    TEST("100000000")            .exception( 100000000l );
    TEST("1000000000000000000")  .exception( 1000000000000000000LL );
    TEST("10000000000000000000") .exception( 10000000000000000000uLL );
    TEST("-1")                   .exception( -1 );
    TEST("-10")                  .exception( -10 );
    TEST("-100")                 .exception( -100 );
    TEST("-10000")               .exception( -10000 );
    TEST("-100000000")           .exception( -100000000l );
    TEST("-1000000000000000000") .exception( -1000000000000000000LL );

    // formatting characters:

    TEST ("_____1234567890") .exception( strf::right(1234567890l, 15, U'_') );

    TEST ("       123")  .exception(  strf::right(123 , 10) );
    TEST (".......123")  .exception(  strf::right(123 , 10, '.') );
    TEST ("......+123")  .exception( +strf::right(123 , 10, '.') );
    TEST ("......-123")  .exception( +strf::right(-123, 10, '.') );
    TEST ("........+0")  .exception( +strf::right(0   , 10, '.') );
    TEST (".......123")  .exception( +strf::right(123u, 10, '.') );

    TEST ("......+123")  .exception( +strf::right(123 , 10, '.') );
    TEST ("......-123")  .exception( +strf::right(-123, 10, '.') );
    TEST ("........+0")  .exception( +strf::right(0   , 10, '.') );
    TEST (".......123")  .exception( +strf::right(123u, 10, '.') );

    TEST (".......123")  .exception(  strf::internal(123,  10, '.') );
    TEST ("+......123")  .exception( +strf::internal(123,  10, '.') );
    TEST ("-......123")  .exception( +strf::internal(-123, 10, '.') );
    TEST ("+........0")  .exception( +strf::internal(0,    10, '.') );
    TEST (".........0")  .exception(  strf::internal(0,    10, '.') );
    TEST (".......123")  .exception( +strf::internal(123u, 10, '.') );

    TEST ("123.......")  .exception(  strf::left(123,  10, '.') );
    TEST ("+123......")  .exception( +strf::left(123,  10, '.') );
    TEST ("-123......")  .exception( +strf::left(-123, 10, '.') );
    TEST ("+0........")  .exception( +strf::left(0,    10, '.') );
    TEST ("0.........")  .exception(  strf::left(0,    10, '.') );
    TEST ("123.......")  .exception( +strf::left(123u, 10, '.') );

    TEST ("...123....")  .exception(  strf::center(123,  10, '.') );
    TEST ("...+123...")  .exception( +strf::center(123,  10, '.') );
    TEST ("...-123...")  .exception( +strf::center(-123, 10, '.') );
    TEST ("....+0....")  .exception( +strf::center(0,    10, '.') );
    TEST ("....0.....")  .exception(  strf::center(0,    10, '.') );
    TEST ("...123....")  .exception( +strf::center(123u, 10, '.') );

    // hexadecimal case
    TEST("0X1234567890ABCDEF") .exception( ~strf::uphex(0x1234567890abcdefLL) );
    TEST("0x1234567890abcdef") .exception( ~strf::hex(0x1234567890abcdefLL) );

    // hexadecimal aligment

    TEST("        aa")   .exception(  strf::hex(0xAA).width(10) );
    TEST("      0xaa")   .exception( ~strf::hex(0xAA)>10 );
    TEST("        aa")   .exception(  strf::hex(0xAA)>10 );
    TEST("      0xaa")   .exception( ~strf::hex(0xAA)>10 );
    TEST("aa        ")   .exception(  strf::hex(0xAA)<10 );
    TEST("0xaa      ")   .exception( ~strf::hex(0xAA)<10 );
    TEST("        aa")   .exception(  strf::hex(0xAA)%10 );
    TEST("0x      aa")   .exception( ~strf::hex(0xAA)%10 );
    TEST("    aa    ")   .exception(  strf::hex(0xAA)^10 );
    TEST("   0xaa   ")   .exception( ~strf::hex(0xAA)^10 );

    // octadecimal aligment

    TEST("        77")   .exception(  strf::oct(077).width(10) );
    TEST("       077")   .exception( ~strf::oct(077)>10 );
    TEST("        77")   .exception(  strf::oct(077)>10 );
    TEST("       077")   .exception( ~strf::oct(077)>10 );
    TEST("77        ")   .exception(  strf::oct(077)<10 );
    TEST("077       ")   .exception( ~strf::oct(077)<10 );
    TEST("        77")   .exception(  strf::oct(077)%10 );
    TEST("0       77")   .exception( ~strf::oct(077)%10 );
    TEST("    77    ")   .exception(  strf::oct(077)^10 );
    TEST("   077    ")   .exception( ~strf::oct(077)^10 );

    // showpos in octadecimal and hexadecimal must not have any effect

    TEST("aa") .exception( +strf::hex(0xAA) );
    TEST("77") .exception( +strf::oct(077) );


    // inside joins

    TEST("     123")   .exception( strf::join_right(8)(123) );
    TEST("...123~~")   .exception( strf::join_right(8, '.')(strf::left(123, 5, U'~')) );
    TEST(".....123")   .exception( strf::join_right(8, '.')(strf::left(123, 3, U'~')) );
    TEST(".....123")   .exception( strf::join_right(8, '.')(strf::left(123, 2, U'~')) );

    TEST("123")    .exception( strf::join_right(3)(123) );
    TEST("123~~")  .exception( strf::join_right(5, '.')(strf::left(123, 5, U'~')) );
    TEST("123")    .exception( strf::join_right(3, '.')(strf::left(123, 3, U'~')) );
    TEST("123")    .exception( strf::join_right(3, '.')(strf::left(123, 2, U'~')) );
    TEST("123")    .exception( strf::join_right(2)(123) );

    TEST("123~~")  .exception( strf::join_right(4, '.')(strf::left(123, 5, U'~')) );
    TEST("123")    .exception( strf::join_right(2, '.')(strf::left(123, 3, U'~')) );
    TEST("123")    .exception( strf::join_right(2, '.')(strf::left(123, 2, U'~')) );

    {
        auto punct = strf::monotonic_grouping<10>{3};

        TEST("       0").facets(punct) .exception(strf::right(0, 8));
        TEST("     100").facets(punct) .exception(strf::right(100, 8));
        TEST("   1,000").facets(punct) .exception(strf::right(1000, 8));
        TEST("    1000").facets(punct) .exception(strf::hex(0x1000) > 8);

        TEST("       0").facets(punct) .exception( strf::join_right(8)(0) );
        TEST("     100").facets(punct) .exception( strf::join_right(8)(100) );
        TEST("   1,000").facets(punct) .exception( strf::join_right(8)(1000) );
        TEST("    1000").facets(punct) .exception( strf::join_right(8)(strf::hex(0x1000)) );
    }

    {
        auto punct = strf::monotonic_grouping<16>{3}.thousands_sep('\'');

        TEST("     0x0").facets(punct) .exception(~strf::hex(0x0) > 8);
        TEST("   0x100").facets(punct) .exception(~strf::hex(0x100) > 8);
        TEST(" 0x1'000").facets(punct) .exception(~strf::hex(0x1000) > 8);
        TEST("   1'000").facets(punct) .exception( strf::hex(0x1000) > 8);

        TEST("     0x0").facets(punct) .exception( strf::join_right(8)(~strf::hex(0x0)) );
        TEST("   0x100").facets(punct) .exception( strf::join_right(8)(~strf::hex(0x100)) );
        TEST(" 0x1'000").facets(punct) .exception( strf::join_right(8)(~strf::hex(0x1000)) );

        TEST("     0x0").facets(punct) .exception( strf::join_right(8)(~strf::hex(0x0)) );
        TEST("   0x100").facets(punct) .exception( strf::join_right(8)(~strf::hex(0x100)) );
        TEST(" 0x1'000").facets(punct) .exception( strf::join_right(8)(~strf::hex(0x1000)) );
    }


    int rc = report_errors() || boost::report_errors();
    return rc;
}






