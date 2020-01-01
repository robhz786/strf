//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>
#include <limits>
#include "test_utils.hpp"

int main()
{

    TEST ( "0") ( 0 );
    TEST (u"0") ( 0 );
    TEST (U"0") ( 0 );
    TEST (L"0") ( 0 );
    TEST ( "0") ( (unsigned)0 );
    TEST (u"0") ( (unsigned)0 );
    TEST (U"0") ( (unsigned)0 );
    TEST (L"0") ( (unsigned)0 );
    TEST ( "123") ( 123 );
    TEST (u"123") ( 123 );
    TEST (U"123") ( 123 );
    TEST (L"123") ( 123 );
    TEST ( "-123") ( -123 );
    TEST (u"-123") ( -123 );
    TEST (U"-123") ( -123 );
    TEST (L"-123") ( -123 );

    TEST ( "0") ( strf::fmt(0) );
    TEST (u"0") ( strf::fmt(0) );
    TEST (U"0") ( strf::fmt(0) );
    TEST (L"0") ( strf::fmt(0) );
    TEST ( "0") ( strf::fmt((unsigned)0) );
    TEST (u"0") ( strf::fmt((unsigned)0) );
    TEST (U"0") ( strf::fmt((unsigned)0) );
    TEST (L"0") ( strf::fmt((unsigned)0) );
    TEST ( "123") ( strf::fmt(123) );
    TEST (u"123") ( strf::fmt(123) );
    TEST (U"123") ( strf::fmt(123) );
    TEST (L"123") ( strf::fmt(123) );
    TEST ( "-123") ( strf::fmt(-123) );
    TEST (u"-123") ( strf::fmt(-123) );
    TEST (U"-123") ( strf::fmt(-123) );
    TEST (L"-123") ( strf::fmt(-123) );

    TEST ( std::to_string(INT32_MAX).c_str()) ( INT32_MAX );
    TEST (std::to_wstring(INT32_MAX).c_str()) ( INT32_MAX );

    TEST ( std::to_string(INT32_MIN).c_str()) ( INT32_MIN );
    TEST (std::to_wstring(INT32_MIN).c_str()) ( INT32_MIN );

    TEST ( std::to_string(UINT32_MAX).c_str()) ( UINT32_MAX );
    TEST (std::to_wstring(UINT32_MAX).c_str()) ( UINT32_MAX );

    TEST ( std::to_string(INT32_MAX).c_str()) ( strf::fmt(INT32_MAX) );
    TEST (std::to_wstring(INT32_MAX).c_str()) ( strf::fmt(INT32_MAX) );

    TEST ( std::to_string(INT32_MIN).c_str()) ( strf::fmt(INT32_MIN) );
    TEST (std::to_wstring(INT32_MIN).c_str()) ( strf::fmt(INT32_MIN) );

    TEST ( std::to_string(UINT32_MAX).c_str()) ( strf::fmt(UINT32_MAX) );
    TEST (std::to_wstring(UINT32_MAX).c_str()) ( strf::fmt(UINT32_MAX) );

    TEST("f")                        ( strf::hex(0xf) );
    TEST("ff")                       ( strf::hex(0xff) );
    TEST("ffff")                     ( strf::hex(0xffff) );
    TEST("fffff")                    ( strf::hex(0xfffffl) );
    TEST("fffffffff")                ( strf::hex(0xfffffffffLL) );
    TEST("ffffffffffffffff")         ( strf::hex(0xffffffffffffffffLL) );
    TEST("0")                        ( strf::hex(0) );
    TEST("1")                        ( strf::hex(0x1) );
    TEST("10")                       ( strf::hex(0x10) );
    TEST("100")                      ( strf::hex(0x100) );
    TEST("10000")                    ( strf::hex(0x10000l) );
    TEST("100000000")                ( strf::hex(0x100000000LL) );
    TEST("1000000000000000")         ( strf::hex(0x1000000000000000LL) );
    TEST("7")                        ( strf::oct(07) );
    TEST("77")                       ( strf::oct(077) );
    TEST("7777")                     ( strf::oct(07777) );
    TEST("777777777")                ( strf::oct(0777777777l) );
    TEST("7777777777777777")         ( strf::oct(07777777777777777LL) );
    TEST("777777777777777777777")    ( strf::oct(0777777777777777777777LL) );
    TEST("0")                        ( strf::oct(0) );
    TEST("1")                        ( strf::oct(01) );
    TEST("10")                       ( strf::oct(010) );
    TEST("100")                      ( strf::oct(0100) );
    TEST("10000")                    ( strf::oct(010000) );
    TEST("100000000")                ( strf::oct(0100000000l) );
    TEST("10000000000000000")        ( strf::oct(010000000000000000LL) );
    TEST("1000000000000000000000")   ( strf::oct(01000000000000000000000LL) );
    TEST("1777777777777777777777")   ( strf::oct(01777777777777777777777LL) );

    TEST("0")                        ( strf::bin(0) );
    TEST("1")                        ( strf::bin(1) );
    TEST("11")                       ( strf::bin(3) );
    TEST("111")                      ( strf::bin(7) );
    TEST("1111")                     ( strf::bin(0xf) );
    TEST("11111111")                 ( strf::bin(0xff) );
    TEST("1010101010101010101010101010101010101010101010101010101010101010")
        ( strf::bin(0xaaaaaaaaaaaaaaaaLL) );
    TEST("1111111111111111111111111111111111111111111111111111111111111111")
        ( strf::bin(0xffffffffffffffffLL) );

    TEST("9")                    ( 9 );
    TEST("99")                   ( 99 );
    TEST("9999")                 ( 9999 );
    TEST("99999999")             ( 99999999l );
    TEST("999999999999999999")   ( 999999999999999999LL );
    TEST("-9")                   ( -9 );
    TEST("-99")                  ( -99 );
    TEST("-9999")                ( -9999 );
    TEST("-99999999")            ( -99999999l );
    TEST("-999999999999999999")  ( -999999999999999999LL );
    TEST("0")                    ( 0 );
    TEST("1")                    ( 1 );
    TEST("10")                   ( 10 );
    TEST("100")                  ( 100 );
    TEST("10000")                ( 10000 );
    TEST("100000000")            ( 100000000l );
    TEST("1000000000000000000")  ( 1000000000000000000LL );
    TEST("10000000000000000000") ( 10000000000000000000uLL );
    TEST("-1")                   ( -1 );
    TEST("-10")                  ( -10 );
    TEST("-100")                 ( -100 );
    TEST("-10000")               ( -10000 );
    TEST("-100000000")           ( -100000000l );
    TEST("-1000000000000000000") ( -1000000000000000000LL );

    // formatting characters:

    TEST ("_____1234567890") ( strf::right(1234567890l, 15, U'_') );

    TEST ("       123")  (  strf::right(123 , 10) );
    TEST (".......123")  (  strf::right(123 , 10, '.') );
    TEST ("......-123")  (  strf::right(-123, 10, '.') );
    TEST (".........0")  (  strf::right(0   , 10, '.') );
    TEST (".......123")  (  strf::right(123u, 10, '.') );
    TEST (".......123")  (  strf::right(123u, 10, '.').p(3) );
    TEST (".....00123")  (  strf::right(123 , 10, '.').p(5) );
    TEST ("....-00123")  (  strf::right(-123 , 10, '.').p(5) );
    TEST ("-000000123")  (  strf::right(-123 , 10, '.').p(9) );
    TEST ("0000000123")  (  strf::right(123 , 10, '.').p(10) );
    TEST ("000000000123")(  strf::right(123 , 10, '.').p(12) );

    TEST ("......+123")  ( +strf::right(123 , 10, '.') );
    TEST ("......-123")  ( +strf::right(-123, 10, '.') );
    TEST ("........+0")  ( +strf::right(0   , 10, '.') );
    TEST (".......123")  (  strf::right(123u, 10, '.') );

    TEST (".......123")  (  strf::split(123,  10, '.') );
    TEST ("+......123")  ( +strf::split(123,  10, '.') );
    TEST ("-......123")  ( +strf::split(-123, 10, '.') );
    TEST ("+........0")  ( +strf::split(0,    10, '.') );
    TEST (".........0")  (  strf::split(0,    10, '.') );
    TEST (".......123")  (  strf::split(123u, 10, '.') );
    TEST ("+.....0123")  ( +strf::split(123,  10, '.').p(4) );
    TEST ("+000000123")  ( +strf::split(123,  10, '.').p(9) );
    TEST ("+0000000123") ( +strf::split(123,  10, '.').p(10) );


    TEST ("123.......")  (  strf::left(123,  10, '.') );
    TEST ("+123......")  ( +strf::left(123,  10, '.') );
    TEST ("-123......")  ( +strf::left(-123, 10, '.') );
    TEST ("+0........")  ( +strf::left(0,    10, '.') );
    TEST ("0.........")  (  strf::left(0,    10, '.') );
    TEST ("123.......")  (  strf::left(123u, 10, '.') );

    TEST ("...123....")  (  strf::center(123,  10, '.') );
    TEST ("...+123...")  ( +strf::center(123,  10, '.') );
    TEST ("...-123...")  ( +strf::center(-123, 10, '.') );
    TEST ("....+0....")  ( +strf::center(0,    10, '.') );
    TEST ("....0.....")  (  strf::center(0,    10, '.') );
    TEST ("...123....")  (  strf::center(123u, 10, '.') );

    // hexadecimal case
    //TEST("0X1234567890ABCDEF") ( ~strf::uphex(0x1234567890abcdefLL) );
    TEST("0x1234567890abcdef") ( ~strf::hex(0x1234567890abcdefLL) );

    // hexadecimal aligment

    TEST("        aa")   (  strf::hex(0xAA)>10 );
    TEST("      0xaa")   ( ~strf::hex(0xAA)>10 );
    TEST("aa        ")   (  strf::hex(0xAA)<10 );
    TEST("0xaa      ")   ( ~strf::hex(0xAA)<10 );
    TEST("        aa")   (  strf::hex(0xAA)%10 );
    TEST("0x      aa")   ( ~strf::hex(0xAA)%10 );
    TEST("    aa    ")   (  strf::hex(0xAA)^10 );
    TEST("   0xaa   ")   ( ~strf::hex(0xAA)^10 );

    TEST("     000aa")   (  strf::hex(0xAA).p(5)>10 );
    TEST("   0x000aa")   ( ~strf::hex(0xAA).p(5)>10 );
    TEST("000aa     ")   (  strf::hex(0xAA).p(5)<10 );
    TEST("0x000aa   ")   ( ~strf::hex(0xAA).p(5)<10 );
    TEST("     000aa")   (  strf::hex(0xAA).p(5)%10 );
    TEST("0x   000aa")   ( ~strf::hex(0xAA).p(5)%10 );
    TEST("  000aa   ")   (  strf::hex(0xAA).p(5)^10 );
    TEST(" 0x000aa  ")   ( ~strf::hex(0xAA).p(5)^10 );

    TEST("00000000aa")   (  strf::hex(0xAA).p(10)>10 );
    TEST("0x000000aa")   ( ~strf::hex(0xAA).p(8)>10 );
    TEST("00000000aa")   (  strf::hex(0xAA).p(10)<10 );
    TEST("0x000000aa")   ( ~strf::hex(0xAA).p(8)<10 );
    TEST("00000000aa")   (  strf::hex(0xAA).p(10)%10 );
    TEST("0x000000aa")   ( ~strf::hex(0xAA).p(8)%10 );
    TEST("00000000aa")   (  strf::hex(0xAA).p(10)^10 );
    TEST("0x000000aa")   ( ~strf::hex(0xAA).p(8)^10 );

    TEST("000000000aa")   (  strf::hex(0xAA).p(11)>10 );
    TEST("0x0000000aa")   ( ~strf::hex(0xAA).p(9)>10 );
    TEST("000000000aa")   (  strf::hex(0xAA).p(11)<10 );
    TEST("0x0000000aa")   ( ~strf::hex(0xAA).p(9)<10 );
    TEST("000000000aa")   (  strf::hex(0xAA).p(11)%10 );
    TEST("0x0000000aa")   ( ~strf::hex(0xAA).p(9)%10 );
    TEST("000000000aa")   (  strf::hex(0xAA).p(11)^10 );
    TEST("0x0000000aa")   ( ~strf::hex(0xAA).p(9)^10 );

    // binary aligment

    TEST("        11")   (  strf::bin(3)>10 );
    TEST("      0b11")   ( ~strf::bin(3)>10 );
    TEST("11        ")   (  strf::bin(3)<10 );
    TEST("0b11      ")   ( ~strf::bin(3)<10 );
    TEST("        11")   (  strf::bin(3)%10 );
    TEST("0b      11")   ( ~strf::bin(3)%10 );
    TEST("    11    ")   (  strf::bin(3)^10 );
    TEST("   0b11   ")   ( ~strf::bin(3)^10 );

    TEST("     00011")   (  strf::bin(3).p(5)>10 );
    TEST("   0b00011")   ( ~strf::bin(3).p(5)>10 );
    TEST("00011     ")   (  strf::bin(3).p(5)<10 );
    TEST("0b00011   ")   ( ~strf::bin(3).p(5)<10 );
    TEST("     00011")   (  strf::bin(3).p(5)%10 );
    TEST("0b   00011")   ( ~strf::bin(3).p(5)%10 );
    TEST("  00011   ")   (  strf::bin(3).p(5)^10 );
    TEST(" 0b00011  ")   ( ~strf::bin(3).p(5)^10 );

    TEST("0000000011")   (  strf::bin(3).p(10)>10 );
    TEST("0b00000011")   ( ~strf::bin(3).p(8)>10 );
    TEST("0000000011")   (  strf::bin(3).p(10)<10 );
    TEST("0b00000011")   ( ~strf::bin(3).p(8)<10 );
    TEST("0000000011")   (  strf::bin(3).p(10)%10 );
    TEST("0b00000011")   ( ~strf::bin(3).p(8)%10 );
    TEST("0000000011")   (  strf::bin(3).p(10)^10 );
    TEST("0b00000011")   ( ~strf::bin(3).p(8)^10 );

    TEST("00000000011")   (  strf::bin(3).p(11)>10 );
    TEST("0b000000011")   ( ~strf::bin(3).p(9)>10 );
    TEST("00000000011")   (  strf::bin(3).p(11)<10 );
    TEST("0b000000011")   ( ~strf::bin(3).p(9)<10 );
    TEST("00000000011")   (  strf::bin(3).p(11)%10 );
    TEST("0b000000011")   ( ~strf::bin(3).p(9)%10 );
    TEST("00000000011")   (  strf::bin(3).p(11)^10 );
    TEST("0b000000011")   ( ~strf::bin(3).p(9)^10 );

    // octadecimal aligment

    TEST("        77")   (  strf::oct(077)>10 );
    TEST("       077")   ( ~strf::oct(077)>10 );
    TEST("77        ")   (  strf::oct(077)<10 );
    TEST("077       ")   ( ~strf::oct(077)<10 );
    TEST("        77")   (  strf::oct(077)%10 );
    TEST("0       77")   ( ~strf::oct(077)%10 );
    TEST("    77    ")   (  strf::oct(077)^10 );
    TEST("   077    ")   ( ~strf::oct(077)^10 );

    TEST("      0077")   (  strf::oct(077).p(4)>10 );
    TEST("     00077")   ( ~strf::oct(077).p(4)>10 );
    TEST("0077      ")   (  strf::oct(077).p(4)<10 );
    TEST("00077     ")   ( ~strf::oct(077).p(4)<10 );
    TEST("      0077")   (  strf::oct(077).p(4)%10 );
    TEST("0     0077")   ( ~strf::oct(077).p(4)%10 );
    TEST("   0077   ")   (  strf::oct(077).p(4)^10 );
    TEST("  00077   ")   ( ~strf::oct(077).p(4)^10 );

    // showpos in octa, binary and hexadecimal must not have any effect

    TEST("aa") ( +strf::hex(0xAA) );
    TEST("11") ( +strf::bin(3) );
    TEST("77") ( +strf::oct(077) );


    // inside joins

    TEST("     123")   ( strf::join_right(8)(123) );
    TEST("...123~~")   ( strf::join_right(8, '.')(strf::left(123, 5, U'~')) );
    TEST(".....123")   ( strf::join_right(8, '.')(strf::left(123, 3, U'~')) );
    TEST(".....123")   ( strf::join_right(8, '.')(strf::left(123, 2, U'~')) );

    TEST("123")    ( strf::join_right(3)(123) );
    TEST("123~~")  ( strf::join_right(5, '.')(strf::left(123, 5, U'~')) );
    TEST("123")    ( strf::join_right(3, '.')(strf::left(123, 3, U'~')) );
    TEST("123")    ( strf::join_right(3, '.')(strf::left(123, 2, U'~')) );
    TEST("123")    ( strf::join_right(2)(123) );

    TEST("123~~")  ( strf::join_right(4, '.')(strf::left(123, 5, U'~')) );
    TEST("123")    ( strf::join_right(2, '.')(strf::left(123, 3, U'~')) );
    TEST("123")    ( strf::join_right(2, '.')(strf::left(123, 2, U'~')) );

    TEST("   00123")   ( strf::join_right(8)(strf::fmt(123).p(5)));
    TEST("..00123~~~")  ( strf::join_right(10, '.')(strf::left(123, 8, U'~').p(5)) );
    TEST(".....00123")  ( strf::join_right(10, '.')(strf::left(123, 5, U'~').p(5)) );

    TEST("00123~~")  ( strf::join_right(7, '.')(strf::left(123, 7, U'~').p(5)) );
    TEST("00123")    ( strf::join_right(5, '.')(strf::left(123, 5, U'~').p(5)) );
    TEST("00123")    ( strf::join_right(5, '.')(strf::left(123, 3, U'~').p(5)) );

    {
        auto punct = strf::monotonic_grouping<10>{3};

        TEST("0").with(punct) (0);
        TEST("1,000").with(punct) (1000);
        TEST("   1,000").with(punct) (strf::join_right(8)(1000ul));
        TEST("-1,000").with(punct) (-1000);

        TEST("       0").with(punct) (strf::right(0, 8));
        TEST("     100").with(punct) (strf::right(100, 8));
        TEST("   1,000").with(punct) (strf::right(1000, 8));
        TEST("   00000000001,000").with(punct) (strf::right(1000,18).p(14));
        TEST("    1000").with(punct) (strf::hex(0x1000) > 8);

        TEST("       0").with(punct) ( strf::join_right(8)(strf::dec(0)) );
        TEST("     100").with(punct) ( strf::join_right(8)(strf::dec(100)) );
        TEST("   1,000").with(punct) ( strf::join_right(8)(strf::dec(1000)) );
        TEST("    1000").with(punct) ( strf::join_right(8)(strf::hex(0x1000)) );
    }

    {
        auto punct = strf::monotonic_grouping<10>{3}.thousands_sep(0x10FFFF);
        TEST(u8"  +1\U0010FFFF000").with(punct) (+strf::right(1000, 8));
        TEST(u8"  +1\U0010FFFF000").with(punct) (strf::join_right(8)(+strf::dec(1000)));
        TEST(u8"----+1\U0010FFFF000").with(punct) (strf::join_right(8)(u8"----", +strf::dec(1000)));
    }

    {
        auto punct = strf::monotonic_grouping<16>{3}.thousands_sep('\'');

        TEST("     0x0").with(punct) (~strf::hex(0x0) > 8);
        TEST("   0x100").with(punct) (~strf::hex(0x100) > 8);
        TEST(" 0x1'000").with(punct) (~strf::hex(0x1000) > 8);
        TEST("   1'000").with(punct) ( strf::hex(0x1000) > 8);

        TEST("     0x0").with(punct) ( strf::join_right(8)(~strf::hex(0x0)) );
        TEST("   0x100").with(punct) ( strf::join_right(8)(~strf::hex(0x100)) );
        TEST(" 0x1'000").with(punct) ( strf::join_right(8)(~strf::hex(0x1000)) );

        TEST("     0x0").with(punct) ( strf::join_right(8)(~strf::hex(0x0)) );
        TEST("   0x100").with(punct) ( strf::join_right(8)(~strf::hex(0x100)) );
        TEST(" 0x1'000").with(punct) ( strf::join_right(8)(~strf::hex(0x1000)) );
    }
   {
        auto punct = strf::monotonic_grouping<2>{3}.thousands_sep('\'');

        TEST("     0b0").with(punct) (~strf::bin(0) > 8);
        TEST("   0b100").with(punct) (~strf::bin(4) > 8);
        TEST(" 0b1'000").with(punct) (~strf::bin(8) > 8);
        TEST("   1'000").with(punct) ( strf::bin(8) > 8);

        TEST("     0b0").with(punct) ( strf::join_right(8)(~strf::bin(0)) );
        TEST("   0b100").with(punct) ( strf::join_right(8)(~strf::bin(4)) );
        TEST(" 0b1'000").with(punct) ( strf::join_right(8)(~strf::bin(8)) );

        TEST("     0b0").with(punct) ( strf::join_right(8)(~strf::bin(0)) );
        TEST("   0b100").with(punct) ( strf::join_right(8)(~strf::bin(4)) );
        TEST(" 0b1'000").with(punct) ( strf::join_right(8)(~strf::bin(8)) );
    }

    {
        auto punct = strf::monotonic_grouping<16>{3}.thousands_sep(0x10FFFF);
        TEST(u8" 0x1\U0010FFFF000").with(punct) (~strf::hex(0x1000) > 8);
        TEST(u8" 0x1\U0010FFFF000").with(punct) (strf::join_right(8)(~strf::hex(0x1000) > 8));
        TEST(u8"---0x1\U0010FFFF000").with(punct)
            (strf::join_right(8)(u8"---", ~strf::hex(0x1000)));
    }

    {
        TEST("1'7'7'7'7'7'7'7'7'7'7'7'7'7'7'7'7'7'7'7'7'7")
            .with(strf::monotonic_grouping<8>{1}.thousands_sep('\''))
            ( strf::oct(01777777777777777777777LL) );
    }
    {
        const auto* expected =
            u8"1\U0010FFFF" u8"7\U0010FFFF" u8"7\U0010FFFF" u8"7\U0010FFFF"
            u8"7\U0010FFFF" u8"7\U0010FFFF" u8"7\U0010FFFF" u8"7\U0010FFFF"
            u8"7\U0010FFFF" u8"7\U0010FFFF" u8"7\U0010FFFF" u8"7\U0010FFFF"
            u8"7\U0010FFFF" u8"7\U0010FFFF" u8"7\U0010FFFF" u8"7\U0010FFFF"
            u8"7\U0010FFFF" u8"7\U0010FFFF" u8"7\U0010FFFF" u8"7\U0010FFFF"
            u8"7\U0010FFFF" u8"7";

        TEST(expected)
            .with(strf::monotonic_grouping<8>{1}.thousands_sep(0x10FFFF))
            ( strf::oct(01777777777777777777777LL) );
    }
    {
        auto punct = strf::monotonic_grouping<8>{3}.thousands_sep(0x10FFFF);
        TEST(u8"  01\U0010FFFF000").with(punct) (~strf::oct(01000) > 8);
        TEST(u8"  01\U0010FFFF000").with(punct) (strf::join_right(8)(~strf::oct(01000)));
        TEST(u8"----01\U0010FFFF000").with(punct) (strf::join_right(8)(u8"----", ~strf::oct(01000)));
    }

    TEST(u8"1\U0010FFFF" u8"1\U0010FFFF" u8"1\U0010FFFF" u8"1")
        .with(strf::monotonic_grouping<2>{1}.thousands_sep(0x10FFFF))
        ( strf::bin(0xF) );
    TEST(u8"1\U0010FFFF" u8"10101010\U0010FFFF" u8"10101010")
        .with(strf::monotonic_grouping<2>{8}.thousands_sep(0x10FFFF))
        ( strf::bin(0x1aaaa) );
    TEST("1'1'1'1")
        .with(strf::monotonic_grouping<2>{1}.thousands_sep('\''))
        ( strf::bin(0xF) );
    TEST("1'10101010'10101010")
        .with(strf::monotonic_grouping<2>{8}.thousands_sep('\''))
        ( strf::bin(0x1aaaa) );
    TEST(u8"1010\U0010FFFF" u8"1010\U0010FFFF" u8"1010\U0010FFFF"
         u8"1010\U0010FFFF" u8"1010\U0010FFFF" u8"1010\U0010FFFF"
         u8"1010\U0010FFFF" u8"1010\U0010FFFF" u8"1010\U0010FFFF"
         u8"1010\U0010FFFF" u8"1010\U0010FFFF" u8"1010\U0010FFFF"
         u8"1010\U0010FFFF" u8"1010\U0010FFFF" u8"1010\U0010FFFF" u8"1010")
        .with(strf::monotonic_grouping<2>{4}.thousands_sep(0x10FFFF))
        ( strf::bin(0xaaaaaaaaaaaaaaaaLL) );

    {
        // Invalid punctuation char. ( They shall be omitted ).

        TEST("9999")
            .with(strf::monotonic_grouping<10>{1}.thousands_sep(0xFFFFFF))
            ( 9999 );

        TEST("ffff")
            .with(strf::monotonic_grouping<10>{1}.thousands_sep(0xFFFFFF))
            ( strf::hex(0xFFFF) );

        TEST("7777")
        .with(strf::monotonic_grouping<10>{1}.thousands_sep(0xFFFFFF))
            ( strf::oct(07777) );

        TEST("1111")
        .with(strf::monotonic_grouping<2>{1}.thousands_sep(0xFFFFFF))
            ( strf::bin(0xF) );
    }

    return boost::report_errors();
}

