//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

void STRF_TEST_FUNC test_input_int()
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

    TEST ( "2147483647")   (2147483647);                //  INT32_MAX
    TEST ("-2147483648")   (-2147483647 - 1);           //  INT32_MIN
    TEST ( "4294967295")   (4294967295u);               // UINT32_MAX
    TEST ("-2147483648")   (-2147483647 - 1);           //  INT32_MIN
    TEST ( "4294967295")   (strf::fmt(4294967295u));    // UINT32_MAX

    TEST ( L"2147483647")   (2147483647);                //  INT32_MAX
    TEST (L"-2147483648")   (-2147483647 - 1);           //  INT32_MIN
    TEST ( L"4294967295")   (4294967295u);               // UINT32_MAX
    TEST (L"-2147483648")  (strf::fmt(-2147483647 - 1)); //  INT32_MIN
    TEST ( L"4294967295")   (strf::fmt(4294967295u));    // UINT32_MAX

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

    // decimal formatting

    TEST ("_____1234567890") ( strf::right(1234567890l, 15, U'_') );

    TEST ("       123")  (  strf::right(123 , 10) );
    TEST (".......123")  (  strf::right(123 , 10, '.') );
    TEST ("......-123")  (  strf::right(-123, 10, '.') );
    TEST (".........0")  (  strf::right(0   , 10, '.') );
    TEST (".......123")  (  strf::right(123u, 10, '.') );

    TEST ("......+123")  ( +strf::right(123 , 10, '.') );
    TEST ("......-123")  ( +strf::right(-123, 10, '.') );
    TEST ("........+0")  ( +strf::right(0   , 10, '.') );
    TEST (".......123")  (  strf::right(123u, 10, '.') );

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

    // decimal with pad0
    TEST (".........1")  (  strf::right(   1, 10, '.').pad0(1) );
    TEST ("....000123")  (  strf::right( 123, 10, '.').pad0(6) );
    TEST (".......123")  (  strf::right( 123, 10, '.').pad0(3) );
    TEST ("....+00123")  ( +strf::right( 123, 10, '.').pad0(6) );
    TEST ("....-00123")  (  strf::right(-123, 10, '.').pad0(6) );
    TEST ("......+123")  ( +strf::right( 123, 10, '.').pad0(4) );
    TEST ("......-123")  (  strf::right(-123, 10, '.').pad0(4) );
    TEST ("....+00000")  ( +strf::right(0   , 10, '.').pad0(6) );
    TEST ("........+0")  ( +strf::right(0   , 10, '.').pad0(2) );
    TEST (".......123")  (  strf::right(123u, 10, '.').pad0(3) );
    TEST ("....000123")  (  strf::right(123u, 10, '.').pad0(6) );

    TEST (".000000123")   (  strf::right(123u, 10, '.').pad0(9) );
    TEST (".000000123")   (  strf::right(123 , 10, '.').pad0(9) );
    TEST ("0000000123")   (  strf::right(123u, 10, '.').pad0(10) );
    TEST ("0000000123")   (  strf::right(123 , 10, '.').pad0(10) );
    TEST ("00000000123")  (  strf::right(123u, 10, '.').pad0(11) );
    TEST ("00000000123")  (  strf::right(123 , 10, '.').pad0(11) );

   {   // decimal with pad0 and precision and punctuation
        auto punct = strf::numpunct<10>{2}.thousands_sep(':');
        TEST("       1:23:45").with(punct)  (  strf::punct( 12345).pad0(7) > 14 );
        TEST("      +1:23:45").with(punct)  ( +strf::punct( 12345).pad0(8) > 14 );
        TEST("      -1:23:45").with(punct)  (  strf::punct(-12345).pad0(8) > 14 );
        TEST("      01:23:45").with(punct)  (  strf::punct( 12345).pad0(8) > 14 );
        TEST("     +01:23:45").with(punct)  ( +strf::punct( 12345).pad0(9) > 14 );

        TEST("       1:23:45").with(punct)  (  strf::punct( 12345).p(5) > 14 );
        TEST("      -1:23:45").with(punct)  ( +strf::punct(-12345).p(5) > 14 );
        TEST("      +1:23:45").with(punct)  ( +strf::punct( 12345).p(5) > 14 );
        TEST("      01:23:45").with(punct)  (  strf::punct( 12345).p(6) > 14 );
        TEST("     +01:23:45").with(punct)  ( +strf::punct( 12345).p(6) > 14 );

        TEST("       1:23:45").with(punct)  (  strf::punct(12345).pad0(7).p(5) > 14 );
        TEST("      01:23:45").with(punct)  (  strf::punct(12345).pad0(7).p(6) > 14 );
        TEST("      01:23:45").with(punct)  (  strf::punct(12345).pad0(8).p(5) > 14 );

        TEST("      +1:23:45").with(punct)  ( +strf::punct( 12345).pad0(8).p(5) > 14 );
        TEST("      -1:23:45").with(punct)  ( +strf::punct(-12345).pad0(8).p(5) > 14 );
        TEST("     +01:23:45").with(punct)  ( +strf::punct( 12345).pad0(8).p(6) > 14 );
        TEST("     +01:23:45").with(punct)  ( +strf::punct( 12345).pad0(9).p(5) > 14 );
        TEST("     -01:23:45").with(punct)  (  strf::punct(-12345).pad0(9).p(5) > 14 );

        auto j = strf::join_right(14);
        TEST("       1:23:45").with(punct)  ( j( strf::punct( 12345).pad0(7)) );
        TEST("      +1:23:45").with(punct)  ( j(+strf::punct( 12345).pad0(8)) );
        TEST("      -1:23:45").with(punct)  ( j( strf::punct(-12345).pad0(8)) );
        TEST("      01:23:45").with(punct)  ( j( strf::punct( 12345).pad0(8)) );
        TEST("     +01:23:45").with(punct)  ( j(+strf::punct( 12345).pad0(9)) );

        TEST("       1:23:45").with(punct)  ( j( strf::punct( 12345).p(5)) );
        TEST("      -1:23:45").with(punct)  ( j(+strf::punct(-12345).p(5)) );
        TEST("      +1:23:45").with(punct)  ( j(+strf::punct( 12345).p(5)) );
        TEST("      01:23:45").with(punct)  ( j( strf::punct( 12345).p(6)) );
        TEST("     +01:23:45").with(punct)  ( j(+strf::punct( 12345).p(6)) );

        TEST("       1:23:45").with(punct)  ( j( strf::punct(12345).pad0(7).p(5)) );
        TEST("      01:23:45").with(punct)  ( j( strf::punct(12345).pad0(7).p(6)) );
        TEST("      01:23:45").with(punct)  ( j( strf::punct(12345).pad0(8).p(5)) );

        TEST("      +1:23:45").with(punct)  ( j(+strf::punct( 12345).pad0(8).p(5)) );
        TEST("      -1:23:45").with(punct)  ( j(+strf::punct(-12345).pad0(8).p(5)) );
        TEST("     +01:23:45").with(punct)  ( j(+strf::punct( 12345).pad0(8).p(6)) );
        TEST("     +01:23:45").with(punct)  ( j(+strf::punct( 12345).pad0(9).p(5)) );
        TEST("     -01:23:45").with(punct)  ( j( strf::punct(-12345).pad0(9).p(5)) );

        // fill_sign
        TEST("  ***-12345***") (j(strf::center(-12345, 12, '*').fill_sign()));
        TEST("  ****12345***") (j(strf::center(12345, 12, '*').fill_sign()));
        TEST("  ***1:23:45**").with(punct) (j(!strf::center(12345, 12, '*').fill_sign()));
        TEST("  ***1:23:45**").with(punct) (j(!strf::center(12345, 12, '*').fill_sign().pad0(8)));
        TEST("  **01:23:45**").with(punct) (j(!strf::center(12345, 12, '*').fill_sign().pad0(9)));
        TEST("  *001:23:45**").with(punct) (j(!strf::center(12345, 12, '*').pad0(9)));

        TEST("  *01:23:45***").with(punct) (j(!strf::left(12345, 12, '*').fill_sign().pad0(9)));
        TEST("  ****01:23:45").with(punct) (j(!strf::right(12345, 12, '*').fill_sign().pad0(9)));
        TEST("  *00001:23:45").with(punct) (j(!strf::left(12345, 0, '*').fill_sign().pad0(12)));
    }

    // hexadecimal letter case
    TEST("0X1234567890ABCDEF").with(strf::lettercase::upper) ( *strf::hex(0x1234567890abcdefLL) );
    TEST("0x1234567890ABCDEF").with(strf::lettercase::mixed) ( *strf::hex(0x1234567890abcdefLL) );
    TEST("0x1234567890abcdef").with(strf::lettercase::lower) ( *strf::hex(0x1234567890abcdefLL) );

    // binary letter case
    TEST("0B111").with(strf::lettercase::upper) ( *strf::bin(7) );
    TEST("0b111").with(strf::lettercase::mixed) ( *strf::bin(7) );
    TEST("0b111").with(strf::lettercase::lower) ( *strf::bin(7) );

    // hexadecimal aligment
    TEST("        aa")   (  strf::hex(0xAA)>10 );
    TEST("      0xaa")   ( *strf::hex(0xAA)>10 );
    TEST("aa        ")   (  strf::hex(0xAA)<10 );
    TEST("0xaa      ")   ( *strf::hex(0xAA)<10 );
    TEST("    aa    ")   (  strf::hex(0xAA)^10 );
    TEST("   0xaa   ")   ( *strf::hex(0xAA)^10 );

    TEST("     000aa")   (  strf::hex(0xAA).p(5)>10 );
    TEST("   0x000aa")   ( *strf::hex(0xAA).p(5)>10 );
    TEST("000aa     ")   (  strf::hex(0xAA).p(5)<10 );
    TEST("0x000aa   ")   ( *strf::hex(0xAA).p(5)<10 );
    TEST("  000aa   ")   (  strf::hex(0xAA).p(5)^10 );
    TEST(" 0x000aa  ")   ( *strf::hex(0xAA).p(5)^10 );

    // // hexadecimal with pad0
    // TEST("        aa")   (  strf::hex(0xAA).pad0(2) >10 );
    // TEST("      0xaa")   ( *strf::hex(0xAA).pad0(4) >10 );
    // TEST("       0aa")   (  strf::hex(0xAA).pad0(3) >10 );
    // TEST("     0x0aa")   ( *strf::hex(0xAA).pad0(5) >10 );
    // TEST("00000000aa")   (  strf::hex(0xAA).pad0(10) >10 );
    // TEST("0x000000aa")   ( *strf::hex(0xAA).pad0(10) >10 );
    // TEST("000000000aa")  (  strf::hex(0xAA).pad0(11) >10 );
    // TEST("0x0000000aa")  ( *strf::hex(0xAA).pad0(11) >10 );

    // // hexadecimal with precision
    // TEST("        aa")   (  strf::hex(0xAA).p(2) >10 );
    // TEST("      0xaa")   ( *strf::hex(0xAA).p(1) >10 );
    // TEST("       0aa")   (  strf::hex(0xAA).p(3) >10 );
    // TEST("     0x0aa")   ( *strf::hex(0xAA).p(3) >10 );
    // TEST("00000000aa")   (  strf::hex(0xAA).p(10) >10 );
    // TEST("0x000000aa")   ( *strf::hex(0xAA).p(8)  >10 );
    // TEST("000000000aa")  (  strf::hex(0xAA).p(11) >10 );
    // TEST("0x0000000aa")  ( *strf::hex(0xAA).p(9)  >10 );

    // // hexadecimal with pad0 and precision
    // TEST("        aa")   (  strf::hex(0xAA).pad0(2).p(2) >10 );
    // TEST("       0aa")   (  strf::hex(0xAA).pad0(2).p(3) >10 );
    // TEST("       0aa")   (  strf::hex(0xAA).pad0(3).p(2) >10 );
    // TEST("      00aa")   (  strf::hex(0xAA).pad0(3).p(4) >10 );
    // TEST("      00aa")   (  strf::hex(0xAA).pad0(4).p(3) >10 );

    {   // hexadecimal with pad0 and precision and punctuation
        auto punct = strf::numpunct<16>{2}.thousands_sep(':');
        TEST("       1:23:45").with(punct)  ( (!strf::hex(0x12345)).pad0(7) > 14 );
        TEST("     0x1:23:45").with(punct)  ( *!strf::hex(0x12345).pad0(9) > 14 );
        TEST("      01:23:45").with(punct)  ( (!strf::hex(0x12345)).pad0(8) > 14 );
        TEST("    0x01:23:45").with(punct)  ( *!strf::hex(0x12345).pad0(10) > 14 );

        TEST("       1:23:45").with(punct)  ( (!strf::hex(0x12345)).p(5) > 14 );
        TEST("     0x1:23:45").with(punct)  ( *!strf::hex(0x12345).p(5) > 14 );
        TEST("      01:23:45").with(punct)  ( (!strf::hex(0x12345)).p(6) > 14 );
        TEST("    0x01:23:45").with(punct)  ( *!strf::hex(0x12345).p(6) > 14 );

        TEST("       1:23:45").with(punct)  ( (!strf::hex(0x12345)).pad0(7).p(5) > 14 );
        TEST("      01:23:45").with(punct)  ( (!strf::hex(0x12345)).pad0(7).p(6) > 14 );
        TEST("      01:23:45").with(punct)  ( (!strf::hex(0x12345)).pad0(8).p(5) > 14 );

        TEST("     0x1:23:45").with(punct)  ( *!strf::hex(0x12345).pad0(9).p(5) > 14 );
        TEST("    0x01:23:45").with(punct)  ( *!strf::hex(0x12345).pad0(7).p(6) > 14 );
        TEST("    0x01:23:45").with(punct)  ( *!strf::hex(0x12345).pad0(10).p(5) > 14 );

        auto j = strf::join_right(14);
        TEST("       1:23:45").with(punct)  ( j( !strf::hex(0x12345).pad0(7)) );
        TEST("     0x1:23:45").with(punct)  ( j(*!strf::hex(0x12345).pad0(9)) );
        TEST("      01:23:45").with(punct)  ( j( !strf::hex(0x12345).pad0(8)) );
        TEST("    0x01:23:45").with(punct)  ( j(*!strf::hex(0x12345).pad0(10)) );

        TEST("       1:23:45").with(punct)  ( j( !strf::hex(0x12345).p(5)) );
        TEST("     0x1:23:45").with(punct)  ( j(*!strf::hex(0x12345).p(5)) );
        TEST("      01:23:45").with(punct)  ( j( !strf::hex(0x12345).p(6)) );
        TEST("    0x01:23:45").with(punct)  ( j(*!strf::hex(0x12345).p(6)) );

        TEST("       1:23:45").with(punct)  ( j( !strf::hex(0x12345).pad0(7).p(5)) );
        TEST("      01:23:45").with(punct)  ( j( !strf::hex(0x12345).pad0(7).p(6)) );
        TEST("      01:23:45").with(punct)  ( j( !strf::hex(0x12345).pad0(8).p(5)) );

        TEST("     0x1:23:45").with(punct)  ( j(*!strf::hex(0x12345).pad0( 9).p(5)) );
        TEST("    0x01:23:45").with(punct)  ( j(*!strf::hex(0x12345).pad0( 7).p(6)) );
        TEST("    0x01:23:45").with(punct)  ( j(*!strf::hex(0x12345).pad0(10).p(5)) );
    }

    // binary aligment
    TEST("        11")   (  strf::bin(3)>10 );
    TEST("      0b11")   ( *strf::bin(3)>10 );
    TEST("11        ")   (  strf::bin(3)<10 );
    TEST("0b11      ")   ( *strf::bin(3)<10 );
    TEST("    11    ")   (  strf::bin(3)^10 );
    TEST("   0b11   ")   ( *strf::bin(3)^10 );

    TEST("     00011")   (  strf::bin(3).p(5)>10 );
    TEST("   0b00011")   ( *strf::bin(3).p(5)>10 );
    TEST("00011     ")   (  strf::bin(3).p(5)<10 );
    TEST("0b00011   ")   ( *strf::bin(3).p(5)<10 );
    TEST("  00011   ")   (  strf::bin(3).p(5)^10 );
    TEST(" 0b00011  ")   ( *strf::bin(3).p(5)^10 );

    // binary with pad0
    TEST("        11")   (  strf::bin(3).pad0(2) >10 );
    TEST("      0b11")   ( *strf::bin(3).pad0(4) >10 );
    TEST("       011")   (  strf::bin(3).pad0(3) >10 );
    TEST("     0b011")   ( *strf::bin(3).pad0(5) >10 );
    TEST("0000000011")   (  strf::bin(3).pad0(10) >10 );
    TEST("0b00000011")   ( *strf::bin(3).pad0(10) >10 );
    TEST("00000000011")  (  strf::bin(3).pad0(11) >10 );
    TEST("0b000000011")  ( *strf::bin(3).pad0(11) >10 );

    // binary with precision
    TEST("        11")   (  strf::bin(3).p(2) >10 );
    TEST("      0b11")   ( *strf::bin(3).p(1) >10 );
    TEST("       011")   (  strf::bin(3).p(3) >10 );
    TEST("     0b011")   ( *strf::bin(3).p(3) >10 );
    TEST("0000000011")   (  strf::bin(3).p(10) >10 );
    TEST("0b00000011")   ( *strf::bin(3).p(8)  >10 );
    TEST("00000000011")  (  strf::bin(3).p(11) >10 );
    TEST("0b000000011")  ( *strf::bin(3).p(9)  >10 );

    // binary with pad0 and precision
    TEST("        11")   (  strf::bin(3).pad0(2).p(2) >10 );
    TEST("       011")   (  strf::bin(3).pad0(2).p(3) >10 );
    TEST("       011")   (  strf::bin(3).pad0(3).p(2) >10 );
    TEST("      0011")   (  strf::bin(3).pad0(3).p(4) >10 );
    TEST("      0011")   (  strf::bin(3).pad0(4).p(3) >10 );

    // octadecimal aligment

    TEST("        77")   (  strf::oct(077)>10 );
    TEST("       077")   ( *strf::oct(077)>10 );
    TEST("77        ")   (  strf::oct(077)<10 );
    TEST("077       ")   ( *strf::oct(077)<10 );
    TEST("    77    ")   (  strf::oct(077)^10 );
    TEST("   077    ")   ( *strf::oct(077)^10 );

    TEST("      0077")   (  strf::oct(077).p(4)>10 );
    TEST("      0077")   ( *strf::oct(077).p(4)>10 );
    TEST("      1234")   (  strf::oct(01234).p(4)>10 );
    TEST("     01234")   ( *strf::oct(01234).p(4)>10 );

    // octal with pad0
    TEST("        12")   (  strf::oct(012).pad0(2) >10 );
    TEST("       012")   (  strf::oct(012).pad0(3) >10 );
    TEST("       012")   ( *strf::oct(012).pad0(3) >10 );
    TEST("     00012")   ( *strf::oct(012).pad0(5) >10 );
    TEST("0000000012")   (  strf::oct(012).pad0(10) >10 );
    TEST("0000000012")   ( *strf::oct(012).pad0(10) >10 );
    TEST("00000000012")  (  strf::oct(012).pad0(11) >10 );
    TEST("00000000012")  ( *strf::oct(012).pad0(11) >10 );

    // octal with precision
    TEST("        12")   (  strf::oct(012).p(2) >10 );
    TEST("       012")   (  strf::oct(012).p(3) >10 );
    TEST("       012")   ( *strf::oct(012).p(3) >10 );
    TEST("     00012")   ( *strf::oct(012).p(5) >10 );
    TEST("0000000012")   (  strf::oct(012).p(10) >10 );
    TEST("0000000012")   ( *strf::oct(012).p(10) >10 );
    TEST("00000000012")  (  strf::oct(012).p(11) >10 );
    TEST("00000000012")  ( *strf::oct(012).p(11) >10 );

    // octal with pad0 and precision
    TEST("        12")   (  strf::oct(012).pad0(2).p(2) >10 );
    TEST("       012")   (  strf::oct(012).pad0(2).p(3) >10 );
    TEST("       012")   (  strf::oct(012).pad0(3).p(2) >10 );
    TEST("       012")   ( *strf::oct(012).pad0(2).p(3) >10 );
    TEST("       012")   ( *strf::oct(012).pad0(3).p(2) >10 );
    TEST("      0012")   ( *strf::oct(012).pad0(3).p(4) >10 );
    TEST("      0012")   ( *strf::oct(012).pad0(4).p(3) >10 );

    {   // octal with pad0 and precision and punctuation
        auto punct = strf::numpunct<8>{2}.thousands_sep(':');
        TEST("    1:23:45:67").with(punct)  (  (!strf::oct(01234567)).pad0(10) > 14 );
        TEST("   01:23:45:67").with(punct)  (  (!strf::oct(01234567)).pad0(11) > 14 );
        TEST("   01:23:45:67").with(punct)  ( *(!strf::oct(01234567)).pad0(11) > 14 );
        TEST("   01:23:45:67").with(punct)  (  (!strf::oct(01234567)).p(8) > 14 );
        TEST("   01:23:45:67").with(punct)  ( *(!strf::oct(01234567)).p(8) > 14 );
        TEST("  001:23:45:67").with(punct)  (  (!strf::oct(01234567)).pad0(11).p(9) > 14 );
        TEST("  001:23:45:67").with(punct)  (  (!strf::oct(01234567)).pad0(12).p(8) > 14 );
        TEST("  001:23:45:67").with(punct)  ( *(!strf::oct(01234567)).pad0(11).p(9) > 14 );
        TEST("  001:23:45:67").with(punct)  ( *(!strf::oct(01234567)).pad0(12).p(8) > 14 );
    }

    // *oct(0) should be printed as "0"
    TEST("0")      (*strf::oct(0));
    TEST("0")      (*strf::oct(0).p(1));
    TEST("    0")  (*strf::oct(0) > 5);
    TEST("    0")  (*strf::oct(0).p(1) > 5);

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
        auto punct = strf::numpunct<10>{3};

        TEST("0").with(punct) (strf::punct(0));
        TEST("1,000").with(punct) (strf::punct(1000));
        TEST("   1,000").with(punct) (strf::join_right(8)(strf::punct(1000ul)));
        TEST("-1,000").with(punct) (strf::punct(-1000));

        TEST("       0").with(punct) (!strf::right(0, 8));
        TEST("     100").with(punct) (!strf::right(100, 8));
        TEST("   1,000").with(punct) (!strf::right(1000, 8));
        TEST("   00000000001,000").with(punct) (!strf::right(1000,18).p(14));
        TEST("    1000").with(punct) ((!strf::hex(0x1000)) > 8);

        TEST("       0").with(punct) ( strf::join_right(8)(!strf::dec(0)) );
        TEST("     100").with(punct) ( strf::join_right(8)(!strf::dec(100)) );
        TEST("   1,000").with(punct) ( strf::join_right(8)(!strf::dec(1000)) );
        TEST("    1000").with(punct) ( strf::join_right(8)(!strf::hex(0x1000)) );
    }
    {
        using punct = strf::numpunct<10>;
        TEST("0") .with(punct(1)) (strf::punct(0));
        TEST("9") .with(punct(1)) (strf::punct(9));
        TEST("9,9") .with(punct(1)) (strf::punct(99));
        TEST("9,9,9") .with(punct(1)) (strf::punct(999));
        TEST("99999") .with(punct(-1)) (strf::punct(99999));

        TEST("1,2345") .with(punct(4, 3, 2)) (strf::punct(12345));
        TEST("12,3456") .with(punct(4, 2, 1)) (strf::punct(123456));

        TEST("1") .with(punct(3, 2)) (strf::punct(1));
        TEST("12") .with(punct(3, 2)) (strf::punct(12));
        TEST("123") .with(punct(3, 2)) (strf::punct(123));
        TEST("1,234") .with(punct(3, 2)) (strf::punct(1234));

        TEST("12,345") .with(punct(3, 2)) (strf::punct(12345));
        TEST("98765,4,321") .with(punct(3, 1, -1)) (strf::punct(987654321));
        TEST("9,87,65,4,321") .with(punct(3, 1, 2)) (strf::punct(987654321));
        TEST("987654,3,21") .with(punct(2, 1, -1)) (strf::punct(987654321));
        TEST("9876,5,4,321") .with(punct(3, 1, 1, -1)) (strf::punct(987654321));
        TEST("9,8,7,6,5,4,3,21") .with(punct(2, 1, 1)) (strf::punct(987654321));
        TEST("1,23") .with(punct(2, 1))  (strf::punct(123));
        TEST("1,234") .with(punct(3, 1))  (strf::punct(1234));

        TEST("18,446,744,073,709,551,615")           .with(punct(3))
            (strf::punct(18446744073709551615ull));
        TEST("8,446,744,073,709,551,615")            .with(punct(3))
            (strf::punct(8446744073709551615ull));
        TEST("446,744,073,709,551,615")              .with(punct(3))
            (strf::punct(446744073709551615ull));
        TEST("18,446,744,073,709,551,61,5")          .with(punct(1, 2, 3))
            (strf::punct(18446744073709551615ull));
        TEST("18446744073709551615")                 .with(punct(-1))
            (strf::punct(18446744073709551615ull));
        TEST("18446744073709,5,51,615")              .with(punct(3, 2, 1, -1))
            (strf::punct(18446744073709551615ull));
        TEST("1,8,4,4,6,7,4,4,0,7,3,7,0,9,5,51,615") .with(punct(3, 2, 1))
            (strf::punct(18446744073709551615ull));
        TEST("1,8,4,4,6,7,4,4,0,7,3,7,09,551,615")   .with(punct(3, 3, 2, 1))
            (strf::punct(18446744073709551615ull));
        TEST("18,44,67,44,07,37,09,55,1615")         .with(punct(4, 2))
            (strf::punct(18446744073709551615ull));
        TEST("1,84,46,74,40,73,70,95,51,615")        .with(punct(3, 2))
            (strf::punct(18446744073709551615ull));

        auto punct2 = [](auto... grps) -> punct
            { return punct{grps...}.thousands_sep(0x10FFFF); };

        TEST(u8"18\U0010FFFF" u8"446\U0010FFFF" u8"744\U0010FFFF" u8"073\U0010FFFF"
             u8"709\U0010FFFF" u8"551\U0010FFFF" u8"61\U0010FFFF" u8"5")
            .with(punct2(1, 2, 3)) (strf::punct(18446744073709551615ull));

        TEST(u8"18446744073709551615")
            .with(punct2(-1)) (strf::punct(18446744073709551615ull));

        TEST(u8"18446744073709\U0010FFFF" u8"5\U0010FFFF" u8"51\U0010FFFF" u8"615")
            .with(punct2(3, 2, 1, -1)) (strf::punct(18446744073709551615ull));

        TEST(u8"1\U0010FFFF" u8"8\U0010FFFF" u8"4\U0010FFFF" u8"4\U0010FFFF"
             u8"6\U0010FFFF" u8"7\U0010FFFF" u8"4\U0010FFFF" u8"4\U0010FFFF"
             u8"0\U0010FFFF" u8"7\U0010FFFF" u8"3\U0010FFFF" u8"7\U0010FFFF"
             u8"0\U0010FFFF" u8"9\U0010FFFF" u8"5\U0010FFFF" u8"51\U0010FFFF" u8"615")
            .with(punct2(3, 2, 1)) (strf::punct(18446744073709551615ull));

        TEST(u8"1\U0010FFFF" u8"8\U0010FFFF" u8"4\U0010FFFF" u8"4\U0010FFFF"
             u8"6\U0010FFFF"
             u8"7\U0010FFFF" u8"4\U0010FFFF" u8"4\U0010FFFF" u8"0\U0010FFFF"
             u8"7\U0010FFFF"
             u8"3\U0010FFFF" u8"7\U0010FFFF" u8"09\U0010FFFF" u8"551\U0010FFFF" u8"615")
            .with(punct2(3, 3, 2, 1)) (strf::punct(18446744073709551615ull));

        TEST(u8"18\U0010FFFF" u8"44\U0010FFFF" u8"67\U0010FFFF" u8"44\U0010FFFF"
             u8"07\U0010FFFF"
             u8"37\U0010FFFF" u8"09\U0010FFFF" u8"55\U0010FFFF" u8"1615")
            .with(punct2(4, 2)) (strf::punct(18446744073709551615ull));

        TEST(u8"1\U0010FFFF" u8"84\U0010FFFF" u8"46\U0010FFFF" u8"74\U0010FFFF"
             u8"40\U0010FFFF" u8"73\U0010FFFF" u8"70\U0010FFFF" u8"95\U0010FFFF"
             u8"51\U0010FFFF" u8"615")
            .with(punct2(3, 2)) (strf::punct(18446744073709551615ull));
    }
    {
        using punct = strf::numpunct<16>;
        TEST("0") .with(punct(1)) (!strf::hex(0));
        TEST("f") .with(punct(1)) (!strf::hex(0xf));
        TEST("a,b") .with(punct(1)) (!strf::hex(0xab));
        TEST("a,b,c") .with(punct(1)) (!strf::hex(0xabc));

        TEST("123") .with(punct(3, 2)) (!strf::hex(0x123));
        TEST("1,234") .with(punct(3, 2)) (!strf::hex(0x1234));
        TEST("12,345") .with(punct(3, 2)) (!strf::hex(0x12345));
        TEST("1,23,456") .with(punct(3, 2)) (!strf::hex(0x123456));

        TEST("1,234,567,89a,bcd,ef0")       .with(punct(3))
            (!strf::hex(0x123456789abcdef0ull));
        TEST("123,456,789,abc,def")         .with(punct(3))
            (!strf::hex(0x123456789abcdefull));
        TEST("12,345,678,9ab,cde")          .with(punct(3))
            (!strf::hex(0x123456789abcdeull));
        TEST("1,234,567,89a,bcd,ef,0")      .with(punct(1, 2, 3))
            (!strf::hex(0x123456789abcdef0ull));
        TEST("123456789abcdef0")            .with(punct(-1))
            (!strf::hex(0x123456789abcdef0ull));
        TEST("123456789a,b,cd,ef0")         .with(punct(3, 2, 1, -1))
            (!strf::hex(0x123456789abcdef0ull));
        TEST("1,2,3,4,5,6,7,8,9,a,b,cd,ef0").with(punct(3, 2, 1))
            (!strf::hex(0x123456789abcdef0ull));
        TEST("1,2,3,4,5,6,7,8,9a,bcd,ef0")  .with(punct(3, 3, 2, 1))
            (!strf::hex(0x123456789abcdef0ull));
        TEST("12,34,56,78,9a,bc,def0")      .with(punct(4, 2))
            (!strf::hex(0x123456789abcdef0ull));
        TEST("1,23,45,67,89,ab,cd,ef0")     .with(punct(3, 2))
            (!strf::hex(0x123456789abcdef0ull));

        auto punct2 = [](auto... grps) -> punct
            { return punct(grps...).thousands_sep(0x10FFFF); };

        TEST(u8"1\U0010FFFF" u8"234\U0010FFFF" u8"567\U0010FFFF" u8"89a\U0010FFFF"
             u8"bcd\U0010FFFF" u8"ef0")
            .with(punct2(3))
            (!strf::hex(0x123456789abcdef0ull));

        TEST(u8"123\U0010FFFF" u8"456\U0010FFFF" u8"789\U0010FFFF" u8"abc\U0010FFFF"
             u8"def")
            .with(punct2(3))
            (!strf::hex(0x123456789abcdefull));

        TEST(u8"12\U0010FFFF" u8"345\U0010FFFF" u8"678\U0010FFFF"
             u8"9ab\U0010FFFF" u8"cde")
            .with(punct2(3))
            (!strf::hex(0x123456789abcdeull));

        TEST(u8"1\U0010FFFF" u8"234\U0010FFFF" u8"567\U0010FFFF"
             u8"89a\U0010FFFF" u8"bcd\U0010FFFF" u8"ef\U0010FFFF" u8"0")
            .with(punct2(1, 2, 3))
            (!strf::hex(0x123456789abcdef0ull));
    }
    {
        using punct = strf::numpunct<8>;
        TEST("0") .with(punct(1)) (!strf::oct(0));
        TEST("7") .with(punct(1)) (!strf::oct(07));
        TEST("1,2") .with(punct(1)) (!strf::oct(012));
        TEST("1,2,3") .with(punct(1)) (!strf::oct(0123));

        TEST("123") .with(punct(3, 2)) (!strf::oct(0123));
        TEST("1,234") .with(punct(3, 2)) (!strf::oct(01234));
        TEST("12,345") .with(punct(3, 2)) (!strf::oct(012345));
        TEST("1,23,456") .with(punct(3, 2)) (!strf::oct(0123456));

        TEST("1,234,567,123,456,712")       .with(punct(3))
            (!strf::oct(01234567123456712ull));
        TEST("123,456,712,345,671")         .with(punct(3))
            (!strf::oct(0123456712345671ull));
        TEST("12,345,671,234,567")          .with(punct(3))
            (!strf::oct(012345671234567ull));
        TEST("12,345,671,234,56,7")         .with(punct(1, 2, 3))
            (!strf::oct(012345671234567ull));
        TEST("1234567")                    .with(punct(-1))
            (!strf::oct(01234567ull));
        TEST("12345671,2,34,567")           .with(punct(3, 2, 1, -1))
            (!strf::oct(012345671234567ull));
        TEST("1,2,3,4,5,6,7,1,2,34,567").with(punct(3, 2, 1))   (!strf::oct(012345671234567ull));
        TEST("1,2,3,4,5,6,71,234,567")  .with(punct(3, 3, 2, 1))(!strf::oct(012345671234567ull));
        TEST("1,23,4567") .with(punct(4, 2)) (!strf::oct(01234567));
        TEST("12,34,567") .with(punct(3, 2)) (!strf::oct(01234567));
    }
    {
        auto punct = [](auto... grps) -> strf::numpunct<2>
            { return strf::numpunct<2>{grps...}.thousands_sep('\''); };

        TEST("0") .with(punct(1)) (!strf::bin(0));
        TEST("1'0") .with(punct(1)) (!strf::bin(2));
        TEST("1'0'1'0") .with(punct(1)) (!strf::bin(0xA));
        TEST("1010'1010'1010'10'10") .with(punct(2,2,4)) (!strf::bin(0xAAAA));
        TEST("10'1010'1010'10'10") .with(punct(2,2,4)) (!strf::bin(0x2AAA));
        TEST("1'0101'0101'01'01") .with(punct(2,2,4)) (!strf::bin(0x1555));

        auto grp_big = [](auto... grps) -> strf::numpunct<2>
            { return strf::numpunct<2>{grps...}.thousands_sep(0x10FFFF); };

        TEST(u8"0") .with(grp_big(1)) (!strf::bin(0));
        TEST(u8"1\U0010FFFF" u8"0") .with(grp_big(1)) (!strf::bin(2));
        TEST(u8"1\U0010FFFF" u8"0\U0010FFFF" u8"1\U0010FFFF" u8"0")
            .with(grp_big(1)) (!strf::bin(0xA));
        TEST(u8"1010\U0010FFFF" u8"1010\U0010FFFF" u8"1010\U0010FFFF"
             u8"10\U0010FFFF" u8"10")
            .with(grp_big(2,2,4)) (!strf::bin(0xAAAA));
        TEST(u8"10\U0010FFFF" u8"1010\U0010FFFF" u8"1010\U0010FFFF"
             u8"10\U0010FFFF" u8"10")
            .with(grp_big(2,2,4)) (!strf::bin(0x2AAA));
        TEST(u8"1\U0010FFFF" u8"0101\U0010FFFF" u8"0101\U0010FFFF"
             u8"01\U0010FFFF" u8"01")
            .with(grp_big(2,2,4)) (!strf::bin(0x1555));
    }
    {
        auto punct = strf::numpunct<10>{3}.thousands_sep(0x10FFFF);
        TEST(u8"  +1\U0010FFFF000").with(punct) (+!strf::right(1000, 8));
        TEST(u8"  +1\U0010FFFF000").with(punct) (strf::join_right(8)(+!strf::dec(1000)));
        TEST(u8"----+1\U0010FFFF000").with(punct) (strf::join_right(8)(u8"----", +!strf::dec(1000)));
    }
    {
        auto punct = strf::numpunct<16>{3}.thousands_sep('\'');

        TEST("     0x0").with(punct) (*!strf::hex(0x0) > 8);
        TEST("   0x100").with(punct) (*!strf::hex(0x100) > 8);
        TEST(" 0x1'000").with(punct) (*!strf::hex(0x1000) > 8);
        TEST("   1'000").with(punct) ((!strf::hex(0x1000)) > 8);

        TEST("     0x0").with(punct) ( strf::join_right(8)(*!strf::hex(0x0)) );
        TEST("   0x100").with(punct) ( strf::join_right(8)(*!strf::hex(0x100)) );
        TEST(" 0x1'000").with(punct) ( strf::join_right(8)(*!strf::hex(0x1000)) );

        TEST("     0x0").with(punct) ( strf::join_right(8)(*!strf::hex(0x0)) );
        TEST("   0x100").with(punct) ( strf::join_right(8)(*!strf::hex(0x100)) );
        TEST(" 0x1'000").with(punct) ( strf::join_right(8)(*!strf::hex(0x1000)) );
    }
    {
        auto punct = strf::numpunct<2>{3}.thousands_sep('\'');

        TEST("     0b0").with(punct) (*!strf::bin(0) > 8);
        TEST("   0b100").with(punct) (*!strf::bin(4) > 8);
        TEST(" 0b1'000").with(punct) (*!strf::bin(8) > 8);
        TEST("   1'000").with(punct) ((!strf::bin(8)) > 8);

        TEST("     0b0").with(punct) ( strf::join_right(8)(*!strf::bin(0)) );
        TEST("   0b100").with(punct) ( strf::join_right(8)(*!strf::bin(4)) );
        TEST(" 0b1'000").with(punct) ( strf::join_right(8)(*!strf::bin(8)) );

        TEST("     0b0").with(punct) ( strf::join_right(8)(*!strf::bin(0)) );
        TEST("   0b100").with(punct) ( strf::join_right(8)(*!strf::bin(4)) );
        TEST(" 0b1'000").with(punct) ( strf::join_right(8)(*!strf::bin(8)) );
    }

    {
        auto punct = strf::numpunct<16>{3}.thousands_sep(0x10FFFF);
        TEST(u8" 0x1\U0010FFFF000").with(punct) (*!strf::hex(0x1000) > 8);
        TEST(u8" 0x1\U0010FFFF000").with(punct) (strf::join_right(8)(*!strf::hex(0x1000) > 8));
        TEST(u8"---0x1\U0010FFFF000").with(punct)
            (strf::join_right(8)(u8"---", *!strf::hex(0x1000)));
    }

    {
        TEST("1'7'7'7'7'7'7'7'7'7'7'7'7'7'7'7'7'7'7'7'7'7")
            .with(strf::numpunct<8>{1}.thousands_sep('\''))
            ( !strf::oct(01777777777777777777777LL) );
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
            .with(strf::numpunct<8>{1}.thousands_sep(0x10FFFF))
            ( !strf::oct(01777777777777777777777LL) );
    }
    {
        auto punct = strf::numpunct<8>{3}.thousands_sep(0x10FFFF);
        TEST(u8"  01\U0010FFFF000").with(punct) (*!strf::oct(01000) > 8);
        TEST(u8"  01\U0010FFFF000").with(punct) (strf::join_right(8)(*!strf::oct(01000)));
        TEST(u8"----01\U0010FFFF000").with(punct) (strf::join_right(8)(u8"----", *!strf::oct(01000)));
    }

    TEST(u8"1\U0010FFFF" u8"1\U0010FFFF" u8"1\U0010FFFF" u8"1")
        .with(strf::numpunct<2>{1}.thousands_sep(0x10FFFF))
        ( !strf::bin(0xF) );
    TEST(u8"1\U0010FFFF" u8"10101010\U0010FFFF" u8"10101010")
        .with(strf::numpunct<2>{8}.thousands_sep(0x10FFFF))
        ( !strf::bin(0x1aaaa) );
    TEST("1'1'1'1")
        .with(strf::numpunct<2>{1}.thousands_sep('\''))
        ( !strf::bin(0xF) );
    TEST("1'10101010'10101010")
        .with(strf::numpunct<2>{8}.thousands_sep('\''))
        ( !strf::bin(0x1aaaa) );
    TEST(u8"1010\U0010FFFF" u8"1010\U0010FFFF" u8"1010\U0010FFFF"
         u8"1010\U0010FFFF" u8"1010\U0010FFFF" u8"1010\U0010FFFF"
         u8"1010\U0010FFFF" u8"1010\U0010FFFF" u8"1010\U0010FFFF"
         u8"1010\U0010FFFF" u8"1010\U0010FFFF" u8"1010\U0010FFFF"
         u8"1010\U0010FFFF" u8"1010\U0010FFFF" u8"1010\U0010FFFF" u8"1010")
        .with(strf::numpunct<2>{4}.thousands_sep(0x10FFFF))
        ( !strf::bin(0xaaaaaaaaaaaaaaaaLL) );

    {
        // Invalid punctuation char. ( They shall be omitted ).

        TEST("9999")
            .with(strf::numpunct<10>{1}.thousands_sep(0xFFFFFF))
            ( !strf::dec(9999) );

        TEST("ffff")
            .with(strf::numpunct<10>{1}.thousands_sep(0xFFFFFF))
            ( !strf::hex(0xFFFF) );

        TEST("7777")
        .with(strf::numpunct<10>{1}.thousands_sep(0xFFFFFF))
            ( !strf::oct(07777) );

        TEST("1111")
        .with(strf::numpunct<2>{1}.thousands_sep(0xFFFFFF))
            ( !strf::bin(0xF) );
    }
}

void STRF_TEST_FUNC test_input_ptr()
{
    void* ptr = strf::detail::bit_cast<void*, std::size_t>(0xABC);

    TEST("0xabc") (ptr);
    TEST("...0xabc") (strf::right(ptr, 8, '.'));
    TEST("...0xabc  ") (strf::join(strf::right(ptr, 8, '.')) < 10);
    TEST("...0xABC").with(strf::lettercase::mixed) (strf::right(ptr, 8, '.'));
    TEST("...0XABC")
        .with(strf::constrain<std::is_pointer>(strf::lettercase::upper))
        (strf::right(ptr, 8, '.'));
}

