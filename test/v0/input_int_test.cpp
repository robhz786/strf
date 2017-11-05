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
    
    TEST ( "0") [{ 0 }];
    TEST (u"0") [{ 0 }];
    TEST (U"0") [{ 0 }];
    TEST (L"0") [{ 0 }];
    
    TEST ( "0") [{ (unsigned)0 }];
    TEST (u"0") [{ (unsigned)0 }];
    TEST (U"0") [{ (unsigned)0 }];
    TEST (L"0") [{ (unsigned)0 }];

    TEST ( "123") [{ 123 }];
    TEST (u"123") [{ 123 }];
    TEST (U"123") [{ 123 }];
    TEST (L"123") [{ 123 }];

    TEST ( "-123") [{ -123 }];
    TEST (u"-123") [{ -123 }];
    TEST (U"-123") [{ -123 }];
    TEST (L"-123") [{ -123 }];

    TEST ( std::to_string(INT32_MAX).c_str()) [{ INT32_MAX }];
    TEST (std::to_wstring(INT32_MAX).c_str()) [{ INT32_MAX }];
    
    TEST ( std::to_string(INT32_MIN).c_str()) [{ INT32_MIN }];
    TEST (std::to_wstring(INT32_MIN).c_str()) [{ INT32_MIN }];

    TEST ( std::to_string(UINT32_MAX).c_str()) [{ UINT32_MAX }];
    TEST (std::to_wstring(UINT32_MAX).c_str()) [{ UINT32_MAX }];

    
    TEST("f")                       .with(strf::hex) [{ 0xf }];
    TEST("ff")                      .with(strf::hex) [{ 0xff }];
    TEST("ffff")                    .with(strf::hex) [{ 0xffff }];
    TEST("fffff")                   .with(strf::hex) [{ 0xfffffl }];
    TEST("fffffffff")               .with(strf::hex) [{ 0xfffffffffLL }];
    TEST("ffffffffffffffff")        .with(strf::hex) [{ 0xffffffffffffffffLL }];
    TEST("0")                       .with(strf::hex) [{ 0 }];
    TEST("1")                       .with(strf::hex) [{ 0x1 }];
    TEST("10")                      .with(strf::hex) [{ 0x10 }];
    TEST("100")                     .with(strf::hex) [{ 0x100 }];
    TEST("10000")                   .with(strf::hex) [{ 0x10000l }];
    TEST("100000000")               .with(strf::hex) [{ 0x100000000LL }];
    TEST("1000000000000000")        .with(strf::hex) [{ 0x1000000000000000LL }];
    TEST("7")                       .with(strf::oct) [{ 07 }];
    TEST("77")                      .with(strf::oct) [{ 077 }];
    TEST("7777")                    .with(strf::oct) [{ 07777 }];
    TEST("777777777")               .with(strf::oct) [{ 0777777777l }];
    TEST("7777777777777777")        .with(strf::oct) [{ 07777777777777777LL }];
    TEST("777777777777777777777")   .with(strf::oct) [{ 0777777777777777777777LL }];
    TEST("0")                       .with(strf::oct) [{ 0 }];
    TEST("1")                       .with(strf::oct) [{ 01 }];
    TEST("10")                      .with(strf::oct) [{ 010 }];
    TEST("100")                     .with(strf::oct) [{ 0100 }];
    TEST("10000")                   .with(strf::oct) [{ 010000 }];
    TEST("100000000")               .with(strf::oct) [{ 0100000000l }];
    TEST("10000000000000000")       .with(strf::oct) [{ 010000000000000000LL }];
    TEST("1000000000000000000000")  .with(strf::oct) [{ 01000000000000000000000LL }];

    TEST("9")                    [{ 9 }];
    TEST("99")                   [{ 99 }];
    TEST("9999")                 [{ 9999 }];
    TEST("99999999")             [{ 99999999l }];
    TEST("999999999999999999")   [{ 999999999999999999LL }];
    TEST("-9")                   [{ -9 }];
    TEST("-99")                  [{ -99 }];
    TEST("-9999")                [{ -9999 }];
    TEST("-99999999")            [{ -99999999l }];
    TEST("-999999999999999999")  [{ -999999999999999999LL }];
    TEST("0")                    [{ 0 }];
    TEST("1")                    [{ 1 }];
    TEST("10")                   [{ 10 }];
    TEST("100")                  [{ 100 }];
    TEST("10000")                [{ 10000 }];
    TEST("100000000")            [{ 100000000l }];
    TEST("1000000000000000000")  [{ 1000000000000000000LL }];
    TEST("10000000000000000000") [{ 10000000000000000000uLL }];
    TEST("-1")                   [{ -1 }];
    TEST("-10")                  [{ -10 }];
    TEST("-100")                 [{ -100 }];
    TEST("-10000")               [{ -10000 }];
    TEST("-100000000")           [{ -100000000l }];
    TEST("-1000000000000000000") [{ -1000000000000000000LL }];

    // formatting characters:

    // < left
    // > right 
    // = internal

    // + show positive sign ( only affects signed integers )
    // - dont show positive sign
    
    // d decimal
    // o octadecimal
    // x hexadecimal
    // X hexadecimal and uppercase

    // c lowercase
    // C uppercase

    // # show base
    // $ dont show base

    
    auto w15 = strf::width(15);
    TEST ("_____1234567890") .with(w15, strf::fill(U'_')) [{ 1234567890l }];
    TEST ("            123") .with(w15, strf::hex)     [{ {123, "d"} }];
    TEST ("  123") .with(w15) [{ {123, 5} }];
    TEST ("  123") [{ {strf::join_right(5), {123}} }];

    // showpos
    TEST ("           +123") .with(w15, strf::showpos) [{ 123 }];
    TEST ("            123") .with(w15, strf::showpos) [{ {123, "-"} }];
    TEST ("           +123") .with(w15)                [{ {123, "+"} }];
    TEST ("           -123") .with(w15, strf::showpos) [{  -123 }];
    TEST ("             +0") .with(w15, strf::showpos) [{ 0 }];
    TEST ("              0") .with(w15, strf::showpos, strf::noshowpos) [{ 0 }];
    TEST ("            123") .with(w15, strf::showpos) [{ (unsigned)123 }];
    TEST ("            123") .with(w15, strf::showpos) [{ {(unsigned)123, "+"} }];

    // width and aligment (decimal only)
    TEST("+  123")        [{ {123, {6, "+="}} }];
    TEST("11  22+  33")   [{ 11, {22, {4, ">"}}, {33, {5, "+="}} }];
    TEST("11__22+__33") .with(strf::fill(U'_'))   [{ 11, {22, {4, ">"}}, {33, {5, "+="}} }];
    TEST("  11   22")   .with(strf::width(4))                   [{ 11, {22, 5} }];
    TEST("   1122  ")   .with(strf::width(5), strf::right)    [{ 11, {22, {4, "<"}} }];
    TEST("11    22  ")  .with(strf::width(6), strf::left)     [{ 11, {22, 4} }];
    TEST("   11+  22")  .with(strf::width(5), strf::internal) [{ 11, {22, "+"} }];

    // hexadecimal case

    TEST("1234567890abcdef") .with(strf::hex)       [{ 0x1234567890abcdefLL }];
    TEST("1234567890ABCDEF") .with(strf::uppercase) [{ {0x1234567890abcdefLL, "x"} }];
    TEST("1234567890ABCDEF")                        [{ {0x1234567890abcdefLL, "xC"} }];
    TEST("1234567890abcdef") .with(strf::uppercase) [{ {0x1234567890abcdefLL, "xc"} }];
    TEST("1234567890abcdef") .with(strf::uppercase) [{ {0x1234567890abcdefLL, "xc"} }];
    TEST("1234567890abcdef") .with(strf::uppercase, strf::lowercase) [{ {0x1234567890abcdefLL, "x"} }];

    // hexadecimal with showbase
    
    TEST("           0XAA") .with(w15) [{ {0xAA, "#xC"} }];                                               
    TEST("           0XAA") .with(w15) [{ {0xAA, "#X"} }];
    TEST("           0xaa") .with(w15) [{ {0xAA, "#x"} }];
    TEST("           0xaa") .with(strf::noshowbase, w15) [{ {0xAA, "#x"} }];
    TEST("             aa") .with(strf::showbase, strf::noshowbase, w15) [{ {0xAA, "x"} }];

    // hexadecimal aligment
    
    TEST("0xaa           ") .with(w15, strf::left)     [{ {0xAA, "#x"} }];
    TEST("0x           aa") .with(w15, strf::internal) [{ {0xAA, "#x"} }];
    TEST("           0xaa") .with(w15, strf::left, strf::right)     [{ {0xAA, "#x"} }];
    TEST("0xaa           ") .with(w15, strf::right) [{ {0xAA, "#x<"} }];
    TEST("0x           aa") .with(w15, strf::right) [{ {0xAA, "#x="} }];
    TEST("           0xaa") .with(w15, strf::left) [{ {0xAA, "#x>"} }];


   // octadecimal with showbase
    
    TEST("            077") .with(w15) [{ {077, "#o"} }];
    TEST("            077") .with(strf::noshowbase, w15) [{ {077, "#o"} }];
    TEST("             77") .with(strf::showbase, strf::noshowbase, w15) [{ {077, "o"} }];

    // octadecimal aligment
    
    TEST("077            ") .with(w15, strf::left)     [{ {077, "#o"} }];
    TEST("0            77") .with(w15, strf::internal) [{ {077, "#o"} }];
    TEST("            077") .with(w15, strf::right)    [{ {077, "#o"} }];
    TEST("077            ") .with(w15, strf::right) [{ {077, "#o<"} }];
    TEST("0            77") .with(w15, strf::right) [{ {077, "#o="} }];
    TEST("            077") .with(w15, strf::left) [{ {077, "#o>"} }];

    // showpos in octadecimal and hexadecimal must not have any effect
    
    TEST("aa") .with(strf::showpos) [{ {0xAA, "+x"} }];
    TEST("77") .with(strf::showpos) [{ {077, "+o"} }];

    int rc = report_errors() || boost::report_errors();
    return rc;
}






