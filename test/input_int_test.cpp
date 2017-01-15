//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"
#include <limits>

#define TEST testf<__LINE__>

int main()
{
    namespace strf = boost::stringify;
    
    TEST ( "0") () (0);
    TEST (u"0") () (0);
    TEST (U"0") () (0);
    TEST (L"0") () (0);
    
    TEST ( "0") () ((unsigned)0);
    TEST (u"0") () ((unsigned)0);
    TEST (U"0") () ((unsigned)0);
    TEST (L"0") () ((unsigned)0);
   
    TEST ( "123") () (123);
    TEST (u"123") () (123);
    TEST (U"123") () (123);
    TEST (L"123") () (123);

    TEST ( "-123") () (-123);
    TEST (u"-123") () (-123);
    TEST (U"-123") () (-123);
    TEST (L"-123") () (-123);

    TEST ( std::to_string(INT32_MAX).c_str()) () (INT32_MAX);
    TEST (std::to_wstring(INT32_MAX).c_str()) () (INT32_MAX);
    
    TEST ( std::to_string(INT32_MIN).c_str()) () (INT32_MIN);
    TEST (std::to_wstring(INT32_MIN).c_str()) () (INT32_MIN);

    TEST ( std::to_string(UINT32_MAX).c_str()) () (UINT32_MAX);
    TEST (std::to_wstring(UINT32_MAX).c_str()) () (UINT32_MAX);
    
    // formatting characters:

    // < left
    // > right 
    // % internal

    // + show positive sign ( only affects signed integers )
    // - dont show positive sign
    
    // d decimal
    // o octadecimal
    // x hexadecimal

    // c lowercase
    // C uppercase

    // # show base
    // $ dont show base

    
    // showpos 

    TEST ("+123") (strf::showpos<>) (123);
    TEST ("123")  (strf::showpos<>) ({123, "-"});
    TEST ("+123") ()                ({123, "+"});
    TEST ("-123") (strf::showpos<>) ( -123);
    TEST ("+0")   (strf::showpos<>) (0);
    TEST ("123")  (strf::showpos<>) ((unsigned)123);
    TEST ("123")  (strf::showpos<>) ({(unsigned)123, "+"});


    // to do:
    
    // width
    // TEST("11  22+  33") ()                   (11, {22, {">", 4}}, {33, {"+%", 5}});
    // TEST("11__22+__33") (strf::fill(U'_')    (11, {22, {">", 4}}, {33, {"+%", 5}});
    // TEST("  11   22")   (strf::width(4))                   (11, {22, 5});
    // TEST("   1122  ")   (strf::width(5), strf::right<>)    (11, {22, {"<", 4}});
    // TEST("11  22  ")    (strf::width(6), strf::left<>)     (11, {22, 4});
    // TEST("   11+  22")  (strf::width(5), strf::internal<>) (11, {22, "+"});

    // hexadecimal

    // TEST("aa") () ({0xAA, "x"});
    // TEST("aa") () ({0xAA, "+x"});
    // TEST("  aa") () ({0xAA, {"x", 4}});
    // TEST("aa") () ({0xAA, {"x", 2}});
    // TEST("aa") () ({0xAA, {"x", 1}});                   
        
    // TEST("  0xbb") () ({0xBB, {"#x", 6}});
    // TEST("0xbb")   () ({0xBB, {"#x", 4}});
    // TEST("0xbb")   () ({0xBB, {"#x", 2}});

    // TEST("0XCC") () ({0xCC, "#xC"});
    
    // TEST("0x  dd") () ({0xDD {"#x%", 6}});
    // TEST("0xdd") () ({0xDD {"#x%", 4}});
    // TEST("0xdd") () ({0xDD {"#x%", 2}});

    // TEST("aabb")     (strf::hex<>) {0xAA, 0xBB);
    // TEST("0xaa0xbb") (strf::hex<>, strf::show_base<>) {0xAA, 0xBB);
    // TEST(" 0XAA   BB 0xcc   dd   11")
    //     (strf::hex<>, strf::upper<>, strf::show_base<>, strf::width(5))
    //     (0xAA, {0xBB, "$"}, {0xCC, "c"}, {0xdd, "$c"}, {11, "d"});


    // octadecimal

    // TEST("11") () ({011, "o"});
    // TEST("11") () ({011, "+o"});
    // TEST("   11") () ({011, {"o", 4}});
    // TEST("11") () ({011, {"o", 2}});
    // TEST("11") () ({011, {"o", 1}});                   
        
    // TEST("  022") () ({0xBB, {"#x", 5}});
    // TEST("022")   () ({0xBB, {"#x", 3}});
    // TEST("022")   () ({0xBB, {"#x", 2}});

    // TEST("0  33") () ({0xDD {"#o%", 5}});
    // TEST("0 33")  () ({0xDD {"#o%", 4}});
    // TEST("033")   () ({0xDD {"#o%", 2}});

    // TEST("aabb")     (strf::oct<>) {0xAA, 0xBB);
    // TEST("0xaa0xbb") (strf::oct<>, strf::show_base<>) {0xAA, 0xBB);
    // TEST(" 011  22  11")
    //     (strf::oct<>,  strf::show_base<>, strf::width(4))
    //     (011, {022, "$"}, {11, "d"});
    



    
    return  boost::report_errors();
}






