//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"

#define TEST(EXPECTED) make_tester((EXPECTED), __FILE__, __LINE__)

template <typename T>
struct is_char32: public std::is_same<T, char32_t>
{
};

int main()
{
    namespace strf = boost::stringify::v0;

    // conversion

    TEST (u8"\ud7ff")     &= {U'\ud7ff'};
    TEST (u8"\ue000")     &= {U'\ue000'};
    TEST (u8"\uffff")     &= {U'\uffff'};
    TEST (u8"\U00010000") &= {U'\U00010000'};
    TEST (u8"\U0010ffff") &= {U'\U0010ffff'};

    TEST (u"\ud7ff")     &= {U'\ud7ff'};
    TEST (u"\ue000")     &= {U'\ue000'};
    TEST (u"\uffff")     &= {U'\uffff'};
    TEST (u"\U00010000") &= {U'\U00010000'};
    TEST (u"\U0010ffff") &= {U'\U0010ffff'};

    TEST (L"\ud7ff")     &= {U'\ud7ff'};
    TEST (L"\ue000")     &= {U'\ue000'};
    TEST (L"\uffff")     &= {U'\uffff'};
    TEST (L"\U00010000") &= {U'\U00010000'};
    TEST (L"\U0010ffff") &= {U'\U0010ffff'};

    TEST (U"\ud7ff")     &= {U'\ud7ff'};
    TEST (U"\ue000")     &= {U'\ue000'};
    TEST (U"\uffff")     &= {U'\uffff'};
    TEST (U"\U00010000") &= {U'\U00010000'};
    TEST (U"\U0010ffff") &= {U'\U0010ffff'};

    TEST("a") &= { U'a' };
    TEST("aaaa") &= { {U'a', {"", 4}} };
    TEST("  aa") &= { {U'a', {4, 2}} };

    TEST("    a") &= { {U'a', 5} };
    TEST("a    ") &= { {U'a', {5, "<"}} };
    TEST("aa   ") &= { {U'a', {5, "<", 2}} };

    TEST("....a") &= { {U'a', {5, U'.'}} };
    TEST("a....") &= { {U'a', {5, U'.', "<"}} };
    TEST("....a") &= { {U'a', {5, U'.', ">"}} };
    TEST("..a..") &= { {U'a', {5, U'.', "^"}} };

    TEST("aa...") &= { {U'a', {5, U'.', "<", 2}} };
    TEST("...aa") &= { {U'a', {5, U'.', ">", 2}} };
    TEST(".aa..") &= { {U'a', {5, U'.', "^", 2}} };

    TEST(".....") &= { {U'a', {5, U'.', "<", 0}} };
    TEST(".....") &= { {U'a', {5, U'.', ">", 0}} };
    TEST(".....") &= { {U'a', {5, U'.', "^", 0}} };

    // width calculations inside joins
    TEST("aaaa|  bb|cccc|  dd|eeee--|  ff--") &=
        {
            {strf::join_left(2, U'-'), {{U'a', {2, "", 4}}}}, U'|',
            {strf::join_left(2, U'-'), {{U'b', {4, "", 2}}}}, U'|',
            {strf::join_left(4, U'-'), {{U'c', {2, "", 4}}}}, U'|',
            {strf::join_left(4, U'-'), {{U'd', {4, "", 2}}}}, U'|',
            {strf::join_left(6, U'-'), {{U'e', {2, "", 4}}}}, U'|',
            {strf::join_left(6, U'-'), {{U'f', {4, "", 2}}}}
        };


    return report_errors() || boost::report_errors();
}













