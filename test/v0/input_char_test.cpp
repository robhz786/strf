//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"

#define TEST(EXPECTED) make_tester((EXPECTED), __FILE__, __LINE__)

template <typename T>
struct is_char: public std::is_same<T, char>
{
};

int main()
{
    namespace strf = boost::stringify::v0;


    TEST("a") &= { 'a' };
    TEST("aaaa") &= { {'a', {"", 4}} };
    TEST("  aa") &= { {'a', {4, 2}} };

    TEST("    a") &= { {'a', 5} };
    TEST("a    ") &= { {'a', {5, "<"}} };
    TEST("aa   ") &= { {'a', {5, "<", 2}} };

    TEST("....a") &= { {'a', {5, U'.'}} };
    TEST("a....") &= { {'a', {5, U'.', "<"}} };
    TEST("....a") &= { {'a', {5, U'.', ">"}} };
    TEST("..a..") &= { {'a', {5, U'.', "^"}} };

    TEST("aa...") &= { {'a', {5, U'.', "<", 2}} };
    TEST("...aa") &= { {'a', {5, U'.', ">", 2}} };
    TEST(".aa..") &= { {'a', {5, U'.', "^", 2}} };

    TEST(".....") &= { {'a', {5, U'.', "<", 0}} };
    TEST(".....") &= { {'a', {5, U'.', ">", 0}} };
    TEST(".....") &= { {'a', {5, U'.', "^", 0}} };

    // width calculations inside joins
    TEST("aaaa|  bb|cccc|  dd|--eeee|  ff--") &=
    {
        {strf::join_left(2, '-'), {{'a', {2, "", 4}}}}, '|',
        {strf::join_left(2, '-'), {{'b', {4, "", 2}}}}, '|',
        {strf::join_left(4, '-'), {{'c', {2, "", 4}}}}, '|',
        {strf::join_left(4, '-'), {{'d', {4, "", 2}}}}, '|',
        {strf::join_right(6, '-'), {{'e', {2, "", 4}}}}, '|',
        {strf::join_left(6, '-'), {{'f', {4, "", 2}}}}
    };

    return report_errors() || boost::report_errors();
}













