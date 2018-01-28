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


    TEST("a")      &= { {strf::join_left(0, '.'), {'a'}} };
    TEST("   a")   &= { {strf::join_left(1, '.'), {{'a', 4}}} };
    TEST("   a..") &= { {strf::join_left(6, '.'), {{'a', 4}}} };

    TEST("  aa")   &= { {strf::join_left(2, '.'), {{'a', {4, "", 2}}}} };
    TEST("  aa")   &= { {strf::join_left(4, '.'), {{'a', {4, "", 2}}}} };
    TEST("  aa..") &= { {strf::join_left(6, '.'), {{'a', {4, "", 2}}}} };

    TEST("aaaa")   &= { {strf::join_left(2, '.'), {{'a', {2, "", 4}}}} };
    TEST("aaaa")   &= { {strf::join_left(4, '.'), {{'a', {2, "", 4}}}} };
    TEST("aaaa..") &= { {strf::join_left(6, '.'), {{'a', {2, "", 4}}}} };

    TEST("aaaa")   &= { {strf::join_left(2, '.'), {{'a', {4, "", 4}}}} };
    TEST("aaaa")   &= { {strf::join_left(4, '.'), {{'a', {4, "", 4}}}} };
    TEST("aaaa..") &= { {strf::join_left(6, '.'), {{'a', {4, "", 4}}}} };

    return report_errors() || boost::report_errors();
}













