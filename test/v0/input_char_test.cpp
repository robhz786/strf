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

    // width, alignment, and repetitions
    TEST("aaaa|bbbb|cccc  |    |aaaa|bbbb|  cccc|    ") &=
    {
            {'a', {2, "<", 4}}, '|',
            {'b', {4, "<", 4}}, '|',
            {'c', {6, "<", 4}}, '|',
            {'d', {4, "<", 0}}, '|',

            {'a', {2, ">", 4}}, '|',
            {'b', {4, ">", 4}}, '|',
            {'c', {6, ">", 4}}, '|',
            {'d', {4, "<", 0}}
    };


    // width calculations inside joins
    TEST("aaaa|  bb|cccc|  dd|eeee--|  ff--") &=
    {
        {strf::join_left(2, '-'), {{'a', {2, "", 4}}}}, '|',
        {strf::join_left(2, '-'), {{'b', {4, "", 2}}}}, '|',
        {strf::join_left(4, '-'), {{'c', {2, "", 4}}}}, '|',
        {strf::join_left(4, '-'), {{'d', {4, "", 2}}}}, '|',
        {strf::join_left(6, '-'), {{'e', {2, "", 4}}}}, '|',
        {strf::join_left(6, '-'), {{'f', {4, "", 2}}}}
    };



    // facets
    TEST("---a--bb----|--a-bb---|a--bb----|--a-bb---|--a-bb---|-a-bb----")
        .with
        ( strf::fill_if<is_char>(U'-')
        , strf::width_if<is_char>(4)
        ) &=
    {
         'a',
         {'b', {"", 2}},
         {'c', {"", 0}},
         {'|', 0},

         {'a', 3},
         {'b', {3, "", 2}},
         {'c', {3, "", 0}},
         {'|', 0},

         {'a', {3, "<"}},
         {'b', {3, "<", 2}},
         {'c', {3, "<", 0}},
         {'|', 0},

         {'a', {3, "="}},
         {'b', {3, "=", 2}},
         {'c', {3, "=", 0}},
         {'|', 0},

         {'a', {3, ">"}},
         {'b', {3, ">", 2}},
         {'c', {3, ">", 0}},
         {'|', 0},

         {'a', {3, "^"}},
         {'b', {3, "^", 2}},
         {'c', {3, "^", 0}}
    };

    TEST("---a--bb----|--a-bb---|a--bb----|--a-bb---|--a-bb---|-a-bb----")
        .with
        ( strf::fill_if<is_char>(U'-')
        , strf::width_if<is_char>(4)
        , strf::internal_if<is_char>
        ) &=
    {
         'a',
         {'b', {"", 2}},
         {'c', {"", 0}},
         {'|', 0},

         {'a', 3},
         {'b', {3, "", 2}},
         {'c', {3, "", 0}},
         {'|', 0},

         {'a', {3, "<"}},
         {'b', {3, "<", 2}},
         {'c', {3, "<", 0}},
         {'|', 0},

         {'a', {3, "="}},
         {'b', {3, "=", 2}},
         {'c', {3, "=", 0}},
         {'|', 0},

         {'a', {3, ">"}},
         {'b', {3, ">", 2}},
         {'c', {3, ">", 0}},
         {'|', 0},

         {'a', {3, "^"}},
         {'b', {3, "^", 2}},
         {'c', {3, "^", 0}}
    };

    TEST("---a--bb----|--a-bb---|a--bb----|--a-bb---|--a-bb---|-a-bb----")
        .with
        ( strf::fill_if<is_char>(U'-')
        , strf::width_if<is_char>(4)
        , strf::right_if<is_char>
        ) &=
    {
         'a',
         {'b', {"", 2}},
         {'c', {"", 0}},
         {'|', 0},

         {'a', 3},
         {'b', {3, "", 2}},
         {'c', {3, "", 0}},
         {'|', 0},

         {'a', {3, "<"}},
         {'b', {3, "<", 2}},
         {'c', {3, "<", 0}},
         {'|', 0},

         {'a', {3, "="}},
         {'b', {3, "=", 2}},
         {'c', {3, "=", 0}},
         {'|', 0},

         {'a', {3, ">"}},
         {'b', {3, ">", 2}},
         {'c', {3, ">", 0}},
         {'|', 0},

         {'a', {3, "^"}},
         {'b', {3, "^", 2}},
         {'c', {3, "^", 0}}
    };


    TEST("a---bb------|a--bb----|a--bb----|--a-bb---|--a-bb---|-a-bb----")
        .with
        ( strf::fill_if<is_char>(U'-')
        , strf::width_if<is_char>(4)
        , strf::left_if<is_char>
        ) &=
    {
         'a',
         {'b', {"", 2}},
         {'c', {"", 0}},
         {'|', 0},

         {'a', 3},
         {'b', {3, "", 2}},
         {'c', {3, "", 0}},
         {'|', 0},

         {'a', {3, "<"}},
         {'b', {3, "<", 2}},
         {'c', {3, "<", 0}},
         {'|', 0},

         {'a', {3, "="}},
         {'b', {3, "=", 2}},
         {'c', {3, "=", 0}},
         {'|', 0},

         {'a', {3, ">"}},
         {'b', {3, ">", 2}},
         {'c', {3, ">", 0}},
         {'|', 0},

         {'a', {3, "^"}},
         {'b', {3, "^", 2}},
         {'c', {3, "^", 0}}
    };

    TEST("-a---bb-----|-a-bb----|a--bb----|--a-bb---|--a-bb---|-a-bb----")
        .with
        ( strf::fill_if<is_char>(U'-')
        , strf::width_if<is_char>(4)
        , strf::center_if<is_char>
        ) &=
    {
         'a',
         {'b', {"", 2}},
         {'c', {"", 0}},
         {'|', 0},

         {'a', 3},
         {'b', {3, "", 2}},
         {'c', {3, "", 0}},
         {'|', 0},

         {'a', {3, "<"}},
         {'b', {3, "<", 2}},
         {'c', {3, "<", 0}},
         {'|', 0},

         {'a', {3, "="}},
         {'b', {3, "=", 2}},
         {'c', {3, "=", 0}},
         {'|', 0},

         {'a', {3, ">"}},
         {'b', {3, ">", 2}},
         {'c', {3, ">", 0}},
         {'|', 0},

         {'a', {3, "^"}},
         {'b', {3, "^", 2}},
         {'c', {3, "^", 0}}
    };

   return report_errors() || boost::report_errors();
}













