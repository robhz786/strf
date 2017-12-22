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

    TEST (u8"\ud7ff")     = {U'\ud7ff'};
    TEST (u8"\ue000")     = {U'\ue000'};
    TEST (u8"\uffff")     = {U'\uffff'};
    TEST (u8"\U00010000") = {U'\U00010000'};
    TEST (u8"\U0010ffff") = {U'\U0010ffff'};

    TEST (u"\ud7ff")     = {U'\ud7ff'};
    TEST (u"\ue000")     = {U'\ue000'};
    TEST (u"\uffff")     = {U'\uffff'};
    TEST (u"\U00010000") = {U'\U00010000'};
    TEST (u"\U0010ffff") = {U'\U0010ffff'};

    TEST (L"\ud7ff")     = {U'\ud7ff'};
    TEST (L"\ue000")     = {U'\ue000'};
    TEST (L"\uffff")     = {U'\uffff'};
    TEST (L"\U00010000") = {U'\U00010000'};
    TEST (L"\U0010ffff") = {U'\U0010ffff'};

    TEST (U"\ud7ff")     = {U'\ud7ff'};
    TEST (U"\ue000")     = {U'\ue000'};
    TEST (U"\uffff")     = {U'\uffff'};
    TEST (U"\U00010000") = {U'\U00010000'};
    TEST (U"\U0010ffff") = {U'\U0010ffff'};

    // width, alignment, and repetitions
    TEST("aaaa|bbbb|cccc  |    |aaaa|bbbb|  cccc|    ") =
        {
            {U'a', {2, "<", 4}}, U'|',
            {U'b', {4, "<", 4}}, U'|',
            {U'c', {6, "<", 4}}, U'|',
            {U'd', {4, "<", 0}}, U'|',

            {U'a', {2, ">", 4}}, U'|',
            {U'b', {4, ">", 4}}, U'|',
            {U'c', {6, ">", 4}}, U'|',
            {U'd', {4, "<", 0}}
        };


    // width calculations inside joins
    TEST("aaaa|  bb|cccc|  dd|eeee--|  ff--") =
        {
            {strf::join_left(2, U'-'), {{U'a', {2, "", 4}}}}, U'|',
            {strf::join_left(2, U'-'), {{U'b', {4, "", 2}}}}, U'|',
            {strf::join_left(4, U'-'), {{U'c', {2, "", 4}}}}, U'|',
            {strf::join_left(4, U'-'), {{U'd', {4, "", 2}}}}, U'|',
            {strf::join_left(6, U'-'), {{U'e', {2, "", 4}}}}, U'|',
            {strf::join_left(6, U'-'), {{U'f', {4, "", 2}}}}
        };



    // facets
    TEST("---a--bb----|--a-bb---|a--bb----|--a-bb---|--a-bb---")
        .with
        ( strf::fill_if<is_char32>(U'-')
        , strf::width_if<is_char32>(4)) =
        {
            U'a',
            {U'b', {"", 2}},
            {U'c', {"", 0}},
            {U'|', 0},

            {U'a', 3},
            {U'b', {3, "", 2}},
            {U'c', {3, "", 0}},
            {U'|', 0},

            {U'a', {3, "<"}},
            {U'b', {3, "<", 2}},
            {U'c', {3, "<", 0}},
            {U'|', 0},

            {U'a', {3, "="}},
            {U'b', {3, "=", 2}},
            {U'c', {3, "=", 0}},
            {U'|', 0},

            {U'a', {3, ">"}},
            {U'b', {3, ">", 2}},
            {U'c', {3, ">", 0}},
        };

    TEST("---a--bb----|--a-bb---|a--bb----|--a-bb---|--a-bb---")
        .with
        ( strf::fill_if<is_char32>(U'-')
        , strf::width_if<is_char32>(4)
        , strf::internal_if<is_char32>
        ) =
    {
         U'a',
         {U'b', {"", 2}},
         {U'c', {"", 0}},
         {U'|', 0},

         {U'a', 3},
         {U'b', {3, "", 2}},
         {U'c', {3, "", 0}},
         {U'|', 0},

         {U'a', {3, "<"}},
         {U'b', {3, "<", 2}},
         {U'c', {3, "<", 0}},
         {U'|', 0},

         {U'a', {3, "="}},
         {U'b', {3, "=", 2}},
         {U'c', {3, "=", 0}},
         {U'|', 0},

         {U'a', {3, ">"}},
         {U'b', {3, ">", 2}},
         {U'c', {3, ">", 0}},
    };

    TEST("---a--bb----|--a-bb---|a--bb----|--a-bb---|--a-bb---")
        .with
        ( strf::fill_if<is_char32>(U'-')
        , strf::width_if<is_char32>(4)
        , strf::right_if<is_char32>
        ) =
    {
         U'a',
         {U'b', {"", 2}},
         {U'c', {"", 0}},
         {U'|', 0},

         {U'a', 3},
         {U'b', {3, "", 2}},
         {U'c', {3, "", 0}},
         {U'|', 0},

         {U'a', {3, "<"}},
         {U'b', {3, "<", 2}},
         {U'c', {3, "<", 0}},
         {U'|', 0},

         {U'a', {3, "="}},
         {U'b', {3, "=", 2}},
         {U'c', {3, "=", 0}},
         {U'|', 0},

         {U'a', {3, ">"}},
         {U'b', {3, ">", 2}},
         {U'c', {3, ">", 0}},
    };


    TEST("a---bb------|a--bb----|a--bb----|--a-bb---|--a-bb---")
        .with
        ( strf::fill_if<is_char32>(U'-')
        , strf::width_if<is_char32>(4)
        , strf::left_if<is_char32>
        ) =
    {
         U'a',
         {U'b', {"", 2}},
         {U'c', {"", 0}},
         {U'|', 0},

         {U'a', 3},
         {U'b', {3, "", 2}},
         {U'c', {3, "", 0}},
         {U'|', 0},

         {U'a', {3, "<"}},
         {U'b', {3, "<", 2}},
         {U'c', {3, "<", 0}},
         {U'|', 0},

         {U'a', {3, "="}},
         {U'b', {3, "=", 2}},
         {U'c', {3, "=", 0}},
         {U'|', 0},

         {U'a', {3, ">"}},
         {U'b', {3, ">", 2}},
         {U'c', {3, ">", 0}},
    };

    return report_errors() || boost::report_errors();
}













