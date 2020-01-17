//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf.hpp>
#include "test_utils.hpp"
#include <array>


int main()
{
    {
        int arr[] = {11, 22, 33, 44};
        TEST("---11223344---")
            ( "---"
            , strf::range(arr)
            , "---" );
        TEST("---+11+22+33+44---")
            ( "---"
            , +strf::fmt_range(arr)
            , "---" );
        TEST("---+11+22+33+44---")
            ( "---"
              , +strf::fmt_range(arr, arr + 4)
            , "---" );
        TEST("--- +11 +22 +33 +44---")
            ( "---"
            , +strf::fmt(strf::range(arr)) > 4
            , "---" );
        TEST("---  11223344---")
            ( "---"
            , strf::join_right(10)(strf::range(arr))
            , "---" );
        TEST("---11223344---")
            ( "---"
            , strf::join_right(1)(strf::range(arr))
            , "---" );
        TEST("---  +11+22+33+44---")
            ( "---"
            , strf::join_right(14)(+strf::fmt_range(arr))
            , "---" );
        TEST("---+11+22+33+44---")
            ( "---"
            , strf::join_right(12)(+strf::fmt_range(arr))
            , "---" );
        TEST("---+11+22+33+44---")
            ( "---"
            , strf::join_right(11)(+strf::fmt_range(arr))
            , "---" );
        TEST("--- +11 +22 +33 +44---")
            ( "---"
            , + strf::fmt_range(arr) > 4
            , "---" );
        TEST("---.. +11 +22 +33 +44---")
            ("---"
            , strf::join_right(18, '.')(+strf::fmt(strf::range(arr)) > 4)
            , "---" );
        TEST("--- +11 +22 +33 +44---")
            ("---"
            , strf::join_right(16, '.')(+strf::fmt(strf::range(arr)) > 4)
            , "---" );
        TEST("--- +11 +22 +33 +44---")
            ( "---"
            , strf::join_right(15, '.')(+strf::fmt(strf::range(arr)) > 4)
            , "---" );
    }
    {
        std::array<const char*, 3> vec{ { "aa", "bb", "cc" } };
        TEST("aabbcc") (strf::range(vec));
        TEST("..aa..bb..cc") (strf::right(strf::range(vec), 4, '.'));
        TEST("..aa..bb..cc--")
            (strf::join_left(14, '-')(strf::right(strf::range(vec), 4, '.')));

        TEST("..aa..bb..cc")
            (strf::join_left(12, '-')(strf::right(strf::range(vec), 4, '.')));

        TEST("..aa..bb..cc")
            (strf::join_left(11, '-')(strf::right(strf::range(vec), 4, '.')));
    }
    {
        char vec[] = {'a', 'b', 'c', 'd'};
        TEST("abcd") (strf::range(vec));
        TEST("a, b, c, d") (strf::separated_range(vec,  ", "));
        TEST("abc") (strf::range(vec,  vec + 3));
        TEST("a, b, c") (strf::separated_range(vec,  vec + 3, ", "));

        TEST("aaabbbcccddd") (strf::fmt_range(vec).multi(3));
        TEST("  --aaabbbcccddd--")
            ( strf::join_right(18)("--", strf::fmt_range(vec).multi(3), "--") );

        TEST("--aaabbbcccddd--")
            ( strf::join_right(16)("--", strf::fmt_range(vec).multi(3), "--") );

        TEST("--aaabbbcccddd--")
            ( strf::join_right(15)("--", strf::fmt_range(vec).multi(3), "--") );

        TEST("--aaabbbcccddd--")
            ( strf::join_right(2)("--", strf::fmt_range(vec).multi(3), "--") );
    }
    {   // With separator
        int vec [3] = {11, 22, 33};

        TEST( "11, 22, 33") (strf::separated_range(vec,  ", "));
        TEST( "11, 22, 33") (strf::separated_range(vec, vec + 3, ", "));
        TEST(u"+11, +22, +33") (+strf::fmt_separated_range(vec,  u", "));
        TEST(u"+11, +22, +33") (+strf::fmt_separated_range(vec, vec + 3, u", "));

        TEST( "0xb, 0x16, 0x21") (~strf::hex(strf::separated_range(vec,   ", ")));
        TEST(u"0xb, 0x16, 0x21") (~strf::hex(strf::separated_range(vec,   u", ")));

        TEST( "  11, 22, 33")
            (strf::join_right(12)(strf::separated_range(vec,  ", ")));
        TEST( "  --11, 22, 33--")
            (strf::join_right(16)("--", strf::separated_range(vec,  ", "), "--"));

        TEST( "   0xb, 0x16, 0x21")
             (strf::join_right(18)(~strf::hex(strf::separated_range(vec, ", "))));
        TEST( "--0xb, 0x16, 0x21--")
             (strf::join_right(8)("--", ~strf::hex(strf::separated_range(vec, ", ")), "--"));
        TEST( "--11, 22, 33--")
             (strf::join_right(8)("--", strf::separated_range(vec, ", "), "--"));
        TEST( "--11, 22, 33--")
             (strf::join_right(7)("--", strf::separated_range(vec, ", "), "--"));
    }
    {
        std::array<int, 3> stl_array{ {11, 22, 33} };
        TEST( "112233")        (strf::range(stl_array));
        TEST( "11, 22, 33")    (strf::separated_range(stl_array,  ", "));
        TEST(u"+11+22+33")     (+strf::fmt_range(stl_array));
        TEST(u"+11, +22, +33") (+strf::fmt_separated_range(stl_array,  u", "));
    }
    {   // range of only one element
        int arr [1] = {11};

        TEST( "11") (strf::range(arr));
        TEST(u"+11") (+strf::fmt_range(arr));
        TEST( "0xb") (~strf::hex(strf::range(arr)));

        TEST( "11") (strf::separated_range(arr,  ", "));
        TEST(u"+11") (+strf::fmt_separated_range(arr,  u", "));
        TEST( "0xb") (~strf::hex(strf::separated_range(arr,   ", ")));

        std::array<int, 1> stl_arr{{11}};

        TEST( "11") (strf::range(stl_arr));
        TEST( "+11") (+strf::fmt_range(stl_arr));
        TEST( "0xb") (~strf::hex(strf::range(stl_arr)));

        TEST( "11") (strf::separated_range(stl_arr,  ", "));
        TEST(u"+11") (+strf::fmt_separated_range(stl_arr,  u", "));
        TEST( "0xb") (~strf::hex(strf::separated_range(stl_arr,   ", ")));

    }
    {  // Emtpy range
        std::array<int, 0> stl_arr{{}};
        TEST( "") (strf::range(stl_arr));
        TEST(u"") (+strf::fmt_range(stl_arr));
        TEST( "") (~strf::hex(strf::range(stl_arr)));

        TEST( "") (strf::separated_range(stl_arr,  ", "));
        TEST(u"") (+strf::fmt_separated_range(stl_arr,  u", "));
        TEST( "") (~strf::hex(strf::separated_range(stl_arr,   ", ")));
    }
    {   // Range transformed by functor
        auto func = [](int x){ return strf::join('<', -x, '>'); };
        int arr [3] = {11, 22, 33};
        std::array<int, 3> stl_arr{{11, 22, 33}};

        TEST("<-11><-22><-33>") ( strf::range(arr, func) );
        TEST("<-11><-22><-33>") ( strf::range(stl_arr, func) );
        TEST("<-11><-22><-33>") ( strf::range(arr, arr + 3, func) );

        TEST("<-11>, <-22>, <-33>") ( strf::separated_range(arr, ", ", func) );
        TEST("<-11>, <-22>, <-33>") ( strf::separated_range(stl_arr, ", ", func) );
        TEST("<-11>, <-22>, <-33>") ( strf::separated_range(arr, arr + 3, ", ", func) );
    }
    return test_finish();
}
