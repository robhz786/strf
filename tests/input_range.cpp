//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

template <typename T>
struct const_iterator
{
    using value_type = T;
    STRF_HD const T& operator*() const { return *ptr; }
    STRF_HD const_iterator operator++() { ++ptr; return *this; }
    STRF_HD const_iterator operator++(int) { ++ptr; return const_iterator{ptr - 1}; }
    STRF_HD bool operator==(const const_iterator& other) const { return ptr == other.ptr; }
    STRF_HD bool operator!=(const const_iterator& other) const { return ptr != other.ptr; }

    const T* ptr;
};

#if ! defined(STRF_FREESTANDING)

namespace std {

template <typename T>
struct iterator_traits<const_iterator<T>>
{
    using value_type = T;
};

} // namespace std

#endif // ! defined(STRF_FREESTANDING)

template <typename T, std::size_t N>
struct simple_array
{
    T array[N != 0 ? N : 1];
    using const_iterator = ::const_iterator<T>;

    STRF_HD const const_iterator begin() const
    {
        return {static_cast<const T*>(array)};
    }
    STRF_HD const const_iterator end() const
    {
        return {array + N};
    }
};


STRF_TEST_FUNC void test_input_range()
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
        simple_array<const char*, 3> vec{ { "aa", "bb", "cc" } };
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

        TEST( "0xb, 0x16, 0x21") (*strf::hex(strf::separated_range(vec,   ", ")));
        TEST(u"0xb, 0x16, 0x21") (*strf::hex(strf::separated_range(vec,   u", ")));

        TEST( "  11, 22, 33")
            (strf::join_right(12)(strf::separated_range(vec,  ", ")));
        TEST( "  --11, 22, 33--")
            (strf::join_right(16)("--", strf::separated_range(vec,  ", "), "--"));

        TEST( "   0xb, 0x16, 0x21")
             (strf::join_right(18)(*strf::hex(strf::separated_range(vec, ", "))));
        TEST( "--0xb, 0x16, 0x21--")
             (strf::join_right(8)("--", *strf::hex(strf::separated_range(vec, ", ")), "--"));
        TEST( "--11, 22, 33--")
             (strf::join_right(8)("--", strf::separated_range(vec, ", "), "--"));
        TEST( "--11, 22, 33--")
             (strf::join_right(7)("--", strf::separated_range(vec, ", "), "--"));
    }
    {
        simple_array<int, 3> stl_array{ {11, 22, 33} };
        TEST( "112233")        (strf::range({11, 22, 33}));
        TEST( "112233")        (strf::range(stl_array));
        TEST( "11, 22, 33")    (strf::separated_range(stl_array,  ", "));
        TEST(u"+11+22+33")     (+strf::fmt_range(stl_array));
        TEST(u"+11, +22, +33") (+strf::fmt_separated_range(stl_array,  u", "));
    }
    {   // range of only one element
        int arr [1] = {11};

        TEST( "11") (strf::range(arr));
        TEST(u"+11") (+strf::fmt_range(arr));
        TEST( "0xb") (*strf::hex(strf::range(arr)));

        TEST( "11") (strf::separated_range(arr,  ", "));
        TEST(u"+11") (+strf::fmt_separated_range(arr,  u", "));
        TEST( "0xb") (*strf::hex(strf::separated_range(arr,   ", ")));

        simple_array<int, 1> stl_arr{{11}};

        TEST( "11") (strf::range(stl_arr));
        TEST( "+11") (+strf::fmt_range(stl_arr));
        TEST( "0xb") (*strf::hex(strf::range(stl_arr)));

        TEST( "11") (strf::separated_range(stl_arr,  ", "));
        TEST(u"+11") (+strf::fmt_separated_range(stl_arr,  u", "));
        TEST( "0xb") (*strf::hex(strf::separated_range(stl_arr,   ", ")));

    }
    {  // Emtpy range
        simple_array<int, 0> stl_arr{{}};
        TEST( "") (strf::range(stl_arr));
        TEST(u"") (+strf::fmt_range(stl_arr));
        TEST( "") (*strf::hex(strf::range(stl_arr)));

        TEST( "") (strf::separated_range(stl_arr,  ", "));
        TEST(u"") (+strf::fmt_separated_range(stl_arr,  u", "));
        TEST( "") (*strf::hex(strf::separated_range(stl_arr,   ", ")));
    }
    {   // Range transformed by functor
        auto func = [](int x){ return strf::join('<', -x, '>'); };
        int arr [3] = {11, 22, 33};
        simple_array<int, 3> stl_arr{{11, 22, 33}};

        TEST("<-11><-22><-33>") ( strf::range(arr, func) );
        TEST("<-11><-22><-33>") ( strf::range(stl_arr, func) );
        TEST("<-11><-22><-33>") ( strf::range(arr, arr + 3, func) );

        TEST("<-11>, <-22>, <-33>") ( strf::separated_range(arr, ", ", func) );
        TEST("<-11>, <-22>, <-33>") ( strf::separated_range(stl_arr, ", ", func) );
        TEST("<-11>, <-22>, <-33>") ( strf::separated_range(arr, arr + 3, ", ", func) );
        TEST("<-11>")               ( strf::separated_range(arr, arr + 1, ", ", func) );
        TEST("")                    ( strf::separated_range(arr, arr, ", ", func) );
    }
}

REGISTER_STRF_TEST(test_input_range);

