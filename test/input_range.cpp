//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify.hpp>
#include "test_utils.hpp"
#include <vector>


int main()
{
    namespace strf = boost::stringify::v0;

    {
        std::vector<int> vec_int = {11, 22, 33, 44};
        TEST("---11223344---")
            ( "---"
            , strf::range(vec_int)
            , "---" );

        TEST("---+11+22+33+44---")
            ( "---"
            , +strf::fmt_range(vec_int)
            , "---" );

        TEST("--- +11 +22 +33 +44---")
            ( "---"
            , +strf::fmt(strf::range(vec_int)) > 4
            , "---" );

        TEST("---  11223344---")
            ( "---"
            , strf::join_right(10)(strf::range(vec_int))
            , "---" );

        TEST("---11223344---")
            ( "---"
            , strf::join_right(1)(strf::range(vec_int))
            , "---" );

        TEST("---  +11+22+33+44---")
            ( "---"
            , strf::join_right(14)(+strf::fmt_range(vec_int))
            , "---" );

        TEST("---+11+22+33+44---")
            ( "---"
            , strf::join_right(12)(+strf::fmt_range(vec_int))
            , "---" );

        TEST("---+11+22+33+44---")
            ( "---"
            , strf::join_right(11)(+strf::fmt_range(vec_int))
            , "---" );

        TEST("--- +11 +22 +33 +44---")
            ( "---"
            , + strf::fmt_range(vec_int) > 4
            , "---" );

        TEST("---.. +11 +22 +33 +44---")
            ("---"
            , strf::join_right(18, '.')(+strf::fmt(strf::range(vec_int)) > 4)
            , "---" );

        TEST("--- +11 +22 +33 +44---")
            ("---"
            , strf::join_right(16, '.')(+strf::fmt(strf::range(vec_int)) > 4)
            , "---" );

        TEST("--- +11 +22 +33 +44---")
            ( "---"
            , strf::join_right(15, '.')(+strf::fmt(strf::range(vec_int)) > 4)
            , "---" );
    }
    {
        std::vector<const char*> vec = { "aa", "bb", "cc" };
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
        std::string vec = "abcd";
        TEST("abcd") (strf::range(vec));
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
    {   // with separator
        int vec [3] = {11, 22, 33};

        TEST( "11, 22, 33") (strf::range(vec,  ", "));
        TEST(u"+11, +22, +33") (+strf::fmt_range(vec,  u", "));

        TEST( "0xb, 0x16, 0x21") (~strf::hex(strf::range(vec,   ", ")));
        TEST(u"0xb, 0x16, 0x21") (~strf::hex(strf::range(vec,   u", ")));

        TEST( "  11, 22, 33")
            (strf::join_right(12)(strf::range(vec,  ", ")));
        TEST( "  --11, 22, 33--")
            (strf::join_right(16)("--", strf::range(vec,  ", "), "--"));

        TEST( "   0xb, 0x16, 0x21")
             (strf::join_right(18)(~strf::hex(strf::range(vec, ", "))));
        TEST( "--0xb, 0x16, 0x21--")
             (strf::join_right(2)("--", ~strf::hex(strf::range(vec, ", ")), "--"));
        TEST( "--0xb, 0x16, 0x21--")
             (strf::join_right(4)("--", ~strf::hex(strf::range(vec, ", ")), "--"));

    }


    return boost::report_errors();
}
