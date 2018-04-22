//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"
#include <vector>

#define TEST(EXPECTED) make_tester((EXPECTED), __FILE__, __LINE__)

int main()
{
    namespace strf = boost::stringify::v0;

    {
        std::vector<int> vec_int = {11, 22, 33, 44};

        TEST("---+11+22+33+44---")
            ( "---"
            , +strf::fmt_iterate(vec_int)
            , "---" );

        TEST("--- +11 +22 +33 +44---")
            ( "---"
            , +strf::fmt(strf::iterate(vec_int)) > 4
            , "---" );

        TEST("---  +11+22+33+44---")
            ( "---"
            , strf::join_right(14)(+strf::fmt_iterate(vec_int))
            , "---" );

        TEST("---+11+22+33+44---")
            ( "---"
            , strf::join_right(12)(+strf::fmt_iterate(vec_int))
            , "---" );

        TEST("---+11+22+33+44---")
            ( "---"
            , strf::join_right(11)(+strf::fmt_iterate(vec_int))
            , "---" );

        TEST("--- +11 +22 +33 +44---")
            ( "---"
            , + strf::fmt_iterate(vec_int) > 4
            , "---" );

        TEST("---.. +11 +22 +33 +44---")
            ("---"
            , strf::join_right(18, '.')(+strf::fmt(strf::iterate(vec_int)) > 4)
            , "---" );

        TEST("--- +11 +22 +33 +44---")
            ("---"
            , strf::join_right(16, '.')(+strf::fmt(strf::iterate(vec_int)) > 4)
            , "---" );

        TEST("--- +11 +22 +33 +44---")
            ( "---"
            , strf::join_right(15, '.')(+strf::fmt(strf::iterate(vec_int)) > 4)
            , "---" );
    }
    {
        std::vector<const char*> vec = { "aa", "bb", "cc" };
        TEST("aabbcc") (strf::iterate(vec));

        TEST("..aa..bb..cc") (strf::right(strf::iterate(vec), 4, '.'));

        TEST("..aa..bb..cc--")
            (strf::join_left(14, '-')(strf::right(strf::iterate(vec), 4, '.')));

        TEST("..aa..bb..cc")
            (strf::join_left(12, '-')(strf::right(strf::iterate(vec), 4, '.')));

        TEST("..aa..bb..cc")
            (strf::join_left(11, '-')(strf::right(strf::iterate(vec), 4, '.')));
    }

    {
        std::string vec = "abcd";
        TEST("abcd") (strf::iterate(vec));
        TEST("aaabbbcccddd") (strf::fmt_iterate(vec).multi(3));
        TEST("  --aaabbbcccddd--")
            ( strf::join_right(18)("--", strf::fmt_iterate(vec).multi(3), "--") );

        TEST("--aaabbbcccddd--")
            ( strf::join_right(16)("--", strf::fmt_iterate(vec).multi(3), "--") );

        TEST("--aaabbbcccddd--")
            ( strf::join_right(15)("--", strf::fmt_iterate(vec).multi(3), "--") );
    }


    return report_errors() || boost::report_errors();
}
