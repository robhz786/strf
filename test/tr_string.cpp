//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <strf/to_string.hpp>
#include "test_utils.hpp"

int main()
{
    // positional argument and automatic arguments
    TEST("0 2 1 2 11")
        .tr( "{ } {2} {} {} {11}"
           , 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

    // auto xstr = strf::to_string.tr("{ } {2} {} {} {} {}", 0, 1, 2, 3);
    // BOOST_TEST(xstr);

    // escape "{" when is followed by another "{"
    TEST("} } { {a } {abc}")
        .tr("} } {{ {{a } {{abc}", "ignored");

    // arguments with comments
    TEST("0 2 0 1 2 3")
        .tr("{ arg0} {2xxx} {0  yyy} {} { arg2} {    }", 0, 1, 2, 3, 4);

    // comments
    TEST("asdfqwert")
        .tr("as{-xxxx}df{-abc{}qwert", "ignored");

    TEST("X aaa Y")      .tr("{} aaa {",      "X", "Y");
    TEST("X aaa Y")      .tr("{} aaa {bbb",   "X", "Y");
    TEST("X aaa {")      .tr("{} aaa {{",     "X", "Y");
    TEST("X aaa {bbb")   .tr("{} aaa {{bbb",  "X", "Y");
    TEST("X aaa ")       .tr("{} aaa {-",     "X", "Y");
    TEST("X aaa ")       .tr("{} aaa {-bbb",  "X", "Y");
    TEST("X aaa Y")      .tr("{} aaa {1",     "X", "Y");
    TEST("X aaa Y")      .tr("{} aaa {1bb",   "X", "Y");
    TEST("X aaa Y")      .tr("{} aaa {}",     "X", "Y");
    TEST("X aaa Y")      .tr("{} aaa {bbb}",  "X", "Y");
    TEST("X aaa {}")     .tr("{} aaa {{}",    "X", "Y");
    TEST("X aaa {bbb}")  .tr("{} aaa {{bbb}", "X", "Y");
    TEST("X aaa ")       .tr("{} aaa {-}",    "X", "Y");
    TEST("X aaa ")       .tr("{} aaa {-bbb}", "X", "Y");
    TEST("X aaa Y")      .tr("{} aaa {1}",    "X", "Y");
    TEST("X aaa Y")      .tr("{} aaa {1bb}",  "X", "Y");

    //
    // now in utf16:
    //

    // positional argument and automatic arguments
    TEST(u"0 2 1 2")
        .tr(u"{ } {2} {} {}", 0, 1, 2, 3);

    // escape "{" when is followed by '{'
    TEST(u"} } { {/ } {abc}")
        .tr(u"} } {{ {{/ } {{abc}", u"ignored");

    // arguments with comments
    TEST(u"0 2 0 1 2 3")
        .tr(u"{ arg0} {2xxx} {0  yyy} {} { arg2} {    }", 0, 1, 2, 3, 4);

    // comments
    TEST(u"asdfqwert")
        .tr(u"as{-xxxx}df{-abc{}qwert", u"ignored");

    TEST(u8"0__2--1==2..3::\uFFFD~~")
        .tr(u8"{ }__{2}--{}=={}..{}::{}~~", 0, 1, 2, 3);
    TEST(u8"0__2--1==2..3::\uFFFD~~")
        .tr(u8"{ }__{2}--{}=={}..{}::{blah}~~", 0, 1, 2, 3);
    TEST(u8"0__2--1==2..3::\uFFFD")
        .tr(u8"{ }__{2}--{}=={}..{}::{", 0, 1, 2, 3);

    TEST("0__2--1==2..3::~~")
        .with(strf::tr_invalid_arg::ignore)
        .tr("{ }__{2}--{}=={}..{}::{}~~", 0, 1, 2, 3);
    TEST("0__2--1==2..3::~~")
        .with(strf::tr_invalid_arg::ignore)
        .tr("{ }__{2}--{}=={}..{}::{blah}~~", 0, 1, 2, 3);
    TEST("0__2--1==2..3::")
        .with(strf::tr_invalid_arg::ignore)
        .tr("{ }__{2}--{}=={}..{}::{", 0, 1, 2, 3);

    TEST(u8"0__\uFFFD--1==2..3::\uFFFD~~")
        .tr(u8"{ }__{10}--{}=={}..{}::{}~~", 0, 1, 2, 3);
    TEST(u8"0__\uFFFD--1==2..3::\uFFFD~~")
        .tr(u8"{ }__{10}--{}=={}..{}::{blah}~~", 0, 1, 2, 3);
    TEST(u8"0__\uFFFD--1==2..3::\uFFFD")
        .tr(u8"{ }__{10}--{}=={}..{}::{", 0, 1, 2, 3);
    TEST("0__--1==2..3::")
        .with(strf::tr_invalid_arg::ignore)
        .tr("{ }__{10}--{}=={}..{}::{", 0, 1, 2, 3);


    return test_finish();

}
