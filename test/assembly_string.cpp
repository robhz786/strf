//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"

int main()
{
    namespace strf = boost::stringify::v0;

    // positional argument and automatic arguments
    TEST("0 2 1 2 11")
        .as( "{ } {2} {} {} {11}"
           , 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);

    // auto xstr = strf::to_string.as("{ } {2} {} {} {} {}", 0, 1, 2, 3);
    // BOOST_TEST(xstr);
    
    // escape "{" when is followed by another "{"
    TEST("} } { {a } {abc}")
        .as("} } {{ {{a } {{abc}", "ignored");

    // arguments with comments
    TEST("0 2 0 1 2 3")
        .as("{ arg0} {2xxx} {0  yyy} {} { arg2} {    }", 0, 1, 2, 3, 4);

    // comments
    TEST("asdfqwert")
        .as("as{-xxxx}df{-abc{}qwert", "ignored");

    TEST("X aaa Y")      .as("{} aaa {",      "X", "Y");
    TEST("X aaa Y")      .as("{} aaa {bbb",   "X", "Y");
    TEST("X aaa {")      .as("{} aaa {{",     "X", "Y");
    TEST("X aaa {bbb")   .as("{} aaa {{bbb",  "X", "Y");
    TEST("X aaa ")       .as("{} aaa {-",     "X", "Y");
    TEST("X aaa ")       .as("{} aaa {-bbb",  "X", "Y");
    TEST("X aaa Y")      .as("{} aaa {1",     "X", "Y");
    TEST("X aaa Y")      .as("{} aaa {1bb",   "X", "Y");
    TEST("X aaa Y")      .as("{} aaa {}",     "X", "Y");
    TEST("X aaa Y")      .as("{} aaa {bbb}",  "X", "Y");
    TEST("X aaa {}")     .as("{} aaa {{}",    "X", "Y");
    TEST("X aaa {bbb}")  .as("{} aaa {{bbb}", "X", "Y");
    TEST("X aaa ")       .as("{} aaa {-}",    "X", "Y");
    TEST("X aaa ")       .as("{} aaa {-bbb}", "X", "Y");
    TEST("X aaa Y")      .as("{} aaa {1}",    "X", "Y");
    TEST("X aaa Y")      .as("{} aaa {1bb}",  "X", "Y");

    //
    // now in utf16:
    //

    // positional argument and automatic arguments
    TEST(u"0 2 1 2")
        .as(u"{ } {2} {} {}", 0, 1, 2, 3);

    // escape "{" when is followed by '{'
    TEST(u"} } { {/ } {abc}")
        .as(u"} } {{ {{/ } {{abc}", u"ignored");
    
    // arguments with comments
    TEST(u"0 2 0 1 2 3")
        .as(u"{ arg0} {2xxx} {0  yyy} {} { arg2} {    }", 0, 1, 2, 3, 4);

    // comments
    TEST(u"asdfqwert")
        .as(u"as{-xxxx}df{-abc{}qwert", u"ignored");

    //
    // errors
    //
    // TEST_ERR("0 2 1 2 3 ", std::make_error_code(std::errc::value_too_large))
    //     .as("{ } {2} {} {} {} {}") (0, 1, 2, 3);
    
    // TEST_ERR("0 1 ", std::make_error_code(std::errc::value_too_large))
    //     .as("{ } {} {10} {} {}") (0, 1, 2, 3);



    // TODO test contrained facets

    
    return report_errors() || boost::report_errors();

}
