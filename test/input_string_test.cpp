//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <boost/stringify.hpp>

#define TEST testf<__LINE__>

int main()
{
    namespace strf = boost::stringify;

    TEST("abc") () ("abc");
    TEST("~~~abc") (strf::fill(U'~'), strf::width(6))   ("abc");
    TEST("~~~abc") (strf::fill(U'~'), strf::width(4))   ({"abc", 6});
    TEST("abc~~~") (strf::fill(U'~'))                   ({"abc", {6, "<"}});
    TEST("abc~~~") (strf::fill(U'~'), strf::left)       ({"abc", 6});
    TEST("~~~abc") (strf::fill(U'~'), strf::internal)   ({"abc", 6});
    TEST("~~~abc") (strf::fill(U'~'), strf::left)       ({"abc", {6, ">"}});
    TEST("~~~abc") (strf::fill(U'~'), strf::left)       ({"abc", {6, "="}});
    TEST("   abc") () ({strf::join > 6, {"abc"}});
    
    TEST("   abcdefghi") (strf::width(3))  ("", {"abc", ">"}, {"def", "<"}, {"ghi", "="});
    TEST("  abcdefghi")  (strf::width(2))  ("", {"abc", ">"}, {"def", "<"}, {"ghi", "="});
    TEST("abcdefghi")    (strf::width(0))  ("", {"abc", ">"}, {"def", "<"}, {"ghi", "="});
    
    std::string abc("abc");
    std::string def("def");
    std::string ghi("ghi");

    TEST("abc") () (abc);
    TEST("~~~abc") (strf::fill(U'~'), strf::width(6))   (abc);
    TEST("~~~abc") (strf::fill(U'~'), strf::width(4))   ({abc, 6});
    TEST("abc~~~") (strf::fill(U'~'))                   ({abc, {6, "<"}});
    TEST("abc~~~") (strf::fill(U'~'), strf::left)       ({abc, 6});
    TEST("~~~abc") (strf::fill(U'~'), strf::internal)   ({abc, 6});
    TEST("~~~abc") (strf::fill(U'~'), strf::left)       ({abc, {6, ">"}});
    TEST("~~~abc") (strf::fill(U'~'), strf::left)       ({abc, {6, "="}});

    TEST("   abcdefghi") (strf::width(3))  ("", {abc, ">"}, {def, "<"}, {ghi, "="});
    TEST("  abcdefghi")  (strf::width(2))  ("", {abc, ">"}, {def, "<"}, {ghi, "="});
    TEST("abcdefghi")    (strf::width(0))  ("", {abc, ">"}, {def, "<"}, {ghi, "="});
    TEST("   abc") () ({strf::join > 6, {abc}});

    return  boost::report_errors();
}





