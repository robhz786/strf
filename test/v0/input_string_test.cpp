//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <boost/stringify.hpp>

int main()
{
    namespace strf = boost::stringify::v0; 

    {
        TEST("abc")     ("abc");
        TEST("~~~abc") .with(strf::fill(U'~'), strf::width(6))   ("abc");
        TEST("~~~abc") .with(strf::fill(U'~'), strf::width(4))   ({"abc", 6});
        TEST("abc~~~") .with(strf::fill(U'~'))                   ({"abc", {6, "<"}});
        TEST("abc~~~") .with(strf::fill(U'~'), strf::left)       ({"abc", 6});
        TEST("~~~abc") .with(strf::fill(U'~'), strf::internal)   ({"abc", 6});
        TEST("~~~abc") .with(strf::fill(U'~'), strf::left)       ({"abc", {6, ">"}});
        TEST("~~~abc") .with(strf::fill(U'~'), strf::left)       ({"abc", {6, "="}});
        TEST("   abc") ({strf::join_right(6), {"abc"}});
    
        TEST("   abcdefghi") .with(strf::width(3))  ("", {"abc", ">"}, {"def", "<"}, {"ghi", "="});
        TEST("  abcdefghi")  .with(strf::width(2))  ("", {"abc", ">"}, {"def", "<"}, {"ghi", "="});
        TEST("abcdefghi")    .with(strf::width(0))  ("", {"abc", ">"}, {"def", "<"}, {"ghi", "="});
    }

    {
        std::string abc("abc");
        std::string def("def");
        std::string ghi("ghi");

        TEST("abc")  (abc);
        TEST("~~~abc") .with(strf::fill(U'~'), strf::width(6))   (abc);
        TEST("~~~abc") .with(strf::fill(U'~'), strf::width(4))   ({abc, 6});
        TEST("abc~~~") .with(strf::fill(U'~'))                   ({abc, {6, "<"}});
        TEST("abc~~~") .with(strf::fill(U'~'), strf::left)       ({abc, 6});
        TEST("~~~abc") .with(strf::fill(U'~'), strf::internal)   ({abc, 6});
        TEST("~~~abc") .with(strf::fill(U'~'), strf::left)       ({abc, {6, ">"}});
        TEST("~~~abc") .with(strf::fill(U'~'), strf::left)       ({abc, {6, "="}});

        TEST("   abcdefghi") .with(strf::width(3))  ("", {abc, ">"}, {def, "<"}, {ghi, "="});
        TEST("  abcdefghi")  .with(strf::width(2))  ("", {abc, ">"}, {def, "<"}, {ghi, "="});
        TEST("abcdefghi")    .with(strf::width(0))  ("", {abc, ">"}, {def, "<"}, {ghi, "="});
        TEST("   abc")     ({strf::join_right(6), {abc}});
    }

    {
        wchar_t abc[] = L"abc";
        wchar_t def[] = L"def";
        wchar_t ghi[] = L"ghi";

        TEST(L"abc")  (abc);
        TEST(L"~~~abc") .with(strf::fill(U'~'), strf::width(6))   (abc);
        TEST(L"~~~abc") .with(strf::fill(U'~'), strf::width(4))   ({abc, 6});
        TEST(L"abc~~~") .with(strf::fill(U'~'))                   ({abc, {6, "<"}});
        TEST(L"abc~~~") .with(strf::fill(U'~'), strf::left)       ({abc, 6});
        TEST(L"~~~abc") .with(strf::fill(U'~'), strf::internal)   ({abc, 6});
        TEST(L"~~~abc") .with(strf::fill(U'~'), strf::left)       ({abc, {6, ">"}});
        TEST(L"~~~abc") .with(strf::fill(U'~'), strf::left)       ({abc, {6, "="}});

        TEST(L"   abcdefghi") .with(strf::width(3))  (L"", {abc, ">"}, {def, "<"}, {ghi, "="});
        TEST(L"  abcdefghi")  .with(strf::width(2))  (L"", {abc, ">"}, {def, "<"}, {ghi, "="});
        TEST(L"abcdefghi")    .with(strf::width(0))  (L"", {abc, ">"}, {def, "<"}, {ghi, "="});
        TEST(L"   abc")     ({strf::join_right(6), {abc}});
    }

    
    {
        std::wstring abc(L"abc");
        std::wstring def(L"def");
        std::wstring ghi(L"ghi");

        TEST(L"abc")  (abc);
        TEST(L"~~~abc") .with(strf::fill(U'~'), strf::width(6))   (abc);
        TEST(L"~~~abc") .with(strf::fill(U'~'), strf::width(4))   ({abc, 6});
        TEST(L"abc~~~") .with(strf::fill(U'~'))                   ({abc, {6, "<"}});
        TEST(L"abc~~~") .with(strf::fill(U'~'), strf::left)       ({abc, 6});
        TEST(L"~~~abc") .with(strf::fill(U'~'), strf::internal)   ({abc, 6});
        TEST(L"~~~abc") .with(strf::fill(U'~'), strf::left)       ({abc, {6, ">"}});
        TEST(L"~~~abc") .with(strf::fill(U'~'), strf::left)       ({abc, {6, "="}});

        TEST(L"   abcdefghi") .with(strf::width(3))  (L"", {abc, ">"}, {def, "<"}, {ghi, "="});
        TEST(L"  abcdefghi")  .with(strf::width(2))  (L"", {abc, ">"}, {def, "<"}, {ghi, "="});
        TEST(L"abcdefghi")    .with(strf::width(0))  (L"", {abc, ">"}, {def, "<"}, {ghi, "="});
        TEST(L"   abc")     ({strf::join_right(6), {abc}});
    }

    
    
    int rc = report_errors();
    return rc;
}





