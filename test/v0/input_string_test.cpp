//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"
#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>

int main()
{
    namespace strf = boost::stringify::v0;

    {
        TEST("abc")      &= { "abc" };
        TEST("   abc")   &= { {"abc", 6} };
        TEST("abc...")   &= { {"abc", {6, '.', "<"}} };
        TEST("...abc")   &= { {"abc", {6, '.', ">"}} };
        TEST("...abc")   &= { {"abc", {6, '.', "="}} };
        TEST(".abc..")   &= { {"abc", {6, '.', "^"}} };
        TEST("     abc")   &= { {strf::join_right(8), {"abc"}} };
        TEST("...abc~~")   &= { {strf::join_right(8, '.'), {{"abc", {5, U'~', "<"}}}} };
        TEST(".....abc")   &= { {strf::join_right(8, '.'), {{"abc", {3, U'~', "<"}}}} };
        TEST(".....abc")   &= { {strf::join_right(8, '.'), {{"abc", {2, U'~', "<"}}}} };


        TEST("   abcdefghijkl") &= { {"", 3}, {"abc", {3, ">"}}, {"def", {3, "<"}}, {"ghi", {3, "="}}, {"jkl", {3, "^"}} };
        TEST("  abcdefghijkl")  &= { {"", 2}, {"abc", {2, ">"}}, {"def", {2, "<"}}, {"ghi", {2, "="}}, {"jkl", {2, "^"}} };
        TEST("abcdefghijkl")    &= { {"", 0}, {"abc", {0, ">"}}, {"def", {0, "<"}}, {"ghi", {0, "="}}, {"jkl", {0, "^"}} };
    }

    {
        wchar_t abc[] = L"abc";
        wchar_t def[] = L"def";
        wchar_t ghi[] = L"ghi";
        TEST(L"abc")    &= { abc };
        TEST(L"   abc") &= { {abc, 6} };
        TEST(L"abc   ") &= { {abc, {6, "<"}} };
        TEST(L"   abc") &= { {abc, {6, ">"}} };
        TEST(L"   abc") &= { {abc, {6, "="}} };
        TEST(L" abc  ") &= { {abc, {6, "^"}} };
        TEST(L"     abc")   &= { {strf::join_right(8), {"abc"}} };
        TEST(L"...abc~~")   &= { {strf::join_right(8, '.'), {{abc, {5, U'~', "<"}}}} };
        TEST(L".....abc")   &= { {strf::join_right(8, '.'), {{abc, {3, U'~', "<"}}}} };
        TEST(L".....abc")   &= { {strf::join_right(8, '.'), {{abc, {2, U'~', "<"}}}} };



        TEST(L"   abcdefghijkl") &= { {L"", 3}, {abc, {3, ">"}}, {def, {3, "<"}}, {ghi, {3, "="}}, {L"jkl", {3, "^"}} };
        TEST(L"  abcdefghijkl")  &= { {L"", 2}, {abc, {2, ">"}}, {def, {2, "<"}}, {ghi, {2, "="}}, {L"jkl", {2, "^"}} };
        TEST(L"abcdefghijkl")    &= { {L"", 0}, {abc, {0, ">"}}, {def, {0, "<"}}, {ghi, {0, "="}}, {L"jkl", {0, "^"}} };
    }


    {
        std::string abc{ "abc" };

        TEST("abc")     &= { abc };
        TEST("   abc")  &= { {abc, 6} };
    }

    {
        std::wstring abc{ L"abc" };

        TEST(L"abc")     &= { abc };
        TEST(L"   abc")  &= { {abc, 6} };
    }

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

    {
        std::string_view abc{"abcdef", 3};

        TEST("abc")    &= { abc };
        TEST("   abc") &= { {abc, 6} };
    }

#endif

    {   // convertion from utf32

        TEST(u8"--\u0080--\u07ff--\u0800--\uffff--\U00010000--\U0010ffff")
            &= { U"--\u0080--\u07ff--\u0800--\uffff--\U00010000--\U0010ffff" };

        TEST(u"--\u0080--\u07ff--\u0800--\uffff--\U00010000--\U0010ffff")
            &= { U"--\u0080--\u07ff--\u0800--\uffff--\U00010000--\U0010ffff" };

        char32_t abc[] = U"abc";
        char32_t def[] = U"def";
        char32_t ghi[] = U"ghi";

        TEST("abc")      &= { abc };
        TEST("   abc")   &= { {abc, 6} };
        TEST("abc   ")   &= { {abc, {6, "<"}} };
        TEST("   abc")   &= { {abc, {6, ">"}} };
        TEST("   abc")   &= { {abc, {6, "="}} };
        TEST(" abc  ")   &= { {abc, {6, "^"}} };
        TEST("     abc")   &= { {strf::join_right(8), {abc}} };
        TEST("...abc~~")   &= { {strf::join_right(8, '.'), {{abc, {5, U'~', "<"}}}} };
        TEST(".....abc")   &= { {strf::join_right(8, '.'), {{abc, {3, U'~', "<"}}}} };
        TEST(".....abc")   &= { {strf::join_right(8, '.'), {{abc, {2, U'~', "<"}}}} };


        TEST("   abcdefghijkl") &= { {U"", 3}, {abc, {3, ">"}}, {def, {3, "<"}}, {ghi, {3, "="}}, {U"jkl", {3, "^"}} };
        TEST("  abcdefghijkl")  &= { {U"", 2}, {abc, {2, ">"}}, {def, {2, "<"}}, {ghi, {2, "="}}, {U"jkl", {2, "^"}} };
        TEST("abcdefghijkl")    &= { {U"", 0}, {abc, {0, ">"}}, {def, {0, "<"}}, {ghi, {0, "="}}, {U"jkl", {0, "^"}} };
    }
    int rc = report_errors() || boost::report_errors();
    return rc;
}





