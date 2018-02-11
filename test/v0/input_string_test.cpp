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
        TEST("   abc")   &= { strf::right("abc", 6) };
        TEST("abc...")   &= { strf::left    ("abc", 6, '.') };
        TEST("...abc")   &= { strf::right   ("abc", 6, '.') };
        TEST(".abc..")   &= { strf::center  ("abc", 6, '.') };
        TEST("     abc")   &= { {strf::join_right(8), {"abc"}} };
        TEST("...abc~~")   &= { {strf::join_right(8, '.'), {strf::left("abc", 5, U'~')}} };
        TEST(".....abc")   &= { {strf::join_right(8, '.'), {strf::left("abc", 3, U'~')}} };
        TEST(".....abc")   &= { {strf::join_right(8, '.'), {strf::left("abc", 2, U'~')}} };

        TEST("   abcdefghi") &= { strf::right("", 3), strf::right("abc", 3), strf::left("def", 3), strf::center("ghi", 3) };
        TEST("  abcdefghi")  &= { strf::right("", 2), strf::right("abc", 2), strf::left("def", 2), strf::center("ghi", 2) };
        TEST("abcdefghi")    &= { strf::right("", 0), strf::right("abc", 0), strf::left("def", 0), strf::center("ghi", 0) };
    }

    {
        wchar_t abc[] = L"abc";
        wchar_t def[] = L"def";
        wchar_t ghi[] = L"ghi";
        TEST(L"abc")      &= { abc };
        TEST(L"   abc")   &= { strf::right(abc, 6) };
        TEST(L"abc...")   &= { strf::left    (abc, 6, '.') };
        TEST(L"...abc")   &= { strf::right   (abc, 6, '.') };
        TEST(L".abc..")   &= { strf::center  (abc, 6, '.') };
        TEST(L"     abc")   &= { {strf::join_right(8), {abc}} };
        TEST(L"...abc~~")   &= { {strf::join_right(8, '.'), {strf::left(abc, 5, U'~')}} };
        TEST(L".....abc")   &= { {strf::join_right(8, '.'), {strf::left(abc, 3, U'~')}} };
        TEST(L".....abc")   &= { {strf::join_right(8, '.'), {strf::left(abc, 2, U'~')}} };

        TEST(L"   abcdefghi") &= { strf::right(L"", 3), strf::right(abc, 3), strf::left(def, 3), strf::center(ghi, 3) };
        TEST(L"  abcdefghi")  &= { strf::right(L"", 2), strf::right(abc, 2), strf::left(def, 2), strf::center(ghi, 2) };
        TEST(L"abcdefghi")    &= { strf::right(L"", 0), strf::right(abc, 0), strf::left(def, 0), strf::center(ghi, 0) };
    }


    {
        std::string abc{ "abc" };

        TEST("abc")     &= { abc };
        TEST("   abc")  &= { strf::right(abc, 6) };
    }

    {
        std::wstring abc{ L"abc" };

        TEST(L"abc")     &= { abc };
        TEST(L"   abc")  &= { strf::right(abc, 6) };
    }

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

    {
        std::string_view abc{"abcdef", 3};

        TEST("abc")    &= { abc };
        TEST("   abc") &= { strf::right(abc, 6) };
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
        TEST("   abc")   &= { strf::right(abc, 6) };
        TEST("abc...")   &= { strf::left    (abc, 6, '.') };
        TEST("...abc")   &= { strf::right   (abc, 6, '.') };
        TEST(".abc..")   &= { strf::center  (abc, 6, '.') };
        TEST("     abc")   &= { {strf::join_right(8), {abc}} };
        TEST("...abc~~")   &= { {strf::join_right(8, '.'), {strf::left(abc, 5, U'~')}} };
        TEST(".....abc")   &= { {strf::join_right(8, '.'), {strf::left(abc, 3, U'~')}} };
        TEST(".....abc")   &= { {strf::join_right(8, '.'), {strf::left(abc, 2, U'~')}} };

        TEST("   abcdefghi") &= { strf::right(U"", 3), strf::right(abc, 3), strf::left(def, 3), strf::center(ghi, 3) };
        TEST("  abcdefghi")  &= { strf::right(U"", 2), strf::right(abc, 2), strf::left(def, 2), strf::center(ghi, 2) };
        TEST("abcdefghi")    &= { strf::right(U"", 0), strf::right(abc, 0), strf::left(def, 0), strf::center(ghi, 0) };

    }
    int rc = report_errors() || boost::report_errors();
    return rc;
}





