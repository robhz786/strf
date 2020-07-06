//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

#if ! defined(__cpp_char8_t)
using char8_t = char;
#endif

void test_input_string()
{
    {
        TEST("   abc")   ( strf::right("abc", 6) );
        TEST("abc...")   ( strf::left    ("abc", 6, '.') );
        TEST("...abc")   ( strf::right   ("abc", 6, '.') );
        TEST(".abc..")   ( strf::center  ("abc", 6, '.') );
        TEST("     abc")   ( strf::join_right(8)("abc") );
        TEST("...abc~~")   ( strf::join_right(8, '.')(strf::left("abc", 5, U'~')) );
        TEST(".....abc")   ( strf::join_right(8, '.')(strf::left("abc", 3, U'~')) );
        TEST(".....abc")   ( strf::join_right(8, '.')(strf::left("abc", 2, U'~')) );

        TEST("   abcdefghi") ( strf::right("", 3), strf::right("abc", 3), strf::left("def", 3), strf::center("ghi", 3) );
        TEST("  abcdefghi")  ( strf::right("", 2), strf::right("abc", 2), strf::left("def", 2), strf::center("ghi", 2) );
        TEST("abcdefghi")    ( strf::right("", 0), strf::right("abc", 0), strf::left("def", 0), strf::center("ghi", 0) );
    }
    {
        wchar_t abc[] = L"abc";
        wchar_t def[] = L"def";
        wchar_t ghi[] = L"ghi";
        TEST(L"abc")      ( abc );
        TEST(L"   abc")   ( strf::right(abc, 6) );
        TEST(L"abc...")   ( strf::left    (abc, 6, '.') );
        TEST(L"...abc")   ( strf::right   (abc, 6, '.') );
        TEST(L".abc..")   ( strf::center  (abc, 6, '.') );
        TEST(L"     abc")   ( strf::join_right(8)(abc) );
        TEST(L"...abc~~")   ( strf::join_right(8, '.')(strf::left(abc, 5, U'~')) );
        TEST(L".....abc")   ( strf::join_right(8, '.')(strf::left(abc, 3, U'~')) );
        TEST(L".....abc")   ( strf::join_right(8, '.')(strf::left(abc, 2, U'~')) );

        TEST(L"   abcdefghi") ( strf::right(L"", 3), strf::right(abc, 3), strf::left(def, 3), strf::center(ghi, 3) );
        TEST(L"  abcdefghi")  ( strf::right(L"", 2), strf::right(abc, 2), strf::left(def, 2), strf::center(ghi, 2) );
        TEST(L"abcdefghi")    ( strf::right(L"", 0), strf::right(abc, 0), strf::left(def, 0), strf::center(ghi, 0) );
    }

#if ! defined(STRF_FREESTANDING)
    
    {
        std::string abc{ "abc" };

        TEST("abc")     ( abc );
        TEST("   abc")  ( strf::right(abc, 6) );
    }

    {
        std::wstring abc{ L"abc" };

        TEST(L"abc")     ( abc );
        TEST(L"   abc")  ( strf::right(abc, 6) );
    }

#endif // ! defined(STRF_FREESTANDING)

#if defined(STRF_HAS_STD_STRING_VIEW)

    {
        std::string_view abc{"abcdef", 3};

        TEST("abc")    ( abc );
        TEST("   abc") ( strf::right(abc, 6) );
    }

#endif // defined(STRF_HAS_STD_STRING_VIEW)

    {   // by-pass encoding sanitization

        TEST("---\x99---") (strf::conv("---\x99---"));
        TEST("---\xA5---")
            .with(strf::iso_8859_3<char>())
            (strf::conv("---\xA5---", strf::iso_8859_3<char>()));
        {
            const char8_t expected[] = { '-', '-', '-', (char8_t)'\xA5', '-', '-', '-', '\0' };
            TEST(expected)
                .with(strf::iso_8859_3<char8_t>())
                (strf::conv("---\xA5---", strf::iso_8859_3<char>()));
        }
        TEST("...---\xA5---")
            .with(strf::iso_8859_3<char>())
            (strf::right("---\xA5---", 10, U'.').conv(strf::iso_8859_3<char>()));
        {
            const char8_t expected[] = { '.', '.', '.', '-', '-', '-', (char8_t)'\xA5', '-', '-', '-', '\0' };
            TEST(expected)
                .with(strf::iso_8859_3<char8_t>())
                (strf::right("---\xA5---", 10, U'.').conv(strf::iso_8859_3<char>()));
        }
    }
    {   // encoding sanitization

        TEST("---\xEF\xBF\xBD---") (strf::sani("---\xFF---"));
        TEST("   ---\xEF\xBF\xBD---") (strf::sani("---\xFF---") > 10);
        TEST("---\xEF\xBF\xBD---") (strf::sani("---\xFF---", strf::utf8<char>()));
        TEST("   ---\xEF\xBF\xBD---") (strf::sani("---\xFF---", strf::utf8<char>()) > 10);
        TEST("---?---")
            .with(strf::iso_8859_3<char>())
            (strf::sani("---\xA5---", strf::iso_8859_3<char>()));
        TEST("  ---?---")
            .with(strf::iso_8859_3<char>())
            (strf::sani("---\xA5---", strf::iso_8859_3<char>()) > 9);

        TEST("...---\x99---") (strf::conv("---\x99---").fill(U'.') > 10);
        TEST("...---\x99---") (strf::conv("---\x99---", strf::utf8<char>()).fill(U'.') > 10);
    }
    {   // encoding conversion

        TEST("--?--\x80--")
            .with(strf::windows_1252<char>())
            (strf::sani("--\xC9\x90--\xE2\x82\xAC--", strf::utf8<char>()));

        TEST("--?--\x80--")
            .with(strf::windows_1252<char>())
            (strf::conv("--\xC9\x90--\xE2\x82\xAC--", strf::utf8<char>()));

        TEST("....--?--\x80--")
            .with(strf::windows_1252<char>())
            (strf::right("--\xC9\x90--\xE2\x82\xAC--", 15, U'.').conv(strf::utf8<char>()));
    }

    {   // convertion from utf32

        TEST(u8"--\u0080--\u07ff--\u0800--\uffff--\U00010000--\U0010ffff")
            ( strf::conv(U"--\u0080--\u07ff--\u0800--\uffff--\U00010000--\U0010ffff") );

        TEST(u"--\u0080--\u07ff--\u0800--\uffff--\U00010000--\U0010ffff")
            ( strf::conv(U"--\u0080--\u07ff--\u0800--\uffff--\U00010000--\U0010ffff") );

        char32_t abc[] = U"abc";
        char32_t def[] = U"def";
        char32_t ghi[] = U"ghi";
        TEST("abc")      ( strf::conv(abc) );
        TEST("   abc")   ( strf::conv(abc) > 6 );
        TEST("abc...")   ( strf::conv(abc).fill('.') < 6 );
        TEST("...abc")   ( strf::conv(abc).fill('.') > 6 );
        TEST(".abc..")   ( strf::conv(abc).fill('.') ^ 6 );
        TEST("     abc")   ( strf::join_right(8)(strf::conv(abc)) );
        TEST("...abc~~")   ( strf::join_right(8, '.')(strf::conv(abc).fill(U'~') < 5));
        TEST(".....abc")   ( strf::join_right(8, '.')(strf::conv(abc).fill(U'~') < 3));
        TEST(".....abc")   ( strf::join_right(8, '.')(strf::conv(abc).fill(U'~') < 2));

        TEST("   abcdefghi") ( strf::conv(U"") > 3, strf::conv(abc) > 3
                             , strf::conv(def) < 3, strf::conv(ghi) ^ 3 );
        TEST("  abcdefghi")  ( strf::conv(U"") > 2, strf::conv(abc) > 2
                             , strf::conv(def) < 2, strf::conv(ghi) ^ 2 );
        TEST("abcdefghi")    ( strf::conv(U"") > 0, strf::conv(abc) > 0
                             , strf::conv(def) < 0, strf::conv(ghi) ^ 0 );

    }
}

