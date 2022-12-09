//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

#if ! defined(__cpp_char8_t)
using char8_t = char;
#endif

STRF_TEST_FUNC void test_input_string()
{
    {
        TEST("abc")      ( strf::fmt("abc") );
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
        const wchar_t abc[] = L"abc";
        const wchar_t def[] = L"def";
        const wchar_t ghi[] = L"ghi";
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

        TEST("---\x99---") (strf::transcode("---\x99---"));
        TEST("---\xA5---")
            .with(strf::iso_8859_3_t<char>{})
            (strf::transcode("---\xA5---", strf::iso_8859_3_t<char>{}));
        {
            const char8_t expected[] = { '-', '-', '-', static_cast<char8_t>('\xA5'), '-', '-', '-', '\0' };
            TEST(expected)
                .with(strf::iso_8859_3_t<char8_t>{})
                (strf::transcode("---\xA5---", strf::iso_8859_3_t<char>{}));
        }
        TEST("...---\xA5---")
            .with(strf::iso_8859_3_t<char>{})
            (strf::right("---\xA5---", 10, U'.').transcode(strf::iso_8859_3_t<char>{}));
        {
            const char8_t expected[] = { '.', '.', '.', '-', '-', '-', static_cast<char8_t>('\xA5'), '-', '-', '-', '\0' };
            TEST(expected)
                .with(strf::iso_8859_3_t<char8_t>{})
                (strf::right("---\xA5---", 10, U'.').transcode(strf::iso_8859_3_t<char>{}));
        }
    }
    {   // encoding sanitization

        TEST("---\xEF\xBF\xBD---") (strf::sani("---\xFF---"));
        TEST("   ---\xEF\xBF\xBD---") (strf::sani("---\xFF---") > 10);
        TEST("---\xEF\xBF\xBD---") (strf::sani("---\xFF---", strf::utf_t<char>{}));
        TEST("   ---\xEF\xBF\xBD---") (strf::sani("---\xFF---", strf::utf_t<char>{}) > 10);
        TEST("---?---")
            .with(strf::iso_8859_3_t<char>{})
            (strf::sani("---\xA5---", strf::iso_8859_3_t<char>{}));
        TEST("  ---?---")
            .with(strf::iso_8859_3_t<char>{})
            (strf::sani("---\xA5---", strf::iso_8859_3_t<char>{}) > 9);

        TEST("...---\x99---") (strf::transcode("---\x99---").fill(U'.') > 10);
        TEST("...---\x99---") (strf::transcode("---\x99---", strf::utf_t<char>{}).fill(U'.') > 10);
    }
    {   // encoding conversion

        TEST("--?--\x80--")
            .with(strf::windows_1252_t<char>{})
            (strf::sani("--\xC9\x90--\xE2\x82\xAC--", strf::utf_t<char>{}));

        TEST("--?--\x80--")
            .with(strf::windows_1252_t<char>{})
            (strf::transcode("--\xC9\x90--\xE2\x82\xAC--", strf::utf_t<char>{}));

        TEST(".......--?--\x80--")
            .with(strf::windows_1252_t<char>{})
            (strf::right("--\xC9\x90--\xE2\x82\xAC--", 15, U'.').transcode(strf::utf_t<char>{}));
    }

    {   // convertion from utf32

        TEST(u8"--\u0080--\u07ff--\u0800--\uffff--\U00010000--\U0010ffff")
            ( strf::transcode(U"--\u0080--\u07ff--\u0800--\uffff--\U00010000--\U0010ffff") );

        TEST(u"--\u0080--\u07ff--\u0800--\uffff--\U00010000--\U0010ffff")
            ( strf::transcode(U"--\u0080--\u07ff--\u0800--\uffff--\U00010000--\U0010ffff") );

        const char32_t abc[] = U"abc";
        const char32_t def[] = U"def";
        const char32_t ghi[] = U"ghi";
        TEST("abc")      ( strf::transcode(abc) );
        TEST("   abc")   ( strf::transcode(abc) > 6 );
        TEST("abc...")   ( strf::transcode(abc).fill('.') < 6 );
        TEST("...abc")   ( strf::transcode(abc).fill('.') > 6 );
        TEST(".abc..")   ( strf::transcode(abc).fill('.') ^ 6 );
        TEST("     abc")   ( strf::join_right(8)(strf::transcode(abc)) );
        TEST("...abc~~")   ( strf::join_right(8, '.')(strf::transcode(abc).fill(U'~') < 5));
        TEST(".....abc")   ( strf::join_right(8, '.')(strf::transcode(abc).fill(U'~') < 3));
        TEST(".....abc")   ( strf::join_right(8, '.')(strf::transcode(abc).fill(U'~') < 2));

        TEST("   abcdefghi") ( strf::transcode(U"") > 3, strf::transcode(abc) > 3
                             , strf::transcode(def) < 3, strf::transcode(ghi) ^ 3 );
        TEST("  abcdefghi")  ( strf::transcode(U"") > 2, strf::transcode(abc) > 2
                             , strf::transcode(def) < 2, strf::transcode(ghi) ^ 2 );
        TEST("abcdefghi")    ( strf::transcode(U"") > 0, strf::transcode(abc) > 0
                             , strf::transcode(def) < 0, strf::transcode(ghi) ^ 0 );

    }
}

REGISTER_STRF_TEST(test_input_string)

