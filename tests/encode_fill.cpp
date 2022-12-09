//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

#if ! defined(__cpp_char8_t)
using char8_t = char;
#endif

#define LEN(STR) (sizeof(STR) / sizeof((STR)[0]) - 1)

#define TEST_FILL(ENC, CODEPOINT, STR)                                     \
    TEST(STR STR STR STR STR STR STR STR) .with(ENC)                       \
        (strf::multi(static_cast<char32_t>(CODEPOINT), 8));                \
                                                                           \
    TEST_CALLING_RECYCLE_AT(4 * LEN(STR), STR STR STR STR STR STR STR STR) \
        .with(ENC)                                                         \
        (strf::multi(static_cast<char32_t>(CODEPOINT), 8));                \
                                                                           \
    TEST_TRUNCATING_AT(4 * LEN(STR), STR STR STR STR)                      \
        .with(ENC)                                                         \
        (strf::multi(static_cast<char32_t>(CODEPOINT), 8));                \

STRF_TEST_FUNC void test_encode_fill()
{
    {
        // UTF-8
        TEST_FILL(strf::utf_t<char>{}, 0x7F, "\x7F");
        TEST_FILL(strf::utf_t<char>{}, 0x80, "\xC2\x80");
        TEST_FILL(strf::utf_t<char>{}, 0x800, "\xE0\xA0\x80");
        TEST_FILL(strf::utf_t<char>{}, 0xFFFF, "\xEF\xBF\xBF");
        TEST_FILL(strf::utf_t<char>{}, 0x10000, "\xF0\x90\x80\x80");
        TEST_FILL(strf::utf_t<char>{}, 0x10FFFF, "\xF4\x8F\xBF\xBF");
        TEST_FILL(strf::utf_t<char>{}, 0x110000, "\xEF\xBF\xBD");
    }

    {
        // UTF-16;
        // test_fill(strf::utf_t<char16_t>{}, U'a', u"a");
        TEST_FILL(strf::utf_t<char16_t>{}, 0x10000, u"\U00010000");
        TEST_FILL(strf::utf_t<char16_t>{}, 0x10000,  u"\U00010000");
        TEST_FILL(strf::utf_t<char16_t>{}, 0x10FFFF, u"\U0010FFFF");
        TEST_FILL(strf::utf_t<char16_t>{}, 0x110000, u"\uFFFD");
    }

    {
        // UTF-32;
        TEST_FILL( strf::utf_t<char32_t>{}, U'a', U"a");
        TEST_FILL(strf::utf_t<char32_t>{}, 0x10000,  U"\U00010000");
        TEST_FILL(strf::utf_t<char32_t>{}, 0x10FFFF, U"\U0010FFFF");
        //TEST_FILL(strf::utf_t<char32_t>{}, 0x110000, U"\uFFFD");
    }

    {
        // single byte encodings
        TEST_FILL(strf::windows_1252_t<char>{}, 0x201A, "\x82");
        TEST_FILL(strf::iso_8859_1_t<char>{}, 0x82, "\x82");
        TEST_FILL(strf::iso_8859_3_t<char>{}, 0x02D8, "\xA2");
        TEST_FILL(strf::iso_8859_15_t<char>{}, 0x20AC, "\xA4");

        TEST_FILL(strf::ascii_t<char>{}, 'a' , "a");
        TEST_FILL(strf::windows_1252_t<char>{}, 'a' , "a");
        TEST_FILL(strf::iso_8859_1_t<char>{}  , 'a' , "a");
        TEST_FILL(strf::iso_8859_3_t<char>{}  , 'a' , "a");
        TEST_FILL(strf::iso_8859_15_t<char>{} , 'a' , "a");

        TEST_FILL(strf::ascii_t<char>{}, 0x800, "?");
        TEST_FILL(strf::windows_1252_t<char>{}, 0x800, "?");
        TEST_FILL(strf::iso_8859_1_t<char>{}  , 0x800, "?");
        TEST_FILL(strf::iso_8859_3_t<char>{}  , 0x800, "?");
        TEST_FILL(strf::iso_8859_15_t<char>{} , 0x800, "?");
    }
}

REGISTER_STRF_TEST(test_encode_fill)
