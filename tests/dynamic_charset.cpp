//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

#if ! defined (__cpp_char8_t)
#   if __GNUC__ >= 11
#       pragma GCC diagnostic ignored "-Wc++20-compat"
#   endif
using char8_t = char;
#endif //! defined (__cpp_char8_t)

void test_dynamic_charset()
{
    strf::dynamic_charset<char> dyn_utf8{strf::utf_t<char>{}};
    strf::dynamic_charset<char> dyn_ascii{strf::ascii_t<char>{}};
    strf::dynamic_charset<char16_t> dyn_utf16{strf::utf_t<char16_t>()};
    strf::dynamic_charset<char32_t> dyn_utf32{strf::utf_t<char32_t>()};

    {
        auto a = dyn_utf8;
        auto b = strf::dynamic_charset<char>{strf::utf_t<char>{}};
        TEST_TRUE(a == b);
        a = dyn_ascii;
        TEST_TRUE(a != b);
    }

    TEST("abc ? def") (dyn_ascii, strf::transcode("abc \xC4\x80 def", dyn_utf8));
    TEST("abc ? def") (dyn_ascii, strf::transcode("abc \xC4\x80 def", strf::utf_t<char>{}));
    TEST("abc ? def") (strf::ascii_t<char>{}, strf::transcode("abc \xC4\x80 def", dyn_utf8));
    TEST("abc \xC4\x80 def") (dyn_utf8, strf::transcode(u"abc \u0100 def"  , dyn_utf16));
    TEST("abc \xC4\x80 def") (dyn_utf8, strf::transcode(U"abc \u0100 def"  , dyn_utf32));
    TEST(u"abcdef") (dyn_utf16, strf::transcode( "abcdef", dyn_ascii));
    TEST( "abcdef") (dyn_ascii, strf::transcode(u"abcdef", dyn_utf16));
    TEST(u"abc \u0100 def") (dyn_utf16, strf::transcode( "abc \xC4\x80 def", dyn_utf8));
    TEST(u"abc \u0100 def") (dyn_utf16, strf::transcode(U"abc \u0100 def"  , dyn_utf32));
    TEST(U"abc \u0100 def") (dyn_utf32, strf::transcode( "abc \xC4\x80 def", dyn_utf8));
    TEST(U"abc \u0100 def") (dyn_utf32, strf::transcode(u"abc \u0100 def"  , dyn_utf16));

    TEST("abc \xC4\x80 def") (dyn_utf8, strf::sani("abc \xC4\x80 def"));
    TEST(u"abc \u0100 def") (dyn_utf16, strf::sani(u"abc \u0100 def"));
    TEST(U"abc \u0100 def") (dyn_utf32, strf::sani(U"abc \u0100 def"));

    {
        auto punct = strf::numpunct<10>(1)
            .thousands_sep(0xFFFFFFFF)
            .decimal_point(0xFFFFFFFF);

        auto input = !strf::fixed(10.5).fill(static_cast<char32_t>(0xFFFFFF)) ^ 6;

        TEST("?10?5?") (punct, dyn_ascii, input);
        TEST("\xEF\xBF\xBD" "10\xEF\xBF\xBD" "5\xEF\xBF\xBD") (punct, dyn_utf8, input);
        TEST(u"\uFFFD" u"10\uFFFD" u"5\uFFFD") (punct, dyn_utf16, input);
        //TEST(U"\uFFFD" U"10\uFFFD" U"5\uFFFD") (punct, dyn_utf32, input);
    }
    {
        const auto invalid_csid = static_cast<strf::charset_id>(0x0);
        auto transc1 = dyn_utf8.find_transcoder_to(strf::tag<wchar_t>{}, invalid_csid);
        TEST_TRUE(transc1.transcode_func() == nullptr);

        auto transc2 = dyn_utf8.find_transcoder_from(strf::tag<wchar_t>{}, invalid_csid);
        TEST_TRUE(transc2.transcode_func() == nullptr);

        auto transc3 = dyn_utf8.find_transcoder_to(strf::tag<wchar_t>{}, invalid_csid);
        TEST_TRUE(transc3.transcode_func() == nullptr);

        auto transc4 = dyn_utf8.find_transcoder_from(strf::tag<wchar_t>{}, invalid_csid);
        TEST_TRUE(transc4.transcode_func() == nullptr);

        const strf::dynamic_charset_data<char> invalid_data = {
            "invalid", invalid_csid, '?', 1, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, {}, {}, {} };

        const strf::dynamic_charset<char> invalid_encoding{invalid_data};

        auto transc5 = invalid_encoding.find_transcoder_from(strf::tag<char>{}, invalid_csid);
        auto transc6 = invalid_encoding.find_transcoder_from(strf::tag<char16_t>{}, invalid_csid);
        auto transc7 = invalid_encoding.find_transcoder_from(strf::tag<wchar_t>{}, invalid_csid);

        auto transc8  = invalid_encoding.find_transcoder_to(strf::tag<char>{}, invalid_csid);
        auto transc9  = invalid_encoding.find_transcoder_to(strf::tag<char16_t>{}, invalid_csid);
        auto transc10 = invalid_encoding.find_transcoder_to(strf::tag<wchar_t>{}, invalid_csid);
        TEST_TRUE(transc5.transcode_func() == nullptr);
        TEST_TRUE(transc6.transcode_func() == nullptr);
        TEST_TRUE(transc7.transcode_func() == nullptr);
        TEST_TRUE(transc8.transcode_func() == nullptr);
        TEST_TRUE(transc9.transcode_func() == nullptr);
        TEST_TRUE(transc10.transcode_func() == nullptr);
    }
}

REGISTER_STRF_TEST(test_dynamic_charset)
