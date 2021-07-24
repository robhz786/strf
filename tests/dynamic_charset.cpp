//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

#if ! defined (__cpp_char8_t)
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

    TEST("abc ? def").with(dyn_ascii) (strf::conv("abc \xC4\x80 def", dyn_utf8));
    TEST("abc ? def").with(dyn_ascii) (strf::conv("abc \xC4\x80 def", strf::utf_t<char>{}));
    TEST("abc ? def").with(strf::ascii_t<char>{}) (strf::conv("abc \xC4\x80 def", dyn_utf8));
    TEST("abc \xC4\x80 def").with(dyn_utf8) (strf::conv(u"abc \u0100 def"  , dyn_utf16));
    TEST("abc \xC4\x80 def").with(dyn_utf8) (strf::conv(U"abc \u0100 def"  , dyn_utf32));
    TEST(u"abcdef").with(dyn_utf16)         (strf::conv( "abcdef", dyn_ascii));
    TEST( "abcdef").with(dyn_ascii)         (strf::conv(u"abcdef", dyn_utf16));
    TEST(u"abc \u0100 def").with(dyn_utf16) (strf::conv( "abc \xC4\x80 def", dyn_utf8));
    TEST(u"abc \u0100 def").with(dyn_utf16) (strf::conv(U"abc \u0100 def"  , dyn_utf32));
    TEST(U"abc \u0100 def").with(dyn_utf32) (strf::conv( "abc \xC4\x80 def", dyn_utf8));
    TEST(U"abc \u0100 def").with(dyn_utf32) (strf::conv(u"abc \u0100 def"  , dyn_utf16));

    TEST("abc \xC4\x80 def").with(dyn_utf8) (strf::sani("abc \xC4\x80 def"));
    TEST(u"abc \u0100 def").with(dyn_utf16) (strf::sani(u"abc \u0100 def"));
    TEST(U"abc \u0100 def").with(dyn_utf32) (strf::sani(U"abc \u0100 def"));

    {
        auto punct = strf::numpunct<10>(1)
            .thousands_sep(0xFFFFFFFF)
            .decimal_point(0xFFFFFFFF);

        auto input = !strf::fixed(10.5).fill((char32_t)0xFFFFFF) ^ 6;

        TEST("?10?5?").with(punct, dyn_ascii) (input);
        TEST("\xEF\xBF\xBD" "10\xEF\xBF\xBD" "5\xEF\xBF\xBD").with(punct, dyn_utf8) (input);
        TEST(u"\uFFFD" u"10\uFFFD" u"5\uFFFD").with(punct, dyn_utf16) (input);
        //TEST(U"\uFFFD" U"10\uFFFD" U"5\uFFFD").with(punct, dyn_utf32) (input);
    }
    {
        auto custom_wcalc = strf::make_width_calculator
            ( [](char32_t ch) -> strf::width_t { return 1 + (ch == U'\u2014'); } );
        TEST( "   x").with(custom_wcalc, dyn_ascii)(strf::fmt('x') > 4);
        TEST( "   x").with(custom_wcalc, dyn_utf8) (strf::fmt('x') > 4);
        TEST(u"   x").with(custom_wcalc, dyn_utf16)(strf::fmt(u'x') > 4);
        TEST(U"   x").with(custom_wcalc, dyn_utf32)(strf::fmt(U'x') > 4);
    }
    {
        const auto invalid_csid = (strf::charset_id) 0x0;
        auto transc1 = dyn_utf8.find_transcoder_to(strf::tag<wchar_t>{}, invalid_csid);
        TEST_TRUE(transc1.transcode_func() == nullptr);

        auto transc2 = dyn_utf8.find_transcoder_from(strf::tag<wchar_t>{}, invalid_csid);
        TEST_TRUE(transc2.transcode_func() == nullptr);

        auto transc3 = dyn_utf8.find_transcoder_to(strf::tag<wchar_t>{}, invalid_csid);
        TEST_TRUE(transc3.transcode_func() == nullptr);

        auto transc4 = dyn_utf8.find_transcoder_from(strf::tag<wchar_t>{}, invalid_csid);
        TEST_TRUE(transc4.transcode_func() == nullptr);

        strf::dynamic_charset_data<char> invalid_data = {
            "invalid", invalid_csid, '?', 1, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, {}, {}, {} };

        strf::dynamic_charset<char> invalid_encoding{invalid_data};

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
