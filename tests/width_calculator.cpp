//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

#define TEST_FAST_WIDTH(STR) TEST(STR) .with( strf::fast_width )
#define TEST_W_AS_FAST_U32LEN(STR) TEST(STR) .with( strf::width_as_fast_u32len )
#define TEST_W_AS_U32LEN(STR) TEST(STR) .with( strf::width_as_u32len )

void STRF_TEST_FUNC test_width_calculator()
{
    using namespace strf::width_literal;
    {
        auto wfunc = [](char32_t ch) -> strf::width_t {
            return ( ch == U'\u2E3A' ? 4: ch == U'\u2014' ? 2: 1 );
        };
        auto wcalc = strf::make_width_calculator(wfunc);
        TEST("abcd").with(wcalc)  (strf::left("abcd", 0));
        TEST("abcd").with(wcalc)  (strf::left("abcd", 3));
        TEST("abcd ").with(wcalc) (strf::left("abcd", 5));

        TEST("abc  ").with(wcalc) (strf::left("abcd", 5).p(3));
        TEST("abcd ").with(wcalc) (strf::left("abcd", 5).p(4));
        TEST("abcd ").with(wcalc) (strf::left("abcd", 5).p(5));

        TEST("123456789-123456789-123456789-   ").with(wcalc)
            (strf::left("123456789-123456789-123456789-", 33).p(32));

        TEST("123456789-123456789-123456789-   ").with(wcalc)
            (strf::left("123456789-123456789-123456789-", 33).p(30));

        TEST("123456789-123456789-123456789    ").with(wcalc)
            (strf::left("123456789-123456789-123456789-", 33).p(29));

        TEST("12345                            ").with(wcalc)
            (strf::left("123456789-123456789-123456789-", 33).p(5));
        TEST("12345                            ").with(wcalc)
            (strf::left("123456789-123456789-123456789-", 33).p(5.99_w));

        TEST(u8"\u2E3A\u2E3A\u2014") .with(wcalc)
            (strf::right(u8"\u2E3A\u2E3A\u2014", 8));

        TEST(u8"  \u2E3A\u2E3A\u2014") .with(wcalc)
            (strf::right(u8"\u2E3A\u2E3A\u2014", 12));

        TEST( u"  \u2E3A\u2E3A\u2014") .with(wcalc)
            (strf::right( u"\u2E3A\u2E3A\u2014", 12));

        // TEST( U"  \u2E3A\u2E3A\u2014") .with(wcalc)
        //     (strf::right( U"\u2E3A\u2E3A\u2014", 12));

        // TEST( L"  \u2E3A\u2E3A\u2014") .with(wcalc)
        //     (strf::right( L"\u2E3A\u2E3A\u2014", 12));

        // TEST(u8"  \u2E3A\u2E3A\u2014") .with(wcalc)
        //     (strf::conv( u"\u2E3A\u2E3A\u2014") > 12);

        // TEST( u"  \u2E3A\u2E3A\u2014") .with(wcalc)
        //     (strf::conv(u8"\u2E3A\u2E3A\u2014") > 12);

    }
    // --------------------------------------------------------------------------------
    // strf::fast_width
    // --------------------------------------------------------------------------------
    TEST_FAST_WIDTH("abcd")   (strf::left("abcd", 0));
    TEST_FAST_WIDTH("abcd")   (strf::left("abcd", 3));
    TEST_FAST_WIDTH("abcd ")  (strf::left("abcd", 5));

    TEST_FAST_WIDTH("abc  ") (strf::left("abcd", 5).p(3));
    TEST_FAST_WIDTH("abcd ") (strf::left("abcd", 5).p(4));
    TEST_FAST_WIDTH("abcd ") (strf::left("abcd", 5).p(5));

    TEST_FAST_WIDTH(u8"   \u2E3A\u2E3A\u2014")
        (strf::fmt(u8"\u2E3A\u2E3A\u2014") > 12);

    TEST_FAST_WIDTH( u"         \u2E3A\u2E3A\u2014")
        (strf::fmt( u"\u2E3A\u2E3A\u2014") > 12.1_w);

    TEST_FAST_WIDTH( U"         \u2E3A\u2E3A\u2014")
        (strf::fmt( U"\u2E3A\u2E3A\u2014") > 12);

    TEST_FAST_WIDTH( u"   \u2E3A\u2E3A\u2014")
        (strf::conv(u8"\u2E3A\u2E3A\u2014") > 12);

    TEST_FAST_WIDTH(u8"         \u2E3A\u2E3A\u2014")
        (strf::conv( u"\u2E3A\u2E3A\u2014") > 12);

    TEST_FAST_WIDTH(u8"         \u2E3A\u2E3A\u2014")
        (strf::conv( U"\u2E3A\u2E3A\u2014") > 12);

    TEST_FAST_WIDTH( u"         \u2E3A\u2E3A\u2014")
        (strf::conv( U"\u2E3A\u2E3A\u2014") > 12);

    // --------------------------------------------------------------------------------
    // strf::width_as_u32len{}
    // --------------------------------------------------------------------------------

    TEST_W_AS_FAST_U32LEN("abcd")  (strf::left("abcd", 0));
    TEST_W_AS_FAST_U32LEN("abcd")  (strf::left("abcd", 4));
    TEST_W_AS_FAST_U32LEN("abcd ") (strf::left("abcd", 5));

    TEST_W_AS_FAST_U32LEN("abc  ") (strf::left("abcd", 5).p(3.999_w));
    TEST_W_AS_FAST_U32LEN("abcd ") (strf::left("abcd", 5).p(4));
    TEST_W_AS_FAST_U32LEN("abcd ") (strf::left("abcd", 5).p(5));

    TEST_W_AS_FAST_U32LEN("            a \xC4\x80 \xE0\xA0\x80 \xF0\x90\x80\x80")
        (strf::right("a \xC4\x80 \xE0\xA0\x80 \xF0\x90\x80\x80", 19));

    TEST_W_AS_FAST_U32LEN(u8"   a \u0080 \u0800 \U00010000")
        (strf::right(u8"a \u0080 \u0800 \U00010000", 10));

    TEST_W_AS_FAST_U32LEN(u8"         \u2E3A\u2E3A\u2014")
        (strf::right(u8"\u2E3A\u2E3A\u2014", 12));

    TEST_W_AS_FAST_U32LEN( u"         \u2E3A\u2E3A\u2014")
        (strf::right( u"\u2E3A\u2E3A\u2014", 12));

    TEST_W_AS_FAST_U32LEN( U"         \u2E3A\u2E3A\u2014")
        (strf::right( U"\u2E3A\u2E3A\u2014", 12));

    TEST_W_AS_FAST_U32LEN( L"         \u2E3A\u2E3A\u2014")
        (strf::right( L"\u2E3A\u2E3A\u2014", 12));

    TEST_W_AS_FAST_U32LEN(u8"         \u2E3A\u2E3A\u2014")
        (strf::conv(u8"\u2E3A\u2E3A\u2014") > 12);

    TEST_W_AS_FAST_U32LEN( u"         \u2E3A\u2E3A\u2014")
        (strf::conv(u"\u2E3A\u2E3A\u2014") > 12);

    // --------------------------------------------------------------------------------
    // strf::width_u32len{}
    // --------------------------------------------------------------------------------

    TEST_W_AS_U32LEN("abcd")  (strf::left("abcd", 0));
    TEST_W_AS_U32LEN("abcd")  (strf::left("abcd", 4));
    TEST_W_AS_U32LEN("abcd ") (strf::left("abcd", 5));

    TEST_W_AS_U32LEN("abc  ") (strf::left("abcd", 5).p(3.999_w));
    TEST_W_AS_U32LEN("abcd ") (strf::left("abcd", 5).p(4));
    TEST_W_AS_U32LEN("abcd ") (strf::left("abcd", 5).p(5));

    // --------------------------------------------------------------------------------
    // Invalid UTF-8 input, when converting to another encoding or sanitizing

    TEST_W_AS_U32LEN(u"   \uFFFD\uFFFD\uFFFD") (strf::sani("\xF1\x80\x80\xE1\x80\xC0") > 6);


    TEST_W_AS_U32LEN(u"   \uFFFD\uFFFD\uFFFD") (strf::sani("\xF1\x80\x80\xE1\x80\xC0") > 6);
    TEST_W_AS_U32LEN(u"   \uFFFD\uFFFD")       (strf::sani("\xC1\xBF") > 5);
    TEST_W_AS_U32LEN(u"   \uFFFD\uFFFD\uFFFD") (strf::sani("\xE0\x9F\x80") > 6);
    TEST_W_AS_U32LEN(u"   \uFFFD\uFFFD\uFFFD") (strf::sani("\xC1\xBF\x80") > 6);
    TEST_W_AS_U32LEN(u"   \uFFFD")             (strf::sani("\xC2") > 4);
    TEST_W_AS_U32LEN(u"   \uFFFD")             (strf::sani("\xE0\xA0") > 4);
    TEST_W_AS_U32LEN(u"   \uFFFD")             (strf::sani("\xF0\x90\xBF") > 4);
    TEST_W_AS_U32LEN(u"   \uFFFD\uFFFD\uFFFD\uFFFD") (strf::sani("\xE0\x9F\x80\x80") > 7);
    TEST_W_AS_U32LEN(u"   \uFFFD\uFFFD\uFFFD\uFFFD") (strf::sani("\xF0\x8F\xBF\xBF") > 7);
    TEST_W_AS_U32LEN(u"   \uFFFD\uFFFD\uFFFD\uFFFD") (strf::sani("\xF4\xBF\xBF\xBF") > 7);
    TEST_W_AS_U32LEN(u"   \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD")
        (strf::sani("\xED\xBF\xBF\x80\x80") > 8);
    TEST_W_AS_U32LEN(u"   \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD")
        (strf::sani("\xF5\x90\x80\x80\x80\x80") > 9);

    // surrogates:
    TEST_W_AS_U32LEN(u"   \uFFFD\uFFFD\uFFFD") (strf::sani("\xED\xA0\x80") > 6);
    TEST_W_AS_U32LEN(u"   \uFFFD\uFFFD\uFFFD") (strf::sani("\xED\xAF\xBF") > 6);
    TEST_W_AS_U32LEN(u"   \uFFFD\uFFFD\uFFFD") (strf::sani("\xED\xB0\x80") > 6);
    TEST_W_AS_U32LEN(u"   \uFFFD\uFFFD\uFFFD") (strf::sani("\xED\xBF\xBF") > 6);

    {
        const char16_t str_0xD800[] = {u' ', 0xD800, u'\0'};
        const char16_t str_0xDBFF[] = {u' ', 0xDBFF, u'\0'};
        const char16_t str_0xDC00[] = {u' ', 0xDC00, u'\0'};
        const char16_t str_0xDFFF[] = {u' ', 0xDFFF, u'\0'};

        char16_t buff[10];
        strf::to(buff)
            .with(strf::width_as_u32len, strf::surrogate_policy::lax )
            (strf::sani("\xED\xA0\x80") > 2);

        TEST_W_AS_U32LEN(str_0xD800).with(strf::surrogate_policy::lax )
            (strf::sani("\xED\xA0\x80") > 2);
        TEST_W_AS_U32LEN(str_0xDBFF).with(strf::surrogate_policy::lax )
            (strf::sani("\xED\xAF\xBF") > 2);
        TEST_W_AS_U32LEN(str_0xDC00).with(strf::surrogate_policy::lax )
            (strf::sani("\xED\xB0\x80") > 2);
        TEST_W_AS_U32LEN(str_0xDFFF).with(strf::surrogate_policy::lax )
            (strf::sani("\xED\xBF\xBF") > 2);
    }
    // --------------------------------------------------------------------------------
    // Invalid UTF-8 input, not sanitizing

    TEST_W_AS_U32LEN("   \xC1\xBF")         (strf::right("\xC1\xBF", 5)); // 2
    TEST_W_AS_U32LEN("   \xE0\x9F\x80")     (strf::right("\xE0\x9F\x80", 6)); // 3
    TEST_W_AS_U32LEN("   \xC1\xBF\x80")     (strf::right("\xC1\xBF\x80", 6)); // 3
    TEST_W_AS_U32LEN("   \xE0\x9F\x80\x80") (strf::right("\xE0\x9F\x80\x80", 7)); // 4
    TEST_W_AS_U32LEN("   \xC2")             (strf::right("\xC2", 4)); // 1
    TEST_W_AS_U32LEN("   \xE0\xA0")         (strf::right("\xE0\xA0", 4)); // 1
    TEST_W_AS_U32LEN("   \xF0\x8F\xBF\xBF") (strf::right("\xF0\x8F\xBF\xBF", 7)); // 4
    TEST_W_AS_U32LEN("   \xF0\x90\xBF")     (strf::right("\xF0\x90\xBF", 4)); // 1
    TEST_W_AS_U32LEN("   \xF4\xBF\xBF\xBF") (strf::right("\xF4\xBF\xBF\xBF", 7)); // 4

    TEST_W_AS_U32LEN("   \xF1\x80\x80\xE1\x80\xC0")
        (strf::right("\xF1\x80\x80\xE1\x80\xC0", 6)); // 3
    TEST_W_AS_U32LEN("   \xED\xBF\xBF\x80\x80")
        (strf::right("\xED\xBF\xBF\x80\x80", 8)); // 5
    TEST_W_AS_U32LEN("   \xF0\x8F\xBF\xBF\x80")
        (strf::right("\xF0\x8F\xBF\xBF\x80", 8)); // 5
    TEST_W_AS_U32LEN("   \xF5\x90\x80\x80\x80\x80")
        (strf::right("\xF5\x90\x80\x80\x80\x80", 9)); // 6

    // surrogates

    TEST_W_AS_U32LEN("   \xED\xA0\x80") (strf::right("\xED\xA0\x80", 6));
    TEST_W_AS_U32LEN("   \xED\xAF\xBF") (strf::right("\xED\xAF\xBF", 6));
    TEST_W_AS_U32LEN("   \xED\xB0\x80") (strf::right("\xED\xB0\x80", 6));
    TEST_W_AS_U32LEN("   \xED\xBF\xBF") (strf::right("\xED\xBF\xBF", 6));
    TEST_W_AS_U32LEN("   \xED\xA0\x80").with(strf::surrogate_policy::lax )
        (strf::right("\xED\xA0\x80", 4));
    TEST_W_AS_U32LEN("   \xED\xAF\xBF").with(strf::surrogate_policy::lax )
        (strf::right("\xED\xAF\xBF", 4));
    TEST_W_AS_U32LEN("   \xED\xB0\x80").with(strf::surrogate_policy::lax )
        (strf::right("\xED\xB0\x80", 4));
    TEST_W_AS_U32LEN("   \xED\xBF\xBF").with(strf::surrogate_policy::lax )
        (strf::right("\xED\xBF\xBF", 4));

    {
        // --------------------------------------------------------------------------------
        // Invalid UTF-16 input, when converting to another encoding or sanitizing
        const char16_t str_0xD800[] = {u' ', 0xD800, u'-', 0xD800, u'\0'};
        const char16_t str_0xDBFF[] = {u' ', 0xDBFF, u'-', 0xDBFF, u'\0'};
        const char16_t str_0xDC00[] = {u' ', 0xDC00, u'-', 0xDC00, u'\0'};
        const char16_t str_0xDFFF[] = {u' ', 0xDFFF, u'-', 0xDFFF, u'\0'};
        const char16_t str_0xDFFF_0xD800[] = {u' ', 0xDFFF, 0xD800, u'-', 0xDFFF, 0xD800, u'\0'};

        TEST_W_AS_U32LEN(u" \uFFFD-\uFFFD")             (strf::sani(str_0xD800 + 1) > 4);
        TEST_W_AS_U32LEN(u" \uFFFD-\uFFFD")             (strf::sani(str_0xDBFF + 1) > 4);
        TEST_W_AS_U32LEN(u" \uFFFD-\uFFFD")             (strf::sani(str_0xDC00 + 1) > 4);
        TEST_W_AS_U32LEN(u" \uFFFD-\uFFFD")             (strf::sani(str_0xDFFF + 1) > 4);
        TEST_W_AS_U32LEN(u" \uFFFD\uFFFD-\uFFFD\uFFFD") (strf::sani(str_0xDFFF_0xD800 + 1) > 6);

        TEST_W_AS_U32LEN(str_0xD800).with(strf::surrogate_policy::lax)
            (strf::right(str_0xD800 + 1, 4));
        TEST_W_AS_U32LEN(str_0xDBFF).with(strf::surrogate_policy::lax)
            (strf::right(str_0xDBFF + 1, 4));
        TEST_W_AS_U32LEN(str_0xDC00).with(strf::surrogate_policy::lax)
            (strf::right(str_0xDC00 + 1, 4));
        TEST_W_AS_U32LEN(str_0xDFFF).with(strf::surrogate_policy::lax)
            (strf::right(str_0xDFFF + 1, 4));
        TEST_W_AS_U32LEN(str_0xDFFF_0xD800) .with(strf::surrogate_policy::lax)
            (strf::right(str_0xDFFF_0xD800 + 1, 6));

        TEST_W_AS_U32LEN(str_0xD800)        (strf::right(str_0xD800 + 1, 4));
        TEST_W_AS_U32LEN(str_0xDBFF)        (strf::right(str_0xDBFF + 1, 4));
        TEST_W_AS_U32LEN(str_0xDC00)        (strf::right(str_0xDC00 + 1, 4));
        TEST_W_AS_U32LEN(str_0xDFFF)        (strf::right(str_0xDFFF + 1, 4));
        TEST_W_AS_U32LEN(str_0xDFFF_0xD800) (strf::right(str_0xDFFF_0xD800 + 1, 6));

    }
}

REGISTER_STRF_TEST(test_width_calculator);

