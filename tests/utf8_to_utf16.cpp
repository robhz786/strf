//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils/transcoding.hpp"

#ifndef __cpp_char8_t
using char8_t = char;
#endif

namespace {

STRF_TEST_FUNC void utf8_to_utf16_unsafe_transcode()
{
    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char16_t)
        .input(u8"ab")
        .expect(u"ab")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char16_t)
        .input(u8"\u0080")
        .expect(u"\u0080")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char16_t)
        .input(u8"\u0800")
        .expect(u"\u0800")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char16_t)
        .input(u8"\uD7FF")
        .expect(u"\uD7FF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char16_t)
        .input(u8"\U00010000")
        .expect(u"\U00010000")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char16_t)
        .input(u8"\U0010FFFF")
        .expect(u"\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRUNCATING_AT(4, u"abcd") (strf::unsafe_transcode("abcdef"));
    TEST_TRUNCATING_AT(2, u"ab")   (strf::unsafe_transcode(u8"ab\u0080"));
    TEST_TRUNCATING_AT(2, u"ab")   (strf::unsafe_transcode(u8"ab\u0800"));
    TEST_TRUNCATING_AT(2, u"ab")   (strf::unsafe_transcode(u8"ab\uD7FF"));
    TEST_TRUNCATING_AT(2, u"ab")   (strf::unsafe_transcode(u8"ab\U00010000"));
    TEST_TRUNCATING_AT(3, u"ab")   (strf::unsafe_transcode(u8"ab\U00010000"));

    TEST_TRUNCATING_AT(3, u"ab\u0080")     (strf::unsafe_transcode(u8"ab\u0080"));
    TEST_TRUNCATING_AT(3, u"ab\u0800")     (strf::unsafe_transcode(u8"ab\u0800"));
    TEST_TRUNCATING_AT(3, u"ab\uD7FF")     (strf::unsafe_transcode(u8"ab\uD7FF"));
    TEST_TRUNCATING_AT(4, u"ab\U00010000") (strf::unsafe_transcode(u8"ab\U00010000"));
    TEST_TRUNCATING_AT(4, u"ab\U0010FFFF") (strf::unsafe_transcode(u8"ab\U0010FFFF"));

    TEST_CALLING_RECYCLE_AT(2, u"ab\u0080")     (strf::unsafe_transcode(u8"ab\u0080"));
    TEST_TRUNCATING_AT     (3, u"ab\u0080")     (strf::unsafe_transcode(u8"ab\u0080"));
    TEST_CALLING_RECYCLE_AT(2, u"ab\u0800")     (strf::unsafe_transcode(u8"ab\u0800"));
    TEST_TRUNCATING_AT     (3, u"ab\u0800")     (strf::unsafe_transcode(u8"ab\u0800"));
    TEST_CALLING_RECYCLE_AT(2, u"ab\uD7FF")     (strf::unsafe_transcode(u8"ab\uD7FF"));
    TEST_TRUNCATING_AT     (3, u"ab\uD7FF")     (strf::unsafe_transcode(u8"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT(2, u"ab\U00010000") (strf::unsafe_transcode(u8"ab\U00010000"));
    TEST_TRUNCATING_AT     (4, u"ab\U00010000") (strf::unsafe_transcode(u8"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(3, u"ab\U0010FFFF") (strf::unsafe_transcode(u8"ab\U0010FFFF"));
    TEST_TRUNCATING_AT     (4, u"ab\U0010FFFF") (strf::unsafe_transcode(u8"ab\U0010FFFF"));

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char16_t)
        .input(u8"ab\U0010FFFF")
        .destination_size(3)
        .expect(u"ab")
        .expect_stop_reason(strf::transcode_stop_reason::reached_limit)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char16_t)
        .input(u8"ab\U0010FFFF")
        .destination_size(0)
        .expect(u"")
        .expect_stop_reason(strf::transcode_stop_reason::reached_limit)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
}


STRF_TEST_FUNC void utf8_to_utf16_valid_sequences()
{
    TEST_UTF_TRANSCODE(char8_t, char16_t)
        .input(u8"ab")
        .expect(u"ab")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char16_t)
        .input(u8"\u0080")
        .expect(u"\u0080")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char16_t)
        .input(u8"\u0800")
        .expect(u"\u0800")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char16_t)
        .input(u8"\uD7FF")
        .expect(u"\uD7FF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char16_t)
        .input(u8"\U00010000")
        .expect(u"\U00010000")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char16_t)
        .input(u8"\U0010FFFF")
        .expect(u"\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST(u" ") (strf::sani("") > 1);
    TEST(u" abc") (strf::sani("abc") > 4);
    TEST(u" ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        (strf::sani(u8"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF") > 8);

    TEST_UTF_TRANSCODE(char8_t, char16_t)
        .input(u8"ab\U0010FFFF")
        .destination_size(3)
        .expect(u"ab")
        .expect_stop_reason(strf::transcode_stop_reason::reached_limit)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char16_t)
        .input(u8"ab\U0010FFFF")
        .destination_size(0)
        .expect(u"")
        .expect_stop_reason(strf::transcode_stop_reason::reached_limit)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});


    TEST_TRUNCATING_AT(4, u"abcd")   (strf::sani("abcdef"));
    TEST_TRUNCATING_AT(6, u"abcdef") (strf::sani("abcdef"));

    TEST_TRUNCATING_AT(2, u"ab") (strf::sani(u8"ab\u0080"));
    TEST_TRUNCATING_AT(2, u"ab") (strf::sani(u8"ab\u0800"));
    TEST_TRUNCATING_AT(2, u"ab") (strf::sani(u8"ab\uD7FF"));
    TEST_TRUNCATING_AT(2, u"ab") (strf::sani(u8"ab\U00010000"));
    TEST_TRUNCATING_AT(3, u"ab") (strf::sani(u8"ab\U00010000"));

    TEST_TRUNCATING_AT(3, u"ab\u0080")     (strf::sani(u8"ab\u0080"));
    TEST_TRUNCATING_AT(3, u"ab\u0800")     (strf::sani(u8"ab\u0800"));
    TEST_TRUNCATING_AT(3, u"ab\uD7FF")     (strf::sani(u8"ab\uD7FF"));
    TEST_TRUNCATING_AT(4, u"ab\U00010000") (strf::sani(u8"ab\U00010000"));
    TEST_TRUNCATING_AT(4, u"ab\U0010FFFF") (strf::sani(u8"ab\U0010FFFF"));

    TEST_CALLING_RECYCLE_AT(2, u"ab\u0080")     (strf::sani(u8"ab\u0080"));
    TEST_TRUNCATING_AT     (3, u"ab\u0080")     (strf::sani(u8"ab\u0080"));
    TEST_CALLING_RECYCLE_AT(2, u"ab\u0800")     (strf::sani(u8"ab\u0800"));
    TEST_TRUNCATING_AT     (3, u"ab\u0800")     (strf::sani(u8"ab\u0800"));
    TEST_CALLING_RECYCLE_AT(2, u"ab\uD7FF")     (strf::sani(u8"ab\uD7FF"));
    TEST_TRUNCATING_AT     (3, u"ab\uD7FF")     (strf::sani(u8"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT(2, u"ab\U00010000") (strf::sani(u8"ab\U00010000"));
    TEST_TRUNCATING_AT     (4, u"ab\U00010000") (strf::sani(u8"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(3, u"ab\U0010FFFF") (strf::sani(u8"ab\U0010FFFF"));
    TEST_TRUNCATING_AT     (4, u"ab\U0010FFFF") (strf::sani(u8"ab\U0010FFFF"));

    {
        // when surrogates are allowed
        const char16_t u16str_D800[] = {u' ', 0xD800, 0};
        const char16_t u16str_DBFF[] = {u' ', 0xDBFF, 0};
        const char16_t u16str_DC00[] = {u' ', 0xDC00, 0};
        const char16_t u16str_DFFF[] = {u' ', 0xDFFF, 0};

        TEST(u16str_D800) .with(strf::surrogate_policy::lax) (strf::sani("\xED\xA0\x80") > 2);
        TEST(u16str_DBFF) .with(strf::surrogate_policy::lax) (strf::sani("\xED\xAF\xBF") > 2);
        TEST(u16str_DC00) .with(strf::surrogate_policy::lax) (strf::sani("\xED\xB0\x80") > 2);
        TEST(u16str_DFFF) .with(strf::surrogate_policy::lax) (strf::sani("\xED\xBF\xBF") > 2);

        TEST_CALLING_RECYCLE_AT(1, u16str_D800)
            .with(strf::surrogate_policy::lax) (strf::sani("\xED\xA0\x80") > 2);
        TEST_TRUNCATING_AT     (2, u16str_D800)
            .with(strf::surrogate_policy::lax) (strf::sani("\xED\xA0\x80") > 2);
        TEST_TRUNCATING_AT     (1, u" ")
            .with(strf::surrogate_policy::lax) (strf::sani("\xED\xA0\x80") > 2);

        TEST_UTF_TRANSCODE(char, char16_t)
            .input(" \xED\xA0\x80")
            .expect(u16str_D800)
            .flags(strf::transcode_flags::lax_surrogate_policy)
            .expect_stop_reason(strf::transcode_stop_reason::completed);
        TEST_UTF_TRANSCODE(char, char16_t)
            .input(" \xED\xAF\xBF")
            .expect(u16str_DBFF)
            .flags(strf::transcode_flags::lax_surrogate_policy)
            .expect_stop_reason(strf::transcode_stop_reason::completed);
        TEST_UTF_TRANSCODE(char, char16_t)
            .input(" \xED\xB0\x80")
            .expect(u16str_DC00)
            .flags(strf::transcode_flags::lax_surrogate_policy)
            .expect_stop_reason(strf::transcode_stop_reason::completed);
        TEST_UTF_TRANSCODE(char, char16_t)
            .input(" \xED\xBF\xBF")
            .expect(u16str_DFFF)
            .flags(strf::transcode_flags::lax_surrogate_policy)
            .expect_stop_reason(strf::transcode_stop_reason::completed);
    }
}

STRF_TEST_FUNC void utf8_to_utf16_invalid_sequences()
{
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xBF")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    const auto flags = strf::transcode_flags::stop_on_unsupported_codepoint; // should have no effect
    // sample from Tabble 3-8 of Unicode standard
    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xF1\x80\x80\xE1\x80\xC0 ")
        .expect(u" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF1',  '\x80',  '\x80'}, {'\xE1',  '\x80'}, {'\xC0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xF1\x80\x80\xE1\x80\xC0 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF1',  '\x80',  '\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // missing leading byte
    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xBF ")
        .expect(u" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xBF ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // missing leading byte
    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \x80\x80 ")
        .expect(u" \uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\x80\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // overlong sequence
    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xC1\xBF ")
        .expect(u" \uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xC1'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xC1\xBF ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xC1'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // overlong sequence
    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xE0\x9F\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xE0'}, {'\x9F'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xE0\x9F\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xE0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // overlong sequence with extra continuation bytes
    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xC1\xBF\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xC1'}, {'\xBF'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xC1\xBF\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xC1'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // overlong sequence with extra continuation bytes
    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xE0\x9F\x80\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xE0'}, {'\x9F'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xE0\x9F\x80\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xE0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // overlong sequence
    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xF0\x8F\xBF\xBF ")
        .expect(u" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF0'}, {'\x8F'}, {'\xBF'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xF0\x8F\xBF\xBF ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // overlong sequence with extra continuation bytes
    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xF0\x8F\xBF\xBF\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF0'}, {'\x8F'}, {'\xBF'}, {'\xBF'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xF0\x8F\xBF\xBF\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // codepoint too big
    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xF4\xBF\xBF\xBF ")
        .expect(u" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF4'}, {'\xBF'}, {'\xBF'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xF4\xBF\xBF\xBF ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF4'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xF4\x90\x80\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF4'}, {'\x90'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xF4\x90\x80\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF4'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xF5\x80\x80\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF5'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xF5\x80\x80\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF5'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xF6\x80\x80\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF6'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xF6\x80\x80\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF6'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xF7\x80\x80\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF7'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xF7\x80\x80\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF7'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xF8\x80\x80\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF8'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xF8\x80\x80\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF8'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xF9\x80\x80\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF9'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xF9\x80\x80\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF9'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xFA\x80\x80\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFA'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xF9\x80\x80\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF9'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xFB\x80\x80\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFB'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xFB\x80\x80\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xFB'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xFC\x80\x80\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFC'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xFC\x80\x80\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xFC'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xFD\x80\x80\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFD'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xFD\x80\x80\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xFD'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xFE\x80\x80\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFE'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xFE\x80\x80\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xFE'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xFF\x80\x80\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFF'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xFF\x80\x80\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xFF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xFF\x80\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFF'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);

    // codepoint too big with extra continuation bytes
    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xF5\x90\x80\x80\x80\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF5'}, {'\x90'}, {'\x80'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xF5\x90\x80\x80\x80\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF5'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // missing continuation byte(s)
    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xC2 ")
        .expect(u" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xC2'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xC2 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xC2'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xE0 ")
        .expect(u" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xE0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xE0 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xE0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xE0\xA0 ")
        .expect(u" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xE0', '\xA0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xE0\xA0 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xE0', '\xA0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xE1 ")
        .expect(u" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xE1'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xE1 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xE1'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xF0\x90\xBF ")
        .expect(u" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF0', '\x90', '\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xF0\x90\xBF ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF0', '\x90', '\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xF1 ")
        .expect(u" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF1'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xF1 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF1'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xF1\x81 ")
        .expect(u" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF1', '\x81'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xF1\x81 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF1', '\x81'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xF1\x81\x81 ")
        .expect(u" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF1', '\x81', '\x81'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xF1\x81\x81 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF1', '\x81', '\x81'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    // surrogate
    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xED\xA0\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xA0'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xED\xA0\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xED\xAF\xBF ")
        .expect(u" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xAF'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xED\xAF\xBF ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xED\xB0\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xB0'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xED\xB0\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xED\xB0\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xB0'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xED\xB0\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xED\xA0\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xA0'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xED\xA0\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xED\xAF\xBF ")
        .expect(u" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xAF'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xED\xAF\xBF ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xED\xB0\x80 ")
        .expect(u" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xB0'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xED\xB0\x80 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xED\xBF\xBF ")
        .expect(u" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xBF'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xED\xBF\xBF ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // missing continuation, but could only be a surrogate.
    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xED\xA0 ")
        .expect(u" \uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xA0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xED\xA0 ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xED\xBF ")
        .expect(u" \uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xED\xBF ")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // missing continuation.
    // It could only be a surrogate, but surrogates are allowed now.
    // So the two bytes are treated as a single invalid sequence
    // (i.e. only one \uFFFD is printed )
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xED\xA0")
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(u"\uFFFD")
        .expect_invalid_sequences({{'\xED', '\xA0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xED\xA0")
        .destination_size(0)
        .flags(strf::transcode_flags::lax_surrogate_policy |
               strf::transcode_flags::stop_on_invalid_sequence)
        .expect(u"")
        .expect_invalid_sequences({{'\xED', '\xA0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xED\xBF")
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(u"\uFFFD")
        .expect_invalid_sequences({{'\xED', '\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xED\xBF")
        .destination_size(0)
        .flags(strf::transcode_flags::lax_surrogate_policy |
               strf::transcode_flags::stop_on_invalid_sequence)
        .expect(u"")
        .expect_invalid_sequences({{'\xED', '\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // missing continuation. Now it starts with \xED, but it is not a surrogate
    TEST_UTF_TRANSCODE(char, char16_t)
        .input(" \xED\x9F ")
        .expect(u" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED', '\x9F'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xED\x9F")
        .destination_size(0)
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect(u"")
        .expect_invalid_sequences({{'\xED', '\x9F'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // cover when recycle() needs to be called
    TEST_CALLING_RECYCLE_AT(2, u" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xA0\x80") > 4);
    TEST_TRUNCATING_AT     (4, u" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xA0\x80") > 4);
    TEST_TRUNCATING_AT     (2, u" \uFFFD")              (strf::sani("\xED\xA0\x80") > 4);

    // When the destination.good() is false
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("\xBF")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);
    TEST_UTF_TRANSCODE(char, char16_t)
        .input("_\xBF")
        .destination_size(0)
        .expect(u"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::reached_limit);
}

struct invalid_seq_counter: strf::transcoding_error_notifier {
    void STRF_HD invalid_sequence(int, const char*, const void*, std::ptrdiff_t) override {
        ++ notifications_count;
    }
    std::ptrdiff_t notifications_count = 0;
};

#if defined(__cpp_exceptions) && __cpp_exceptions  && ! defined(__CUDACC__)

struct dummy_exception: public std::exception {};

struct notifier_that_throws : strf::transcoding_error_notifier {
    void STRF_HD invalid_sequence(int, const char*, const void*, std::ptrdiff_t) override {
        throw dummy_exception{};
    }
};

#endif // __cpp_exceptions

STRF_TEST_FUNC void utf8_to_utf16_error_notifier()
{
    {
        invalid_seq_counter notifier;
        strf::transcoding_error_notifier_ptr notifier_ptr{&notifier};

        TEST(u"\uFFFD\uFFFD\uFFFD").with(notifier_ptr) (strf::sani("\xED\xA0\x80"));
        TEST_EQ(notifier.notifications_count, 3);

        notifier.notifications_count = 0;
        TEST_TRUNCATING_AT(1, u"\uFFFD").with(notifier_ptr) (strf::sani("\xED\xA0\x80"));
        TEST_TRUE(notifier.notifications_count > 0);
    }
#if defined(__cpp_exceptions) && __cpp_exceptions  && ! defined(__CUDACC__)
    {
        // check that an exception can be thrown, i.e,
        // ensure there is no `noexcept` blocking it
        notifier_that_throws notifier;
        strf::transcoding_error_notifier_ptr notifier_ptr{&notifier};
        bool thrown = false;
        try {
            char16_t buff[10];
            strf::to(buff) .with(notifier_ptr) (strf::sani("\xED\xA0\x80"));
        } catch (dummy_exception&) {
            thrown = true;
        } catch(...) {
        }
        TEST_TRUE(thrown);
    }

#endif // defined(__cpp_exceptions)
}

STRF_TEST_FUNC void utf8_to_utf16_find_transcoder()
{
#if ! defined(__CUDACC__)

    using static_transcoder_type = strf::static_transcoder
        <char, char16_t, strf::csid_utf8, strf::csid_utf16>;

    const strf::dynamic_charset<char>     dyn_utf8  = strf::utf8_t<char>{}.to_dynamic();
    const strf::dynamic_charset<char16_t> dyn_utf16 = strf::utf16_t<char16_t>{}.to_dynamic();
    const strf::dynamic_transcoder<char, char16_t> tr = strf::find_transcoder(dyn_utf8, dyn_utf16);

    TEST_TRUE(tr.transcode_func()      == static_transcoder_type::transcode);
    TEST_TRUE(tr.transcode_size_func() == static_transcoder_type::transcode_size);

#endif // defined(__CUDACC__)

    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char, char16_t, strf::csid_utf8, strf::csid_utf16 >
                       , decltype(strf::find_transcoder( strf::utf_t<char>{}
                                                       , strf::utf_t<char16_t>{})) >
                  :: value));
}


} // unnamed namespace


STRF_TEST_FUNC void test_utf8_to_utf16()
{
    utf8_to_utf16_unsafe_transcode();
    utf8_to_utf16_valid_sequences();
    utf8_to_utf16_invalid_sequences();
    utf8_to_utf16_error_notifier();
    utf8_to_utf16_find_transcoder();
}

REGISTER_STRF_TEST(test_utf8_to_utf16)
