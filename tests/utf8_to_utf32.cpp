//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils/transcoding.hpp"

#ifndef __cpp_char8_t
#   if __GNUC__ >= 11
#       pragma GCC diagnostic ignored "-Wc++20-compat"
#   endif
using char8_t = char;
#endif

namespace {

STRF_TEST_FUNC void utf8_to_utf32_unsafe_transcode()
{
    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char32_t)
        .input(u8"ab")
        .expect(U"ab")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char32_t)
        .input(u8"\u0080")
        .expect(U"\u0080")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char32_t)
        .input(u8"\u0800")
        .expect(U"\u0800")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char32_t)
        .input(u8"\uD7FF")
        .expect(U"\uD7FF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char32_t)
        .input(u8"\U00010000")
        .expect(U"\U00010000")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char32_t)
        .input(u8"\U0010FFFF")
        .expect(U"\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST(U"\U00010000") (strf::unsafe_transcode(u8"\U00010000"));

    TEST(U" ") (strf::unsafe_transcode("") > 1);
    TEST(U"\U0010FFFF") (strf::unsafe_transcode("\xF4\x8F\xBF\xBF"));

    TEST(U" abc") (strf::unsafe_transcode("abc") > 4);
    TEST(U" ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        (strf::unsafe_transcode(u8"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF") > 8);

    TEST_TRUNCATING_AT(2, U"ab")     (strf::unsafe_transcode("abcdef"));
    TEST_TRUNCATING_AT(6, U"abcdef") (strf::unsafe_transcode("abcdef"));

    TEST_TRUNCATING_AT(2, U"ab") (strf::unsafe_transcode(u8"ab\u0080"));
    TEST_TRUNCATING_AT(2, U"ab") (strf::unsafe_transcode(u8"ab\u0800"));
    TEST_TRUNCATING_AT(2, U"ab") (strf::unsafe_transcode(u8"ab\uD7FF"));
    TEST_TRUNCATING_AT(2, U"ab") (strf::unsafe_transcode(u8"ab\U00010000"));

    TEST_TRUNCATING_AT(3, U"ab\u0080")     (strf::unsafe_transcode(u8"ab\u0080"));
    TEST_TRUNCATING_AT(3, U"ab\u0800")     (strf::unsafe_transcode(u8"ab\u0800"));
    TEST_TRUNCATING_AT(3, U"ab\uD7FF")     (strf::unsafe_transcode(u8"ab\uD7FF"));
    TEST_TRUNCATING_AT(3, U"ab\U00010000") (strf::unsafe_transcode(u8"ab\U00010000"));
    TEST_TRUNCATING_AT(3, U"ab\U0010FFFF") (strf::unsafe_transcode(u8"ab\U0010FFFF"));

    TEST_CALLING_RECYCLE_AT(2, U"ab\u0080")     (strf::unsafe_transcode(u8"ab\u0080"));
    TEST_TRUNCATING_AT     (3, U"ab\u0080")     (strf::unsafe_transcode(u8"ab\u0080"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\u0800")     (strf::unsafe_transcode(u8"ab\u0800"));
    TEST_TRUNCATING_AT     (3, U"ab\u0800")     (strf::unsafe_transcode(u8"ab\u0800"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\uD7FF")     (strf::unsafe_transcode(u8"ab\uD7FF"));
    TEST_TRUNCATING_AT     (3, U"ab\uD7FF")     (strf::unsafe_transcode(u8"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\U00010000") (strf::unsafe_transcode(u8"ab\U00010000"));
    TEST_TRUNCATING_AT     (4, U"ab\U00010000") (strf::unsafe_transcode(u8"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\U0010FFFF") (strf::unsafe_transcode(u8"ab\U0010FFFF"));
    TEST_TRUNCATING_AT     (3, U"ab\U0010FFFF") (strf::unsafe_transcode(u8"ab\U0010FFFF"));

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char32_t)
        .input(u8"ab\U0010FFFF")
        .destination_size(2)
        .expect(U"ab")
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char32_t)
        .input(u8"ab\U0010FFFF")
        .destination_size(0)
        .expect(U"")
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
}

STRF_TEST_FUNC void utf8_to_utf32_valid_sequences()
{
    TEST_UTF_TRANSCODE(char8_t, char32_t)
        .input(u8"ab")
        .expect(U"ab")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char32_t)
        .input(u8"\u0080")
        .expect(U"\u0080")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char32_t)
        .input(u8"\u0800")
        .expect(U"\u0800")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char32_t)
        .input(u8"\uD7FF")
        .expect(U"\uD7FF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char32_t)
        .input(u8"\U00010000")
        .expect(U"\U00010000")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char32_t)
        .input(u8"\U0010FFFF")
        .expect(U"\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST(U"\U00010000") (strf::sani(u8"\U00010000"));

    TEST(U" ") (strf::sani("") > 1);
    TEST(U"\U0010FFFF") (strf::sani("\xF4\x8F\xBF\xBF"));

    TEST(U" abc") (strf::sani("abc") > 4);
    TEST(U" ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        (strf::sani(u8"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF") > 8);

    TEST_TRUNCATING_AT(2, U"ab")     (strf::sani("abcdef"));
    TEST_TRUNCATING_AT(6, U"abcdef") (strf::sani("abcdef"));

    TEST_TRUNCATING_AT(2, U"ab") (strf::sani(u8"ab\u0080"));
    TEST_TRUNCATING_AT(2, U"ab") (strf::sani(u8"ab\u0800"));
    TEST_TRUNCATING_AT(2, U"ab") (strf::sani(u8"ab\uD7FF"));
    TEST_TRUNCATING_AT(2, U"ab") (strf::sani(u8"ab\U00010000"));

    TEST_TRUNCATING_AT(3, U"ab\u0080")     (strf::sani(u8"ab\u0080"));
    TEST_TRUNCATING_AT(3, U"ab\u0800")     (strf::sani(u8"ab\u0800"));
    TEST_TRUNCATING_AT(3, U"ab\uD7FF")     (strf::sani(u8"ab\uD7FF"));
    TEST_TRUNCATING_AT(3, U"ab\U00010000") (strf::sani(u8"ab\U00010000"));
    TEST_TRUNCATING_AT(3, U"ab\U0010FFFF") (strf::sani(u8"ab\U0010FFFF"));

    TEST_CALLING_RECYCLE_AT(2, U"ab\u0080")     (strf::sani(u8"ab\u0080"));
    TEST_TRUNCATING_AT     (3, U"ab\u0080")     (strf::sani(u8"ab\u0080"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\u0800")     (strf::sani(u8"ab\u0800"));
    TEST_TRUNCATING_AT     (3, U"ab\u0800")     (strf::sani(u8"ab\u0800"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\uD7FF")     (strf::sani(u8"ab\uD7FF"));
    TEST_TRUNCATING_AT     (3, U"ab\uD7FF")     (strf::sani(u8"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\U00010000") (strf::sani(u8"ab\U00010000"));
    TEST_TRUNCATING_AT     (4, U"ab\U00010000") (strf::sani(u8"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\U0010FFFF") (strf::sani(u8"ab\U0010FFFF"));
    TEST_TRUNCATING_AT     (3, U"ab\U0010FFFF") (strf::sani(u8"ab\U0010FFFF"));

    TEST_UTF_TRANSCODE(char8_t, char32_t)
        .input(u8"ab\U0010FFFF")
        .destination_size(2)
        .expect(U"ab")
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char32_t)
        .input(u8"ab\U0010FFFF")
        .destination_size(0)
        .expect(U"")
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    {
        // when surrogates are allowed
        const char32_t u32str_D800[] = {U' ', 0xD800, 0};
        const char32_t u32str_DBFF[] = {U' ', 0xDBFF, 0};
        const char32_t u32str_DC00[] = {U' ', 0xDC00, 0};
        const char32_t u32str_DFFF[] = {U' ', 0xDFFF, 0};

        TEST_UTF_TRANSCODE(char, char32_t)
            .input(" \xED\xA0\x80")
            .expect(u32str_D800)
            .flags(strf::transcode_flags::lax_surrogate_policy)
            .expect_stop_reason(strf::transcode_stop_reason::completed);
        TEST_UTF_TRANSCODE(char, char32_t)
            .input(" \xED\xAF\xBF")
            .expect(u32str_DBFF)
            .flags(strf::transcode_flags::lax_surrogate_policy)
            .expect_stop_reason(strf::transcode_stop_reason::completed);
        TEST_UTF_TRANSCODE(char, char32_t)
            .input(" \xED\xB0\x80")
            .expect(u32str_DC00)
            .flags(strf::transcode_flags::lax_surrogate_policy)
            .expect_stop_reason(strf::transcode_stop_reason::completed);
        TEST_UTF_TRANSCODE(char, char32_t)
            .input(" \xED\xBF\xBF")
            .expect(u32str_DFFF)
            .flags(strf::transcode_flags::lax_surrogate_policy)
            .expect_stop_reason(strf::transcode_stop_reason::completed);
    }

    // with flag stop_on_invalid_sequence
    TEST_UTF_TRANSCODE(char8_t, char32_t)
        .input(u8"ab")
        .expect(U"ab")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char32_t)
        .input(u8"\u0080")
        .expect(U"\u0080")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char32_t)
        .input(u8"\u0800")
        .expect(U"\u0800")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char32_t)
        .input(u8"\uD7FF")
        .expect(U"\uD7FF")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char32_t)
        .input(u8"\U00010000")
        .expect(U"\U00010000")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char32_t)
        .input(u8"\U0010FFFF")
        .expect(U"\U0010FFFF")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
}

STRF_TEST_FUNC void utf8_to_utf32_invalid_sequences()
{

    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xBF")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    const auto flags = strf::transcode_flags::stop_on_unsupported_codepoint; // should have no effect
    // sample from Tabble 3-8 of Unicode standard
    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xF1\x80\x80\xE1\x80\xC0 ")
        .expect(U" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF1',  '\x80',  '\x80'}, {'\xE1',  '\x80'}, {'\xC0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xF1\x80\x80\xE1\x80\xC0 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF1',  '\x80',  '\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // missing leading byte
    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xBF ")
        .expect(U" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xBF ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // missing leading byte
    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \x80\x80 ")
        .expect(U" \uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\x80\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // overlong sequence
    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xC1\xBF ")
        .expect(U" \uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xC1'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xC1\xBF ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xC1'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // overlong sequence
    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xE0\x9F\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xE0'}, {'\x9F'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xE0\x9F\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xE0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // overlong sequence with extra continuation bytes
    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xC1\xBF\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xC1'}, {'\xBF'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xC1\xBF\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xC1'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // overlong sequence with extra continuation bytes
    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xE0\x9F\x80\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xE0'}, {'\x9F'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xE0\x9F\x80\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xE0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // overlong sequence
    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xF0\x8F\xBF\xBF ")
        .expect(U" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF0'}, {'\x8F'}, {'\xBF'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xF0\x8F\xBF\xBF ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // overlong sequence with extra continuation bytes
    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xF0\x8F\xBF\xBF\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF0'}, {'\x8F'}, {'\xBF'}, {'\xBF'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xF0\x8F\xBF\xBF\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // codepoint too big
    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xF4\xBF\xBF\xBF ")
        .expect(U" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF4'}, {'\xBF'}, {'\xBF'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xF4\xBF\xBF\xBF ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF4'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xF4\x90\x80\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF4'}, {'\x90'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xF4\x90\x80\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF4'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xF5\x80\x80\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF5'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xF5\x80\x80\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF5'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xF6\x80\x80\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF6'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xF6\x80\x80\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF6'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xF7\x80\x80\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF7'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xF7\x80\x80\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF7'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xF8\x80\x80\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF8'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xF8\x80\x80\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF8'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xF9\x80\x80\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF9'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xF9\x80\x80\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF9'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xFA\x80\x80\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFA'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xF9\x80\x80\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF9'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xFB\x80\x80\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFB'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xFB\x80\x80\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xFB'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xFC\x80\x80\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFC'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xFC\x80\x80\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xFC'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xFD\x80\x80\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFD'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xFD\x80\x80\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xFD'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xFE\x80\x80\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFE'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xFE\x80\x80\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xFE'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xFF\x80\x80\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFF'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xFF\x80\x80\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xFF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xFF\x80\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFF'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);

    // codepoint too big with extra continuation bytes
    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xF5\x90\x80\x80\x80\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF5'}, {'\x90'}, {'\x80'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xF5\x90\x80\x80\x80\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF5'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // missing continuation byte(s)
    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xC2 ")
        .expect(U" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xC2'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xC2 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xC2'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xE0 ")
        .expect(U" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xE0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xE0 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xE0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xE0\xA0 ")
        .expect(U" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xE0', '\xA0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xE0\xA0 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xE0', '\xA0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xE1 ")
        .expect(U" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xE1'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xE1 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xE1'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xF0\x90\xBF ")
        .expect(U" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF0', '\x90', '\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xF0\x90\xBF ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF0', '\x90', '\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xF1 ")
        .expect(U" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF1'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xF1 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF1'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xF1\x81 ")
        .expect(U" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF1', '\x81'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xF1\x81 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF1', '\x81'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xF1\x81\x81 ")
        .expect(U" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF1', '\x81', '\x81'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xF1\x81\x81 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF1', '\x81', '\x81'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    // surrogate
    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xED\xA0\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xA0'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xED\xA0\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xED\xAF\xBF ")
        .expect(U" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xAF'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xED\xAF\xBF ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xED\xB0\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xB0'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xED\xB0\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xED\xB0\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xB0'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xED\xB0\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xED\xA0\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xA0'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xED\xA0\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xED\xAF\xBF ")
        .expect(U" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xAF'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xED\xAF\xBF ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xED\xB0\x80 ")
        .expect(U" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xB0'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xED\xB0\x80 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xED\xBF\xBF ")
        .expect(U" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xBF'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xED\xBF\xBF ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // missing continuation, but could only be a surrogate.
    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xED\xA0 ")
        .expect(U" \uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xA0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xED\xA0 ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xED\xBF ")
        .expect(U" \uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xED\xBF ")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // missing continuation.
    // It could only be a surrogate, but surrogates are allowed now.
    // So the two bytes are treated as a single invalid sequence
    // (i.e. only one \uFFFD is printed )
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xED\xA0")
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(U"\uFFFD")
        .expect_invalid_sequences({{'\xED', '\xA0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xED\xA0")
        .destination_size(0)
        .flags(strf::transcode_flags::lax_surrogate_policy |
               strf::transcode_flags::stop_on_invalid_sequence)
        .expect(U"")
        .expect_invalid_sequences({{'\xED', '\xA0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xED\xBF")
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(U"\uFFFD")
        .expect_invalid_sequences({{'\xED', '\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xED\xBF")
        .destination_size(0)
        .flags(strf::transcode_flags::lax_surrogate_policy |
               strf::transcode_flags::stop_on_invalid_sequence)
        .expect(U"")
        .expect_invalid_sequences({{'\xED', '\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // missing continuation. Now it starts with \xED, but it is not a surrogate
    TEST_UTF_TRANSCODE(char, char32_t)
        .input(" \xED\x9F ")
        .expect(U" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED', '\x9F'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xED\x9F")
        .destination_size(0)
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect(U"")
        .expect_invalid_sequences({{'\xED', '\x9F'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // cover when recycle() needs to be called
    TEST_CALLING_RECYCLE_AT(2, U" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xA0\x80") > 4);
    TEST_TRUNCATING_AT     (4, U" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xA0\x80") > 4);
    TEST_TRUNCATING_AT     (2, U" \uFFFD")              (strf::sani("\xED\xA0\x80") > 4);

    // When the destination.good() is false
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("\xBF")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);
    TEST_UTF_TRANSCODE(char, char32_t)
        .input("_\xBF")
        .destination_size(0)
        .expect(U"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
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

STRF_TEST_FUNC void utf8_to_utf32_error_notifier()
{
    {
        invalid_seq_counter notifier;
        strf::transcoding_error_notifier_ptr notifier_ptr{&notifier};

        TEST(U"\uFFFD\uFFFD\uFFFD").with(notifier_ptr) (strf::sani("\xED\xA0\x80"));
        TEST_EQ(notifier.notifications_count, 3);

        notifier.notifications_count = 0;
        TEST_TRUNCATING_AT(1, U"\uFFFD").with(notifier_ptr) (strf::sani("\xED\xA0\x80"));
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
            char32_t buff[10];
            strf::to(buff) .with(notifier_ptr) (strf::sani("\xED\xA0\x80"));
        } catch (dummy_exception&) {
            thrown = true;
        }
        TEST_TRUE(thrown);
    }
#endif // __cpp_exceptions
}

STRF_TEST_FUNC void utf8_to_utf32_find_transcoder()
{
#if ! defined(__CUDACC__)

    using static_transcoder_type = strf::static_transcoder
        <char, char32_t, strf::csid_utf8, strf::csid_utf32>;

    const strf::dynamic_charset<char>     dyn_utf8  = strf::utf8_t<char>{}.to_dynamic();
    const strf::dynamic_charset<char32_t> dyn_utf32 = strf::utf32_t<char32_t>{}.to_dynamic();
    const strf::dynamic_transcoder<char, char32_t> tr = strf::find_transcoder(dyn_utf8, dyn_utf32);

    TEST_TRUE(tr.transcode_func()      == static_transcoder_type::transcode);
    TEST_TRUE(tr.transcode_size_func() == static_transcoder_type::transcode_size);

#endif // defined(__CUDACC__)

    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char, char32_t, strf::csid_utf8, strf::csid_utf32 >
                       , decltype(strf::find_transcoder( strf::utf_t<char>{}
                                                       , strf::utf_t<char32_t>{})) >
                  :: value));
}

} // unnamed namespace

STRF_TEST_FUNC void test_utf8_to_utf32()
{
    utf8_to_utf32_unsafe_transcode();
    utf8_to_utf32_valid_sequences();
    utf8_to_utf32_invalid_sequences();
    utf8_to_utf32_error_notifier();
    utf8_to_utf32_find_transcoder();
}

REGISTER_STRF_TEST(test_utf8_to_utf32)
