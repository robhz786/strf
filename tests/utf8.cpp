//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


// // This test file aims to cover the following class template specializations:
//
// // The UTF-8 charset
// template <typename CharT>
// class static_charset<CharT, strf::csid_utf8>;
//
// // The UTF-8 sanitizer:
// template <typename SrcCharT, typename DstCharT>
// class static_transcoder<SrcCharT, DstCharT, strf::csid_utf8, strf::csid_utf8>;


#include "test_utils/transcoding.hpp"

#ifndef __cpp_char8_t
#   if __GNUC__ >= 11
#       pragma GCC diagnostic ignored "-Wc++20-compat"
#   endif
using char8_t = char;
#endif

namespace {
STRF_TEST_FUNC void utf8_to_utf8_unsafe_transcode()
{
    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char8_t)
        .input(u8"ab")
        .expect(u8"ab")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char8_t)
        .input(u8"\u0080")
        .expect(u8"\u0080")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char8_t)
        .input(u8"\u0800")
        .expect(u8"\u0800")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char8_t)
        .input(u8"\uD7FF")
        .expect(u8"\uD7FF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char8_t)
        .input(u8"\U00010000")
        .expect(u8"\U00010000")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char8_t)
        .input(u8"\U0010FFFF")
        .expect(u8"\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char8_t)
        .input (u8"ab\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF")
        .expect(u8"ab\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char8_t)
        .input(u8"abc")
        .expect(u8"ab")
        .destination_size(2)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char8_t)
        .input(u8"\U00010000")
        .expect(u8"")
        .destination_size(0)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char8_t)
        .input(u8"\U00010000")
        .expect(u8"")
        .destination_size(3)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char8_t)
        .input(u8"\U00010000")
        .expect(u8"")
        .destination_size(2)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char8_t)
        .input(u8"\U00010000")
        .expect(u8"")
        .destination_size(1)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char8_t)
        .input(u8"\uE000")
        .expect(u8"")
        .destination_size(2)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char8_t)
        .input(u8"\uE000")
        .expect(u8"")
        .destination_size(1)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);

    TEST_UTF_UNSAFE_TRANSCODE(char8_t, char8_t)
        .input(u8"\u0080")
        .expect(u8"")
        .destination_size(1)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);


    TEST_CALLING_RECYCLE_AT(7, u8"abc\U00010000_def") (strf::unsafe_transcode(u8"abc\U00010000_def"));
    TEST_CALLING_RECYCLE_AT(6, u8"abc\U00010000_def") (strf::unsafe_transcode(u8"abc\U00010000_def"));
    TEST_CALLING_RECYCLE_AT(5, u8"abc\U00010000_def") (strf::unsafe_transcode(u8"abc\U00010000_def"));
    TEST_CALLING_RECYCLE_AT(4, u8"abc\U00010000_def") (strf::unsafe_transcode(u8"abc\U00010000_def"));
    TEST_CALLING_RECYCLE_AT(3, u8"abc\U00010000_def") (strf::unsafe_transcode(u8"abc\U00010000_def"));

    TEST_CALLING_RECYCLE_AT(6, u8"abc\uE000_def") (strf::unsafe_transcode(u8"abc\uE000_def"));
    TEST_CALLING_RECYCLE_AT(5, u8"abc\uE000_def") (strf::unsafe_transcode(u8"abc\uE000_def"));
    TEST_CALLING_RECYCLE_AT(4, u8"abc\uE000_def") (strf::unsafe_transcode(u8"abc\uE000_def"));
    TEST_CALLING_RECYCLE_AT(3, u8"abc\uE000_def") (strf::unsafe_transcode(u8"abc\uE000_def"));

    TEST_CALLING_RECYCLE_AT(4, u8"abc\u0080_def") (strf::unsafe_transcode(u8"abc\u0080_def"));
    TEST_CALLING_RECYCLE_AT(3, u8"abc\u0080_def") (strf::unsafe_transcode(u8"abc\u0080_def"));

    TEST_CALLING_RECYCLE_AT(3, u8"abcdef") (strf::unsafe_transcode(u8"abcdef"));


    TEST_TRUNCATING_AT(7, u8"abc\U00010000") (strf::unsafe_transcode(u8"abc\U00010000_def"));
    TEST_TRUNCATING_AT(6, u8"abc") (strf::unsafe_transcode(u8"abc\U00010000_def"));
    TEST_TRUNCATING_AT(5, u8"abc") (strf::unsafe_transcode(u8"abc\U00010000_def"));
    TEST_TRUNCATING_AT(4, u8"abc") (strf::unsafe_transcode(u8"abc\U00010000_def"));
    TEST_TRUNCATING_AT(3, u8"abc") (strf::unsafe_transcode(u8"abc\U00010000_def"));

    TEST_TRUNCATING_AT(6, u8"abc\uE000") (strf::unsafe_transcode(u8"abc\uE000_def"));
    TEST_TRUNCATING_AT(5, u8"abc") (strf::unsafe_transcode(u8"abc\uE000_def"));
    TEST_TRUNCATING_AT(4, u8"abc") (strf::unsafe_transcode(u8"abc\uE000_def"));
    TEST_TRUNCATING_AT(3, u8"abc") (strf::unsafe_transcode(u8"abc\uE000_def"));

    TEST_TRUNCATING_AT(5, u8"abc\u0080") (strf::unsafe_transcode(u8"abc\u0080_def"));
    TEST_TRUNCATING_AT(4, u8"abc") (strf::unsafe_transcode(u8"abc\u0080_def"));
    TEST_TRUNCATING_AT(3, u8"abc") (strf::unsafe_transcode(u8"abc\u0080_def"));

    TEST_TRUNCATING_AT(3, u8"abc") (strf::unsafe_transcode(u8"abcdef"));


    // when using strf::transcode_flags::lax_surrogate_policy
    const char str_D800[] = "\xED\xA0\x80";
    const char str_DBFF[] = "\xED\xAF\xBF";
    const char str_DC00[] = "\xED\xB0\x80";
    const char str_DFFF[] = "\xED\xBF\xBF";
    //const char str_DFFF_D800_[] = {0xDFFF, 0xD800, u'_', 0};

    TEST_UTF_UNSAFE_TRANSCODE(char, char)
        .input(str_D800, str_DBFF, str_DC00, str_DFFF, str_D800, '_')
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(str_D800, str_DBFF, str_DC00, str_DFFF, str_D800, '_')
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char, char)
        .input("hello")
        .destination_size(0)
        .expect("")
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
}


STRF_TEST_FUNC void utf8_sani_valid_sequences()
{
    TEST_UTF_TRANSCODE(char8_t, char8_t)
        .input(u8"ab")
        .expect(u8"ab")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char8_t)
        .input(u8"\u0080")
        .expect(u8"\u0080")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char8_t)
        .input(u8"\u0800")
        .expect(u8"\u0800")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char8_t)
        .input(u8"\uD7FF")
        .expect(u8"\uD7FF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char8_t)
        .input(u8"\U00010000")
        .expect(u8"\U00010000")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char8_t)
        .input(u8"\U0010FFFF")
        .expect(u8"\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST(" ") (strf::sani("") > 1);
    TEST(" abc") (strf::sani("abc") > 4);
    TEST(u8"ab\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF")
        (strf::sani(u8"ab\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF") > 8);

    TEST_TRUNCATING_AT(4, u8"abcd") (strf::sani("abcdef"));

    TEST_TRUNCATING_AT(3, u8"ab") (strf::sani(u8"ab\u0080"));
    TEST_TRUNCATING_AT(4, u8"ab") (strf::sani(u8"ab\u0800"));
    TEST_TRUNCATING_AT(4, u8"ab") (strf::sani(u8"ab\uD7FF"));
    TEST_TRUNCATING_AT(5, u8"ab") (strf::sani(u8"ab\U00010000"));
    TEST_TRUNCATING_AT(5, u8"ab") (strf::sani(u8"ab\U0010FFFF"));

    TEST_TRUNCATING_AT(4, u8"ab\u0080")     (strf::sani(u8"ab\u0080"));
    TEST_TRUNCATING_AT(5, u8"ab\u0800")     (strf::sani(u8"ab\u0800"));
    TEST_TRUNCATING_AT(5, u8"ab\uD7FF")     (strf::sani(u8"ab\uD7FF"));
    TEST_TRUNCATING_AT(6, u8"ab\U00010000") (strf::sani(u8"ab\U00010000"));
    TEST_TRUNCATING_AT(6, u8"ab\U0010FFFF") (strf::sani(u8"ab\U0010FFFF"));

    TEST_CALLING_RECYCLE_AT(3, u8"ab\u0080")     (strf::sani(u8"ab\u0080"));
    TEST_CALLING_RECYCLE_AT(4, u8"ab\uD7FF")     (strf::sani(u8"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT(5, u8"ab\U00010000") (strf::sani(u8"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(5, u8"ab\U0010FFFF") (strf::sani(u8"ab\U0010FFFF"));

    TEST_UTF_TRANSCODE(char8_t, char)
        .input(u8"ab\U0010FFFF")
        .destination_size(3)
        .expect("ab")
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char8_t, char)
        .input(u8"ab\U0010FFFF")
        .destination_size(0)
        .expect("")
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    {
        // with flags stop_on_invalid_sequence (though sequences are still valid)

        TEST_UTF_TRANSCODE(char8_t, char8_t)
            .input(u8"\u0080")
            .expect(u8"\u0080")
            .flags(strf::transcode_flags::stop_on_invalid_sequence)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});

        TEST_UTF_TRANSCODE(char8_t, char8_t)
            .input(u8"\u0800")
            .expect(u8"\u0800")
            .flags(strf::transcode_flags::stop_on_invalid_sequence)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});

        TEST_UTF_TRANSCODE(char8_t, char8_t)
            .input(u8"\uD7FF")
            .expect(u8"\uD7FF")
            .flags(strf::transcode_flags::stop_on_invalid_sequence)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});

        TEST_UTF_TRANSCODE(char8_t, char8_t)
            .input(u8"\U00010000")
            .expect(u8"\U00010000")
            .flags(strf::transcode_flags::stop_on_invalid_sequence)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});

        TEST_UTF_TRANSCODE(char8_t, char8_t)
            .input(u8"\U0010FFFF")
            .expect(u8"\U0010FFFF")
            .flags(strf::transcode_flags::stop_on_invalid_sequence)
            .flags(strf::transcode_flags::stop_on_invalid_sequence)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
    }

    {
        // when surrogates are allowed
        const char str_D800[] = "\xED\xA0\x80";
        const char str_DBFF[] = "\xED\xAF\xBF";
        const char str_DC00[] = "\xED\xB0\x80";
        const char str_DFFF[] = "\xED\xBF\xBF";

        TEST_UTF_TRANSCODE(char, char)
            .input(str_D800)
            .flags(strf::transcode_flags::lax_surrogate_policy)
            .expect(str_D800)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char, char)
            .input(str_DBFF)
            .flags(strf::transcode_flags::lax_surrogate_policy)
            .expect(str_DBFF)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char, char)
            .input(str_DC00)
            .flags(strf::transcode_flags::lax_surrogate_policy)
            .expect(str_DC00)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char, char)
            .input(str_DFFF)
            .flags(strf::transcode_flags::lax_surrogate_policy)
            .expect(str_DFFF)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});

        const auto flags = ( strf::transcode_flags::lax_surrogate_policy |
                             strf::transcode_flags::stop_on_invalid_sequence |
                             strf::transcode_flags::stop_on_unsupported_codepoint );

        TEST_UTF_TRANSCODE(char, char)
            .input(str_D800)
            .flags(flags)
            .expect(str_D800)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char, char)
            .input(str_DBFF)
            .flags(flags)
            .expect(str_DBFF)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char, char)
            .input(str_DC00)
            .flags(flags)
            .expect(str_DC00)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char, char)
            .input(str_DFFF)
            .flags(flags)
            .expect(str_DFFF)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        // TEST_UTF_TRANSCODE(char, char8_t)
        //     .input(str_DBFF, str_DFFF)
        //     .flags(flags)
        //     .expect(u8"10FFFF")
        //     .expect_stop_reason(strf::transcode_stop_reason::completed)
        //     .expect_unsupported_codepoints({})
        //     .expect_invalid_sequences({});
    }
}

// TEST_INVALID_SEQS(\("[^"]+\)", \(.+\));
// TEST_UTF_TRANSCODE(char, char)
//     .input(\1)
//     .expect("")
//     .flags()
//     .expect_invalid_sequences({\2})
//     .expect_unsupported_codepoints({})
//     .expec_stop_reason(strf::transcode_stop_reason::completed);

// \(\\x[0-9A-Fa-f][0-9A-Fa-f]\)
// '\1',

// {" {{
// ,  *"} → }}
// , *", *" → }, {


STRF_TEST_FUNC void utf8_sani_invalid_sequences()
{
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xBF")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    const auto flags = strf::transcode_flags::stop_on_unsupported_codepoint; // should have no effect
    // sample from Tabble 3-8 of Unicode standard
    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xF1\x80\x80\xE1\x80\xC0 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF1',  '\x80',  '\x80'}, {'\xE1',  '\x80'}, {'\xC0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xF1\x80\x80\xE1\x80\xC0 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF1',  '\x80',  '\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // missing leading byte
    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xBF ")
        .expect(u8" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xBF ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // missing leading byte
    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \x80\x80 ")
        .expect(u8" \uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\x80\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // overlong sequence
    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xC1\xBF ")
        .expect(u8" \uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xC1'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xC1\xBF ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xC1'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // overlong sequence
    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xE0\x9F\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xE0'}, {'\x9F'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xE0\x9F\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xE0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // overlong sequence with extra continuation bytes
    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xC1\xBF\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xC1'}, {'\xBF'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xC1\xBF\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xC1'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // overlong sequence with extra continuation bytes
    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xE0\x9F\x80\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xE0'}, {'\x9F'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xE0\x9F\x80\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xE0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // overlong sequence
    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xF0\x8F\xBF\xBF ")
        .expect(u8" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF0'}, {'\x8F'}, {'\xBF'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xF0\x8F\xBF\xBF ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // overlong sequence with extra continuation bytes
    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xF0\x8F\xBF\xBF\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF0'}, {'\x8F'}, {'\xBF'}, {'\xBF'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xF0\x8F\xBF\xBF\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // codepoint too big
    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xF4\xBF\xBF\xBF ")
        .expect(u8" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF4'}, {'\xBF'}, {'\xBF'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xF4\xBF\xBF\xBF ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF4'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xF4\x90\x80\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF4'}, {'\x90'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xF4\x90\x80\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF4'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xF5\x80\x80\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF5'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xF5\x80\x80\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF5'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xF6\x80\x80\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF6'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xF6\x80\x80\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF6'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xF7\x80\x80\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF7'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xF7\x80\x80\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF7'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xF8\x80\x80\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF8'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xF8\x80\x80\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF8'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xF9\x80\x80\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF9'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xF9\x80\x80\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF9'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xFA\x80\x80\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFA'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xF9\x80\x80\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF9'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xFB\x80\x80\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFB'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xFB\x80\x80\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xFB'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xFC\x80\x80\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFC'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xFC\x80\x80\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xFC'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xFD\x80\x80\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFD'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xFD\x80\x80\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xFD'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xFE\x80\x80\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFE'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xFE\x80\x80\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xFE'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xFF\x80\x80\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFF'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xFF\x80\x80\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xFF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xFF\x80\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xFF'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);

    // codepoint too big with extra continuation bytes
    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xF5\x90\x80\x80\x80\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF5'}, {'\x90'}, {'\x80'}, {'\x80'}, {'\x80'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xF5\x90\x80\x80\x80\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF5'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // missing continuation byte(s)
    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xC2 ")
        .expect(u8" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xC2'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xC2 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xC2'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xE0 ")
        .expect(u8" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xE0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xE0 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xE0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xE0\xA0 ")
        .expect(u8" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xE0', '\xA0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xE0\xA0 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xE0', '\xA0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xE1 ")
        .expect(u8" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xE1'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xE1 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xE1'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xF0\x90\xBF ")
        .expect(u8" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF0', '\x90', '\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xF0\x90\xBF ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF0', '\x90', '\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xF1 ")
        .expect(u8" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF1'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xF1 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF1'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xF1\x81 ")
        .expect(u8" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF1', '\x81'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xF1\x81 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF1', '\x81'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xF1\x81\x81 ")
        .expect(u8" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xF1', '\x81', '\x81'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xF1\x81\x81 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xF1', '\x81', '\x81'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);


    // surrogate
    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xED\xA0\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xA0'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xED\xA0\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xED\xAF\xBF ")
        .expect(u8" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xAF'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xED\xAF\xBF ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xED\xB0\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xB0'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xED\xB0\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xED\xB0\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xB0'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xED\xB0\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xED\xA0\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xA0'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xED\xA0\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xED\xAF\xBF ")
        .expect(u8" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xAF'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xED\xAF\xBF ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xED\xB0\x80 ")
        .expect(u8" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xB0'}, {'\x80'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xED\xB0\x80 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xED\xBF\xBF ")
        .expect(u8" \uFFFD\uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xBF'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xED\xBF\xBF ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // missing continuation, but could only be a surrogate.
    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xED\xA0 ")
        .expect(u8" \uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xA0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xED\xA0 ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xED\xBF ")
        .expect(u8" \uFFFD\uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED'}, {'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xED\xBF ")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xED'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // missing continuation.
    // It could only be a surrogate, but surrogates are allowed now.
    // So the two bytes are treated as a single invalid sequence
    // (i.e. only one \uFFFD is printed )
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xED\xA0")
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(u8"\uFFFD")
        .expect_invalid_sequences({{'\xED', '\xA0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xED\xA0")
        .destination_size(0)
        .flags(strf::transcode_flags::lax_surrogate_policy |
               strf::transcode_flags::stop_on_invalid_sequence)
        .expect(u8"")
        .expect_invalid_sequences({{'\xED', '\xA0'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xED\xBF")
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(u8"\uFFFD")
        .expect_invalid_sequences({{'\xED', '\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xED\xBF")
        .destination_size(0)
        .flags(strf::transcode_flags::lax_surrogate_policy |
               strf::transcode_flags::stop_on_invalid_sequence)
        .expect(u8"")
        .expect_invalid_sequences({{'\xED', '\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // missing continuation. Now it starts with \xED, but it is not a surrogate
    TEST_UTF_TRANSCODE(char, char8_t)
        .input(" \xED\x9F ")
        .expect(u8" \uFFFD ")
        .flags(flags)
        .expect_invalid_sequences({{'\xED', '\x9F'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xED\x9F")
        .destination_size(0)
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect(u8"")
        .expect_invalid_sequences({{'\xED', '\x9F'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);

    // cover when recycle() needs to be called
    TEST_CALLING_RECYCLE_AT(3, u8" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xA0\x80") > 4);
    TEST_TRUNCATING_AT     (10, u8" \uFFFD\uFFFD\uFFFD") (strf::sani("\xED\xA0\x80") > 4);
    TEST_TRUNCATING_AT     (4, u8" \uFFFD")              (strf::sani("\xED\xA0\x80") > 4);


    // When the destination.good() is false
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("\xBF")
        .destination_size(0)
        .expect(u8"")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_invalid_sequences({{'\xBF'}})
        .expect_unsupported_codepoints({})
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence);
    TEST_UTF_TRANSCODE(char, char8_t)
        .input("_\xBF")
        .destination_size(0)
        .expect(u8"")
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

struct dummy_exception : std::exception {};
struct notifier_that_throws : strf::transcoding_error_notifier {
    void STRF_HD invalid_sequence(int, const char*, const void*, std::ptrdiff_t) override {
        throw dummy_exception{};
    }
};

#endif // __cpp_exceptions


STRF_TEST_FUNC void utf8_sani_error_notifier()
{
    {
        invalid_seq_counter notifier;
        strf::transcoding_error_notifier_ptr notifier_ptr{&notifier};

        TEST(u8"\uFFFD\uFFFD\uFFFD").with(notifier_ptr) (strf::sani("\xED\xA0\x80"));
        TEST_EQ(notifier.notifications_count, 3);

        notifier.notifications_count = 0;
        TEST_TRUNCATING_AT(3, u8"\uFFFD").with(notifier_ptr) (strf::sani("\xED\xA0\x80"));
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
            char buff[40];
            strf::to(buff) .with(notifier_ptr) (strf::sani("\xED\xA0\x80"));
        } catch (dummy_exception&) {
            thrown = true;
        }
        TEST_TRUE(thrown);
    }

#endif // defined(__cpp_exceptions)
}

STRF_TEST_FUNC void utf8_sani_find_transcoder()
{
#if ! defined(__CUDACC__)

    using static_transcoder_type = strf::static_transcoder
        <char, char, strf::csid_utf8, strf::csid_utf8>;

    const strf::dynamic_charset<char> dyn_cs  = strf::utf8_t<char>{}.to_dynamic();
    const strf::dynamic_transcoder<char, char> tr = strf::find_transcoder(dyn_cs, dyn_cs);

    TEST_TRUE(tr.transcode_func()      == static_transcoder_type::transcode);
    TEST_TRUE(tr.transcode_size_func() == static_transcoder_type::transcode_size);

#endif // defined(__CUDACC__)

    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char, char, strf::csid_utf8, strf::csid_utf8 >
                   , decltype(strf::find_transcoder( strf::utf_t<char>{}
                                                   , strf::utf_t<char>{})) >
                  :: value));
}

template <typename CharT, std::size_t N>
STRF_HD std::ptrdiff_t utf8_count_codepoints(const CharT (&str)[N])
{
    return strf::utf8_t<CharT>::count_codepoints(str, str + N - 1, 100000).count;
}

template <typename CharT, std::size_t N>
STRF_HD std::ptrdiff_t utf8_count_codepoints_fast(const CharT (&str)[N])
{
    return strf::utf8_t<CharT>::count_codepoints_fast(str, str + N - 1, 100000).count;
}

STRF_HD void utf8_codepoints_count()
{
    {   // test valid input
        TEST_EQ(0, utf8_count_codepoints(""));
        TEST_EQ(3, utf8_count_codepoints(u8"ab\u0080"));
        TEST_EQ(3, utf8_count_codepoints(u8"ab\u0800"));
        TEST_EQ(3, utf8_count_codepoints(u8"ab\uD7FF"));
        TEST_EQ(3, utf8_count_codepoints(u8"ab\uE000"));
        TEST_EQ(3, utf8_count_codepoints(u8"ab\U00010000"));
        TEST_EQ(3, utf8_count_codepoints(u8"ab\U0010FFFF"));

        TEST_EQ(0, utf8_count_codepoints_fast(u8""));
        TEST_EQ(3, utf8_count_codepoints_fast(u8"ab\u0080"));
        TEST_EQ(3, utf8_count_codepoints_fast(u8"ab\u0800"));
        TEST_EQ(3, utf8_count_codepoints_fast(u8"ab\uD7FF"));
        TEST_EQ(3, utf8_count_codepoints_fast(u8"ab\uE000"));
        TEST_EQ(3, utf8_count_codepoints_fast(u8"ab\U00010000"));
        TEST_EQ(3, utf8_count_codepoints_fast(u8"ab\U0010FFFF"));
    }
    {   // test invalid sequences
        TEST_EQ(3, utf8_count_codepoints("\xF1\x80\x80\xE1\x80\xC0"));
        TEST_EQ(1, utf8_count_codepoints("\xBF"));
        TEST_EQ(2, utf8_count_codepoints("\x80\x80"));
        TEST_EQ(2, utf8_count_codepoints("\xC1\xBF"));
        TEST_EQ(3, utf8_count_codepoints("\xE0\x9F\x80"));
        TEST_EQ(3, utf8_count_codepoints("\xC1\xBF\x80"));
        TEST_EQ(4, utf8_count_codepoints("\xE0\x9F\x80\x80"));
        TEST_EQ(4, utf8_count_codepoints("\xF0\x8F\xBF\xBF"));
        TEST_EQ(5, utf8_count_codepoints("\xF0\x8F\xBF\xBF\x80"));
        TEST_EQ(1, utf8_count_codepoints("\xF0\x90\xBF"));
        TEST_EQ(4, utf8_count_codepoints("\xF4\xBF\xBF\xBF"));
        TEST_EQ(6, utf8_count_codepoints("\xF5\x90\x80\x80\x80\x80"));
        TEST_EQ(1, utf8_count_codepoints("\xC2"));
        TEST_EQ(1, utf8_count_codepoints("\xE0"));
        TEST_EQ(1, utf8_count_codepoints("\xE0\xA0"));
        TEST_EQ(1, utf8_count_codepoints("\xF"));
        TEST_EQ(1, utf8_count_codepoints("\xF1\x81"));
        TEST_EQ(1, utf8_count_codepoints("\xF1\x81\x81"));
        TEST_EQ(3, utf8_count_codepoints("\xED\xA0\x80"));
        TEST_EQ(3, utf8_count_codepoints("\xED\xAF\xBF"));
        TEST_EQ(3, utf8_count_codepoints("\xED\xB0\x80"));
        TEST_EQ(3, utf8_count_codepoints("\xED\xBF\xBF"));
        TEST_EQ(2, utf8_count_codepoints("\xED\xA0"));
        TEST_EQ(1, utf8_count_codepoints("\xED\x9F"));

        TEST_EQ(4, utf8_count_codepoints("\xF4\x90\x80\x80"));
        TEST_EQ(4, utf8_count_codepoints("\xF5\x80\x80\x80"));
        TEST_EQ(4, utf8_count_codepoints("\xF6\x80\x80\x80"));
        TEST_EQ(4, utf8_count_codepoints("\xF7\x80\x80\x80"));
        TEST_EQ(4, utf8_count_codepoints("\xF8\x80\x80\x80"));
        TEST_EQ(4, utf8_count_codepoints("\xF9\x80\x80\x80"));
        TEST_EQ(4, utf8_count_codepoints("\xFA\x80\x80\x80"));
        TEST_EQ(4, utf8_count_codepoints("\xFB\x80\x80\x80"));
        TEST_EQ(4, utf8_count_codepoints("\xFC\x80\x80\x80"));
        TEST_EQ(4, utf8_count_codepoints("\xFD\x80\x80\x80"));
        TEST_EQ(4, utf8_count_codepoints("\xFE\x80\x80\x80"));
        TEST_EQ(4, utf8_count_codepoints("\xFF\x80\x80\x80"));
        TEST_EQ(3, utf8_count_codepoints("\xF9\x80\x80"));
        TEST_EQ(2, utf8_count_codepoints("\xF9\x80"));
        TEST_EQ(1, utf8_count_codepoints("\xF9"));
    }

    {   // when limit is less than or equal to count

        const char8_t str[] = u8"a\0\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF";
        const auto * const str_end = str + sizeof(str) - 1;
        const strf::utf8_t<char8_t> charset;

        {
            auto r = charset.count_codepoints(str, str_end, 8);
            TEST_EQ((const void*)r.ptr, (const void*)str_end);
            TEST_EQ(r.count, 8);
        }
        {
            auto r = charset.count_codepoints(str, str_end, 7);
            TEST_EQ((const void*)r.ptr, (const void*)(str_end - 4));
            TEST_EQ(r.count, 7);
        }
        {
            auto r = charset.count_codepoints(str, str_end, 0);
            TEST_EQ((const void*)r.ptr, (const void*)str);
            TEST_EQ(r.count, 0);
        }
        {
            auto r = charset.count_codepoints_fast(str, str_end, 8);
            TEST_EQ((const void*)r.ptr, (const void*)str_end);
            TEST_EQ(r.count, 8);
        }
        {
            auto r = charset.count_codepoints_fast(str, str_end, 7);
            TEST_EQ((const void*)r.ptr, (const void*)(str_end - 4));
            TEST_EQ(r.count, 7);
        }
        {
            auto r = charset.count_codepoints_fast(str, str_end, 0);
            TEST_EQ((const void*)r.ptr, (const void*)str);
            TEST_EQ(r.count, 0);
        }
    }
}

STRF_TEST_FUNC void utf8_miscellaneous()
{
    {  // cover write_replacement_char(x);
        TEST(u8"\uFFFD")                       .tr(u8"{10}");
        TEST_CALLING_RECYCLE_AT(2, u8"\uFFFD") .tr(u8"{10}");
        TEST_TRUNCATING_AT     (3, u8"\uFFFD") .tr(u8"{10}");
        TEST_TRUNCATING_AT     (2, u8"")       .tr(u8"{10}");
    }
    const strf::utf_t<char> charset;
    {
        using utf16_to_utf8 = strf::static_transcoder
            <char16_t, char, strf::csid_utf16, strf::csid_utf8>;

        auto tr = charset.find_transcoder_from<char16_t>(strf::csid_utf16);
        TEST_TRUE(tr.transcode_func()      == utf16_to_utf8::transcode);
        TEST_TRUE(tr.transcode_size_func() == utf16_to_utf8::transcode_size);
    }

    TEST_EQ(1, charset.validate(U'a'));
    TEST_EQ(2, charset.validate(0x80));
    TEST_EQ(2, charset.validate(0x7FF));
    TEST_EQ(3, charset.validate(0x800));
    TEST_EQ(3, charset.validate(0xFFFF));
    TEST_EQ(4, charset.validate(0x10000));
    TEST_EQ( 4, charset.validate(0x10FFFF));

    TEST_EQ(-1, charset.validate(0x110000));
    TEST_EQ( 3, charset.validate(0xD7FF));
    TEST_EQ(-1, charset.validate(0xD800)); // surrogate
    TEST_EQ(-1, charset.validate(0xDFFF)); // surrogate
    TEST_EQ( 3, charset.validate(0xE000));

}

} // unnamed namespace

STRF_TEST_FUNC void test_utf8()
{
    utf8_to_utf8_unsafe_transcode();
    utf8_sani_valid_sequences();
    utf8_sani_invalid_sequences();
    utf8_sani_error_notifier();
    utf8_sani_find_transcoder();
    utf8_codepoints_count();
    utf8_miscellaneous();
}

REGISTER_STRF_TEST(test_utf8)
