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

STRF_TEST_FUNC void utf32_to_utf8_unsafe_transcode()
{
    const char32_t u32str_D800[] = {0xD800, 0};
    const char32_t u32str_DBFF[] = {0xDBFF, 0};
    const char32_t u32str_DC00[] = {0xDC00, 0};
    const char32_t u32str_DFFF[] = {0xDFFF, 0};
    const char32_t u32str_DFFF_D800_[] = {0xDFFF, 0xD800, u'_', 0};

    const char u8str_D800[] = "\xED\xA0\x80";
    const char u8str_DBFF[] = "\xED\xAF\xBF";
    const char u8str_DC00[] = "\xED\xB0\x80";
    const char u8str_DFFF[] = "\xED\xBF\xBF";
    const char u8str_DFFF_D800_[] = "\xED\xBF\xBF\xED\xA0\x80_";

    TEST_UTF_UNSAFE_TRANSCODE(char32_t, char8_t)
        .input(U"")
        .expect(u8"")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char32_t, char8_t)
        .input(U"ab")
        .expect(u8"ab")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char32_t, char8_t)
        .input(U"\u0080")
        .expect(u8"\u0080")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char32_t, char8_t)
        .input(U"\u0800")
        .expect(u8"\u0800")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char32_t, char8_t)
        .input(U"\uD7FF")
        .expect(u8"\uD7FF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char32_t, char8_t)
        .input(U"\U00010000")
        .expect(u8"\U00010000")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char32_t, char8_t)
        .input(U"\U0010FFFF")
        .expect(u8"\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST(u8" ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        (strf::unsafe_transcode(U"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF") > 8);

    TEST("\xF4\x8F\xBF\xBF") (strf::unsafe_transcode(U"\U0010FFFF"));

    TEST_TRUNCATING_AT(2, u8"ab") (strf::unsafe_transcode(U"abc"));
    TEST_TRUNCATING_AT(2, u8"ab") (strf::unsafe_transcode(U"ab\u0080"));
    TEST_TRUNCATING_AT(2, u8"ab") (strf::unsafe_transcode(U"ab\uD7FF"));
    TEST_TRUNCATING_AT(2, u8"ab") (strf::unsafe_transcode(U"ab\U00010000"));

    TEST_TRUNCATING_AT(3, u8"abc")          (strf::unsafe_transcode(U"abc"));
    TEST_TRUNCATING_AT(4, u8"ab\u0080")     (strf::unsafe_transcode(U"ab\u0080"));
    TEST_TRUNCATING_AT(5, u8"ab\uD7FF")     (strf::unsafe_transcode(U"ab\uD7FF"));
    TEST_TRUNCATING_AT(6, u8"ab\U00010000") (strf::unsafe_transcode(U"ab\U00010000"));
    TEST_TRUNCATING_AT(6, u8"ab\U0010FFFF") (strf::unsafe_transcode(U"ab\U0010FFFF"));

    TEST_CALLING_RECYCLE_AT(2, u8"abc")          (strf::unsafe_transcode(U"abc"));
    TEST_CALLING_RECYCLE_AT(3, u8"ab\u0080")     (strf::unsafe_transcode(U"ab\u0080"));
    TEST_CALLING_RECYCLE_AT(4, u8"ab\uD7FF")     (strf::unsafe_transcode(U"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT(4, u8"ab\U00010000") (strf::unsafe_transcode(U"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(5, u8"ab\U0010FFFF") (strf::unsafe_transcode(U"ab\U0010FFFF"));

    TEST_UTF_UNSAFE_TRANSCODE(char32_t, char8_t)
        .input(U"ab\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF")
        .expect(u8"ab\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_UNSAFE_TRANSCODE(char32_t, char8_t)
        .input(U"abc")
        .expect(u8"ab")
        .destination_size(2)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UTF_UNSAFE_TRANSCODE(char32_t, char8_t)
        .input(U"\U00010000")
        .expect(u8"")
        .destination_size(1)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UTF_UNSAFE_TRANSCODE(char32_t, char8_t)
        .input(U"\U00010000")
        .expect(u8"")
        .destination_size(0)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UTF_UNSAFE_TRANSCODE(char32_t, char8_t)
        .input(U"abc")
        .expect(u8"")
        .destination_size(0)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);

    // when using strf::transcode_flags::lax_surrogate_policy
    TEST_UTF_UNSAFE_TRANSCODE(char32_t, char)
        .input(u32str_D800)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(u8str_D800)
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UTF_UNSAFE_TRANSCODE(char32_t, char)
        .input(u32str_DBFF)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(u8str_DBFF)
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UTF_UNSAFE_TRANSCODE(char32_t, char)
        .input(u32str_DC00)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(u8str_DC00)
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UTF_UNSAFE_TRANSCODE(char32_t, char)
        .input(u32str_DFFF)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(u8str_DFFF)
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UTF_UNSAFE_TRANSCODE(char32_t, char)
        .input(u32str_DFFF_D800_)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(u8str_DFFF_D800_)
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
}

STRF_TEST_FUNC void utf32_to_utf8_valid_sequences()
{
    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(U"ab")
        .expect(u8"ab")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(U"\u0080")
        .expect(u8"\u0080")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(U"\u0800")
        .expect(u8"\u0800")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(U"\uD7FF")
        .expect(u8"\uD7FF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(U"\U00010000")
        .expect(u8"\U00010000")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(U"\U0010FFFF")
        .expect(u8"\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST(u8" ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        (strf::sani(U"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF") > 8);

    TEST("\xF4\x8F\xBF\xBF") (strf::sani(U"\U0010FFFF"));

    TEST_TRUNCATING_AT(2, u8"ab") (strf::sani(U"abc"));
    TEST_TRUNCATING_AT(2, u8"ab") (strf::sani(U"ab\u0080"));
    TEST_TRUNCATING_AT(2, u8"ab") (strf::sani(U"ab\uD7FF"));
    TEST_TRUNCATING_AT(2, u8"ab") (strf::sani(U"ab\U00010000"));

    TEST_TRUNCATING_AT(3, u8"abc")          (strf::sani(U"abc"));
    TEST_TRUNCATING_AT(4, u8"ab\u0080")     (strf::sani(U"ab\u0080"));
    TEST_TRUNCATING_AT(5, u8"ab\uD7FF")     (strf::sani(U"ab\uD7FF"));
    TEST_TRUNCATING_AT(6, u8"ab\U00010000") (strf::sani(U"ab\U00010000"));
    TEST_TRUNCATING_AT(6, u8"ab\U0010FFFF") (strf::sani(U"ab\U0010FFFF"));

    TEST_CALLING_RECYCLE_AT(2, u8"abc")          (strf::sani(U"abc"));
    TEST_CALLING_RECYCLE_AT(3, u8"ab\u0080")     (strf::sani(U"ab\u0080"));
    TEST_CALLING_RECYCLE_AT(4, u8"ab\uD7FF")     (strf::sani(U"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT(4, u8"ab\U00010000") (strf::sani(U"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(5, u8"ab\U0010FFFF") (strf::sani(U"ab\U0010FFFF"));
    {
        // when surrogates are allowed

        const auto flags = ( strf::transcode_flags::lax_surrogate_policy |
                             strf::transcode_flags::stop_on_invalid_sequence |
                             strf::transcode_flags::stop_on_unsupported_codepoint );

        const char32_t u32str_D800[] = {0xD800, 0};
        const char32_t u32str_DBFF[] = {0xDBFF, 0};
        const char32_t u32str_DC00[] = {0xDC00, 0};
        const char32_t u32str_DFFF[] = {0xDFFF, 0};
        const char32_t u32str_DFFF_D800_[] = {0xDFFF, 0xD800, u'_', 0};

        const char u8str_D800[] = "\xED\xA0\x80";
        const char u8str_DBFF[] = "\xED\xAF\xBF";
        const char u8str_DC00[] = "\xED\xB0\x80";
        const char u8str_DFFF[] = "\xED\xBF\xBF";
        const char u8str_DFFF_D800_[] = "\xED\xBF\xBF\xED\xA0\x80_";

        TEST_UTF_TRANSCODE(char32_t, char)
            .input(u32str_D800)
            .flags(flags)
            .expect(u8str_D800)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char32_t, char)
            .input(u32str_DBFF)
            .flags(flags)
            .expect(u8str_DBFF)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char32_t, char)
            .input(u32str_DC00)
            .flags(flags)
            .expect(u8str_DC00)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char32_t, char)
            .input(u32str_DFFF)
            .flags(flags)
            .expect(u8str_DFFF)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char32_t, char)
            .input(u32str_DFFF_D800_)
            .flags(flags)
            .expect(u8str_DFFF_D800_)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
    }
}

STRF_TEST_FUNC void utf32_to_utf8_valid_sequences_with_destination_too_small()
{
    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(U"abc")
        .expect(u8"ab")
        .destination_size(2)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(U"\u0080")
        .expect(u8"")
        .destination_size(1)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(U"\u0800")
        .expect(u8"")
        .destination_size(2)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(U"\uD7FF")
        .expect(u8"")
        .destination_size(2)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(U"\U00010000")
        .expect(u8"")
        .destination_size(3)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(U"\U0010FFFF")
        .expect(u8"")
        .destination_size(3)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    {
        // when surrogates are allowed
        const char32_t u32str_D800[] = {0xD800, 0};
        const char32_t u32str_DBFF[] = {0xDBFF, 0};
        const char32_t u32str_DC00[] = {0xDC00, 0};
        const char32_t u32str_DFFF[] = {0xDFFF, 0};
        const char32_t u32str_DFFF_D800_[] = {0xDFFF, 0xD800, u'_', 0};

        TEST_UTF_TRANSCODE(char32_t, char8_t)
            .input(u32str_D800)
            .flags(strf::transcode_flags::lax_surrogate_policy)
            .expect(u8"")
            .destination_size(2)
            .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char32_t, char8_t)
            .input(u32str_DBFF)
            .flags(strf::transcode_flags::lax_surrogate_policy)
            .expect(u8"")
            .destination_size(2)
            .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char32_t, char8_t)
            .input(u32str_DC00)
            .flags(strf::transcode_flags::lax_surrogate_policy)
            .expect(u8"")
            .destination_size(2)
            .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char32_t, char8_t)
            .input(u32str_DFFF)
            .flags(strf::transcode_flags::lax_surrogate_policy)
            .expect(u8"")
            .destination_size(2)
            .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char32_t, char8_t)
            .input(u32str_DFFF_D800_)
            .flags(strf::transcode_flags::lax_surrogate_policy)
            .expect(u8"")
            .destination_size(2)
            .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});

        const auto flags = ( strf::transcode_flags::lax_surrogate_policy |
                             strf::transcode_flags::stop_on_invalid_sequence |
                             strf::transcode_flags::stop_on_unsupported_codepoint );
        TEST_UTF_TRANSCODE(char32_t, char8_t)
            .input(u32str_D800)
            .flags(flags)
            .expect(u8"")
            .destination_size(2)
            .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char32_t, char8_t)
            .input(u32str_DBFF)
            .flags(flags)
            .expect(u8"")
            .destination_size(2)
            .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char32_t, char8_t)
            .input(u32str_DC00)
            .flags(flags)
            .expect(u8"")
            .destination_size(2)
            .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char32_t, char8_t)
            .input(u32str_DFFF)
            .flags(flags)
            .expect(u8"")
            .destination_size(2)
            .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char32_t, char8_t)
            .input(u32str_DFFF_D800_)
            .flags(flags)
            .expect(u8"")
            .destination_size(2)
            .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
    }
}

STRF_TEST_FUNC void test_not_allowed_surrogate(char32_t surrogate_char)
{
    TEST_SCOPE_DESCRIPTION("codepoint is ", strf::hex(static_cast<unsigned>(surrogate_char)));

    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(U"abc_", surrogate_char, U"_def")
        .expect(u8"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{surrogate_char}});
    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(U"abc_", surrogate_char, U"_def")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect(u8"abc_")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{surrogate_char}});
    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(U"abc_", surrogate_char, U"_def")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .destination_size(4)
        .expect(u8"abc_")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{surrogate_char}});
    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(surrogate_char, U"_def")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .destination_size(0)
        .expect(u8"")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{surrogate_char}});
    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(surrogate_char, U"_def")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect(u8"")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{surrogate_char}});
}

STRF_TEST_FUNC void utf32_to_utf8_invalid_sequences()
{
    test_not_allowed_surrogate(static_cast<char32_t>(0xD800)) ;
    test_not_allowed_surrogate(static_cast<char32_t>(0xDBFF)) ;
    test_not_allowed_surrogate(static_cast<char32_t>(0xDC00)) ;
    test_not_allowed_surrogate(static_cast<char32_t>(0xDFFF)) ;

    // codepoint too big
    const char32_t str_110000[] = {0x110000, 0};
    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(str_110000)
        .expect(u8"\uFFFD")
        .flags(strf::transcode_flags::lax_surrogate_policy ) // should have no effect
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{static_cast<char32_t>(0x110000)}});
    TEST_UTF_TRANSCODE(char32_t, char8_t)
        .input(str_110000)
        .expect(u8"")
        .destination_size(0)
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{static_cast<char32_t>(0x110000)}});
}

struct invalid_seq_counter: strf::transcoding_error_notifier {
    void STRF_HD invalid_sequence(int, const char*, const void*, std::ptrdiff_t) override {
        ++ notifications_count;
    }
    std::ptrdiff_t notifications_count = 0;
};

#if defined(__cpp_exceptions) && __cpp_exceptions  && ! defined(__CUDACC__)

struct dummy_exception: std::exception {};

struct notifier_that_throws : strf::transcoding_error_notifier {
    void STRF_HD invalid_sequence(int, const char*, const void*, std::ptrdiff_t) override {
        throw dummy_exception{};
    }
};

#endif // __cpp_exceptions

STRF_TEST_FUNC void utf32_to_utf8_error_notifier()
{
    const char32_t invalid_input[] = {0xD800, 0xDFFF, 0};
    {
        invalid_seq_counter notifier;
        strf::transcoding_error_notifier_ptr notifier_ptr{&notifier};

        TEST(u8"\uFFFD\uFFFD").with(notifier_ptr) (strf::sani(invalid_input));
        TEST_EQ(notifier.notifications_count, 2);

        notifier.notifications_count = 0;
        TEST_TRUNCATING_AT(5, u8"\uFFFD").with(notifier_ptr) (strf::sani(invalid_input));
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
            char buff[16];
            strf::to(buff) .with(notifier_ptr) (strf::sani(invalid_input));
        } catch (dummy_exception&) {
            thrown = true;
        }
        TEST_TRUE(thrown);
    }
#endif // __cpp_exceptions

    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char32_t, char, strf::csid_utf32, strf::csid_utf8 >
                   , decltype(strf::find_transcoder( strf::utf_t<char32_t>{}
                                                   , strf::utf_t<char>{})) >
                  :: value));
}

STRF_TEST_FUNC void utf32_to_utf8_find_transcoder()
{
#if ! defined(__CUDACC__)

    using static_transcoder_type = strf::static_transcoder
        <char32_t, char, strf::csid_utf32, strf::csid_utf8>;

    const strf::dynamic_charset<char32_t> dyn_utf32 = strf::utf32_t<char32_t>{}.to_dynamic();
    const strf::dynamic_charset<char>     dyn_utf8  = strf::utf8_t<char>{}.to_dynamic();
    const strf::dynamic_transcoder<char32_t, char> tr = strf::find_transcoder(dyn_utf32, dyn_utf8);

    TEST_TRUE(tr.transcode_func()      == static_transcoder_type::transcode);
    TEST_TRUE(tr.transcode_size_func() == static_transcoder_type::transcode_size);

#endif // defined(__CUDACC__)
}

} // unnamed namespace

STRF_TEST_FUNC void test_utf32_to_utf8()
{
    utf32_to_utf8_unsafe_transcode();
    utf32_to_utf8_valid_sequences();
    utf32_to_utf8_valid_sequences_with_destination_too_small();
    utf32_to_utf8_invalid_sequences();
    utf32_to_utf8_error_notifier();
    utf32_to_utf8_find_transcoder();
}

REGISTER_STRF_TEST(test_utf32_to_utf8)
