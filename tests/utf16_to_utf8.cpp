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

STRF_TEST_FUNC void utf16_to_utf8_unsafe_transcode()
{
    TEST_UTF_UNSAFE_TRANSCODE(char16_t, char8_t)
        .input(u"ab")
        .expect(u8"ab")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char16_t, char8_t)
        .input(u"\u0080")
        .expect(u8"\u0080")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char16_t, char8_t)
        .input(u"\u0800")
        .expect(u8"\u0800")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char16_t, char8_t)
        .input(u"\uD7FF")
        .expect(u8"\uD7FF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char16_t, char8_t)
        .input(u"\U00010000")
        .expect(u8"\U00010000")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char16_t, char8_t)
        .input(u"\U0010FFFF")
        .expect(u8"\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_UNSAFE_TRANSCODE(char16_t, char8_t)
        .input(u"ab\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF")
        .expect(u8"ab\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_UNSAFE_TRANSCODE(char16_t, char)
        .input(u"")
        .expect("")
        .destination_size(0)
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_UNSAFE_TRANSCODE(char16_t, char)
        .input(u"a")
        .expect("")
        .destination_size(0)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UTF_UNSAFE_TRANSCODE(char16_t, char)
        .input(u"\u0080")
        .expect("")
        .destination_size(1)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UTF_UNSAFE_TRANSCODE(char16_t, char)
        .input(u"\u0800")
        .expect("")
        .destination_size(2)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UTF_UNSAFE_TRANSCODE(char16_t, char)
        .input(u"\U00010000")
        .expect("")
        .destination_size(3)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);

    TEST_CALLING_RECYCLE_AT(2, u8"abc")          (strf::unsafe_transcode(u"abc"));
    TEST_CALLING_RECYCLE_AT(2, u8"ab\u0080")     (strf::unsafe_transcode(u"ab\u0080"));
    TEST_CALLING_RECYCLE_AT(3, u8"ab\u0080")     (strf::unsafe_transcode(u"ab\u0080"));
    TEST_CALLING_RECYCLE_AT(4, u8"ab\uD7FF")     (strf::unsafe_transcode(u"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT(4, u8"ab\U00010000") (strf::unsafe_transcode(u"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(5, u8"ab\U0010FFFF") (strf::unsafe_transcode(u"ab\U0010FFFF"));

    TEST_UTF_UNSAFE_TRANSCODE(char16_t, char)
        .input(u"ab\U0010FFFF")
        .destination_size(5)
        .expect("ab")
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    // when using strf::transcode_flags::lax_surrogate_policy
    const char16_t u16str_D800[] = {0xD800, 0};
    const char16_t u16str_DBFF[] = {0xDBFF, 0};
    const char16_t u16str_DC00[] = {0xDC00, 0};
    const char16_t u16str_DFFF[] = {0xDFFF, 0};
    const char16_t u16str_DFFF_D800_[] = {0xDFFF, 0xD800, u'_', 0};

    TEST_UTF_UNSAFE_TRANSCODE(char16_t, char)
        .input(u16str_D800)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect("\xED\xA0\x80")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UTF_UNSAFE_TRANSCODE(char16_t, char)
        .input(u16str_DBFF)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect("\xED\xAF\xBF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UTF_UNSAFE_TRANSCODE(char16_t, char)
        .input(u16str_DC00)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect("\xED\xB0\x80")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UTF_UNSAFE_TRANSCODE(char16_t, char)
        .input(u16str_DFFF)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect("\xED\xBF\xBF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UTF_UNSAFE_TRANSCODE(char16_t, char)
        .input(u16str_DFFF_D800_)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect("\xED\xBF\xBF\xED\xA0\x80_")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
}

STRF_TEST_FUNC void utf16_to_utf8_valid_sequences()
{
    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"ab")
        .expect(u8"ab")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"\u0080")
        .expect(u8"\u0080")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"\u0800")
        .expect(u8"\u0800")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"\uD7FF")
        .expect(u8"\uD7FF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"\U00010000")
        .expect(u8"\U00010000")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"\U0010FFFF")
        .expect(u8"\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"ab\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF")
        .expect(u8"ab\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char16_t, char)
        .input(u"")
        .expect("")
        .destination_size(0)
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UTF_TRANSCODE(char16_t, char)
        .input(u"a")
        .expect("")
        .destination_size(0)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UTF_TRANSCODE(char16_t, char)
        .input(u"\u0080")
        .expect("")
        .destination_size(1)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UTF_TRANSCODE(char16_t, char)
        .input(u"\u0800")
        .expect("")
        .destination_size(2)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UTF_TRANSCODE(char16_t, char)
        .input(u"\U00010000")
        .expect("")
        .destination_size(3)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);

    TEST_CALLING_RECYCLE_AT(2, u8"abc")          (strf::sani(u"abc"));
    TEST_CALLING_RECYCLE_AT(2, u8"ab\u0080")     (strf::sani(u"ab\u0080"));
    TEST_CALLING_RECYCLE_AT(3, u8"ab\u0080")     (strf::sani(u"ab\u0080"));
    TEST_CALLING_RECYCLE_AT(4, u8"ab\uD7FF")     (strf::sani(u"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT(4, u8"ab\U00010000") (strf::sani(u"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(5, u8"ab\U0010FFFF") (strf::sani(u"ab\U0010FFFF"));

    TEST_UTF_TRANSCODE(char16_t, char)
        .input(u"ab\U0010FFFF")
        .destination_size(5)
        .expect("ab")
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    {
        // when surrogates are allowed
        const char16_t u16str_D800[] = {0xD800, 0};
        const char16_t u16str_DBFF[] = {0xDBFF, 0};
        const char16_t u16str_DC00[] = {0xDC00, 0};
        const char16_t u16str_DFFF[] = {0xDFFF, 0};

        const char16_t u16str_DFFF_D800_[] = {0xDFFF, 0xD800, u'_', 0};

        const auto flags = ( strf::transcode_flags::lax_surrogate_policy |
                             strf::transcode_flags::stop_on_invalid_sequence |
                             strf::transcode_flags::stop_on_unsupported_codepoint );

        TEST_UTF_TRANSCODE(char16_t, char)
            .input(u16str_D800)
            .flags(flags)
            .expect("\xED\xA0\x80")
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char16_t, char)
            .input(u16str_DBFF)
            .flags(flags)
            .expect("\xED\xAF\xBF")
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char16_t, char)
            .input(u16str_DC00)
            .flags(flags)
            .expect("\xED\xB0\x80")
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char16_t, char)
            .input(u16str_DFFF)
            .flags(flags)
            .expect("\xED\xBF\xBF")
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_UTF_TRANSCODE(char16_t, char)
            .input(u16str_DFFF_D800_)
            .flags(flags)
            .expect("\xED\xBF\xBF\xED\xA0\x80_")
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
    }
}

STRF_TEST_FUNC void utf16_to_utf8_invalid_sequences()
{
    constexpr auto high_surrogate_sample1 = static_cast<char16_t>(0xD800);
    constexpr auto high_surrogate_sample2 = static_cast<char16_t>(0xDBFF);
    constexpr auto low_surrogate_sample1  = static_cast<char16_t>(0xDC00);
    constexpr auto low_surrogate_sample2  = static_cast<char16_t>(0xDFFF);

    // high surrogate not followed by low surrogate
    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"abc_", high_surrogate_sample1, u"_def")
        .expect(u8"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{high_surrogate_sample1}});
    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"abc_", high_surrogate_sample2, u"_def")
        .expect(u8"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{high_surrogate_sample2}});
    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"abc_", high_surrogate_sample1, u"_def")
        .expect(u8"abc_")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{high_surrogate_sample1}});
    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"abc_", high_surrogate_sample1, u"_def")
        .destination_size(6)
        .expect(u8"abc_")
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{high_surrogate_sample1}});

    // low surrogate not preceded by high surrogate
    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"abc_", low_surrogate_sample1, u"_def")
        .expect(u8"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1}});
    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"abc_", low_surrogate_sample2, u"_def")
        .expect(u8"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample2}});
    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"abc_", low_surrogate_sample1, u"_def")
        .expect(u8"abc_")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1}});

    // high surrogate followed by another high surrogate
    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"abc_", high_surrogate_sample1, high_surrogate_sample2, u"_def")
        .expect(u8"abc_\uFFFD\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{high_surrogate_sample1},
                                  {high_surrogate_sample2}});
    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"abc_", high_surrogate_sample1, high_surrogate_sample2, u"_def")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect(u8"abc_")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{high_surrogate_sample1}});

    // low surrogate followed by a high surrogate
    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"abc_", low_surrogate_sample1, high_surrogate_sample1, u"_def")
        .expect(u8"abc_\uFFFD\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1},
                                  {high_surrogate_sample1}});
    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"abc_", low_surrogate_sample1, high_surrogate_sample1, u"_def")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect(u8"abc_")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1}});

    // just a low surrogate
    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"abc_", low_surrogate_sample1, u"_def")
        .expect(u8"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1}});
    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"abc_", low_surrogate_sample2, u"_def")
        .expect(u8"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample2}});

    TEST_UTF_TRANSCODE(char16_t, char8_t)
        .input(u"abc_", low_surrogate_sample1, u"_def")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect(u8"abc_")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1}});

    const char16_t str[] = {0xDFFF, 0};
    TEST_TRUNCATING_AT     (4, u8" \uFFFD") (strf::sani(str) > 2);
    TEST_CALLING_RECYCLE_AT(3, u8" \uFFFD") (strf::sani(str) > 2);
    TEST_TRUNCATING_AT     (4, u8" \uFFFD") (strf::sani(str) > 2);
    TEST_TRUNCATING_AT     (3, u8" ")       (strf::sani(str) > 2);
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

STRF_TEST_FUNC void utf16_to_utf8_error_notifier()
{
    const char16_t invalid_input[] = {0xDFFF, 0xD800, 0};
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
            char buff[40];
            strf::to(buff) .with(notifier_ptr) (strf::sani(invalid_input));
        } catch (dummy_exception&) {
            thrown = true;
        } catch(...) {
        }
        TEST_TRUE(thrown);
    }

#endif // __cpp_exceptions
}

STRF_TEST_FUNC void utf16_to_utf8_find_transcoder()
{
#if ! defined(__CUDACC__)

    using static_transcoder_type = strf::static_transcoder
        <char16_t, char, strf::csid_utf16, strf::csid_utf8>;

    const strf::dynamic_charset<char16_t> dyn_utf16 = strf::utf16_t<char16_t>{}.to_dynamic();
    const strf::dynamic_charset<char>     dyn_utf8  = strf::utf8_t<char>{}.to_dynamic();
    const strf::dynamic_transcoder<char16_t, char> tr = strf::find_transcoder(dyn_utf16, dyn_utf8);

    TEST_TRUE(tr.transcode_func()      == static_transcoder_type::transcode);
    TEST_TRUE(tr.transcode_size_func() == static_transcoder_type::transcode_size);

#endif // defined(__CUDACC__)

    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char16_t, char, strf::csid_utf16, strf::csid_utf8 >
                   , decltype(strf::find_transcoder( strf::utf_t<char16_t>{}
                                                   , strf::utf_t<char>{} )) >
                  :: value));
}

} // unnamed namespace

STRF_TEST_FUNC void test_utf16_to_utf8()
{
    utf16_to_utf8_unsafe_transcode();
    utf16_to_utf8_valid_sequences();
    utf16_to_utf8_invalid_sequences();
    utf16_to_utf8_error_notifier();
    utf16_to_utf8_find_transcoder();
}

REGISTER_STRF_TEST(test_utf16_to_utf8)
