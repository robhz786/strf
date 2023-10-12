//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils/transcoding.hpp"

#define TEST_TRANSCODE                                                  \
    test_utils::transcode_tester_caller(BOOST_CURRENT_FUNCTION, __FILE__, __LINE__) \
    << test_utils::transcoding_test_data_maker<strf::utf_t<char16_t>, strf::utf_t<char32_t>> \
    (strf::utf<char16_t>, strf::utf<char32_t>, true)

#define TEST_UNSAFE_TRANSCODE                                           \
    test_utils::transcode_tester_caller(BOOST_CURRENT_FUNCTION, __FILE__, __LINE__) \
    << test_utils::transcoding_test_data_maker<strf::utf_t<char16_t>, strf::utf_t<char32_t>> \
    (strf::utf<char16_t>, strf::utf<char32_t>, false)


namespace {

STRF_TEST_FUNC void utf16_to_utf32_unsafe_transcode()
{
    TEST_UNSAFE_TRANSCODE
        .input(u"ab")
        .expect(U"ab")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UNSAFE_TRANSCODE
        .input(u"\u0080")
        .expect(U"\u0080")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UNSAFE_TRANSCODE
        .input(u"\u0800")
        .expect(U"\u0800")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UNSAFE_TRANSCODE
        .input(u"\uD7FF")
        .expect(U"\uD7FF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UNSAFE_TRANSCODE
        .input(u"\U00010000")
        .expect(U"\U00010000")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UNSAFE_TRANSCODE
        .input(u"\U0010FFFF")
        .expect(U"\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UNSAFE_TRANSCODE
        .input(u"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        .expect(U"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UNSAFE_TRANSCODE
        .input(u"abc")
        .expect(U"ab")
        .destination_size(2)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UNSAFE_TRANSCODE
        .input(u"\U00010000")
        .expect(U"")
        .destination_size(0)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);

    TEST_CALLING_RECYCLE_AT(2, U"ab\uD7FF")     (strf::unsafe_transcode(u"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\U00010000") (strf::unsafe_transcode(u"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\U0010FFFF") (strf::unsafe_transcode(u"ab\U0010FFFF"));

    // when using strf::transcode_flags::lax_surrogate_policy
    const char16_t u16str_D800[] = {0xD800, 0};
    const char16_t u16str_DBFF[] = {0xDBFF, 0};
    const char16_t u16str_DC00[] = {0xDC00, 0};
    const char16_t u16str_DFFF[] = {0xDFFF, 0};
    const char16_t u16str_DFFF_D800_[] = {0xDFFF, 0xD800, u'_', 0};

    TEST_UNSAFE_TRANSCODE
        .input(u16str_D800)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(static_cast<char32_t>(0xD800))
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UNSAFE_TRANSCODE
        .input(u16str_DBFF)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(static_cast<char32_t>(0xDBFF))
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UNSAFE_TRANSCODE
        .input(u16str_DC00)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(static_cast<char32_t>(0xDC00))
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UNSAFE_TRANSCODE
        .input(u16str_DFFF)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(static_cast<char32_t>(0xDFFF))
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UNSAFE_TRANSCODE
        .input(u16str_DFFF_D800_)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(static_cast<char32_t>(0xDFFF), static_cast<char32_t>(0xD800), U'_')
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
}

STRF_TEST_FUNC void utf16_to_utf32_valid_sequences()
{
    TEST_TRANSCODE
        .input(u"ab")
        .expect(U"ab")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRANSCODE
        .input(u"\u0080")
        .expect(U"\u0080")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRANSCODE
        .input(u"\u0800")
        .expect(U"\u0800")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRANSCODE
        .input(u"\uD7FF")
        .expect(U"\uD7FF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRANSCODE
        .input(u"\U00010000")
        .expect(U"\U00010000")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRANSCODE
        .input(u"\U0010FFFF")
        .expect(U"\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRANSCODE
        .input(u"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        .expect(U"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_TRANSCODE
        .input(u"abc")
        .expect(U"ab")
        .destination_size(2)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_TRANSCODE
        .input(u"\U00010000")
        .expect(U"")
        .destination_size(0)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);

    TEST_CALLING_RECYCLE_AT(2, U"ab\uD7FF")     (strf::transcode(u"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\U00010000") (strf::transcode(u"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\U0010FFFF") (strf::transcode(u"ab\U0010FFFF"));

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

        TEST_TRANSCODE
            .input(u16str_D800)
            .flags(flags)
            .expect(static_cast<char32_t>(0xD800))
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_TRANSCODE
            .input(u16str_DBFF)
            .flags(flags)
            .expect(static_cast<char32_t>(0xDBFF))
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_TRANSCODE
            .input(u16str_DC00)
            .flags(flags)
            .expect(static_cast<char32_t>(0xDC00))
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_TRANSCODE
            .input(u16str_DFFF)
            .flags(flags)
            .expect(static_cast<char32_t>(0xDFFF))
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_TRANSCODE
            .input(u16str_DFFF_D800_)
            .flags(flags)
            .expect(static_cast<char32_t>(0xDFFF), static_cast<char32_t>(0xD800), U'_')
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
    }
}

STRF_TEST_FUNC void utf16_to_utf32_invalid_sequences()
{
    constexpr auto high_surrogate_sample1 = static_cast<char16_t>(0xD800);
    constexpr auto high_surrogate_sample2 = static_cast<char16_t>(0xDBFF);
    constexpr auto low_surrogate_sample1  = static_cast<char16_t>(0xDC00);
    constexpr auto low_surrogate_sample2  = static_cast<char16_t>(0xDFFF);

    // high surrogate not followed by low surrogate
    TEST_TRANSCODE
        .input(u"abc_", high_surrogate_sample1, u"_def")
        .expect(U"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{high_surrogate_sample1}});
    TEST_TRANSCODE
        .input(u"abc_", high_surrogate_sample2, u"_def")
        .expect(U"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{high_surrogate_sample2}});
    TEST_TRANSCODE
        .input(u"abc_", high_surrogate_sample1, u"_def")
        .expect(U"abc_")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{high_surrogate_sample1}});
    TEST_TRANSCODE
        .input(u"abc_", high_surrogate_sample1, u"_def")
        .destination_size(4)
        .expect(U"abc_")
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{high_surrogate_sample1}});

    // low surrogate not preceded by high surrogate
    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample1, u"_def")
        .expect(U"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1}});
    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample2, u"_def")
        .expect(U"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample2}});
    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample1, u"_def")
        .expect(U"abc_")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1}});

    // high surrogate followed by another high surrogate
    TEST_TRANSCODE
        .input(u"abc_", high_surrogate_sample1, high_surrogate_sample2, u"_def")
        .expect(U"abc_\uFFFD\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{high_surrogate_sample1},
                                  {high_surrogate_sample2}});
    TEST_TRANSCODE
        .input(u"abc_", high_surrogate_sample1, high_surrogate_sample2, u"_def")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect(U"abc_")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{high_surrogate_sample1}});

    // low surrogate followed by a high surrogate
    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample1, high_surrogate_sample1, u"_def")
        .expect(U"abc_\uFFFD\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1},
                                  {high_surrogate_sample1}});
    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample1, high_surrogate_sample1, u"_def")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect(U"abc_")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1}});

    // just a low surrogate
    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample1, u"_def")
        .expect(U"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1}});
    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample2, u"_def")
        .expect(U"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample2}});

    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample1, u"_def")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect(U"abc_")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1}});
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

STRF_TEST_FUNC void utf16_to_utf32_error_notifier()
{
    const char16_t invalid_input[] = {0xDFFF, 0xD800, 0};
    {
        invalid_seq_counter notifier;
        strf::transcoding_error_notifier_ptr notifier_ptr{&notifier};

        TEST(U"\uFFFD\uFFFD").with(notifier_ptr) (strf::sani(invalid_input));
        TEST_EQ(notifier.notifications_count, 2);

        notifier.notifications_count = 0;
        TEST_TRUNCATING_AT(1, U"\uFFFD").with(notifier_ptr) (strf::sani(invalid_input));
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
            strf::to(buff) .with(notifier_ptr) (strf::sani(invalid_input));
        } catch (dummy_exception&) {
            thrown = true;
        } catch(...) {
        }
        TEST_TRUE(thrown);
    }
#endif // __cpp_exceptions
}

STRF_TEST_FUNC void utf16_to_utf32_find_transcoder()
{
#if ! defined(__CUDACC__)

    using static_transcoder_type = strf::static_transcoder
        <char16_t, char32_t, strf::csid_utf16, strf::csid_utf32>;

    const strf::dynamic_charset<char16_t> dyn_utf16 = strf::utf16_t<char16_t>{}.to_dynamic();
    const strf::dynamic_charset<char32_t> dyn_utf32 = strf::utf32_t<char32_t>{}.to_dynamic();
    const strf::dynamic_transcoder<char16_t, char32_t> tr = strf::find_transcoder(dyn_utf16, dyn_utf32);

    TEST_TRUE(tr.transcode_func()      == static_transcoder_type::transcode);
    TEST_TRUE(tr.transcode_size_func() == static_transcoder_type::transcode_size);

#endif // defined(__CUDACC__)
}

} // unnamed namespace

STRF_TEST_FUNC void test_utf16_to_utf32()
{
    utf16_to_utf32_unsafe_transcode();
    utf16_to_utf32_valid_sequences();
    utf16_to_utf32_invalid_sequences();
    utf16_to_utf32_error_notifier();
    utf16_to_utf32_find_transcoder();
}

REGISTER_STRF_TEST(test_utf16_to_utf32)
