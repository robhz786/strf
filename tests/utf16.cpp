//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils/transcoding.hpp"

#define TEST_TRANSCODE                                                  \
    test_utils::transcode_tester_caller(BOOST_CURRENT_FUNCTION, __FILE__, __LINE__) \
    << test_utils::transcoding_test_data_maker<strf::utf_t<char16_t>, strf::utf_t<char16_t>> \
    (strf::utf<char16_t>, strf::utf<char16_t>, true)

#define TEST_UNSAFE_TRANSCODE                                           \
    test_utils::transcode_tester_caller(BOOST_CURRENT_FUNCTION, __FILE__, __LINE__) \
    << test_utils::transcoding_test_data_maker<strf::utf_t<char16_t>, strf::utf_t<char16_t>> \
    (strf::utf<char16_t>, strf::utf<char16_t>, false)


namespace {

STRF_TEST_FUNC void utf16_to_utf16_unsafe_transcode()
{
    TEST_UNSAFE_TRANSCODE
        .input(u"ab")
        .expect(u"ab")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UNSAFE_TRANSCODE
        .input(u"\u0080")
        .expect(u"\u0080")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UNSAFE_TRANSCODE
        .input(u"\u0800")
        .expect(u"\u0800")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UNSAFE_TRANSCODE
        .input(u"\uD7FF")
        .expect(u"\uD7FF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UNSAFE_TRANSCODE
        .input(u"\U00010000")
        .expect(u"\U00010000")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UNSAFE_TRANSCODE
        .input(u"\U0010FFFF")
        .expect(u"\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UNSAFE_TRANSCODE
        .input (u"ab\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF")
        .expect(u"ab\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UNSAFE_TRANSCODE
        .input(u"abc")
        .expect(u"ab")
        .destination_size(2)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UNSAFE_TRANSCODE
        .input(u"\U00010000")
        .expect(u"")
        .destination_size(1)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UNSAFE_TRANSCODE
        .input(u"\U00010000")
        .expect(u"")
        .destination_size(0)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);


    TEST_CALLING_RECYCLE_AT(2, u"ab\U00010000") (strf::unsafe_transcode(u"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(2, u"ab\U0010FFFF") (strf::unsafe_transcode(u"ab\U0010FFFF"));
    TEST_CALLING_RECYCLE_AT(3, u"ab\U00010000") (strf::unsafe_transcode(u"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(3, u"ab\U0010FFFF") (strf::unsafe_transcode(u"ab\U0010FFFF"));


    // when using strf::transcode_flags::lax_surrogate_policy
    const char16_t u16str_D800[] = {0xD800, 0};
    const char16_t u16str_DBFF[] = {0xDBFF, 0};
    const char16_t u16str_DC00[] = {0xDC00, 0};
    const char16_t u16str_DFFF[] = {0xDFFF, 0};
    const char16_t u16str_DFFF_D800_[] = {0xDFFF, 0xD800, u'_', 0};

    TEST_UNSAFE_TRANSCODE
        .input(u16str_D800)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(u16str_D800)
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UNSAFE_TRANSCODE
        .input(u16str_DBFF)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(u16str_DBFF)
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UNSAFE_TRANSCODE
        .input(u16str_DC00)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(u16str_DC00)
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UNSAFE_TRANSCODE
        .input(u16str_DFFF)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(u16str_DFFF)
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UNSAFE_TRANSCODE
        .input(u16str_DFFF_D800_)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(u16str_DFFF_D800_)
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    // The result should be as when in the input is UTF-32
    TEST_UNSAFE_TRANSCODE
        .input(u"ab\U00010000")
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(u"ab")
        .destination_size(3)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UNSAFE_TRANSCODE
        .input(u16str_DFFF_D800_)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(u16str_DFFF)
        .destination_size(1)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
}

STRF_TEST_FUNC void utf16_sani_valid_sequences()
{
    TEST_TRANSCODE
        .input(u"ab")
        .expect(u"ab")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRANSCODE
        .input(u"\u0080")
        .expect(u"\u0080")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRANSCODE
        .input(u"\u0800")
        .expect(u"\u0800")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRANSCODE
        .input(u"\uD7FF")
        .expect(u"\uD7FF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRANSCODE
        .input(u"\U00010000")
        .expect(u"\U00010000")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRANSCODE
        .input(u"\U0010FFFF")
        .expect(u"\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRANSCODE
        .input(u"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        .expect(u"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_TRANSCODE
        .input(u"abc")
        .expect(u"ab")
        .destination_size(2)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_TRANSCODE
        .input(u"\U00010000")
        .expect(u"")
        .destination_size(1)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_TRANSCODE
        .input(u"\U00010000")
        .expect(u"")
        .destination_size(0)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);




    TEST_CALLING_RECYCLE_AT(2, u"ab\U00010000") (strf::sani(u"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(2, u"ab\U0010FFFF") (strf::sani(u"ab\U0010FFFF"));
    TEST_CALLING_RECYCLE_AT(3, u"ab\U00010000") (strf::sani(u"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(3, u"ab\U0010FFFF") (strf::sani(u"ab\U0010FFFF"));

    TEST_CALLING_RECYCLE_AT(2, u"ab\uD7FF")     (strf::sani(u"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT(3, u"ab\U00010000") (strf::sani(u"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(3, u"ab\U0010FFFF") (strf::sani(u"ab\U0010FFFF"));

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
            .expect(u16str_D800)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_TRANSCODE
            .input(u16str_DBFF)
            .flags(flags)
            .expect(u16str_DBFF)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_TRANSCODE
            .input(u16str_DC00)
            .flags(flags)
            .expect(u16str_DC00)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_TRANSCODE
            .input(u16str_DFFF)
            .flags(flags)
            .expect(u16str_DFFF)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_TRANSCODE
            .input(u16str_DFFF_D800_)
            .flags(flags)
            .expect(u16str_DFFF_D800_)
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
    }
}

STRF_TEST_FUNC void utf16_sani_invalid_sequences()
{
    constexpr auto high_surrogate_sample1 = static_cast<char16_t>(0xD800);
    constexpr auto high_surrogate_sample2 = static_cast<char16_t>(0xDBFF);
    constexpr auto low_surrogate_sample1  = static_cast<char16_t>(0xDC00);
    constexpr auto low_surrogate_sample2  = static_cast<char16_t>(0xDFFF);

    // high surrogate not followed by low surrogate
    TEST_TRANSCODE
        .input(u"abc_", high_surrogate_sample1, u"_def")
        .expect(u"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{high_surrogate_sample1}});
    TEST_TRANSCODE
        .input(u"abc_", high_surrogate_sample2, u"_def")
        .expect(u"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{high_surrogate_sample2}});
    TEST_TRANSCODE
        .input(u"abc_", high_surrogate_sample1, u"_def")
        .expect(u"abc_")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{high_surrogate_sample1}});
    TEST_TRANSCODE
        .input(u"abc_", high_surrogate_sample1, u"_def")
        .destination_size(4)
        .expect(u"abc_")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{high_surrogate_sample1}});

    // low surrogate not preceded by high surrogate
    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample1, u"_def")
        .expect(u"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1}});
    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample2, u"_def")
        .expect(u"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample2}});
    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample1, u"_def")
        .expect(u"abc_")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1}});
    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample1, u"_def")
        .destination_size(4)
        .expect(u"abc_")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1}});

    // high surrogate followed by another high surrogate
    TEST_TRANSCODE
        .input(u"abc_", high_surrogate_sample1, high_surrogate_sample2, u"_def")
        .expect(u"abc_\uFFFD\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{high_surrogate_sample1},
                                  {high_surrogate_sample2}});
    TEST_TRANSCODE
        .input(u"abc_", high_surrogate_sample1, high_surrogate_sample2, u"_def")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect(u"abc_")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{high_surrogate_sample1}});

    // low surrogate followed by a high surrogate
    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample1, high_surrogate_sample1, u"_def")
        .expect(u"abc_\uFFFD\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1},
                                  {high_surrogate_sample1}});
    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample1, high_surrogate_sample1, u"_def")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect(u"abc_")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1}});
    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample1, high_surrogate_sample1, u"_def")
        .destination_size(4)
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect(u"abc_")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1}});

    // just a low surrogate
    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample1, u"_def")
        .expect(u"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1}});
    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample2, u"_def")
        .expect(u"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample2}});

    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample1, u"_def")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect(u"abc_")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1}});

    //
    TEST_TRANSCODE
        .input(u"abc_", low_surrogate_sample1, u"_def")
        .destination_size(4)
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect(u"abc_")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{low_surrogate_sample1}});

    {
        const char16_t str[] = {'_', 0xD800, '_', 0};
        TEST(u" _\uFFFD_")                       (strf::sani(str) > 4);
        TEST_CALLING_RECYCLE_AT(2, u" _\uFFFD_") (strf::sani(str) > 4);
        TEST_TRUNCATING_AT     (2, u" _")        (strf::sani(str) > 4);
        TEST_TRUNCATING_AT     (4, u" _\uFFFD_") (strf::sani(str) > 4);
    }
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

STRF_TEST_FUNC void utf16_sani_error_notifier()
{
    const char16_t invalid_input[] = {0xDFFF, 0xD800, 0};
    {
        invalid_seq_counter notifier;
        strf::transcoding_error_notifier_ptr notifier_ptr{&notifier};

        TEST(u"\uFFFD\uFFFD").with(notifier_ptr) (strf::sani(invalid_input));
        TEST_EQ(notifier.notifications_count, 2);

        notifier.notifications_count = 0;
        TEST_TRUNCATING_AT(1, u"\uFFFD").with(notifier_ptr) (strf::sani(invalid_input));
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
            strf::to(buff) .with(notifier_ptr) (strf::sani(invalid_input));
        } catch (dummy_exception&) {
            thrown = true;
        } catch(...) {
        }
        TEST_TRUE(thrown);
    }

#endif // __cpp_exceptions
}

STRF_TEST_FUNC void utf16_sani_find_transcoder()
{
#if ! defined(__CUDACC__)

    using static_transcoder_type = strf::static_transcoder
        <char16_t, char16_t, strf::csid_utf16, strf::csid_utf16>;

    const strf::dynamic_charset<char16_t> dyn_cs = strf::utf16_t<char16_t>{}.to_dynamic();
    const strf::dynamic_transcoder<char16_t, char16_t> tr = strf::find_transcoder(dyn_cs, dyn_cs);

    TEST_TRUE(tr.transcode_func()      == static_transcoder_type::transcode);
    TEST_TRUE(tr.transcode_size_func() == static_transcoder_type::transcode_size);

#endif // defined(__CUDACC__)

    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char16_t, char16_t, strf::csid_utf16, strf::csid_utf16 >
                   , decltype(strf::find_transcoder( strf::utf_t<char16_t>()
                                                   , strf::utf_t<char16_t>())) >
                  :: value));
    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char16_t, char32_t, strf::csid_utf16, strf::csid_utf32 >
                   , decltype(strf::find_transcoder( strf::utf_t<char16_t>()
                                                   , strf::utf_t<char32_t>())) >
                  :: value));
    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char32_t, char16_t,  strf::csid_utf32, strf::csid_utf16 >
                   , decltype(strf::find_transcoder( strf::utf_t<char32_t>()
                                                   , strf::utf_t<char16_t>())) >
                  :: value));
}

template <std::size_t N>
STRF_HD std::ptrdiff_t utf16_count_codepoints(const char16_t (&str)[N])
{
    return strf::utf16_t<char16_t>::count_codepoints
        (str, str + N - 1, 100000)
        .count;
}

template <std::size_t N>
STRF_HD std::ptrdiff_t utf16_count_codepoints_fast(const char16_t (&str)[N])
{
    return strf::utf16_t<char16_t>::count_codepoints_fast(str, str + N - 1, 100000).count;
}

STRF_HD void utf16_codepoints_count()
{
    {   // test valid input
        TEST_EQ(0, utf16_count_codepoints(u""));
        TEST_EQ(3, utf16_count_codepoints(u"abc"));
        TEST_EQ(1, utf16_count_codepoints(u"\uD7FF"));
        TEST_EQ(1, utf16_count_codepoints(u"\uE000"));
        TEST_EQ(1, utf16_count_codepoints(u"\U0010FFFF"));

        TEST_EQ(0, utf16_count_codepoints_fast(u""));
        TEST_EQ(3, utf16_count_codepoints_fast(u"abc"));
        TEST_EQ(1, utf16_count_codepoints_fast(u"\uD7FF"));
        TEST_EQ(1, utf16_count_codepoints_fast(u"\uE000"));
        TEST_EQ(1, utf16_count_codepoints_fast(u"\U0010FFFF"));
    }
    {   // when limit is less than or equal to count

        const char16_t str[] = u"a\0\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF";
        const auto str_len = sizeof(str)/2 - 1;
        const auto* const str_end = str + str_len;
        const strf::utf16_t<char16_t> charset;

        {
            auto r = charset.count_codepoints(str, str_end, 8);
            TEST_EQ((const void*)r.ptr, (const void*)str_end);
            TEST_EQ(r.count, 8);
        }
        {
            auto r = charset.count_codepoints(str, str_end, 7);
            TEST_EQ((const void*)r.ptr, (const void*)(str_end - 2));
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
            TEST_EQ((const void*)r.ptr, (const void*)(str_end - 2));
            TEST_EQ(r.count, 7);
        }
        {
            auto r = charset.count_codepoints_fast(str, str_end, 0);
            TEST_EQ((const void*)r.ptr, (const void*)str);
            TEST_EQ(r.count, 0);
        }
    }
}

STRF_TEST_FUNC void utf16_miscellaneous()
{
    const strf::utf16_t<char16_t> charset;
    {  // cover write_replacement_char(x);
        TEST(u"\uFFFD")                         .tr(u"{10}");
        TEST_CALLING_RECYCLE_AT(2, u"  \uFFFD").tr(u"  {10}");
        TEST_TRUNCATING_AT     (3, u"  \uFFFD").tr(u"  {10}");
        TEST_TRUNCATING_AT     (2, u"  ")      .tr(u"  {10}");
    }

    TEST_EQ(1, charset.validate('a'));
    TEST_EQ(1, charset.validate(0xFFFF));
    TEST_EQ(2, charset.validate(0x10000));
    TEST_EQ(2, charset.validate(0x10FFFF));
    TEST_EQ((std::size_t)-1, charset.validate(0x110000));

    {
        using utf16_to_utf16 = strf::static_transcoder
            <char16_t, char16_t, strf::csid_utf16, strf::csid_utf16>;

        auto tr = charset.find_transcoder_from<char16_t>(strf::csid_utf16);
        TEST_TRUE(tr.transcode_func()      == utf16_to_utf16::transcode);
        TEST_TRUE(tr.transcode_size_func() == utf16_to_utf16::transcode_size);
    }
}


} // unnamed namespace

STRF_TEST_FUNC void test_utf16()
{
    utf16_to_utf16_unsafe_transcode();
    utf16_sani_valid_sequences();
    utf16_sani_invalid_sequences();
    utf16_sani_error_notifier();
    utf16_sani_find_transcoder();
    utf16_codepoints_count();
    utf16_miscellaneous();
}

REGISTER_STRF_TEST(test_utf16)
