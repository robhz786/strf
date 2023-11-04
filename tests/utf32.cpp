//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils/transcoding.hpp"

#define TEST_TRANSCODE                                                  \
    test_utils::transcode_tester_caller(BOOST_CURRENT_FUNCTION, __FILE__, __LINE__) \
    << test_utils::transcoding_test_data_maker<strf::utf_t<char32_t>, strf::utf_t<char32_t>> \
    (strf::utf<char32_t>, strf::utf<char32_t>, true)

#define TEST_UNSAFE_TRANSCODE                                           \
    test_utils::transcode_tester_caller(BOOST_CURRENT_FUNCTION, __FILE__, __LINE__) \
    << test_utils::transcoding_test_data_maker<strf::utf_t<char32_t>, strf::utf_t<char32_t>> \
    (strf::utf<char32_t>, strf::utf<char32_t>, false)

namespace {

STRF_TEST_FUNC void utf32_to_utf32_unsafe_transcode()
{
    TEST_UNSAFE_TRANSCODE
        .input(U"ab")
        .expect(U"ab")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UNSAFE_TRANSCODE
        .input(U"\u0080")
        .expect(U"\u0080")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UNSAFE_TRANSCODE
        .input(U"\u0800")
        .expect(U"\u0800")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UNSAFE_TRANSCODE
        .input(U"\uD7FF")
        .expect(U"\uD7FF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UNSAFE_TRANSCODE
        .input(U"\U00010000")
        .expect(U"\U00010000")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UNSAFE_TRANSCODE
        .input(U"\U0010FFFF")
        .expect(U"\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_UNSAFE_TRANSCODE
        .input(U"ab\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF")
        .expect(U"ab\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UNSAFE_TRANSCODE
        .input(U"abc")
        .expect(U"ab")
        .destination_size(2)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UNSAFE_TRANSCODE
        .input(U"\U00010000")
        .expect(U"")
        .destination_size(0)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_UNSAFE_TRANSCODE
        .input(U"abc\U00010000")
        .expect(U"")
        .destination_size(0)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);

    const char32_t str_D800[] = {0xD800, 0};
    const char32_t str_DBFF[] = {0xDBFF, 0};
    const char32_t str_DC00[] = {0xDC00, 0};
    const char32_t str_DFFF[] = {0xDFFF, 0};
    const char32_t str_DFFF_D800_[] = {0xDFFF, 0xD800, u'_', 0};

    TEST_UNSAFE_TRANSCODE
        .input(str_D800)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(static_cast<char32_t>(0xD800))
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UNSAFE_TRANSCODE
        .input(str_DBFF)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(static_cast<char32_t>(0xDBFF))
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UNSAFE_TRANSCODE
        .input(str_DC00)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(static_cast<char32_t>(0xDC00))
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UNSAFE_TRANSCODE
        .input(str_DFFF)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(static_cast<char32_t>(0xDFFF))
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
    TEST_UNSAFE_TRANSCODE
        .input(str_DFFF_D800_)
        .flags(strf::transcode_flags::lax_surrogate_policy)
        .expect(static_cast<char32_t>(0xDFFF), static_cast<char32_t>(0xD800), U'_')
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});
}


STRF_TEST_FUNC void utf32_valid_sequences()
{
    TEST_TRANSCODE
        .input(U"ab")
        .expect(U"ab")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRANSCODE
        .input(U"\u0080")
        .expect(U"\u0080")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRANSCODE
        .input(U"\u0800")
        .expect(U"\u0800")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRANSCODE
        .input(U"\uD7FF")
        .expect(U"\uD7FF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRANSCODE
        .input(U"\U00010000")
        .expect(U"\U00010000")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRANSCODE
        .input(U"\U0010FFFF")
        .expect(U"\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({});

    TEST_TRANSCODE
        .input(U"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        .expect(U"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_TRANSCODE
        .input(U"abc")
        .expect(U"ab")
        .destination_size(2)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);
    TEST_TRANSCODE
        .input(U"\U00010000")
        .expect(U"")
        .destination_size(0)
        .expect_stop_reason(strf::transcode_stop_reason::insufficient_output_space);

    TEST_TRUNCATING_AT(2, U"ab") (strf::sani(U"ab\U0010FFFF"));
    TEST_TRUNCATING_AT(3, U"ab\U0010FFFF") (strf::sani(U"ab\U0010FFFF"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\U0010FFFF") (strf::sani(U"ab\U0010FFFF"));

    {
        // when surrogates are allowed
        const char32_t str_D800[] = {0xD800, 0};
        const char32_t str_DBFF[] = {0xDBFF, 0};
        const char32_t str_DC00[] = {0xDC00, 0};
        const char32_t str_DFFF[] = {0xDFFF, 0};

        const char32_t str_DFFF_D800_[] = {0xDFFF, 0xD800, u'_', 0};

        const auto flags = ( strf::transcode_flags::lax_surrogate_policy |
                             strf::transcode_flags::stop_on_invalid_sequence |
                             strf::transcode_flags::stop_on_unsupported_codepoint );
        TEST_TRANSCODE
            .input(str_D800)
            .flags(flags)
            .expect(static_cast<char32_t>(0xD800))
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_TRANSCODE
            .input(str_DBFF)
            .flags(flags)
            .expect(static_cast<char32_t>(0xDBFF))
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_TRANSCODE
            .input(str_DC00)
            .flags(flags)
            .expect(static_cast<char32_t>(0xDC00))
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_TRANSCODE
            .input(str_DFFF)
            .flags(flags)
            .expect(static_cast<char32_t>(0xDFFF))
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
        TEST_TRANSCODE
            .input(str_DFFF_D800_)
            .flags(flags)
            .expect(static_cast<char32_t>(0xDFFF), static_cast<char32_t>(0xD800), U'_')
            .expect_stop_reason(strf::transcode_stop_reason::completed)
            .expect_unsupported_codepoints({})
            .expect_invalid_sequences({});
    }
}

STRF_TEST_FUNC void test_not_allowed_surrogate(char32_t surrogate_char)
{
    TEST_TRANSCODE
        .input(U"abc_", surrogate_char, U"_def")
        .expect(U"abc_\uFFFD_def")
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{surrogate_char}});
    TEST_TRANSCODE
        .input(U"abc_", surrogate_char, U"_def")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect(U"abc_")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{surrogate_char}});
    TEST_TRANSCODE
        .input(U"abc_", surrogate_char, U"_def")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .destination_size(4)
        .expect(U"abc_")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{surrogate_char}});
    TEST_TRANSCODE
        .input(surrogate_char, U"_def")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .destination_size(0)
        .expect(U"")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{surrogate_char}});
    TEST_TRANSCODE
        .input(surrogate_char, U"_def")
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect(U"")
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{surrogate_char}});
}

STRF_TEST_FUNC void utf32_invalid_sequences()
{
    // codepoint too big
    const char32_t str_110000[] = {0x110000, 0};
    TEST_TRANSCODE
        .input(str_110000)
        .expect(U"\uFFFD")
        .flags(strf::transcode_flags::lax_surrogate_policy ) // should have no effect
        .expect_stop_reason(strf::transcode_stop_reason::completed)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{static_cast<char32_t>(0x110000)}});
    TEST_TRANSCODE
        .input(str_110000)
        .expect(U"")
        .destination_size(0)
        .flags(strf::transcode_flags::stop_on_invalid_sequence)
        .expect_stop_reason(strf::transcode_stop_reason::invalid_sequence)
        .expect_unsupported_codepoints({})
        .expect_invalid_sequences({{static_cast<char32_t>(0x110000)}});

    test_not_allowed_surrogate(static_cast<char32_t>(0xD800)) ;
    test_not_allowed_surrogate(static_cast<char32_t>(0xDBFF)) ;
    test_not_allowed_surrogate(static_cast<char32_t>(0xDC00)) ;
    test_not_allowed_surrogate(static_cast<char32_t>(0xDFFF)) ;
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

STRF_TEST_FUNC void utf32_error_notifier()
{
    {
        invalid_seq_counter notifier;
        strf::transcoding_error_notifier_ptr notifier_ptr{&notifier};

        {
            const char32_t invalid_input[] = {0xD800, 0xDFFF, 0x110000, 0};
            TEST(U"\uFFFD\uFFFD\uFFFD")
                .with(notifier_ptr)
                (strf::sani(invalid_input));
            TEST_EQ(notifier.notifications_count, 3);

            notifier.notifications_count = 0;
            TEST_TRUNCATING_AT(1, U"\uFFFD")
                .with(notifier_ptr)
                (strf::sani(invalid_input));
            TEST_TRUE(notifier.notifications_count > 0);
        }
    }

#if defined(__cpp_exceptions) && __cpp_exceptions  && ! defined(__CUDACC__)

    {
        // check that an exception can be thrown, i.e,
        // ensure there is no `noexcept` blocking it
        notifier_that_throws notifier;
        strf::transcoding_error_notifier_ptr notifier_ptr{&notifier};

        {
            const char32_t invalid_input[] = {0xD800, 0xDFFF, 0x110000, 0};
            bool thrown = false;
            try {
                char32_t buff[10];
                strf::to(buff)
                    .with(notifier_ptr)
                    (strf::sani(invalid_input));
            } catch (dummy_exception&) {
                thrown = true;
            }
            TEST_TRUE(thrown);
        }

    }
#endif // __cpp_exceptions

}

template <std::size_t N>
STRF_HD std::ptrdiff_t utf32_count_codepoints(const char32_t (&str)[N])
{
    return strf::utf32_t<char32_t>::count_codepoints(str, str + N - 1, 100000).count;
}

template <std::size_t N>
STRF_HD std::ptrdiff_t utf32_count_codepoints_fast(const char32_t (&str)[N])
{
    return strf::utf32_t<char32_t>::count_codepoints_fast(str, str + N - 1, 100000).count;
}

STRF_HD void utf32_codepoints_count()
{
    {   // test valid input
        TEST_EQ(0, utf32_count_codepoints(U""));
        TEST_EQ(3, utf32_count_codepoints(U"abc"));
        TEST_EQ(1, utf32_count_codepoints(U"\uD7FF"));
        TEST_EQ(1, utf32_count_codepoints(U"\uE000"));
        TEST_EQ(1, utf32_count_codepoints(U"\U0010FFFF"));

        TEST_EQ(0, utf32_count_codepoints_fast(U""));
        TEST_EQ(3, utf32_count_codepoints_fast(U"abc"));
        TEST_EQ(1, utf32_count_codepoints_fast(U"\uD7FF"));
        TEST_EQ(1, utf32_count_codepoints_fast(U"\uE000"));
        TEST_EQ(1, utf32_count_codepoints_fast(U"\U0010FFFF"));
    }
    {   // when limit is less than or equal to count

        const char32_t str[] = U"a\0\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF";
        const auto str_len = sizeof(str)/4 - 1;
        const auto * const str_end = str + str_len;
        const strf::utf32_t<char32_t> charset;

        {
            auto r = charset.count_codepoints(str, str_end, 8);
            TEST_EQ((const void*)r.ptr, (const void*)str_end);
            TEST_EQ(r.count, 8);
        }
        {
            auto r = charset.count_codepoints(str, str_end, 7);
            TEST_EQ((const void*)r.ptr, (const void*)(str_end - 1));
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
            TEST_EQ((const void*)r.ptr, (const void*)(str_end - 1));
            TEST_EQ(r.count, 7);
        }
        {
            auto r = charset.count_codepoints_fast(str, str_end, 0);
            TEST_EQ((const void*)r.ptr, (const void*)str);
            TEST_EQ(r.count, 0);
        }
    }
}

STRF_TEST_FUNC void utf32_miscellaneous()
{
    {   // cover write_replacement_char(x);
        TEST(U"\uFFFD")                         .tr(U"{10}");
        TEST_CALLING_RECYCLE_AT(2, U"  \uFFFD").tr(U"  {10}");
        TEST_TRUNCATING_AT     (3, U"  \uFFFD").tr(U"  {10}");
        TEST_TRUNCATING_AT     (2, U"  ")      .tr(U"  {10}");
    }
    const strf::utf32_t<char32_t> charset;
    TEST_EQ(1, charset.validate(U'a'));
    TEST_EQ(1, charset.validate(0x10FFFF));
    TEST_EQ(-1, charset.validate(0x110000));
    TEST_EQ( 1, charset.validate(0xD7FF));
    TEST_EQ(-1, charset.validate(0xD800)); // surrogate
    TEST_EQ(-1, charset.validate(0xDFFF)); // surrogate
    TEST_EQ( 1, charset.validate(0xE000));

    {
        using utf32_to_utf32 = strf::static_transcoder
            <char32_t, char32_t, strf::csid_utf32, strf::csid_utf32>;
        auto tr = charset.find_transcoder_from(strf::tag<char32_t>{}, strf::csid_utf32);
        TEST_TRUE(tr.transcode_func()      == utf32_to_utf32::transcode);
        TEST_TRUE(tr.transcode_size_func() == utf32_to_utf32::transcode_size);
    }
    {
        using utf32_to_utf32 = strf::static_transcoder
            <char32_t, char32_t, strf::csid_utf32, strf::csid_utf32>;
        auto tr = charset.find_transcoder_to(strf::tag<char32_t>{}, strf::csid_utf32);
        TEST_TRUE(tr.transcode_func()      == utf32_to_utf32::transcode);
        TEST_TRUE(tr.transcode_size_func() == utf32_to_utf32::transcode_size);
    }
    {
        auto invalid_scid = static_cast<strf::charset_id>(123456);
        auto tr = charset.find_transcoder_from(strf::tag<char32_t>{}, invalid_scid);
        TEST_TRUE(tr.transcode_func()      == nullptr);
        TEST_TRUE(tr.transcode_size_func() == nullptr);
    }

    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char32_t, char32_t, strf::csid_utf32, strf::csid_utf32 >
                   , decltype(strf::find_transcoder( strf::utf_t<char32_t>()
                                                   , strf::utf_t<char32_t>())) >
                  :: value));
}

} // unnamed namespace

STRF_TEST_FUNC void test_utf32()
{
    utf32_to_utf32_unsafe_transcode();
    utf32_valid_sequences();
    utf32_invalid_sequences();
    utf32_error_notifier();
    utf32_codepoints_count();
    utf32_miscellaneous();
}

REGISTER_STRF_TEST(test_utf32)
