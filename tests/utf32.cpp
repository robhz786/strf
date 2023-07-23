//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils/transcoding.hpp"

#define TEST_TRANSCODE                                                  \
    test_utils::trancode_tester_caller(BOOST_CURRENT_FUNCTION, __FILE__, __LINE__) \
    << test_utils::transcoding_test_data_maker<strf::utf_t<char32_t>, strf::utf_t<char32_t>> \
    (strf::utf<char32_t>, strf::utf<char32_t>, true)

#define TEST_UNSAFE_TRANSCODE                                           \
    test_utils::trancode_tester_caller(BOOST_CURRENT_FUNCTION, __FILE__, __LINE__) \
    << test_utils::transcoding_test_data_maker<strf::utf_t<char32_t>, strf::utf_t<char32_t>> \
    (strf::utf<char32_t>, strf::utf<char32_t>, false)

namespace {

STRF_TEST_FUNC void utf32_to_utf32_unsafe_transcode()
{
    TEST_UNSAFE_TRANSCODE
        .input(U"ab\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF")
        .expect(U"ab\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_UNSAFE_TRANSCODE
        .input(U"abc")
        .expect(U"ab")
        .destination_size(2)
        .expect_stop_reason(strf::transcode_stop_reason::bad_destination);
    TEST_UNSAFE_TRANSCODE
        .input(U"\U00010000")
        .expect(U"")
        .bad_destination()
        .expect_stop_reason(strf::transcode_stop_reason::bad_destination);
    TEST_UNSAFE_TRANSCODE
        .input(U"abc\U00010000")
        .expect(U"")
        .bad_destination()
        .expect_stop_reason(strf::transcode_stop_reason::bad_destination);

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
        .input(U"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        .expect(U"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        .expect_stop_reason(strf::transcode_stop_reason::completed);
    TEST_TRANSCODE
        .input(U"abc")
        .expect(U"ab")
        .destination_size(2)
        .expect_stop_reason(strf::transcode_stop_reason::bad_destination);
    TEST_TRANSCODE
        .input(U"\U00010000")
        .expect(U"")
        .bad_destination()
        .expect_stop_reason(strf::transcode_stop_reason::bad_destination);

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

        TEST(U" \U00010000") .with(strf::surrogate_policy::lax) (strf::sani(U"\U00010000") > 2);

        const char32_t str_spc_D800[] = {' ', 0xD800, 0};
        TEST_CALLING_RECYCLE_AT(1, str_spc_D800)
            .with(strf::surrogate_policy::lax) (strf::sani(str_spc_D800));
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
        .bad_destination()
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
    TEST_TRANSCODE
        .input(str_110000)
        .expect(U"")
        .bad_destination()
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

        {   // strf::surrogate_policy::strict

            const char32_t invalid_input[] = {0xD800, 0xDFFF, 0x110000, 0};
            TEST(U"\uFFFD\uFFFD\uFFFD")
                .with(notifier_ptr, strf::surrogate_policy::strict)
                (strf::sani(invalid_input));
            TEST_EQ(notifier.notifications_count, 3);

            notifier.notifications_count = 0;
            TEST_TRUNCATING_AT(1, U"\uFFFD")
                .with(notifier_ptr, strf::surrogate_policy::strict)
                (strf::sani(invalid_input));
            TEST_TRUE(notifier.notifications_count > 0);
        }

        {   // using strf::surrogate_policy::lax
            const char32_t invalid_input[] = {0x110000, 0x110001, 0};
            notifier.notifications_count = 0;
            TEST(U"\uFFFD\uFFFD")
                .with(notifier_ptr, strf::surrogate_policy::lax)
                (strf::sani(invalid_input));
            TEST_EQ(notifier.notifications_count, 2);

            notifier.notifications_count = 0;
            TEST_TRUNCATING_AT(1, U"\uFFFD")
                .with(notifier_ptr, strf::surrogate_policy::lax)
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

        {   // using strf::surrogate_policy::strict
            const char32_t invalid_input[] = {0xD800, 0xDFFF, 0x110000, 0};
            bool thrown = false;
            try {
                char32_t buff[10];
                strf::to(buff)
                    .with(notifier_ptr, strf::surrogate_policy::strict)
                    (strf::sani(invalid_input));
            } catch (dummy_exception&) {
                thrown = true;
            } catch(...) {
            }
            TEST_TRUE(thrown);
        }
        {   // using strf::surrogate_policy::lax
            const char32_t invalid_input[] = {0x110000, 0x110001, 0};
            bool thrown = false;
            try {
                char32_t buff[10];
                strf::to(buff)
                    .with(notifier_ptr, strf::surrogate_policy::lax)
                    (strf::sani(invalid_input));
            } catch (dummy_exception&) {
                thrown = true;
            } catch(...) {
            }
            TEST_TRUE(thrown);
        }

    }
#endif // __cpp_exceptions

}

template <std::size_t N>
STRF_HD std::ptrdiff_t utf32_count_codepoints_strict(const char32_t (&str)[N])
{
    return strf::utf32_t<char32_t>::count_codepoints
        (str, str + N - 1, 100000, strf::surrogate_policy::strict)
        .count;
}

template <std::size_t N>
STRF_HD std::ptrdiff_t utf32_count_codepoints_lax(const char32_t (&str)[N])
{
    return strf::utf32_t<char32_t>::count_codepoints
        (str, str + N - 1, 100000, strf::surrogate_policy::lax)
        .count;
}

template <std::size_t N>
STRF_HD std::ptrdiff_t utf32_count_codepoints_fast(const char32_t (&str)[N])
{
    return strf::utf32_t<char32_t>::count_codepoints_fast(str, str + N - 1, 100000).count;
}

STRF_HD void utf32_codepoints_count()
{
    {   // test valid input
        TEST_EQ(0, utf32_count_codepoints_strict(U""));
        TEST_EQ(3, utf32_count_codepoints_strict(U"abc"));
        TEST_EQ(1, utf32_count_codepoints_strict(U"\uD7FF"));
        TEST_EQ(1, utf32_count_codepoints_strict(U"\uE000"));
        TEST_EQ(1, utf32_count_codepoints_strict(U"\U0010FFFF"));

        TEST_EQ(1, utf32_count_codepoints_lax(U"\uD7FF"));
        TEST_EQ(1, utf32_count_codepoints_lax(U"\uE000"));
        TEST_EQ(1, utf32_count_codepoints_lax(U"\U0010FFFF"));

        TEST_EQ(0, utf32_count_codepoints_fast(U""));
        TEST_EQ(3, utf32_count_codepoints_fast(U"abc"));
        TEST_EQ(1, utf32_count_codepoints_fast(U"\uD7FF"));
        TEST_EQ(1, utf32_count_codepoints_fast(U"\uE000"));
        TEST_EQ(1, utf32_count_codepoints_fast(U"\U0010FFFF"));
    }
    {   // when surrogates are allowed
        const char32_t u32str_D800[] = {0xD800, 0};
        const char32_t u32str_DBFF[] = {0xDBFF, 0};
        const char32_t u32str_DC00[] = {0xDC00, 0};
        const char32_t u32str_DFFF[] = {0xDFFF, 0};

        TEST_EQ(1, utf32_count_codepoints_lax(u32str_D800));
        TEST_EQ(1, utf32_count_codepoints_lax(u32str_DBFF));
        TEST_EQ(1, utf32_count_codepoints_lax(u32str_DC00));
        TEST_EQ(1, utf32_count_codepoints_lax(u32str_DFFF));
    }
    {   // invalid sequences
        {
            // high surrogate followed by another high surrogate
            const char32_t str[] = {0xD800, 0xD800, 0};
            TEST_EQ(2, utf32_count_codepoints_lax(str));
        }
        {
            // low surrogate followed by a high surrogate
            const char32_t str[] = {0xDFFF, 0xD800, 0};
            TEST_EQ(2, utf32_count_codepoints_lax(str));
        }
        {
            // a low surrogate
            const char32_t str[] = {0xDFFF, 0};
            TEST_EQ(1, utf32_count_codepoints_lax(str));
        }
        {
            // a high surrogate
            const char32_t str[] = {0xD800, 0};
            TEST_EQ(1, utf32_count_codepoints_lax(str));
        }
    }
    {   // when limit is less than or equal to count

        const char32_t str[] = U"a\0\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF";
        const auto str_len = sizeof(str)/4 - 1;
        const auto * const str_end = str + str_len;
        const strf::utf32_t<char32_t> charset;
        constexpr auto strict = strf::surrogate_policy::strict;

        {
            auto r = charset.count_codepoints(str, str_end, 8, strict);
            TEST_EQ((const void*)r.ptr, (const void*)str_end);
            TEST_EQ(r.count, 8);
        }
        {
            auto r = charset.count_codepoints(str, str_end, 7, strict);
            TEST_EQ((const void*)r.ptr, (const void*)(str_end - 1));
            TEST_EQ(r.count, 7);
        }
        {
            auto r = charset.count_codepoints(str, str_end, 0, strict);
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
    TEST_EQ(1, charset.validate(0xFFFFFF));

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
