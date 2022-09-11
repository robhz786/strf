//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_invalid_sequences.hpp"

namespace {

STRF_TEST_FUNC void utf32_valid_sequences()
{
    TEST(U" ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        (strf::sani(U"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF") > 8);

    TEST_TRUNCATING_AT(2, U"ab") (strf::sani(U"ab\U0010FFFF"));
    TEST_TRUNCATING_AT(3, U"ab\U0010FFFF") (strf::sani(U"ab\U0010FFFF"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\U0010FFFF") (strf::sani(U"ab\U0010FFFF"));

    TEST_TRUNCATING_AT(2, U"ab")
        .with(strf::surrogate_policy::lax) (strf::sani(U"ab\U0010FFFF"));
    TEST_TRUNCATING_AT(3, U"ab\U0010FFFF")
        .with(strf::surrogate_policy::lax) (strf::sani(U"ab\U0010FFFF"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\U0010FFFF")
        .with(strf::surrogate_policy::lax) (strf::sani(U"ab\U0010FFFF"));

    {
        // when surrogates are allowed
        const char32_t str_D800[] = {0xD800, 0};
        const char32_t str_DBFF[] = {0xDBFF, 0};
        const char32_t str_DC00[] = {0xDC00, 0};
        const char32_t str_DFFF[] = {0xDFFF, 0};

        const char32_t str_D800_[] = {0xD800, U'_', 0};
        const char32_t str_DBFF_[] = {0xDBFF, U'_', 0};
        const char32_t str_DC00_[] = {0xDC00, U'_', 0};
        const char32_t str_DFFF_[] = {0xDFFF, U'_', 0};

        const char32_t _str_D800[] = {U' ', 0xD800, 0};
        const char32_t _str_DBFF[] = {U' ', 0xDBFF, 0};
        const char32_t _str_DC00[] = {U' ', 0xDC00, 0};
        const char32_t _str_DFFF[] = {U' ', 0xDFFF, 0};

        const char32_t _str_D800_[] = {U' ', 0xD800, U'_', 0};
        const char32_t _str_DBFF_[] = {U' ', 0xDBFF, U'_', 0};
        const char32_t _str_DC00_[] = {U' ', 0xDC00, U'_', 0};
        const char32_t _str_DFFF_[] = {U' ', 0xDFFF, U'_', 0};

        TEST(_str_D800) .with(strf::surrogate_policy::lax) (strf::sani(str_D800) > 2);
        TEST(_str_DBFF) .with(strf::surrogate_policy::lax) (strf::sani(str_DBFF) > 2);
        TEST(_str_DC00) .with(strf::surrogate_policy::lax) (strf::sani(str_DC00) > 2);
        TEST(_str_DFFF) .with(strf::surrogate_policy::lax) (strf::sani(str_DFFF) > 2);

        TEST(_str_D800_) .with(strf::surrogate_policy::lax) (strf::sani(str_D800_) > 3);
        TEST(_str_DBFF_) .with(strf::surrogate_policy::lax) (strf::sani(str_DBFF_) > 3);
        TEST(_str_DC00_) .with(strf::surrogate_policy::lax) (strf::sani(str_DC00_) > 3);
        TEST(_str_DFFF_) .with(strf::surrogate_policy::lax) (strf::sani(str_DFFF_) > 3);

        TEST_TRUNCATING_AT(2, _str_D800)
            .with(strf::surrogate_policy::lax) (strf::sani(str_D800) > 2);
        TEST_TRUNCATING_AT(1, U" ")
            .with(strf::surrogate_policy::lax) (strf::sani(str_D800) > 2);
    }
}

#define TEST_INVALID_SEQS(INPUT, ...)                                   \
    test_utils::test_invalid_sequences                                  \
        <strf::csid_utf32, strf::csid_utf32, char32_t, char32_t>        \
        ( BOOST_CURRENT_FUNCTION, __FILE__, __LINE__                    \
        , strf::surrogate_policy::strict, (INPUT), __VA_ARGS__ );

#define TEST_INVALID_SEQS_LAX(INPUT, ...)                               \
    test_utils::test_invalid_sequences                                  \
        <strf::csid_utf32, strf::csid_utf32, char32_t, char32_t>        \
        ( BOOST_CURRENT_FUNCTION, __FILE__, __LINE__                    \
        , strf::surrogate_policy::lax, (INPUT), __VA_ARGS__ );

STRF_TEST_FUNC void utf32_invalid_sequences()
{
    const char32_t str_dfff[] = {0xDFFF, 0};
    const char32_t str_d800[] = {0xD800, 0};
    const char32_t str_110000[] = {0x110000, 0};
    {
        // surrogates
        const char32_t str[] = {0xD800, 0xDFFF, 0};
        TEST(U" \uFFFD\uFFFD") (strf::sani(str) > 3);

        TEST_TRUNCATING_AT(2, U" \uFFFD") (strf::sani(str) > 3);
        TEST_TRUNCATING_AT(3, U" \uFFFD\uFFFD") (strf::sani(str) > 3);
        TEST_TRUNCATING_AT(1, U" ") (strf::sani(str) > 3);
    }
    {   // codepoint too big
        const char32_t str[] = {0xD800, 0xDFFF, 0x110000, 0};
        TEST(U" \uFFFD\uFFFD\uFFFD") (strf::sani(str) > 4);
        TEST_INVALID_SEQS(str, str_d800, str_dfff, str_110000);

        const char32_t expected_lax[] = {0xD800, 0xDFFF, 0xFFFD, 0};
        TEST(expected_lax).with(strf::surrogate_policy::lax) (strf::sani(str) > 2);
        TEST_INVALID_SEQS_LAX(str, str_110000);
    }
}

struct invalid_seq_counter: strf::transcoding_error_notifier {
    void STRF_HD invalid_sequence(std::size_t, const char*, const void*, std::size_t) override {
        ++ notifications_count;
    }
    std::size_t notifications_count = 0;
};

#if defined(__cpp_exceptions) && __cpp_exceptions  && ! defined(__CUDACC__)

struct dummy_exception: std::exception {};

struct notifier_that_throws : strf::transcoding_error_notifier {
    void STRF_HD invalid_sequence(std::size_t, const char*, const void*, std::size_t) override {
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
STRF_HD std::size_t utf32_count_codepoints_strict(const char32_t (&str)[N])
{
    return strf::utf32_t<char32_t>::count_codepoints
        (str, N - 1, 100000, strf::surrogate_policy::strict)
        .count;
}

template <std::size_t N>
STRF_HD std::size_t utf32_count_codepoints_lax(const char32_t (&str)[N])
{
    return strf::utf32_t<char32_t>::count_codepoints
        (str, N - 1, 100000, strf::surrogate_policy::lax)
        .count;
}

template <std::size_t N>
STRF_HD std::size_t utf32_count_codepoints_fast(const char32_t (&str)[N])
{
    return strf::utf32_t<char32_t>::count_codepoints_fast(str, N - 1, 100000).count;
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
        strf::utf32_t<char32_t> charset;
        constexpr auto strict = strf::surrogate_policy::strict;

        {
            auto r = charset.count_codepoints(str, str_len, 8, strict);
            TEST_EQ(r.pos, str_len);
            TEST_EQ(r.count, 8);
        }
        {
            auto r = charset.count_codepoints(str, str_len, 7, strict);
            TEST_EQ(r.pos, str_len - 1);
            TEST_EQ(r.count, 7);
        }
        {
            auto r = charset.count_codepoints(str, str_len, 0, strict);
            TEST_EQ(r.pos, 0);
            TEST_EQ(r.count, 0);
        }
        {
            auto r = charset.count_codepoints_fast(str, str_len, 8);
            TEST_EQ(r.pos, str_len);
            TEST_EQ(r.count, 8);
        }
        {
            auto r = charset.count_codepoints_fast(str, str_len, 7);
            TEST_EQ(r.pos, str_len - 1);
            TEST_EQ(r.count, 7);
        }
        {
            auto r = charset.count_codepoints_fast(str, str_len, 0);
            TEST_EQ(r.pos, 0);
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
    strf::utf32_t<char32_t> charset;
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
    utf32_valid_sequences();
    utf32_invalid_sequences();
    utf32_error_notifier();
    utf32_codepoints_count();
    utf32_miscellaneous();
}

REGISTER_STRF_TEST(test_utf32);
