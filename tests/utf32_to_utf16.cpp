//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_invalid_sequences.hpp"

namespace {

STRF_TEST_FUNC void utf32_to_utf16_valid_sequences()
{
    TEST(u" ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        (strf::sani(U"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF") > 8);

    TEST_TRUNCATING_AT(2, u"ab") (strf::sani(U"ab\uD7FF"));
    TEST_TRUNCATING_AT(2, u"ab") (strf::sani(U"ab\U00010000"));

    TEST_TRUNCATING_AT(3, u"ab\uD7FF")     (strf::sani(U"ab\uD7FF"));
    TEST_TRUNCATING_AT(4, u"ab\U00010000") (strf::sani(U"ab\U00010000"));
    TEST_TRUNCATING_AT(4, u"ab\U0010FFFF") (strf::sani(U"ab\U0010FFFF"));

    TEST_CALLING_RECYCLE_AT(2, u"ab\uD7FF")     (strf::sani(U"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT(2, u"ab\U00010000") (strf::sani(U"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(3, u"ab\U0010FFFF") (strf::sani(U"ab\U0010FFFF"));
    {
        // when surrogates are allowed
        const char32_t u32str_D800[] = {0xD800, 0};
        const char32_t u32str_DBFF[] = {0xDBFF, 0};
        const char32_t u32str_DC00[] = {0xDC00, 0};
        const char32_t u32str_DFFF[] = {0xDFFF, 0};

        const char32_t u32str_D800_[] = {0xD800, u'_', 0};
        const char32_t u32str_DBFF_[] = {0xDBFF, u'_', 0};
        const char32_t u32str_DC00_[] = {0xDC00, u'_', 0};
        const char32_t u32str_DFFF_[] = {0xDFFF, u'_', 0};

        const char32_t u32str_DFFF_D800_[] = {0xDFFF, 0xD800, u'_', 0};

        const char16_t u16str_D800[] = {u' ', 0xD800, 0};
        const char16_t u16str_DBFF[] = {u' ', 0xDBFF, 0};
        const char16_t u16str_DC00[] = {u' ', 0xDC00, 0};
        const char16_t u16str_DFFF[] = {u' ', 0xDFFF, 0};

        const char16_t u16str_D800_[] = {u' ', 0xD800, u'_', 0};
        const char16_t u16str_DBFF_[] = {u' ', 0xDBFF, u'_', 0};
        const char16_t u16str_DC00_[] = {u' ', 0xDC00, u'_', 0};
        const char16_t u16str_DFFF_[] = {u' ', 0xDFFF, u'_', 0};

        const char16_t u16str_DFFF_D800_[] = {u' ', 0xDFFF, 0xD800, u'_', 0};

        TEST(u16str_D800) .with(strf::surrogate_policy::lax) (strf::sani(u32str_D800) > 2);
        TEST(u16str_DBFF) .with(strf::surrogate_policy::lax) (strf::sani(u32str_DBFF) > 2);
        TEST(u16str_DC00) .with(strf::surrogate_policy::lax) (strf::sani(u32str_DC00) > 2);
        TEST(u16str_DFFF) .with(strf::surrogate_policy::lax) (strf::sani(u32str_DFFF) > 2);

        TEST(u16str_D800_) .with(strf::surrogate_policy::lax) (strf::sani(u32str_D800_) > 3);
        TEST(u16str_DBFF_) .with(strf::surrogate_policy::lax) (strf::sani(u32str_DBFF_) > 3);
        TEST(u16str_DC00_) .with(strf::surrogate_policy::lax) (strf::sani(u32str_DC00_) > 3);
        TEST(u16str_DFFF_) .with(strf::surrogate_policy::lax) (strf::sani(u32str_DFFF_) > 3);

        TEST(u16str_DFFF_D800_) .with(strf::surrogate_policy::lax)
            (strf::sani(u32str_DFFF_D800_) > 4);

        TEST_CALLING_RECYCLE_AT(1, u16str_D800)
            .with(strf::surrogate_policy::lax) (strf::sani(u32str_D800) > 2);
        TEST_TRUNCATING_AT     (2, u16str_D800)
            .with(strf::surrogate_policy::lax) (strf::sani(u32str_D800) > 2);
        TEST_TRUNCATING_AT     (1, u" ")
            .with(strf::surrogate_policy::lax) (strf::sani(u32str_D800) > 2);
    }
}

#define TEST_INVALID_SEQS(INPUT, ...)                                   \
    test_utils::test_invalid_sequences                                  \
        <strf::csid_utf32, strf::csid_utf16, char32_t, char16_t>        \
        ( BOOST_CURRENT_FUNCTION, __FILE__, __LINE__                    \
        , strf::surrogate_policy::strict, (INPUT), __VA_ARGS__ );

#define TEST_INVALID_SEQS_LAX(INPUT, ...)                               \
    test_utils::test_invalid_sequences                                  \
        <strf::csid_utf32, strf::csid_utf16, char32_t, char16_t>        \
        ( BOOST_CURRENT_FUNCTION, __FILE__, __LINE__                    \
        , strf::surrogate_policy::lax, (INPUT), __VA_ARGS__ );

STRF_TEST_FUNC void utf32_to_utf16_invalid_sequences()
{
    const char32_t str_dfff[] = {0xDFFF, 0};
    const char32_t str_d800[] = {0xD800, 0};
    const char32_t str_110000[] = {0x110000, 0};
    {
        // surrogates
        const char32_t str[] = {0xD800, 0xDFFF, 0};
        TEST(u" \uFFFD\uFFFD") (strf::sani(str) > 3);

        TEST_TRUNCATING_AT     (2, u" \uFFFD")       (strf::sani(str) > 3);
        TEST_TRUNCATING_AT     (3, u" \uFFFD\uFFFD") (strf::sani(str) > 3);
        TEST_CALLING_RECYCLE_AT(2, u" \uFFFD\uFFFD") (strf::sani(str) > 3);
    }
    {   // codepoint too big
        const char32_t str[] = {0xD800, 0xDFFF, 0x110000, 0};
        TEST(u" \uFFFD\uFFFD\uFFFD") (strf::sani(str) > 4);
        TEST_INVALID_SEQS(str, str_d800, str_dfff, str_110000);

        const char16_t expected_lax[] = {0xD800, 0xDFFF, 0xFFFD, 0};
        TEST(expected_lax).with(strf::surrogate_policy::lax) (strf::sani(str));
        TEST_INVALID_SEQS_LAX(str, str_110000);
    }
}

struct invalid_seq_counter: strf::transcoding_error_notifier {
    void STRF_HD invalid_sequence(const char*, const void*, std::size_t, std::size_t) override {
        ++ notifications_count;
    }
    std::size_t notifications_count = 0;
};

#if defined(__cpp_exceptions) && __cpp_exceptions  && ! defined(__CUDACC__)

struct dummy_exception {};

struct notifier_that_throws : strf::transcoding_error_notifier {
    void STRF_HD invalid_sequence(const char*, const void*, std::size_t, std::size_t) override {
        throw dummy_exception{};
    }
};

#endif // __cpp_exceptions

STRF_TEST_FUNC void utf32_to_utf16_error_notifier()
{
    const char32_t invalid_input[] = {0x110000, 0xFFFFFF, 0};
    {
        invalid_seq_counter notifier;
        strf::transcoding_error_notifier_ptr notifier_ptr{&notifier};

        TEST(u"\uFFFD\uFFFD").with(notifier_ptr) (strf::sani(invalid_input));
        TEST_EQ(notifier.notifications_count, 2);

        notifier.notifications_count = 0;
        TEST_TRUNCATING_AT(1, u"\uFFFD").with(notifier_ptr) (strf::sani(invalid_input));
        TEST_TRUE(notifier.notifications_count > 0);

        notifier.notifications_count = 0;
        TEST(u"\uFFFD\uFFFD")
            .with(notifier_ptr, strf::surrogate_policy::lax)
            (strf::sani(invalid_input));
        TEST_EQ(notifier.notifications_count, 2);

        notifier.notifications_count = 0;
        TEST_TRUNCATING_AT(1, u"\uFFFD")
            .with(notifier_ptr, strf::surrogate_policy::lax)
            (strf::sani(invalid_input));
        TEST_TRUE(notifier.notifications_count > 0);
    }

#if defined(__cpp_exceptions) && __cpp_exceptions  && ! defined(__CUDACC__)

    {
        // check that an exception can be thrown, i.e,
        // ensure there is no `noexcept` blocking it
        notifier_that_throws notifier;
        strf::transcoding_error_notifier_ptr notifier_ptr{&notifier};
        {
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
        {
            bool thrown = false;
            try {
                char16_t buff[10];
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

STRF_TEST_FUNC void utf32_to_utf16_find_transcoder()
{
#if ! defined(__CUDACC__)

    using static_transcoder_type = strf::static_transcoder
        <char32_t, char, strf::csid_utf32, strf::csid_utf8>;

    strf::dynamic_charset<char32_t> dyn_utf32 = strf::utf32_t<char32_t>{}.to_dynamic();
    strf::dynamic_charset<char>     dyn_utf8  = strf::utf8_t<char>{}.to_dynamic();
    strf::dynamic_transcoder<char32_t, char> tr = strf::find_transcoder(dyn_utf32, dyn_utf8);

    TEST_TRUE(tr.transcode_func()      == static_transcoder_type::transcode);
    TEST_TRUE(tr.transcode_size_func() == static_transcoder_type::transcode_size);

#endif // defined(__CUDACC__)
}

} // unnamed namespace

STRF_TEST_FUNC void test_utf32_to_utf16()
{
    utf32_to_utf16_valid_sequences();
    utf32_to_utf16_invalid_sequences();
    utf32_to_utf16_error_notifier();
    utf32_to_utf16_find_transcoder();
}

REGISTER_STRF_TEST(test_utf32_to_utf16);
