//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_invalid_sequences.hpp"

namespace {

STRF_TEST_FUNC void utf16_to_utf32_valid_sequences()
{
    TEST(U" ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        (strf::sani(u"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF") > 8);

    TEST_TRUNCATING_AT(2, U"ab") (strf::sani(u"ab\uD7FF"));
    TEST_TRUNCATING_AT(2, U"ab") (strf::sani(u"ab\U00010000"));

    TEST_TRUNCATING_AT(3, U"ab\uD7FF")     (strf::sani(u"ab\uD7FF"));
    TEST_TRUNCATING_AT(3, U"ab\U00010000") (strf::sani(u"ab\U00010000"));
    TEST_TRUNCATING_AT(3, U"ab\U0010FFFF") (strf::sani(u"ab\U0010FFFF"));

    TEST_CALLING_RECYCLE_AT(2, U"ab\uD7FF")     (strf::sani(u"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\U00010000") (strf::sani(u"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\U0010FFFF") (strf::sani(u"ab\U0010FFFF"));
{
        // when surrogates are allowed
        const char32_t u32str_D800[] = {U' ', 0xD800, 0};
        const char32_t u32str_DBFF[] = {U' ', 0xDBFF, 0};
        const char32_t u32str_DC00[] = {U' ', 0xDC00, 0};
        const char32_t u32str_DFFF[] = {U' ', 0xDFFF, 0};

        const char32_t u32str_D800_[] = {U' ', 0xD800, U'_', 0};
        const char32_t u32str_DBFF_[] = {U' ', 0xDBFF, U'_', 0};
        const char32_t u32str_DC00_[] = {U' ', 0xDC00, U'_', 0};
        const char32_t u32str_DFFF_[] = {U' ', 0xDFFF, U'_', 0};

        const char32_t u32str_DFFF_D800_[] = {U' ', 0xDFFF, 0xD800, U'_', 0};

        const char16_t u16str_D800[] = {0xD800, 0};
        const char16_t u16str_DBFF[] = {0xDBFF, 0};
        const char16_t u16str_DC00[] = {0xDC00, 0};
        const char16_t u16str_DFFF[] = {0xDFFF, 0};

        const char16_t u16str_D800_[] = {0xD800, u'_', 0};
        const char16_t u16str_DBFF_[] = {0xDBFF, u'_', 0};
        const char16_t u16str_DC00_[] = {0xDC00, u'_', 0};
        const char16_t u16str_DFFF_[] = {0xDFFF, u'_', 0};

        const char16_t u16str_DFFF_D800_[] = {0xDFFF, 0xD800, u'_', 0};

        TEST(u32str_D800) .with(strf::surrogate_policy::lax) (strf::sani(u16str_D800) > 2);
        TEST(u32str_DBFF) .with(strf::surrogate_policy::lax) (strf::sani(u16str_DBFF) > 2);
        TEST(u32str_DC00) .with(strf::surrogate_policy::lax) (strf::sani(u16str_DC00) > 2);
        TEST(u32str_DFFF) .with(strf::surrogate_policy::lax) (strf::sani(u16str_DFFF) > 2);

        TEST(u32str_D800_) .with(strf::surrogate_policy::lax) (strf::sani(u16str_D800_) > 3);
        TEST(u32str_DBFF_) .with(strf::surrogate_policy::lax) (strf::sani(u16str_DBFF_) > 3);
        TEST(u32str_DC00_) .with(strf::surrogate_policy::lax) (strf::sani(u16str_DC00_) > 3);
        TEST(u32str_DFFF_) .with(strf::surrogate_policy::lax) (strf::sani(u16str_DFFF_) > 3);

        TEST(u32str_DFFF_D800_) .with(strf::surrogate_policy::lax)
            (strf::sani(u16str_DFFF_D800_) > 4);

        TEST(U" \U00010000") .with(strf::surrogate_policy::lax) (strf::sani(u"\U00010000") > 2);

        TEST_CALLING_RECYCLE_AT(1, u32str_D800)
            .with(strf::surrogate_policy::lax) (strf::sani(u16str_D800) > 2);
        TEST_TRUNCATING_AT     (2, u32str_D800)
            .with(strf::surrogate_policy::lax) (strf::sani(u16str_D800) > 2);
        TEST_TRUNCATING_AT     (1, U" ")
            .with(strf::surrogate_policy::lax) (strf::sani(u16str_D800) > 2);
    }

}

#define TEST_INVALID_SEQS(INPUT, ...)                                   \
    test_utils::test_invalid_sequences                                  \
        <strf::csid_utf16, strf::csid_utf32, char16_t, char32_t>        \
        ( BOOST_CURRENT_FUNCTION, __FILE__, __LINE__                    \
        , strf::surrogate_policy::strict, (INPUT), __VA_ARGS__ );

#define TEST_INVALID_SEQS_LAX(INPUT, ...)                               \
    test_utils::test_invalid_sequences                                  \
        <strf::csid_utf16, strf::csid_utf32, char16_t, char32_t>        \
        ( BOOST_CURRENT_FUNCTION, __FILE__, __LINE__                    \
        , strf::surrogate_policy::lax, (INPUT), __VA_ARGS__ );


STRF_TEST_FUNC void utf16_to_utf32_invalid_sequences()
{
    const char16_t str_dfff[] = {0xDFFF, 0};
    const char16_t str_d800[] = {0xD800, 0};
    {
        // high surrogate followed by another high surrogate
        const char16_t str[] = {0xD800, 0xD800, 0};
        TEST(U" \uFFFD\uFFFD") (strf::sani(str) > 3);
        TEST_INVALID_SEQS(str, str_d800, str_d800);
    }
    {
        // low surrogate followed by a high surrogate
        const char16_t str[] = {0xDFFF, 0xD800, 0};
        TEST(U" \uFFFD\uFFFD") (strf::sani(str) > 3);
        TEST_INVALID_SEQS(str, str_dfff, str_d800);
    }
    {
        // a low surrogate
        const char16_t str[] = {0xDFFF, 0};
        TEST(U" \uFFFD") (strf::sani(str) > 2);
        TEST_INVALID_SEQS(str, str_dfff);
    }
    {
        // a high surrogate
        const char16_t str[] = {0xD800, 0};
        TEST(U" \uFFFD") (strf::sani(str) > 2);
        TEST_INVALID_SEQS(str, str_d800);
    }
    {
        // low surrogate followed by a high surrogate
        const char16_t str[] = {'_', 0xDFFF, 0xD800, '_', 0};
        TEST(U" _\uFFFD\uFFFD_") (strf::sani(str) > 5);
        TEST_INVALID_SEQS(str, str_dfff, str_d800);
    }
    {
        const char16_t str[] = {'_', 0xDFFF, '_', 0};
        TEST(U" _\uFFFD_") (strf::sani(str) > 4);
        TEST_INVALID_SEQS(str, str_dfff);
    }
    {
        const char16_t str[] = {'_', 0xD800, '_', 0};
        TEST(U" _\uFFFD_") (strf::sani(str) > 4);
        TEST_INVALID_SEQS(str, str_d800);
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

    strf::dynamic_charset<char16_t> dyn_utf16 = strf::utf16_t<char16_t>{}.to_dynamic();
    strf::dynamic_charset<char32_t> dyn_utf32 = strf::utf32_t<char32_t>{}.to_dynamic();
    strf::dynamic_transcoder<char16_t, char32_t> tr = strf::find_transcoder(dyn_utf16, dyn_utf32);

    TEST_TRUE(tr.transcode_func()      == static_transcoder_type::transcode);
    TEST_TRUE(tr.transcode_size_func() == static_transcoder_type::transcode_size);

#endif // defined(__CUDACC__)
}

} // unnamed namespace

STRF_TEST_FUNC void test_utf16_to_utf32()
{
    utf16_to_utf32_valid_sequences();
    utf16_to_utf32_invalid_sequences();
    utf16_to_utf32_error_notifier();
    utf16_to_utf32_find_transcoder();
}

REGISTER_STRF_TEST(test_utf16_to_utf32);
