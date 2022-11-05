//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_invalid_sequences.hpp"

namespace {

STRF_TEST_FUNC void utf16_to_utf8_valid_sequences()
{
    TEST(u8" ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        (strf::sani(u"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF") > 8);

    TEST_TRUNCATING_AT(2, u8"ab") (strf::sani(u"abc"));
    TEST_TRUNCATING_AT(2, u8"ab") (strf::sani(u"ab\u0080"));
    TEST_TRUNCATING_AT(3, u8"ab") (strf::sani(u"ab\u0080"));
    TEST_TRUNCATING_AT(4, u8"ab") (strf::sani(u"ab\uD7FF"));
    TEST_TRUNCATING_AT(5, u8"ab") (strf::sani(u"ab\U00010000"));

    TEST_TRUNCATING_AT(3, u8"abc")          (strf::sani(u"abc"));
    TEST_TRUNCATING_AT(4, u8"ab\u0080")     (strf::sani(u"ab\u0080"));
    TEST_TRUNCATING_AT(5, u8"ab\uD7FF")     (strf::sani(u"ab\uD7FF"));
    TEST_TRUNCATING_AT(6, u8"ab\U00010000") (strf::sani(u"ab\U00010000"));
    TEST_TRUNCATING_AT(6, u8"ab\U0010FFFF") (strf::sani(u"ab\U0010FFFF"));

    TEST_CALLING_RECYCLE_AT(2, u8"abc")          (strf::sani(u"abc"));
    TEST_CALLING_RECYCLE_AT(2, u8"ab\u0080")     (strf::sani(u"ab\u0080"));
    TEST_CALLING_RECYCLE_AT(3, u8"ab\u0080")     (strf::sani(u"ab\u0080"));
    TEST_CALLING_RECYCLE_AT(4, u8"ab\uD7FF")     (strf::sani(u"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT(4, u8"ab\U00010000") (strf::sani(u"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(5, u8"ab\U0010FFFF") (strf::sani(u"ab\U0010FFFF"));
    {
        // when surrogates are allowed
        const char16_t u16str_D800[] = {0xD800, 0};
        const char16_t u16str_DBFF[] = {0xDBFF, 0};
        const char16_t u16str_DC00[] = {0xDC00, 0};
        const char16_t u16str_DFFF[] = {0xDFFF, 0};

        const char16_t u16str_D800_[] = {0xD800, u'_', 0};
        const char16_t u16str_DBFF_[] = {0xDBFF, u'_', 0};
        const char16_t u16str_DC00_[] = {0xDC00, u'_', 0};
        const char16_t u16str_DFFF_[] = {0xDFFF, u'_', 0};

        const char16_t u16str_DFFF_D800_[] = {0xDFFF, 0xD800, u'_', 0};

        TEST(" \xED\xA0\x80") .with(strf::surrogate_policy::lax) (strf::sani(u16str_D800) > 2);
        TEST(" \xED\xAF\xBF") .with(strf::surrogate_policy::lax) (strf::sani(u16str_DBFF) > 2);
        TEST(" \xED\xB0\x80") .with(strf::surrogate_policy::lax) (strf::sani(u16str_DC00) > 2);
        TEST(" \xED\xBF\xBF") .with(strf::surrogate_policy::lax) (strf::sani(u16str_DFFF) > 2);

        TEST(" \xED\xA0\x80_") .with(strf::surrogate_policy::lax) (strf::sani(u16str_D800_) > 3);
        TEST(" \xED\xAF\xBF_") .with(strf::surrogate_policy::lax) (strf::sani(u16str_DBFF_) > 3);
        TEST(" \xED\xB0\x80_") .with(strf::surrogate_policy::lax) (strf::sani(u16str_DC00_) > 3);
        TEST(" \xED\xBF\xBF_") .with(strf::surrogate_policy::lax) (strf::sani(u16str_DFFF_) > 3);

        TEST(" \xED\xBF\xBF\xED\xA0\x80_") .with(strf::surrogate_policy::lax)
            (strf::sani(u16str_DFFF_D800_) > 4);
        TEST(u8" \U00010000") .with(strf::surrogate_policy::lax) (strf::sani(u"\U00010000") > 2);

        TEST_TRUNCATING_AT(4, " \xED\xA0\x80")
            .with(strf::surrogate_policy::lax) (strf::sani(u16str_D800) > 2);
        TEST_TRUNCATING_AT(3, u8" ")
            .with(strf::surrogate_policy::lax) (strf::sani(u16str_D800) > 2);
    }

}

#define TEST_INVALID_SEQS(INPUT, ...)                                   \
    test_utils::test_invalid_sequences                                  \
        <strf::csid_utf16, strf::csid_utf8, char16_t, char>             \
        ( BOOST_CURRENT_FUNCTION, __FILE__, __LINE__                    \
        , strf::surrogate_policy::strict, (INPUT), __VA_ARGS__ );

#define TEST_INVALID_SEQS_LAX(INPUT, ...)                               \
    test_utils::test_invalid_sequences                                  \
        <strf::csid_utf16, strf::csid_utf8, char16_t, char>             \
        ( BOOST_CURRENT_FUNCTION, __FILE__, __LINE__                    \
        , strf::surrogate_policy::lax, (INPUT), __VA_ARGS__ );

STRF_TEST_FUNC void utf16_to_utf8_invalid_sequences()
{
    const char16_t str_dfff[] = {0xDFFF, 0};
    const char16_t str_d800[] = {0xD800, 0};
    {
        // high surrogate at the end
        const char16_t str[] = {0xD800, 0};
        TEST(u8" \uFFFD") (strf::sani(str) > 2);
        TEST_INVALID_SEQS(str, str_d800);
    }
    {
        // high surrogate followed by another high surrogate
        const char16_t str[] = {0xD800, 0xD800, 0};
        TEST(u8" \uFFFD\uFFFD") (strf::sani(str) > 3);
        TEST_INVALID_SEQS(str, str_d800, str_d800);
    }
    {
        // low surrogate followed by a high surrogate
        const char16_t str[] = {0xDFFF, 0xD800, 0};
        TEST(u8" \uFFFD\uFFFD") (strf::sani(str) > 3);
        TEST_INVALID_SEQS(str, str_dfff, str_d800);
    }
    {
        // a low surrogate
        const char16_t str[] = {0xDFFF, 0};
        TEST(u8" \uFFFD") (strf::sani(str) > 2);
        TEST_INVALID_SEQS(str, str_dfff);

        TEST_TRUNCATING_AT     (4, u8" \uFFFD") (strf::sani(str) > 2);
        TEST_CALLING_RECYCLE_AT(3, u8" \uFFFD") (strf::sani(str) > 2);
        TEST_TRUNCATING_AT     (4, u8" \uFFFD") (strf::sani(str) > 2);
        TEST_TRUNCATING_AT     (3, u8" ")       (strf::sani(str) > 2);
    }
    {
        // a high surrogate
        const char16_t str[] = {'_', 0xD800, '_', 0};
        TEST(u8" _\uFFFD_") (strf::sani(str) > 4);
        TEST_INVALID_SEQS(str, str_d800);
    }
    {
        // low surrogate followed by a high surrogate
        const char16_t str[] = {'_', 0xDFFF, 0xD800, '_', 0};
        TEST(u8" _\uFFFD\uFFFD_") (strf::sani(str) > 5);
        TEST_INVALID_SEQS(str, str_dfff, str_d800);
    }
    {
        const char16_t str[] = {'_', 0xDFFF, '_', 0};
        TEST(u8" _\uFFFD_") (strf::sani(str) > 4);
        TEST_INVALID_SEQS(str, str_dfff);
    }
    {
        const char16_t str[] = {'_', 0xD800, '_', 0};
        TEST(u8" _\uFFFD_") (strf::sani(str) > 4);
    }
}

struct invalid_seq_counter: strf::transcoding_error_notifier {
    void STRF_HD invalid_sequence(std::size_t, const char*, const void*, std::size_t) override {
        ++ notifications_count;
    }
    std::size_t notifications_count = 0;
};

#if defined(__cpp_exceptions) && __cpp_exceptions  && ! defined(__CUDACC__)

struct dummy_exception : std::exception {};

struct notifier_that_throws : strf::transcoding_error_notifier {
    void STRF_HD invalid_sequence(std::size_t, const char*, const void*, std::size_t) override {
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
    utf16_to_utf8_valid_sequences();
    utf16_to_utf8_invalid_sequences();
    utf16_to_utf8_error_notifier();
    utf16_to_utf8_find_transcoder();
}

REGISTER_STRF_TEST(test_utf16_to_utf8);
