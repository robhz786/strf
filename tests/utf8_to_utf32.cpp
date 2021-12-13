//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

namespace {

STRF_TEST_FUNC void utf8_to_utf32_valid_sequences()
{
    TEST(U"\U00010000") (strf::sani(u8"\U00010000"));

    TEST(U" ") (strf::sani("") > 1);
    TEST(U"\U0010FFFF") (strf::sani("\xF4\x8F\xBF\xBF"));

    TEST(U" abc") (strf::sani("abc") > 4);
    TEST(U" ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        (strf::sani(u8"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF") > 8);

    TEST_TRUNCATING_AT(2, U"ab")     (strf::sani("abcdef"));
    TEST_TRUNCATING_AT(6, U"abcdef") (strf::sani("abcdef"));

    TEST_TRUNCATING_AT(2, U"ab") (strf::sani(u8"ab\u0080"));
    TEST_TRUNCATING_AT(2, U"ab") (strf::sani(u8"ab\u0800"));
    TEST_TRUNCATING_AT(2, U"ab") (strf::sani(u8"ab\uD7FF"));
    TEST_TRUNCATING_AT(2, U"ab") (strf::sani(u8"ab\U00010000"));

    TEST_TRUNCATING_AT(3, U"ab\u0080")     (strf::sani(u8"ab\u0080"));
    TEST_TRUNCATING_AT(3, U"ab\u0800")     (strf::sani(u8"ab\u0800"));
    TEST_TRUNCATING_AT(3, U"ab\uD7FF")     (strf::sani(u8"ab\uD7FF"));
    TEST_TRUNCATING_AT(3, U"ab\U00010000") (strf::sani(u8"ab\U00010000"));
    TEST_TRUNCATING_AT(3, U"ab\U0010FFFF") (strf::sani(u8"ab\U0010FFFF"));

    TEST_CALLING_RECYCLE_AT(2, U"ab\u0080")     (strf::sani(u8"ab\u0080"));
    TEST_TRUNCATING_AT     (3, U"ab\u0080")     (strf::sani(u8"ab\u0080"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\u0800")     (strf::sani(u8"ab\u0800"));
    TEST_TRUNCATING_AT     (3, U"ab\u0800")     (strf::sani(u8"ab\u0800"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\uD7FF")     (strf::sani(u8"ab\uD7FF"));
    TEST_TRUNCATING_AT     (3, U"ab\uD7FF")     (strf::sani(u8"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\U00010000") (strf::sani(u8"ab\U00010000"));
    TEST_TRUNCATING_AT     (4, U"ab\U00010000") (strf::sani(u8"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(2, U"ab\U0010FFFF") (strf::sani(u8"ab\U0010FFFF"));
    TEST_TRUNCATING_AT     (3, U"ab\U0010FFFF") (strf::sani(u8"ab\U0010FFFF"));

    {
        // when surrogates are allowed
        const char32_t u32str_D800[] = {U' ', 0xD800, 0};
        const char32_t u32str_DBFF[] = {U' ', 0xDBFF, 0};
        const char32_t u32str_DC00[] = {U' ', 0xDC00, 0};
        const char32_t u32str_DFFF[] = {U' ', 0xDFFF, 0};

        TEST(u32str_D800) .with(strf::surrogate_policy::lax) (strf::sani("\xED\xA0\x80") > 2);
        TEST(u32str_DBFF) .with(strf::surrogate_policy::lax) (strf::sani("\xED\xAF\xBF") > 2);
        TEST(u32str_DC00) .with(strf::surrogate_policy::lax) (strf::sani("\xED\xB0\x80") > 2);
        TEST(u32str_DFFF) .with(strf::surrogate_policy::lax) (strf::sani("\xED\xBF\xBF") > 2);

        TEST_CALLING_RECYCLE_AT(1, u32str_D800)
            .with(strf::surrogate_policy::lax) (strf::sani("\xED\xA0\x80") > 2);
        TEST_TRUNCATING_AT     (2, u32str_D800)
            .with(strf::surrogate_policy::lax) (strf::sani("\xED\xA0\x80") > 2);
        TEST_TRUNCATING_AT     (1, U" ")
            .with(strf::surrogate_policy::lax) (strf::sani("\xED\xA0\x80") > 2);
    }
}

STRF_TEST_FUNC void utf8_to_utf32_invalid_sequences()
{
    // sample from Tabble 3-8 of Unicode standard
    TEST(U" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xF1\x80\x80\xE1\x80\xC0") > 4);
    TEST(U" \uFFFD\uFFFD\uFFFD_") (strf::sani("\xF1\x80\x80\xE1\x80\xC0_") > 5);

    // missing leading byte
    TEST(U" \uFFFD")  (strf::sani("\xBF") > 2);
    TEST(U" \uFFFD_") (strf::sani("\xBF_") > 3);

    // missing leading byte
    TEST(U" \uFFFD\uFFFD")  (strf::sani("\x80\x80") > 3);
    TEST(U" \uFFFD\uFFFD_") (strf::sani("\x80\x80_") > 4);

    // overlong sequence
    TEST(U" \uFFFD\uFFFD")  (strf::sani("\xC1\xBF") > 3);
    TEST(U" \uFFFD\uFFFD_") (strf::sani("\xC1\xBF_") > 4);

    // overlong sequence
    TEST(U" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xE0\x9F\x80") > 4);
    TEST(U" \uFFFD\uFFFD\uFFFD_") (strf::sani("\xE0\x9F\x80_") > 5);

    // overlong sequence with extra continuation bytes
    TEST(U" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xC1\xBF\x80") > 4);
    TEST(U" \uFFFD\uFFFD\uFFFD_") (strf::sani("\xC1\xBF\x80_") > 5);

    // overlong sequence with extra continuation bytes
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD")  (strf::sani("\xE0\x9F\x80\x80") > 5);
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD_") (strf::sani("\xE0\x9F\x80\x80_") > 6);

    // overlong sequence
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD")  (strf::sani("\xF0\x8F\xBF\xBF" ) > 5);
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD_") (strf::sani("\xF0\x8F\xBF\xBF_" ) > 6);

    // overlong sequence with extra continuation bytes
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD")  (strf::sani("\xF0\x8F\xBF\xBF\x80" ) > 6);
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD_") (strf::sani("\xF0\x8F\xBF\xBF\x80_" ) > 7);

    // codepoint too big.
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xF4\x90\x80\x80_") > 6);
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xF5\x80\x80\x80_") > 6);
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xF6\x80\x80\x80_") > 6);
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xF7\x80\x80\x80_") > 6);
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xF8\x80\x80\x80_") > 6);
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xF9\x80\x80\x80_") > 6);
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xFA\x80\x80\x80_") > 6);
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xFB\x80\x80\x80_") > 6);
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xFC\x80\x80\x80_") > 6);
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xFD\x80\x80\x80_") > 6);
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xFE\x80\x80\x80_") > 6);
    TEST(U" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xFF\x80\x80\x80_") > 6);

    // missing continuation
    TEST(U" \uFFFD")  (strf::sani("\xF0\x90\xBF" ) > 2);
    TEST(U" \uFFFD_") (strf::sani("\xF0\x90\xBF_" ) > 3);

    TEST(U" \uFFFD")  (strf::sani("\xC2") > 2);
    TEST(U" \uFFFD_") (strf::sani("\xC2_") > 3);

    TEST(U" \uFFFD")  (strf::sani("\xE0") > 2);
    TEST(U" \uFFFD_") (strf::sani("\xE0_") > 3);

    TEST(U" \uFFFD")  (strf::sani("\xE0\xA0") > 2);
    TEST(U" \uFFFD_") (strf::sani("\xE0\xA0_") > 3);

    TEST(U" \uFFFD")  (strf::sani("\xE1") > 2);
    TEST(U" \uFFFD_") (strf::sani("\xE1_") > 3);

    TEST(U" \uFFFD")  (strf::sani("\xF1") > 2);
    TEST(U" \uFFFD_") (strf::sani("\xF1_") > 3);

    TEST(U" \uFFFD")  (strf::sani("\xF1\x81") > 2);
    TEST(U" \uFFFD_") (strf::sani("\xF1\x81_") > 3);

    TEST(U" \uFFFD")  (strf::sani("\xF1\x81\x81") > 2);
    TEST(U" \uFFFD_") (strf::sani("\xF1\x81\x81_") > 3);

    // surrogate
    TEST(U" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xA0\x80") > 4);
    TEST(U" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xAF\xBF") > 4);
    TEST(U" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xB0\x80") > 4);
    TEST(U" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xBF\xBF") > 4);
    TEST(U" \uFFFD\uFFFD\uFFFD_")  (strf::sani("\xED\xA0\x80_") > 5);
    TEST(U" \uFFFD\uFFFD\uFFFD_")  (strf::sani("\xED\xAF\xBF_") > 5);
    TEST(U" \uFFFD\uFFFD\uFFFD_")  (strf::sani("\xED\xB0\x80_") > 5);
    TEST(U" \uFFFD\uFFFD\uFFFD_")  (strf::sani("\xED\xBF\xBF_") > 5);

    // missing continuation, but could only be a surrogate.
    TEST(U" \uFFFD\uFFFD")  (strf::sani("\xED\xA0") > 3);
    TEST(U" \uFFFD\uFFFD_") (strf::sani("\xED\xBF_") > 4);

    // missing continuation. It could only be a surrogate, but surrogates are allowed
    auto allow_surr = strf::surrogate_policy::lax;
    TEST(U" \uFFFD")  .with(allow_surr) (strf::sani("\xED\xA0") > 2);
    TEST(U" \uFFFD_") .with(allow_surr) (strf::sani("\xED\xBF_") > 3);

    // missing continuation. Now it starts with \xED, but it is not a surrogate
    TEST(U" \uFFFD")  (strf::sani("\xED\x9F") > 2);
    TEST(U" \uFFFD_") (strf::sani("\xED\x9F_") > 3);

    // cover when recycle_buffer() needs to be called
    TEST_CALLING_RECYCLE_AT(2, U" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xA0\x80") > 4);
    TEST_TRUNCATING_AT     (4, U" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xA0\x80") > 4);
    TEST_TRUNCATING_AT     (2, U" \uFFFD")              (strf::sani("\xED\xA0\x80") > 4);
}

struct invalid_seq_counter: strf::invalid_seq_notifier {
    void STRF_HD notify() override {
        ++ notifications_count;
    }
    std::size_t notifications_count = 0;
};

#if defined(__cpp_exceptions) && __cpp_exceptions  && ! defined(__CUDACC__)

struct dummy_exception {};

struct notifier_that_throws : strf::invalid_seq_notifier {
    void STRF_HD notify() override {
        throw dummy_exception{};
    }
};

#endif // __cpp_exceptions

STRF_TEST_FUNC void utf8_to_utf32_error_notifier()
{
    {
        invalid_seq_counter notifier;
        strf::invalid_seq_notifier_ptr notifier_ptr{&notifier};

        TEST(U"\uFFFD\uFFFD\uFFFD").with(notifier_ptr) (strf::sani("\xED\xA0\x80"));
        TEST_EQ(notifier.notifications_count, 3);

        notifier.notifications_count = 0;
        TEST_TRUNCATING_AT(1, U"\uFFFD").with(notifier_ptr) (strf::sani("\xED\xA0\x80"));
        TEST_TRUE(notifier.notifications_count > 0);
    }

#if defined(__cpp_exceptions) && __cpp_exceptions  && ! defined(__CUDACC__)
    {
        // check that an exception can be thrown, i.e,
        // ensure there is no `noexcept` blocking it
        notifier_that_throws notifier;
        strf::invalid_seq_notifier_ptr notifier_ptr{&notifier};
        bool thrown = false;
        try {
            char32_t buff[10];
            strf::to(buff) .with(notifier_ptr) (strf::sani("\xED\xA0\x80"));
        } catch (dummy_exception&) {
            thrown = true;
        } catch(...) {
        }
        TEST_TRUE(thrown);
    }
#endif // __cpp_exceptions
}

STRF_TEST_FUNC void utf8_to_utf32_find_transcoder()
{
#if ! defined(__CUDACC__)

    using static_transcoder_type = strf::static_transcoder
        <char, char32_t, strf::csid_utf8, strf::csid_utf32>;

    strf::dynamic_charset<char>     dyn_utf8  = strf::utf8_t<char>{}.to_dynamic();
    strf::dynamic_charset<char32_t> dyn_utf32 = strf::utf32_t<char32_t>{}.to_dynamic();
    strf::dynamic_transcoder<char, char32_t> tr = strf::find_transcoder(dyn_utf8, dyn_utf32);

    TEST_TRUE(tr.transcode_func()      == static_transcoder_type::transcode);
    TEST_TRUE(tr.transcode_size_func() == static_transcoder_type::transcode_size);

#endif // defined(__CUDACC__)

    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char, char32_t, strf::csid_utf8, strf::csid_utf32 >
                       , decltype(strf::find_transcoder( strf::utf_t<char>{}
                                                       , strf::utf_t<char32_t>{})) >
                  :: value));
}

} // unnamed namespace

STRF_TEST_FUNC void test_utf8_to_utf32()
{
    utf8_to_utf32_valid_sequences();
    utf8_to_utf32_invalid_sequences();
    utf8_to_utf32_error_notifier();
    utf8_to_utf32_find_transcoder();
}

REGISTER_STRF_TEST(test_utf8_to_utf32);
