//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

namespace {

STRF_TEST_FUNC void utf8_to_utf16_valid_sequences()
{
    TEST(u" ") (strf::sani("") > 1);
    TEST(u" abc") (strf::sani("abc") > 4);
    TEST(u" ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        (strf::sani(u8"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF") > 8);

    TEST_CALLING_RECYCLE_AT<2,2>(u"abcd") (strf::sani("abcdef"));

    TEST_CALLING_RECYCLE_AT<2> (u"ab") (strf::sani(u8"ab\u0080"));
    TEST_CALLING_RECYCLE_AT<2> (u"ab") (strf::sani(u8"ab\u0800"));
    TEST_CALLING_RECYCLE_AT<2> (u"ab") (strf::sani(u8"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT<2> (u"ab") (strf::sani(u8"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT<3> (u"ab") (strf::sani(u8"ab\U00010000"));

    TEST_CALLING_RECYCLE_AT<3> (u"ab\u0080")     (strf::sani(u8"ab\u0080"));
    TEST_CALLING_RECYCLE_AT<3> (u"ab\u0800")     (strf::sani(u8"ab\u0800"));
    TEST_CALLING_RECYCLE_AT<3> (u"ab\uD7FF")     (strf::sani(u8"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT<5> (u"ab\U00010000") (strf::sani(u8"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT<5> (u"ab\U0010FFFF") (strf::sani(u8"ab\U0010FFFF"));

    TEST_CALLING_RECYCLE_AT<2, 1> (u"ab\u0080")     (strf::sani(u8"ab\u0080"));
    TEST_CALLING_RECYCLE_AT<2, 1> (u"ab\u0800")     (strf::sani(u8"ab\u0800"));
    TEST_CALLING_RECYCLE_AT<2, 1> (u"ab\uD7FF")     (strf::sani(u8"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT<2, 2> (u"ab\U00010000") (strf::sani(u8"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT<3, 1> (u"ab\U0010FFFF") (strf::sani(u8"ab\U0010FFFF"));

    {
        // when surrogates are allowed
        const char16_t u16str_D800[] = {u' ', 0xD800, 0};
        const char16_t u16str_DBFF[] = {u' ', 0xDBFF, 0};
        const char16_t u16str_DC00[] = {u' ', 0xDC00, 0};
        const char16_t u16str_DFFF[] = {u' ', 0xDFFF, 0};

        TEST(u16str_D800) .with(strf::surrogate_policy::lax) (strf::sani("\xED\xA0\x80") > 2);
        TEST(u16str_DBFF) .with(strf::surrogate_policy::lax) (strf::sani("\xED\xAF\xBF") > 2);
        TEST(u16str_DC00) .with(strf::surrogate_policy::lax) (strf::sani("\xED\xB0\x80") > 2);
        TEST(u16str_DFFF) .with(strf::surrogate_policy::lax) (strf::sani("\xED\xBF\xBF") > 2);

        TEST_CALLING_RECYCLE_AT<1, 1> (u16str_D800)
            .with(strf::surrogate_policy::lax) (strf::sani("\xED\xA0\x80") > 2);
        TEST_CALLING_RECYCLE_AT<1> (u" ")
            .with(strf::surrogate_policy::lax) (strf::sani("\xED\xA0\x80") > 2);
    }
}

void STRF_TEST_FUNC utf8_to_utf16_invalid_sequences()
{
    // sample from Tabble 3-8 of Unicode standard
    TEST(u" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xF1\x80\x80\xE1\x80\xC0") > 4);
    TEST(u" \uFFFD\uFFFD\uFFFD_") (strf::sani("\xF1\x80\x80\xE1\x80\xC0_") > 5);

    // missing leading byte
    TEST(u" \uFFFD")  (strf::sani("\xBF") > 2);
    TEST(u" \uFFFD_") (strf::sani("\xBF_") > 3);

    // missing leading byte
    TEST(u" \uFFFD\uFFFD")  (strf::sani("\x80\x80") > 3);
    TEST(u" \uFFFD\uFFFD_") (strf::sani("\x80\x80_") > 4);

    // overlong sequence
    TEST(u" \uFFFD\uFFFD")  (strf::sani("\xC1\xBF") > 3);
    TEST(u" \uFFFD\uFFFD_") (strf::sani("\xC1\xBF_") > 4);

    // overlong sequence
    TEST(u" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xE0\x9F\x80") > 4);
    TEST(u" \uFFFD\uFFFD\uFFFD_") (strf::sani("\xE0\x9F\x80_") > 5);

    // overlong sequence with extra continuation bytes
    TEST(u" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xC1\xBF\x80") > 4);
    TEST(u" \uFFFD\uFFFD\uFFFD_") (strf::sani("\xC1\xBF\x80_") > 5);

    // overlong sequence with extra continuation bytes
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD")  (strf::sani("\xE0\x9F\x80\x80") > 5);
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD_") (strf::sani("\xE0\x9F\x80\x80_") > 6);

    // overlong sequence
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD")  (strf::sani("\xF0\x8F\xBF\xBF" ) > 5);
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD_") (strf::sani("\xF0\x8F\xBF\xBF_" ) > 6);

    // overlong sequence with extra continuation bytes
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD")  (strf::sani("\xF0\x8F\xBF\xBF\x80" ) > 6);
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD_") (strf::sani("\xF0\x8F\xBF\xBF\x80_" ) > 7);

    // missing continuation
    TEST(u" \uFFFD")  (strf::sani("\xF0\x90\xBF" ) > 2);
    TEST(u" \uFFFD_") (strf::sani("\xF0\x90\xBF_" ) > 3);

    // codepoint too big
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD")  (strf::sani("\xF4\xBF\xBF\xBF") > 5);
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD_") (strf::sani("\xF4\xBF\xBF\xBF_") > 6);

    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xF4\x90\x80\x80_") > 6);
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xF5\x80\x80\x80_") > 6);
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xF6\x80\x80\x80_") > 6);
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xF7\x80\x80\x80_") > 6);
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xF8\x80\x80\x80_") > 6);
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xF9\x80\x80\x80_") > 6);
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xFA\x80\x80\x80_") > 6);
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xFB\x80\x80\x80_") > 6);
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xFC\x80\x80\x80_") > 6);
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xFD\x80\x80\x80_") > 6);
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xFE\x80\x80\x80_") > 6);
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xFF\x80\x80\x80_") > 6);

    // codepoint too big with extra continuation bytes
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD")  (strf::sani("\xF5\x90\x80\x80\x80\x80") > 7);
    TEST(u" \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD_") (strf::sani("\xF5\x90\x80\x80\x80\x80_") > 8);

    // missing continuation
    TEST(u" \uFFFD")  (strf::sani("\xC2") > 2);
    TEST(u" \uFFFD_") (strf::sani("\xC2_") > 3);

    TEST(u" \uFFFD")  (strf::sani("\xE0") > 2);
    TEST(u" \uFFFD_") (strf::sani("\xE0_") > 3);

    TEST(u" \uFFFD")  (strf::sani("\xE0\xA0") > 2);
    TEST(u" \uFFFD_") (strf::sani("\xE0\xA0_") > 3);

    TEST(u" \uFFFD")  (strf::sani("\xE1") > 2);
    TEST(u" \uFFFD_") (strf::sani("\xE1_") > 3);

    TEST(u" \uFFFD")  (strf::sani("\xF1") > 2);
    TEST(u" \uFFFD_") (strf::sani("\xF1_") > 3);

    TEST(u" \uFFFD")  (strf::sani("\xF1\x81") > 2);
    TEST(u" \uFFFD_") (strf::sani("\xF1\x81_") > 3);

    TEST(u" \uFFFD")  (strf::sani("\xF1\x81\x81") > 2);
    TEST(u" \uFFFD_") (strf::sani("\xF1\x81\x81_") > 3);

    // surrogate
    TEST(u" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xA0\x80") > 4);
    TEST(u" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xAF\xBF") > 4);
    TEST(u" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xB0\x80") > 4);
    TEST(u" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xBF\xBF") > 4);
    TEST(u" \uFFFD\uFFFD\uFFFD_")  (strf::sani("\xED\xA0\x80_") > 5);
    TEST(u" \uFFFD\uFFFD\uFFFD_")  (strf::sani("\xED\xAF\xBF_") > 5);
    TEST(u" \uFFFD\uFFFD\uFFFD_")  (strf::sani("\xED\xB0\x80_") > 5);
    TEST(u" \uFFFD\uFFFD\uFFFD_")  (strf::sani("\xED\xBF\xBF_") > 5);

    // missing continuation, but could only be a surrogate.
    TEST(u" \uFFFD\uFFFD")  (strf::sani("\xED\xA0") > 3);
    TEST(u" \uFFFD\uFFFD_") (strf::sani("\xED\xBF_") > 4);

    // missing continuation. It could only be a surrogate, but surrogates are allowed
    auto allow_surr = strf::surrogate_policy::lax;
    TEST(u" \uFFFD")  .with(allow_surr) (strf::sani("\xED\xA0") > 2);
    TEST(u" \uFFFD_") .with(allow_surr) (strf::sani("\xED\xBF_") > 3);

    // missing continuation. Now it starts with \xED, but it is not a surrogate
    TEST(u" \uFFFD")  (strf::sani("\xED\x9F") > 2);
    TEST(u" \uFFFD_") (strf::sani("\xED\x9F_") > 3);

    // cover when recycle needs to be called
    TEST_CALLING_RECYCLE_AT<2,2>(u" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xA0\x80") > 4);
    TEST_CALLING_RECYCLE_AT<2>  (u" \uFFFD")              (strf::sani("\xED\xA0\x80") > 4);
}

STRF_TEST_FUNC int error_handler_calls_count = 0 ;
struct dummy_exception {};

STRF_TEST_FUNC void utf8_to_utf16_error_notifier()
{
    strf::invalid_seq_notifier notifier{ [](){++error_handler_calls_count;} };

    ::error_handler_calls_count = 0;
    TEST(u"\uFFFD\uFFFD\uFFFD").with(notifier) (strf::sani("\xED\xA0\x80"));
    TEST_EQ(::error_handler_calls_count, 3);

    ::error_handler_calls_count = 0;
    TEST_CALLING_RECYCLE_AT<1>(u"\uFFFD").with(notifier) (strf::sani("\xED\xA0\x80"));
    TEST_TRUE(::error_handler_calls_count > 0);

#if defined(__cpp_exceptions) && __cpp_exceptions  && ! defined(__CUDACC__)

    // check that an exception can be thrown, i.e,
    // ensure there is no `noexcept` blocking it
    strf::invalid_seq_notifier notifier_that_throws{ [](){ throw dummy_exception{}; } };
    bool thrown = false;
    try {
        char16_t buff[10];
        strf::to(buff) .with(notifier_that_throws) (strf::sani("\xED\xA0\x80"));
    } catch (dummy_exception&) {
        thrown = true;
    } catch(...) {
    }
    TEST_TRUE(thrown);

#endif // defined(__cpp_exceptions)
}

STRF_TEST_FUNC void utf8_to_utf16_find_transcoder()
{
#if ! defined(__CUDACC__)

    using static_transcoder_type = strf::static_transcoder
        <char, char16_t, strf::csid_utf8, strf::csid_utf16>;

    strf::dynamic_charset<char>     dyn_utf8  = strf::utf8_t<char>{}.to_dynamic();
    strf::dynamic_charset<char16_t> dyn_utf16 = strf::utf16_t<char16_t>{}.to_dynamic();
    strf::dynamic_transcoder<char, char16_t> tr = strf::find_transcoder(dyn_utf8, dyn_utf16);

    TEST_TRUE(tr.transcode_func()      == static_transcoder_type::transcode);
    TEST_TRUE(tr.transcode_size_func() == static_transcoder_type::transcode_size);

#endif // defined(__CUDACC__)

    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char, char16_t, strf::csid_utf8, strf::csid_utf16 >
                       , decltype(strf::find_transcoder( strf::utf_t<char>{}
                                                       , strf::utf_t<char16_t>{})) >
                  :: value));
}


} // unnamed namespace


void STRF_TEST_FUNC test_utf8_to_utf16()
{
    utf8_to_utf16_valid_sequences();
    utf8_to_utf16_invalid_sequences();
    utf8_to_utf16_error_notifier();
    utf8_to_utf16_find_transcoder();
}

REGISTER_STRF_TEST(test_utf8_to_utf16);
