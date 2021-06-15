//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

namespace {

STRF_TEST_FUNC void utf32_to_utf8_valid_sequences()
{
    TEST(u8" ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        (strf::sani(U"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF") > 8);

    TEST("\xF4\x8F\xBF\xBF") (strf::sani(U"\U0010FFFF"));

    TEST_CALLING_RECYCLE_AT<2> (u8"ab") (strf::sani(U"abc"));
    TEST_CALLING_RECYCLE_AT<2> (u8"ab") (strf::sani(U"ab\u0080"));
    TEST_CALLING_RECYCLE_AT<2> (u8"ab") (strf::sani(U"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT<2> (u8"ab") (strf::sani(U"ab\U00010000"));

    TEST_CALLING_RECYCLE_AT<3> (u8"abc")          (strf::sani(U"abc"));
    TEST_CALLING_RECYCLE_AT<4> (u8"ab\u0080")     (strf::sani(U"ab\u0080"));
    TEST_CALLING_RECYCLE_AT<5> (u8"ab\uD7FF")     (strf::sani(U"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT<6> (u8"ab\U00010000") (strf::sani(U"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT<6> (u8"ab\U0010FFFF") (strf::sani(U"ab\U0010FFFF"));

    TEST_CALLING_RECYCLE_AT<2, 1> (u8"abc")          (strf::sani(U"abc"));
    TEST_CALLING_RECYCLE_AT<3, 1> (u8"ab\u0080")     (strf::sani(U"ab\u0080"));
    TEST_CALLING_RECYCLE_AT<4, 1> (u8"ab\uD7FF")     (strf::sani(U"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT<4, 2> (u8"ab\U00010000") (strf::sani(U"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT<5, 1> (u8"ab\U0010FFFF") (strf::sani(U"ab\U0010FFFF"));
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

        TEST(" \xED\xA0\x80") .with(strf::surrogate_policy::lax) (strf::sani(u32str_D800) > 2);
        TEST(" \xED\xAF\xBF") .with(strf::surrogate_policy::lax) (strf::sani(u32str_DBFF) > 2);
        TEST(" \xED\xB0\x80") .with(strf::surrogate_policy::lax) (strf::sani(u32str_DC00) > 2);
        TEST(" \xED\xBF\xBF") .with(strf::surrogate_policy::lax) (strf::sani(u32str_DFFF) > 2);

        TEST(" \xED\xA0\x80_") .with(strf::surrogate_policy::lax) (strf::sani(u32str_D800_) > 3);
        TEST(" \xED\xAF\xBF_") .with(strf::surrogate_policy::lax) (strf::sani(u32str_DBFF_) > 3);
        TEST(" \xED\xB0\x80_") .with(strf::surrogate_policy::lax) (strf::sani(u32str_DC00_) > 3);
        TEST(" \xED\xBF\xBF_") .with(strf::surrogate_policy::lax) (strf::sani(u32str_DFFF_) > 3);

        TEST(" \xED\xBF\xBF\xED\xA0\x80_") .with(strf::surrogate_policy::lax)
            (strf::sani(u32str_DFFF_D800_) > 4);
        TEST(u8" \U00010000") .with(strf::surrogate_policy::lax) (strf::sani(U"\U00010000") > 2);

        TEST_CALLING_RECYCLE_AT<4> (" \xED\xA0\x80")
            .with(strf::surrogate_policy::lax) (strf::sani(u32str_D800) > 2);
        TEST_CALLING_RECYCLE_AT<3> (u8" ")
            .with(strf::surrogate_policy::lax) (strf::sani(u32str_D800) > 2);
    }
}

STRF_TEST_FUNC void utf32_to_utf8_invalid_sequences()
{
    {
        // surrogates
        const char32_t str[] = {0xD800, 0xDFFF, 0};
        TEST(u8" \uFFFD\uFFFD") (strf::sani(str) > 3);

        TEST_CALLING_RECYCLE_AT<6>    (u8" \uFFFD") (strf::sani(str) > 3);
        TEST_CALLING_RECYCLE_AT<3, 4> (u8" \uFFFD\uFFFD") (strf::sani(str) > 3);
        TEST_CALLING_RECYCLE_AT<3>    (u8" ") (strf::sani(str) > 3);
    }
    {   // codepoint too big
        const char32_t str[] = {0x110000, 0};
        TEST(u8" \uFFFD") (strf::sani(str) > 2);
    }
}

STRF_TEST_FUNC int error_handler_calls_count = 0 ;
struct dummy_exception {};

STRF_TEST_FUNC void utf32_to_utf8_error_notifier()
{

    strf::invalid_seq_notifier notifier{ [](){++error_handler_calls_count;} };

    const char32_t invalid_input[] = {0xD800, 0xDFFF, 0};

    ::error_handler_calls_count = 0;
    TEST(u8"\uFFFD\uFFFD").with(notifier) (strf::sani(invalid_input));
    TEST_EQ(::error_handler_calls_count, 2);

    ::error_handler_calls_count = 0;
    TEST_CALLING_RECYCLE_AT<5>(u8"\uFFFD").with(notifier) (strf::sani(invalid_input));
    TEST_TRUE(::error_handler_calls_count > 0);

#if defined(__cpp_exceptions) && __cpp_exceptions  && ! defined(__CUDACC__)

    // check that an exception can be thrown, i.e,
    // ensure there is no `noexcept` blocking it
    strf::invalid_seq_notifier notifier_that_throws{ [](){ throw dummy_exception{}; } };
    bool thrown = false;
    try {
        char buff[16];
        strf::to(buff) .with(notifier_that_throws) (strf::sani(invalid_input));
    } catch (dummy_exception&) {
        thrown = true;
    } catch(...) {
    }
    TEST_TRUE(thrown);

#endif // __cpp_exceptions

    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char32_t, char, strf::csid_utf32, strf::csid_utf8 >
                   , decltype(strf::find_transcoder( strf::utf_t<char32_t>{}
                                                   , strf::utf_t<char>{})) >
                  :: value));
}

STRF_TEST_FUNC void utf32_to_utf8_find_transcoder()
{
#if ! defined(__CUDACC__)

    using static_transcoder_type = strf::static_transcoder
        <char32_t, char, strf::csid_utf32, strf::csid_utf8>;

    strf::dynamic_charset<char32_t> dyn_utf32 = strf::utf32<char32_t>.to_dynamic();
    strf::dynamic_charset<char>     dyn_utf8  = strf::utf8<char>.to_dynamic();
    strf::dynamic_transcoder<char32_t, char> tr = strf::find_transcoder(dyn_utf32, dyn_utf8);

    TEST_TRUE(tr.transcode_func()      == static_transcoder_type::transcode);
    TEST_TRUE(tr.transcode_size_func() == static_transcoder_type::transcode_size);

#endif // defined(__CUDACC__)
}

} // unnamed namespace

void STRF_TEST_FUNC test_utf32_to_utf8()
{
    utf32_to_utf8_valid_sequences();
    utf32_to_utf8_invalid_sequences();
    utf32_to_utf8_error_notifier();
    utf32_to_utf8_find_transcoder();
}

REGISTER_STRF_TEST(test_utf32_to_utf8);
