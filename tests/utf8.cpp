//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

#ifndef __cpp_char8_t
using char8_t = char;
#endif

/* This test file aims to cover the following class template specializations:

// The UTF-8 charset
template <typename CharT>
class static_charset<CharT, strf::csid_utf8>;

// The UTF-8 sanitizer:
template <typename SrcCharT, typename DestCharT>
class static_transcoder<SrcCharT, DestCharT, strf::csid_utf8, strf::csid_utf8>;

*/

namespace {

STRF_TEST_FUNC void utf8_sani_valid_sequences()
{
    TEST(" ") (strf::sani("") > 1);
    TEST(" abc") (strf::sani("abc") > 4);
    TEST(u8"ab\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF")
        (strf::sani(u8"ab\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF") > 8);

    TEST_CALLING_RECYCLE_AT<2,2>(u8"abcd") (strf::sani("abcdef"));

    TEST_CALLING_RECYCLE_AT<3> (u8"ab") (strf::sani(u8"ab\u0080"));
    TEST_CALLING_RECYCLE_AT<4> (u8"ab") (strf::sani(u8"ab\u0800"));
    TEST_CALLING_RECYCLE_AT<4> (u8"ab") (strf::sani(u8"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT<5> (u8"ab") (strf::sani(u8"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT<5> (u8"ab") (strf::sani(u8"ab\U0010FFFF"));

    TEST_CALLING_RECYCLE_AT<4> (u8"ab\u0080")     (strf::sani(u8"ab\u0080"));
    TEST_CALLING_RECYCLE_AT<5> (u8"ab\u0800")     (strf::sani(u8"ab\u0800"));
    TEST_CALLING_RECYCLE_AT<5> (u8"ab\uD7FF")     (strf::sani(u8"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT<6> (u8"ab\U00010000") (strf::sani(u8"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT<6> (u8"ab\U0010FFFF") (strf::sani(u8"ab\U0010FFFF"));

    TEST_CALLING_RECYCLE_AT<3, 1> (u8"ab\u0080")     (strf::sani(u8"ab\u0080"));
    TEST_CALLING_RECYCLE_AT<4, 1> (u8"ab\u0800")     (strf::sani(u8"ab\u0800"));
    TEST_CALLING_RECYCLE_AT<4, 1> (u8"ab\uD7FF")     (strf::sani(u8"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT<5, 2> (u8"ab\U00010000") (strf::sani(u8"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT<5, 1> (u8"ab\U0010FFFF") (strf::sani(u8"ab\U0010FFFF"));

    {
        // when surrogates are allowed
        TEST(" \xED\xA0\x80") .with(strf::surrogate_policy::lax) (strf::sani("\xED\xA0\x80") > 2);
        TEST(" \xED\xAF\xBF") .with(strf::surrogate_policy::lax) (strf::sani("\xED\xAF\xBF") > 2);
        TEST(" \xED\xB0\x80") .with(strf::surrogate_policy::lax) (strf::sani("\xED\xB0\x80") > 2);
        TEST(" \xED\xBF\xBF") .with(strf::surrogate_policy::lax) (strf::sani("\xED\xBF\xBF") > 2);

        TEST_CALLING_RECYCLE_AT<4> (" \xED\xA0\x80")
            .with(strf::surrogate_policy::lax) (strf::sani("\xED\xA0\x80") > 2);
        TEST_CALLING_RECYCLE_AT<3> (u8" ")
            .with(strf::surrogate_policy::lax) (strf::sani("\xED\xA0\x80") > 2);
    }
}

STRF_TEST_FUNC void utf8_sani_invalid_sequences()
{
    // sample from Tabble 3-8 of Unicode standard
    TEST(u8" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xF1\x80\x80\xE1\x80\xC0") > 4);
    TEST(u8" \uFFFD\uFFFD\uFFFD_") (strf::sani("\xF1\x80\x80\xE1\x80\xC0_") > 5);

    // missing leading byte
    TEST(u8" \uFFFD")  (strf::sani("\xBF") > 2);
    TEST(u8" \uFFFD_") (strf::sani("\xBF_") > 3);

    // missing leading byte
    TEST(u8" \uFFFD\uFFFD")  (strf::sani("\x80\x80") > 3);
    TEST(u8" \uFFFD\uFFFD_") (strf::sani("\x80\x80_") > 4);

    // overlong sequence
    TEST(u8" \uFFFD\uFFFD")  (strf::sani("\xC1\xBF") > 3);
    TEST(u8" \uFFFD\uFFFD_") (strf::sani("\xC1\xBF_") > 4);

    // overlong sequence
    TEST(u8" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xE0\x9F\x80") > 4);
    TEST(u8" \uFFFD\uFFFD\uFFFD_") (strf::sani("\xE0\x9F\x80_") > 5);

    // overlong sequence with extra continuation bytes
    TEST(u8" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xC1\xBF\x80") > 4);
    TEST(u8" \uFFFD\uFFFD\uFFFD_") (strf::sani("\xC1\xBF\x80_") > 5);

    // overlong sequence with extra continuation bytes
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD")  (strf::sani("\xE0\x9F\x80\x80") > 5);
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD_") (strf::sani("\xE0\x9F\x80\x80_") > 6);

    // overlong sequence
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD")  (strf::sani("\xF0\x8F\xBF\xBF" ) > 5);
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD_") (strf::sani("\xF0\x8F\xBF\xBF_" ) > 6);

    // overlong sequence with extra continuation bytes
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD")  (strf::sani("\xF0\x8F\xBF\xBF\x80" ) > 6);
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD_") (strf::sani("\xF0\x8F\xBF\xBF\x80_" ) > 7);


    // codepoint too big
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD")  (strf::sani("\xF4\xBF\xBF\xBF") > 5);
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD_") (strf::sani("\xF4\xBF\xBF\xBF_") > 6);

    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xF4\x90\x80\x80_") > 6);
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xF5\x80\x80\x80_") > 6);
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xF6\x80\x80\x80_") > 6);
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xF7\x80\x80\x80_") > 6);
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xF8\x80\x80\x80_") > 6);
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xF9\x80\x80\x80_") > 6);
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xFA\x80\x80\x80_") > 6);
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xFB\x80\x80\x80_") > 6);
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xFC\x80\x80\x80_") > 6);
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xFD\x80\x80\x80_") > 6);
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xFE\x80\x80\x80_") > 6);
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD_")  (strf::sani("\xFF\x80\x80\x80_") > 6);
    TEST(u8"  \uFFFD\uFFFD\uFFFD_")        (strf::sani("\xFF\x80\x80_") > 6);

    // codepoint too big with extra continuation bytes
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD")  (strf::sani("\xF5\x90\x80\x80\x80\x80") > 7);
    TEST(u8" \uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD_") (strf::sani("\xF5\x90\x80\x80\x80\x80_") > 8);

    // missing continuation byte(s)
    TEST(u8" \uFFFD")  (strf::sani("\xC2") > 2);
    TEST(u8" \uFFFD_") (strf::sani("\xC2_") > 3);

    TEST(u8" \uFFFD")  (strf::sani("\xE0") > 2);
    TEST(u8" \uFFFD_") (strf::sani("\xE0_") > 3);

    TEST(u8" \uFFFD")  (strf::sani("\xE0\xA0") > 2);
    TEST(u8" \uFFFD_") (strf::sani("\xE0\xA0_") > 3);

    TEST(u8" \uFFFD")  (strf::sani("\xE1") > 2);
    TEST(u8" \uFFFD_") (strf::sani("\xE1_") > 3);

    TEST(u8" \uFFFD")  (strf::sani("\xF0\x90\xBF" ) > 2);
    TEST(u8" \uFFFD_") (strf::sani("\xF0\x90\xBF_" ) > 3);

    TEST(u8" \uFFFD")  (strf::sani("\xF1") > 2);
    TEST(u8" \uFFFD_") (strf::sani("\xF1_") > 3);

    TEST(u8" \uFFFD")  (strf::sani("\xF1\x81") > 2);
    TEST(u8" \uFFFD_") (strf::sani("\xF1\x81_") > 3);

    TEST(u8" \uFFFD")  (strf::sani("\xF1\x81\x81") > 2);
    TEST(u8" \uFFFD_") (strf::sani("\xF1\x81\x81_") > 3);

    // surrogate
    TEST(u8" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xA0\x80") > 4);
    TEST(u8" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xAF\xBF") > 4);
    TEST(u8" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xB0\x80") > 4);
    TEST(u8" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xBF\xBF") > 4);
    TEST(u8" \uFFFD\uFFFD\uFFFD_")  (strf::sani("\xED\xA0\x80_") > 5);
    TEST(u8" \uFFFD\uFFFD\uFFFD_")  (strf::sani("\xED\xAF\xBF_") > 5);
    TEST(u8" \uFFFD\uFFFD\uFFFD_")  (strf::sani("\xED\xB0\x80_") > 5);
    TEST(u8" \uFFFD\uFFFD\uFFFD_")  (strf::sani("\xED\xBF\xBF_") > 5);

    // missing continuation, but could only be a surrogate.
    TEST(u8" \uFFFD\uFFFD")  (strf::sani("\xED\xA0") > 3);
    TEST(u8" \uFFFD\uFFFD_") (strf::sani("\xED\xBF_") > 4);

    // missing continuation. It could only be a surrogate, but surrogates are allowed
    auto allow_surr = strf::surrogate_policy::lax;
    TEST(u8" \uFFFD")  .with(allow_surr) (strf::sani("\xED\xA0") > 2);
    TEST(u8" \uFFFD_") .with(allow_surr) (strf::sani("\xED\xBF_") > 3);

    // missing continuation. Now it starts with \xED, but it is not a surrogate
    TEST(u8" \uFFFD")  (strf::sani("\xED\x9F") > 2);
    TEST(u8" \uFFFD_") (strf::sani("\xED\x9F_") > 3);

    // cover when recycle needs to be called
    TEST_CALLING_RECYCLE_AT<3,7>(u8" \uFFFD\uFFFD\uFFFD")  (strf::sani("\xED\xA0\x80") > 4);
    TEST_CALLING_RECYCLE_AT<4>  (u8" \uFFFD")              (strf::sani("\xED\xA0\x80") > 4);
}

STRF_TEST_FUNC int error_handler_calls_count = 0 ;
struct dummy_exception {};

STRF_TEST_FUNC void utf8_sani_error_notifier()
{
    strf::invalid_seq_notifier notifier{ [](){++error_handler_calls_count;} };

    ::error_handler_calls_count = 0;
    TEST(u8"\uFFFD\uFFFD\uFFFD").with(notifier) (strf::sani("\xED\xA0\x80"));
    TEST_EQ(::error_handler_calls_count, 3);

    ::error_handler_calls_count = 0;
    TEST_CALLING_RECYCLE_AT<3>(u8"\uFFFD").with(notifier) (strf::sani("\xED\xA0\x80"));
    TEST_TRUE(::error_handler_calls_count > 0);

#if defined(__cpp_exceptions) && __cpp_exceptions  && ! defined(__CUDACC__)

    // check that an exception can be thrown, i.e,
    // ensure there is no `noexcept` blocking it
    strf::invalid_seq_notifier notifier_that_throws{ [](){ throw dummy_exception{}; } };
    bool thrown = false;
    try {
        char buff[40];
        strf::to(buff) .with(notifier_that_throws) (strf::sani("\xED\xA0\x80"));
    } catch (dummy_exception&) {
        thrown = true;
    } catch(...) {
    }
    TEST_TRUE(thrown);

#endif // defined(__cpp_exceptions)
}

STRF_TEST_FUNC void utf8_sani_find_transcoder()
{
#if ! defined(__CUDACC__)

    using static_transcoder_type = strf::static_transcoder
        <char, char, strf::csid_utf8, strf::csid_utf8>;

    strf::dynamic_charset<char> dyn_cs  = strf::utf8_t<char>{}.to_dynamic();
    strf::dynamic_transcoder<char, char> tr = strf::find_transcoder(dyn_cs, dyn_cs);

    TEST_TRUE(tr.transcode_func()      == static_transcoder_type::transcode);
    TEST_TRUE(tr.transcode_size_func() == static_transcoder_type::transcode_size);

#endif // defined(__CUDACC__)

    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char, char, strf::csid_utf8, strf::csid_utf8 >
                   , decltype(strf::find_transcoder( strf::utf_t<char>{}
                                                   , strf::utf_t<char>{})) >
                  :: value));
}

template <typename CharT, std::size_t N>
STRF_HD std::size_t utf8_codepoints_robust_count_strict(const CharT (&str)[N])
{
    return strf::utf8_t<CharT>::codepoints_robust_count
        (str, N - 1, 100000, strf::surrogate_policy::strict)
        .count;
}

template <typename CharT, std::size_t N>
STRF_HD std::size_t utf8_codepoints_robust_count_lax(const CharT (&str)[N])
{
    return strf::utf8_t<CharT>::codepoints_robust_count
        (str, N - 1, 100000, strf::surrogate_policy::lax)
        .count;
}

template <typename CharT, std::size_t N>
STRF_HD std::size_t utf8_codepoints_fast_count(const CharT (&str)[N])
{
    return strf::utf8_t<CharT>::codepoints_fast_count(str, N - 1, 100000).count;
}

STRF_HD void utf8_codepoints_count()
{
    {   // test valid input
        TEST_EQ(0, utf8_codepoints_robust_count_strict(""));
        TEST_EQ(3, utf8_codepoints_robust_count_strict(u8"ab\u0080"));
        TEST_EQ(3, utf8_codepoints_robust_count_strict(u8"ab\u0800"));
        TEST_EQ(3, utf8_codepoints_robust_count_strict(u8"ab\uD7FF"));
        TEST_EQ(3, utf8_codepoints_robust_count_strict(u8"ab\uE000"));
        TEST_EQ(3, utf8_codepoints_robust_count_strict(u8"ab\U00010000"));
        TEST_EQ(3, utf8_codepoints_robust_count_strict(u8"ab\U0010FFFF"));

        TEST_EQ(0, utf8_codepoints_robust_count_lax(u8""));
        TEST_EQ(3, utf8_codepoints_robust_count_lax(u8"ab\u0080"));
        TEST_EQ(3, utf8_codepoints_robust_count_lax(u8"ab\u0800"));
        TEST_EQ(3, utf8_codepoints_robust_count_lax(u8"ab\uD7FF"));
        TEST_EQ(3, utf8_codepoints_robust_count_lax(u8"ab\uE000"));
        TEST_EQ(3, utf8_codepoints_robust_count_lax(u8"ab\U00010000"));
        TEST_EQ(3, utf8_codepoints_robust_count_lax(u8"ab\U0010FFFF"));

        TEST_EQ(0, utf8_codepoints_fast_count(u8""));
        TEST_EQ(3, utf8_codepoints_fast_count(u8"ab\u0080"));
        TEST_EQ(3, utf8_codepoints_fast_count(u8"ab\u0800"));
        TEST_EQ(3, utf8_codepoints_fast_count(u8"ab\uD7FF"));
        TEST_EQ(3, utf8_codepoints_fast_count(u8"ab\uE000"));
        TEST_EQ(3, utf8_codepoints_fast_count(u8"ab\U00010000"));
        TEST_EQ(3, utf8_codepoints_fast_count(u8"ab\U0010FFFF"));
    }
    {   // when surrogates are allowed
        TEST_EQ(1, utf8_codepoints_robust_count_lax("\xED\xA0\x80"));
        TEST_EQ(1, utf8_codepoints_robust_count_lax("\xED\xAF\xBF"));
        TEST_EQ(1, utf8_codepoints_robust_count_lax("\xED\xB0\x80"));
        TEST_EQ(1, utf8_codepoints_robust_count_lax("\xED\xBF\xBF"));
    }
    {   // test invalid sequences
        TEST_EQ(3, utf8_codepoints_robust_count_strict("\xF1\x80\x80\xE1\x80\xC0"));
        TEST_EQ(1, utf8_codepoints_robust_count_strict("\xBF"));
        TEST_EQ(2, utf8_codepoints_robust_count_strict("\x80\x80"));
        TEST_EQ(2, utf8_codepoints_robust_count_strict("\xC1\xBF"));
        TEST_EQ(3, utf8_codepoints_robust_count_strict("\xE0\x9F\x80"));
        TEST_EQ(3, utf8_codepoints_robust_count_strict("\xC1\xBF\x80"));
        TEST_EQ(4, utf8_codepoints_robust_count_strict("\xE0\x9F\x80\x80"));
        TEST_EQ(4, utf8_codepoints_robust_count_strict("\xF0\x8F\xBF\xBF"));
        TEST_EQ(5, utf8_codepoints_robust_count_strict("\xF0\x8F\xBF\xBF\x80"));
        TEST_EQ(1, utf8_codepoints_robust_count_strict("\xF0\x90\xBF"));
        TEST_EQ(4, utf8_codepoints_robust_count_strict("\xF4\xBF\xBF\xBF"));
        TEST_EQ(6, utf8_codepoints_robust_count_strict("\xF5\x90\x80\x80\x80\x80"));
        TEST_EQ(1, utf8_codepoints_robust_count_strict("\xC2"));
        TEST_EQ(1, utf8_codepoints_robust_count_strict("\xE0"));
        TEST_EQ(1, utf8_codepoints_robust_count_strict("\xE0\xA0"));
        TEST_EQ(1, utf8_codepoints_robust_count_strict("\xF"));
        TEST_EQ(1, utf8_codepoints_robust_count_strict("\xF1\x81"));
        TEST_EQ(1, utf8_codepoints_robust_count_strict("\xF1\x81\x81"));
        TEST_EQ(3, utf8_codepoints_robust_count_strict("\xED\xA0\x80"));
        TEST_EQ(3, utf8_codepoints_robust_count_strict("\xED\xAF\xBF"));
        TEST_EQ(3, utf8_codepoints_robust_count_strict("\xED\xB0\x80"));
        TEST_EQ(3, utf8_codepoints_robust_count_strict("\xED\xBF\xBF"));
        TEST_EQ(2, utf8_codepoints_robust_count_strict("\xED\xA0"));
        TEST_EQ(1, utf8_codepoints_robust_count_lax("\xED\xA0"));
        TEST_EQ(1, utf8_codepoints_robust_count_strict("\xED\x9F"));

        TEST_EQ(4, utf8_codepoints_robust_count_strict("\xF4\x90\x80\x80"));
        TEST_EQ(4, utf8_codepoints_robust_count_strict("\xF5\x80\x80\x80"));
        TEST_EQ(4, utf8_codepoints_robust_count_strict("\xF6\x80\x80\x80"));
        TEST_EQ(4, utf8_codepoints_robust_count_strict("\xF7\x80\x80\x80"));
        TEST_EQ(4, utf8_codepoints_robust_count_strict("\xF8\x80\x80\x80"));
        TEST_EQ(4, utf8_codepoints_robust_count_strict("\xF9\x80\x80\x80"));
        TEST_EQ(4, utf8_codepoints_robust_count_strict("\xFA\x80\x80\x80"));
        TEST_EQ(4, utf8_codepoints_robust_count_strict("\xFB\x80\x80\x80"));
        TEST_EQ(4, utf8_codepoints_robust_count_strict("\xFC\x80\x80\x80"));
        TEST_EQ(4, utf8_codepoints_robust_count_strict("\xFD\x80\x80\x80"));
        TEST_EQ(4, utf8_codepoints_robust_count_strict("\xFE\x80\x80\x80"));
        TEST_EQ(4, utf8_codepoints_robust_count_strict("\xFF\x80\x80\x80"));
        TEST_EQ(3, utf8_codepoints_robust_count_strict("\xF9\x80\x80"));
        TEST_EQ(2, utf8_codepoints_robust_count_strict("\xF9\x80"));
        TEST_EQ(1, utf8_codepoints_robust_count_strict("\xF9"));
    }

    constexpr auto strict = strf::surrogate_policy::strict;

    {   // when limit is less than or equal to count

        const char8_t str[] = u8"a\0\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF";
        const auto str_len = sizeof(str) - 1;
        strf::utf8_t<char8_t> charset;

        {
            auto r = charset.codepoints_robust_count(str, str_len, 8, strict);
            TEST_EQ(r.pos, str_len);
            TEST_EQ(r.count, 8);
        }
        {
            auto r = charset.codepoints_robust_count(str, str_len, 7, strict);
            TEST_EQ(r.pos, str_len - 4);
            TEST_EQ(r.count, 7);
        }
        {
            auto r = charset.codepoints_robust_count(str, str_len, 0, strict);
            TEST_EQ(r.pos, 0);
            TEST_EQ(r.count, 0);
        }
        {
            auto r = charset.codepoints_fast_count(str, str_len, 8);
            TEST_EQ(r.pos, str_len);
            TEST_EQ(r.count, 8);
        }
        {
            auto r = charset.codepoints_fast_count(str, str_len, 7);
            TEST_EQ(r.pos, str_len - 4);
            TEST_EQ(r.count, 7);
        }
        {
            auto r = charset.codepoints_fast_count(str, str_len, 0);
            TEST_EQ(r.pos, 0);
            TEST_EQ(r.count, 0);
        }
    }
}

STRF_TEST_FUNC void utf8_miscellaneous()
{
    {  // cover write_replacement_char(x);
        TEST(u8"\uFFFD")                         .tr(u8"{10}");
        TEST_CALLING_RECYCLE_AT<2,1> (u8"\uFFFD").tr(u8"{10}");
        TEST_CALLING_RECYCLE_AT<2>   (u8"")      .tr(u8"{10}");
    }
    strf::utf_t<char> charset;
    {
        using utf16_to_utf8 = strf::static_transcoder
            <char16_t, char, strf::csid_utf16, strf::csid_utf8>;

        auto tr = charset.find_transcoder_from<char16_t>(strf::csid_utf16);
        TEST_TRUE(tr.transcode_func()      == utf16_to_utf8::transcode);
        TEST_TRUE(tr.transcode_size_func() == utf16_to_utf8::transcode_size);
    }
}

} // unnamed namespace

STRF_TEST_FUNC void test_utf8()
{
    utf8_sani_valid_sequences();
    utf8_sani_invalid_sequences();
    utf8_sani_error_notifier();
    utf8_sani_find_transcoder();
    utf8_codepoints_count();
    utf8_miscellaneous();
}

REGISTER_STRF_TEST(test_utf8);
