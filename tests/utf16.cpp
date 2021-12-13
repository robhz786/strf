//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

namespace {

STRF_TEST_FUNC void utf16_sani_valid_sequences()
{
    TEST(u" ab\u0080\u0800\uD7FF\U00010000\U0010FFFF")
        (strf::sani(u"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF") > 8);

    TEST_TRUNCATING_AT(2, u"ab") (strf::sani(u"ab\uD7FF"));
    TEST_TRUNCATING_AT(3, u"ab") (strf::sani(u"ab\U00010000"));

    TEST_TRUNCATING_AT(3, u"ab\uD7FF")     (strf::sani(u"ab\uD7FF"));
    TEST_TRUNCATING_AT(4, u"ab\U00010000") (strf::sani(u"ab\U00010000"));
    TEST_TRUNCATING_AT(4, u"ab\U0010FFFF") (strf::sani(u"ab\U0010FFFF"));

    TEST_CALLING_RECYCLE_AT(2, u"ab\uD7FF") (strf::sani(u"ab\uD7FF"));
    TEST_CALLING_RECYCLE_AT(3, u"ab\U00010000") (strf::sani(u"ab\U00010000"));
    TEST_CALLING_RECYCLE_AT(3, u"ab\U0010FFFF") (strf::sani(u"ab\U0010FFFF"));

    {
        // when surrogates are allowed
        const char16_t _u16str_D800[] = {u' ', 0xD800, 0};
        const char16_t _u16str_DBFF[] = {u' ', 0xDBFF, 0};
        const char16_t _u16str_DC00[] = {u' ', 0xDC00, 0};
        const char16_t _u16str_DFFF[] = {u' ', 0xDFFF, 0};

        const char16_t _u16str_D800_[] = {u' ', 0xD800, u'_', 0};
        const char16_t _u16str_DBFF_[] = {u' ', 0xDBFF, u'_', 0};
        const char16_t _u16str_DC00_[] = {u' ', 0xDC00, u'_', 0};
        const char16_t _u16str_DFFF_[] = {u' ', 0xDFFF, u'_', 0};

        const char16_t _u16str_DFFF_D800_[] = {u' ', 0xDFFF, 0xD800, u'_', 0};

        const char16_t u16str_D800[] = {0xD800, 0};
        const char16_t u16str_DBFF[] = {0xDBFF, 0};
        const char16_t u16str_DC00[] = {0xDC00, 0};
        const char16_t u16str_DFFF[] = {0xDFFF, 0};

        const char16_t u16str_D800_[] = {0xD800, u'_', 0};
        const char16_t u16str_DBFF_[] = {0xDBFF, u'_', 0};
        const char16_t u16str_DC00_[] = {0xDC00, u'_', 0};
        const char16_t u16str_DFFF_[] = {0xDFFF, u'_', 0};

        const char16_t u16str_DFFF_D800_[] = {0xDFFF, 0xD800, u'_', 0};

        TEST(_u16str_D800) .with(strf::surrogate_policy::lax) (strf::sani(u16str_D800) > 2);
        TEST(_u16str_DBFF) .with(strf::surrogate_policy::lax) (strf::sani(u16str_DBFF) > 2);
        TEST(_u16str_DC00) .with(strf::surrogate_policy::lax) (strf::sani(u16str_DC00) > 2);
        TEST(_u16str_DFFF) .with(strf::surrogate_policy::lax) (strf::sani(u16str_DFFF) > 2);

        TEST(_u16str_D800_) .with(strf::surrogate_policy::lax) (strf::sani(u16str_D800_) > 3);
        TEST(_u16str_DBFF_) .with(strf::surrogate_policy::lax) (strf::sani(u16str_DBFF_) > 3);
        TEST(_u16str_DC00_) .with(strf::surrogate_policy::lax) (strf::sani(u16str_DC00_) > 3);
        TEST(_u16str_DFFF_) .with(strf::surrogate_policy::lax) (strf::sani(u16str_DFFF_) > 3);

        TEST(_u16str_DFFF_D800_) .with(strf::surrogate_policy::lax)
            (strf::sani(u16str_DFFF_D800_) > 4);

        TEST(u" \U00010000") .with(strf::surrogate_policy::lax) (strf::sani(u"\U00010000") > 2);

        TEST_CALLING_RECYCLE_AT(1, _u16str_D800)
            .with(strf::surrogate_policy::lax) (strf::sani(u16str_D800) > 2);
        TEST_TRUNCATING_AT     (2, _u16str_D800)
            .with(strf::surrogate_policy::lax) (strf::sani(u16str_D800) > 2);
        TEST_TRUNCATING_AT     (1, u" ")
            .with(strf::surrogate_policy::lax) (strf::sani(u16str_D800) > 2);
    }

}

STRF_TEST_FUNC void utf16_sani_invalid_sequences()
{
    {
        // high surrogate followed by another high surrogate
        const char16_t str[] = {0xD800, 0xD800, 0};
        TEST(u" \uFFFD\uFFFD") (strf::sani(str) > 3);
    }
    {
        // low surrogate followed by a high surrogate
        const char16_t str[] = {0xDFFF, 0xD800, 0};
        TEST(u" \uFFFD\uFFFD") (strf::sani(str) > 3);
    }
    {
        // a low surrogate
        const char16_t str[] = {0xDFFF, 0};
        TEST(u" \uFFFD") (strf::sani(str) > 2);
    }
    {
        // a high surrogate
        const char16_t str[] = {0xD800, 0};
        TEST(u" \uFFFD") (strf::sani(str) > 2);
    }
    {
        // low surrogate followed by a high surrogate
        const char16_t str[] = {'_', 0xDFFF, 0xD800, '_', 0};
        TEST(u" _\uFFFD\uFFFD_") (strf::sani(str) > 5);
    }
    {
        const char16_t str[] = {'_', 0xDFFF, '_', 0};
        TEST(u" _\uFFFD_") (strf::sani(str) > 4);
    }
    {
        const char16_t str[] = {'_', 0xD800, '_', 0};
        TEST(u" _\uFFFD_") (strf::sani(str) > 4);
        TEST_CALLING_RECYCLE_AT(2, u" _\uFFFD_") (strf::sani(str) > 4);
        TEST_TRUNCATING_AT     (2, u" _")             (strf::sani(str) > 4);
        TEST_TRUNCATING_AT     (4, u" _\uFFFD_")      (strf::sani(str) > 4);
    }
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

STRF_TEST_FUNC void utf16_sani_error_notifier()
{
    const char16_t invalid_input[] = {0xDFFF, 0xD800, 0};
    {
        invalid_seq_counter notifier;
        strf::invalid_seq_notifier_ptr notifier_ptr{&notifier};

        TEST(u"\uFFFD\uFFFD").with(notifier_ptr) (strf::sani(invalid_input));
        TEST_EQ(notifier.notifications_count, 2);

        notifier.notifications_count = 0;
        TEST_TRUNCATING_AT(1, u"\uFFFD").with(notifier_ptr) (strf::sani(invalid_input));
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
            char16_t buff[10];
            strf::to(buff) .with(notifier_ptr) (strf::sani(invalid_input));
        } catch (dummy_exception&) {
            thrown = true;
        } catch(...) {
        }
        TEST_TRUE(thrown);
    }

#endif // __cpp_exceptions
}

STRF_TEST_FUNC void utf16_sani_find_transcoder()
{
#if ! defined(__CUDACC__)

    using static_transcoder_type = strf::static_transcoder
        <char16_t, char16_t, strf::csid_utf16, strf::csid_utf16>;

    strf::dynamic_charset<char16_t> dyn_cs = strf::utf16_t<char16_t>{}.to_dynamic();
    strf::dynamic_transcoder<char16_t, char16_t> tr = strf::find_transcoder(dyn_cs, dyn_cs);

    TEST_TRUE(tr.transcode_func()      == static_transcoder_type::transcode);
    TEST_TRUE(tr.transcode_size_func() == static_transcoder_type::transcode_size);

#endif // defined(__CUDACC__)

    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char16_t, char16_t, strf::csid_utf16, strf::csid_utf16 >
                   , decltype(strf::find_transcoder( strf::utf_t<char16_t>()
                                                   , strf::utf_t<char16_t>())) >
                  :: value));
    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char16_t, char32_t, strf::csid_utf16, strf::csid_utf32 >
                   , decltype(strf::find_transcoder( strf::utf_t<char16_t>()
                                                   , strf::utf_t<char32_t>())) >
                  :: value));
    TEST_TRUE((std::is_same
                   < strf::static_transcoder
                       < char32_t, char16_t,  strf::csid_utf32, strf::csid_utf16 >
                   , decltype(strf::find_transcoder( strf::utf_t<char32_t>()
                                                   , strf::utf_t<char16_t>())) >
                  :: value));
}

template <std::size_t N>
STRF_HD std::size_t utf16_codepoints_robust_count_strict(const char16_t (&str)[N])
{
    return strf::utf16_t<char16_t>::codepoints_robust_count
        (str, N - 1, 100000, strf::surrogate_policy::strict)
        .count;
}

template <std::size_t N>
STRF_HD std::size_t utf16_codepoints_robust_count_lax(const char16_t (&str)[N])
{
    return strf::utf16_t<char16_t>::codepoints_robust_count
        (str, N - 1, 100000, strf::surrogate_policy::lax)
        .count;
}

template <std::size_t N>
STRF_HD std::size_t utf16_codepoints_fast_count(const char16_t (&str)[N])
{
    return strf::utf16_t<char16_t>::codepoints_fast_count(str, N - 1, 100000).count;
}

STRF_HD void utf16_codepoints_count()
{
    {   // test valid input
        TEST_EQ(0, utf16_codepoints_robust_count_strict(u""));
        TEST_EQ(3, utf16_codepoints_robust_count_strict(u"abc"));
        TEST_EQ(1, utf16_codepoints_robust_count_strict(u"\uD7FF"));
        TEST_EQ(1, utf16_codepoints_robust_count_strict(u"\uE000"));
        TEST_EQ(1, utf16_codepoints_robust_count_strict(u"\U0010FFFF"));

        TEST_EQ(1, utf16_codepoints_robust_count_lax(u"\uD7FF"));
        TEST_EQ(1, utf16_codepoints_robust_count_lax(u"\uE000"));
        TEST_EQ(1, utf16_codepoints_robust_count_lax(u"\U0010FFFF"));

        TEST_EQ(0, utf16_codepoints_fast_count(u""));
        TEST_EQ(3, utf16_codepoints_fast_count(u"abc"));
        TEST_EQ(1, utf16_codepoints_fast_count(u"\uD7FF"));
        TEST_EQ(1, utf16_codepoints_fast_count(u"\uE000"));
        TEST_EQ(1, utf16_codepoints_fast_count(u"\U0010FFFF"));
    }
    {   // when surrogates are allowed
        const char16_t u16str_D800[] = {0xD800, 0};
        const char16_t u16str_DBFF[] = {0xDBFF, 0};
        const char16_t u16str_DC00[] = {0xDC00, 0};
        const char16_t u16str_DFFF[] = {0xDFFF, 0};

        TEST_EQ(1, utf16_codepoints_robust_count_lax(u16str_D800));
        TEST_EQ(1, utf16_codepoints_robust_count_lax(u16str_DBFF));
        TEST_EQ(1, utf16_codepoints_robust_count_lax(u16str_DC00));
        TEST_EQ(1, utf16_codepoints_robust_count_lax(u16str_DFFF));
    }
    {   // invalid sequences
        {
            // high surrogate followed by another high surrogate
            const char16_t str[] = {0xD800, 0xD800, 0};
            TEST_EQ(2, utf16_codepoints_robust_count_lax(str));
        }
        {
            // low surrogate followed by a high surrogate
            const char16_t str[] = {0xDFFF, 0xD800, 0};
            TEST_EQ(2, utf16_codepoints_robust_count_lax(str));
        }
        {
            // a low surrogate
            const char16_t str[] = {0xDFFF, 0};
            TEST_EQ(1, utf16_codepoints_robust_count_lax(str));
        }
        {
            // a high surrogate
            const char16_t str[] = {0xD800, 0};
            TEST_EQ(1, utf16_codepoints_robust_count_lax(str));
        }
    }
    {   // when limit is less than or equal to count

        const char16_t str[] = u"a\0\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF";
        const auto str_len = sizeof(str)/2 - 1;
        strf::utf16_t<char16_t> charset;
        constexpr auto strict = strf::surrogate_policy::strict;

        {
            auto r = charset.codepoints_robust_count(str, str_len, 8, strict);
            TEST_EQ(r.pos, str_len);
            TEST_EQ(r.count, 8);
        }
        {
            auto r = charset.codepoints_robust_count(str, str_len, 7, strict);
            TEST_EQ(r.pos, str_len - 2);
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
            TEST_EQ(r.pos, str_len - 2);
            TEST_EQ(r.count, 7);
        }
        {
            auto r = charset.codepoints_fast_count(str, str_len, 0);
            TEST_EQ(r.pos, 0);
            TEST_EQ(r.count, 0);
        }
    }
}

STRF_TEST_FUNC void utf16_miscellaneous()
{
    strf::utf16_t<char16_t> charset;
    {  // cover write_replacement_char(x);
        TEST(u"\uFFFD")                         .tr(u"{10}");
        TEST_CALLING_RECYCLE_AT(2, u"  \uFFFD").tr(u"  {10}");
        TEST_TRUNCATING_AT     (3, u"  \uFFFD").tr(u"  {10}");
        TEST_TRUNCATING_AT     (2, u"  ")      .tr(u"  {10}");
    }

    TEST_EQ(1, charset.validate('a'));
    TEST_EQ(1, charset.validate(0xFFFF));
    TEST_EQ(2, charset.validate(0x10000));
    TEST_EQ(2, charset.validate(0x10FFFF));
    TEST_EQ((std::size_t)-1, charset.validate(0x110000));

    {
        using utf8_to_utf16 = strf::static_transcoder
            <char, char16_t, strf::csid_utf8, strf::csid_utf16>;

        auto tr = charset.find_transcoder_from<char>(strf::csid_utf8);
        TEST_TRUE(tr.transcode_func()      == utf8_to_utf16::transcode);
        TEST_TRUE(tr.transcode_size_func() == utf8_to_utf16::transcode_size);
    }
}


} // unnamed namespace

STRF_TEST_FUNC void test_utf16()
{
    utf16_sani_valid_sequences();
    utf16_sani_invalid_sequences();
    utf16_sani_error_notifier();
    utf16_sani_find_transcoder();
    utf16_codepoints_count();
    utf16_miscellaneous();
}

REGISTER_STRF_TEST(test_utf16);
