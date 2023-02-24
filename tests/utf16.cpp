//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_invalid_sequences.hpp"

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

#define TEST_INVALID_SEQS(INPUT, ...)                                   \
    test_utils::test_invalid_sequences                                  \
        <strf::csid_utf16, strf::csid_utf16, char16_t, char16_t>        \
        ( BOOST_CURRENT_FUNCTION, __FILE__, __LINE__                    \
        , strf::surrogate_policy::strict, (INPUT), __VA_ARGS__ );

#define TEST_INVALID_SEQS_LAX(INPUT, ...)                               \
    test_utils::test_invalid_sequences                                  \
        <strf::csid_utf16, strf::csid_utf16, char16_t, char16_t>        \
        ( BOOST_CURRENT_FUNCTION, __FILE__, __LINE__                    \
        , strf::surrogate_policy::lax, (INPUT), __VA_ARGS__ );

STRF_TEST_FUNC void utf16_sani_invalid_sequences()
{
    const char16_t str_dfff[] = {0xDFFF, 0};
    const char16_t str_d800[] = {0xD800, 0};
    {
        // high surrogate followed by another high surrogate
        const char16_t str[] = {0xD800, 0xD800, 0};
        TEST(u" \uFFFD\uFFFD") (strf::sani(str) > 3);
        TEST_INVALID_SEQS(str, str_d800, str_d800);
    }
    {
        // low surrogate followed by a high surrogate
        const char16_t str[] = {0xDFFF, 0xD800, 0};
        TEST(u" \uFFFD\uFFFD") (strf::sani(str) > 3);
        TEST_INVALID_SEQS(str, str_dfff, str_d800);
    }
    {
        // a low surrogate
        const char16_t str[] = {0xDFFF, 0};
        TEST(u" \uFFFD") (strf::sani(str) > 2);
        TEST_INVALID_SEQS(str, str_dfff);
    }
    {
        // a high surrogate
        const char16_t str[] = {0xD800, 0};
        TEST(u" \uFFFD") (strf::sani(str) > 2);
        TEST_INVALID_SEQS(str, str_d800);
    }
    {
        // low surrogate followed by a high surrogate
        const char16_t str[] = {'_', 0xDFFF, 0xD800, '_', 0};
        TEST(u" _\uFFFD\uFFFD_") (strf::sani(str) > 5);
        TEST_INVALID_SEQS(str, str_dfff, str_d800);
    }
    {
        const char16_t str[] = {'_', 0xDFFF, '_', 0};
        TEST(u" _\uFFFD_") (strf::sani(str) > 4);
        TEST_INVALID_SEQS(str, str_dfff);
    }
    {
        const char16_t str[] = {'_', 0xD800, '_', 0};
        TEST(u" _\uFFFD_") (strf::sani(str) > 4);
        TEST_INVALID_SEQS(str, str_d800);
        TEST_CALLING_RECYCLE_AT(2, u" _\uFFFD_") (strf::sani(str) > 4);
        TEST_TRUNCATING_AT     (2, u" _")             (strf::sani(str) > 4);
        TEST_TRUNCATING_AT     (4, u" _\uFFFD_")      (strf::sani(str) > 4);
    }
}

struct invalid_seq_counter: strf::transcoding_error_notifier {
    void STRF_HD invalid_sequence(int, const char*, const void*, std::ptrdiff_t) override {
        ++ notifications_count;
    }
    std::ptrdiff_t notifications_count = 0;
};

#if defined(__cpp_exceptions) && __cpp_exceptions  && ! defined(__CUDACC__)

struct dummy_exception : std::exception {};
struct notifier_that_throws : strf::transcoding_error_notifier {
    void STRF_HD invalid_sequence(int, const char*, const void*, std::ptrdiff_t) override {
        throw dummy_exception{};
    }
};

#endif // __cpp_exceptions

STRF_TEST_FUNC void utf16_sani_error_notifier()
{
    const char16_t invalid_input[] = {0xDFFF, 0xD800, 0};
    {
        invalid_seq_counter notifier;
        strf::transcoding_error_notifier_ptr notifier_ptr{&notifier};

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
        strf::transcoding_error_notifier_ptr notifier_ptr{&notifier};
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

    const strf::dynamic_charset<char16_t> dyn_cs = strf::utf16_t<char16_t>{}.to_dynamic();
    const strf::dynamic_transcoder<char16_t, char16_t> tr = strf::find_transcoder(dyn_cs, dyn_cs);

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
STRF_HD std::ptrdiff_t utf16_count_codepoints_strict(const char16_t (&str)[N])
{
    return strf::utf16_t<char16_t>::count_codepoints
        (str, str + N - 1, 100000, strf::surrogate_policy::strict)
        .count;
}

template <std::size_t N>
STRF_HD std::ptrdiff_t utf16_count_codepoints_lax(const char16_t (&str)[N])
{
    return strf::utf16_t<char16_t>::count_codepoints
        (str, str + N - 1, 100000, strf::surrogate_policy::lax)
        .count;
}

template <std::size_t N>
STRF_HD std::ptrdiff_t utf16_count_codepoints_fast(const char16_t (&str)[N])
{
    return strf::utf16_t<char16_t>::count_codepoints_fast(str, str + N - 1, 100000).count;
}

STRF_HD void utf16_codepoints_count()
{
    {   // test valid input
        TEST_EQ(0, utf16_count_codepoints_strict(u""));
        TEST_EQ(3, utf16_count_codepoints_strict(u"abc"));
        TEST_EQ(1, utf16_count_codepoints_strict(u"\uD7FF"));
        TEST_EQ(1, utf16_count_codepoints_strict(u"\uE000"));
        TEST_EQ(1, utf16_count_codepoints_strict(u"\U0010FFFF"));

        TEST_EQ(1, utf16_count_codepoints_lax(u"\uD7FF"));
        TEST_EQ(1, utf16_count_codepoints_lax(u"\uE000"));
        TEST_EQ(1, utf16_count_codepoints_lax(u"\U0010FFFF"));

        TEST_EQ(0, utf16_count_codepoints_fast(u""));
        TEST_EQ(3, utf16_count_codepoints_fast(u"abc"));
        TEST_EQ(1, utf16_count_codepoints_fast(u"\uD7FF"));
        TEST_EQ(1, utf16_count_codepoints_fast(u"\uE000"));
        TEST_EQ(1, utf16_count_codepoints_fast(u"\U0010FFFF"));
    }
    {   // when surrogates are allowed
        const char16_t u16str_D800[] = {0xD800, 0};
        const char16_t u16str_DBFF[] = {0xDBFF, 0};
        const char16_t u16str_DC00[] = {0xDC00, 0};
        const char16_t u16str_DFFF[] = {0xDFFF, 0};

        TEST_EQ(1, utf16_count_codepoints_lax(u16str_D800));
        TEST_EQ(1, utf16_count_codepoints_lax(u16str_DBFF));
        TEST_EQ(1, utf16_count_codepoints_lax(u16str_DC00));
        TEST_EQ(1, utf16_count_codepoints_lax(u16str_DFFF));
    }
    {   // invalid sequences
        {
            // high surrogate followed by another high surrogate
            const char16_t str[] = {0xD800, 0xD800, 0};
            TEST_EQ(2, utf16_count_codepoints_lax(str));
        }
        {
            // low surrogate followed by a high surrogate
            const char16_t str[] = {0xDFFF, 0xD800, 0};
            TEST_EQ(2, utf16_count_codepoints_lax(str));
        }
        {
            // a low surrogate
            const char16_t str[] = {0xDFFF, 0};
            TEST_EQ(1, utf16_count_codepoints_lax(str));
        }
        {
            // a high surrogate
            const char16_t str[] = {0xD800, 0};
            TEST_EQ(1, utf16_count_codepoints_lax(str));
        }
    }
    {   // when limit is less than or equal to count

        const char16_t str[] = u"a\0\u0080\u0800\uD7FF\uE000\U00010000\U0010FFFF";
        const auto str_len = sizeof(str)/2 - 1;
        const auto* const str_end = str + str_len;
        const strf::utf16_t<char16_t> charset;
        constexpr auto strict = strf::surrogate_policy::strict;

        {
            auto r = charset.count_codepoints(str, str_end, 8, strict);
            TEST_EQ((const void*)r.ptr, (const void*)str_end);
            TEST_EQ(r.count, 8);
        }
        {
            auto r = charset.count_codepoints(str, str_end, 7, strict);
            TEST_EQ((const void*)r.ptr, (const void*)(str_end - 2));
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
            TEST_EQ((const void*)r.ptr, (const void*)(str_end - 2));
            TEST_EQ(r.count, 7);
        }
        {
            auto r = charset.count_codepoints_fast(str, str_end, 0);
            TEST_EQ((const void*)r.ptr, (const void*)str);
            TEST_EQ(r.count, 0);
        }
    }
}

STRF_TEST_FUNC void utf16_miscellaneous()
{
    const strf::utf16_t<char16_t> charset;
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

REGISTER_STRF_TEST(test_utf16)
