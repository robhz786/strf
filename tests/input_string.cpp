//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

#if ! defined(__cpp_char8_t)
#   if __GNUC__ >= 11
#       pragma GCC diagnostic ignored "-Wc++20-compat"
#   endif
using char8_t = char;
#endif

namespace {

STRF_TEST_FUNC void test_basic()
{
    {
        TEST("abc")      ( strf::fmt("abc") );
        TEST("   abc")   ( strf::right("abc", 6) );
        TEST("abc...")   ( strf::left    ("abc", 6, '.') );
        TEST("...abc")   ( strf::right   ("abc", 6, '.') );
        TEST(".abc..")   ( strf::center  ("abc", 6, '.') );
        TEST("     abc")   ( strf::join_right(8)("abc") );
        TEST("...abc~~")   ( strf::join_right(8, '.')(strf::left("abc", 5, U'~')) );
        TEST(".....abc")   ( strf::join_right(8, '.')(strf::left("abc", 3, U'~')) );
        TEST(".....abc")   ( strf::join_right(8, '.')(strf::left("abc", 2, U'~')) );

        TEST("   abcdefghi") ( strf::right("", 3), strf::right("abc", 3), strf::left("def", 3), strf::center("ghi", 3) );
        TEST("  abcdefghi")  ( strf::right("", 2), strf::right("abc", 2), strf::left("def", 2), strf::center("ghi", 2) );
        TEST("abcdefghi")    ( strf::right("", 0), strf::right("abc", 0), strf::left("def", 0), strf::center("ghi", 0) );

        // string precision
        TEST("abc")      ( strf::fmt("abcd").p(3) );
        TEST("   abc")   ( strf::right("abcd", 6).p(3) );
        TEST("abc...")   ( strf::left    ("abcd", 6, '.').p(3) );
        TEST("...abc")   ( strf::right   ("abcd", 6, '.').p(3) );
        TEST(".abc..")   ( strf::center  ("abcd", 6, '.').p(3) );
        TEST("     abc")   ( strf::join_right(8)( strf::fmt("abcd").p(3) ) );
        TEST("...abc~~")   ( strf::join_right(8, '.')(strf::left("abcd", 5, U'~').p(3)) );
        TEST(".....abc")   ( strf::join_right(8, '.')(strf::left("abcd", 3, U'~').p(3)) );
        TEST(".....abc")   ( strf::join_right(8, '.')(strf::left("abcd", 2, U'~').p(3)) );
    }
    {
        const wchar_t abc[] = L"abc";
        const wchar_t def[] = L"def";
        const wchar_t ghi[] = L"ghi";
        TEST(L"abc")      ( abc );
        TEST(L"   abc")   ( strf::right(abc, 6) );
        TEST(L"abc...")   ( strf::left    (abc, 6, '.') );
        TEST(L"...abc")   ( strf::right   (abc, 6, '.') );
        TEST(L".abc..")   ( strf::center  (abc, 6, '.') );
        TEST(L"     abc")   ( strf::join_right(8)(abc) );
        TEST(L"...abc~~")   ( strf::join_right(8, '.')(strf::left(abc, 5, U'~')) );
        TEST(L".....abc")   ( strf::join_right(8, '.')(strf::left(abc, 3, U'~')) );
        TEST(L".....abc")   ( strf::join_right(8, '.')(strf::left(abc, 2, U'~')) );

        TEST(L"   abcdefghi") ( strf::right(L"", 3), strf::right(abc, 3), strf::left(def, 3), strf::center(ghi, 3) );
        TEST(L"  abcdefghi")  ( strf::right(L"", 2), strf::right(abc, 2), strf::left(def, 2), strf::center(ghi, 2) );
        TEST(L"abcdefghi")    ( strf::right(L"", 0), strf::right(abc, 0), strf::left(def, 0), strf::center(ghi, 0) );
    }

#if ! defined(STRF_FREESTANDING)

    {
        std::string abc{ "abc" };

        TEST("abc")     ( abc );
        TEST("   abc")  ( strf::right(abc, 6) );
    }

    {
        std::wstring abc{ L"abc" };

        TEST(L"abc")     ( abc );
        TEST(L"   abc")  ( strf::right(abc, 6) );
    }

#endif // ! defined(STRF_FREESTANDING)

#if defined(STRF_HAS_STD_STRING_VIEW)

    {
        std::string_view abc{"abcdef", 3};

        TEST("abc")    ( abc );
        TEST("   abc") ( strf::right(abc, 6) );
    }

#endif // defined(STRF_HAS_STD_STRING_VIEW)
}

struct transcoding_errors_counter: strf::transcoding_error_notifier {
    void STRF_HD invalid_sequence(int, const char*, const void*, std::ptrdiff_t) override {
        ++ invalid_sequences;
    }
    void STRF_HD unsupported_codepoint(const char*, unsigned) override {
        ++ unsupported_codepoints;
    }

    std::ptrdiff_t unsupported_codepoints = 0;
    std::ptrdiff_t invalid_sequences = 0;
};

STRF_TEST_FUNC void sanitize()
{
    TEST("---\xEF\xBF\xBD---")    (strf::sani("---\xFF---"));
    TEST("   ---\xEF\xBF\xBD---") (strf::sani("---\xFF---") > 10);
    TEST("---\xEF\xBF\xBD---")    (strf::sani("---\xFF---", strf::utf_t<char>{}));
    TEST("   ---\xEF\xBF\xBD---") (strf::sani("---\xFF---", strf::utf_t<char>{}) > 10);

    TEST("---\xEF\xBF\xBD-")      (strf::sani("---\xFF---").p(5) );
    TEST("     ---\xEF\xBF\xBD-") (strf::sani("---\xFF---").p(5) > 10);
    TEST("---\xEF\xBF\xBD-")      (strf::sani("---\xFF---", strf::utf_t<char>{}).p(5));
    TEST("     ---\xEF\xBF\xBD-") (strf::sani("---\xFF---", strf::utf_t<char>{}).p(5) > 10);


    TEST("---?---")
        ( strf::iso_8859_3_t<char>{}
        , strf::sani("---\xA5---", strf::iso_8859_3_t<char>{}) );
    TEST("  ---?---")
        ( strf::iso_8859_3_t<char>{}
        , strf::sani("---\xA5---", strf::iso_8859_3_t<char>{}) > 9 );

    // todo

}

STRF_TEST_FUNC void transcode()
{
    // trancoding UTF-8 to UTF-16

    {
        // "\xBF" is an invali sequence
        // "\xF0\x9F\x91\x88" encodes full-width codepoint
        const char* input = "\xBF_\xF0\x9F\x91\x88";

        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};

        TEST(u"\uFFFD_\U0001F448")      (notifier, strf::transcode(input) );
        TEST(u"   \uFFFD_\U0001F448")   (notifier, strf::transcode(input) > 7 );
        TEST(u"\uFFFD_\U0001F448...")   (notifier, strf::transcode(input).fill('.') < 7 );
        TEST(u"...\uFFFD_\U0001F448")   (notifier, strf::transcode(input).fill('.') > 7 );
        TEST(u".\uFFFD_\U0001F448..")   (notifier, strf::transcode(input).fill('.') ^ 7 );

        TEST(u"   \uFFFD_\U0001F448")    (notifier, strf::join_right(7)(strf::transcode(input)) );
        TEST(u"...\uFFFD_\U0001F448~~~") (notifier, strf::join_right(10, '.')(strf::transcode(input).fill('~') < 7) );
        TEST(u"......\uFFFD_\U0001F448") (notifier, strf::join_right(10, '.')(strf::transcode(input).fill('~') < 4) );
        TEST(u"......\uFFFD_\U0001F448") (notifier, strf::join_right(10, '.')(strf::transcode(input).fill('~') < 3) );

        TEST_EQ(counter.invalid_sequences, 9);
    }
    {
        // "\xBF" is an invali sequence
        // "\xF0\x9F\x91\x88" encodes full-width codepoint
        const char* input = "\xBF_\xF0\x9F\x91\x88___";

        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};

        TEST(u"\uFFFD_\U0001F448")      (notifier, strf::transcode(input).p(4) );
        TEST(u"   \uFFFD_\U0001F448")   (notifier, strf::transcode(input).p(4) > 7 );
        TEST(u"\uFFFD_\U0001F448...")   (notifier, strf::transcode(input).p(4).fill('.') < 7 );
        TEST(u"...\uFFFD_\U0001F448")   (notifier, strf::transcode(input).p(4).fill('.') > 7 );
        TEST(u".\uFFFD_\U0001F448..")   (notifier, strf::transcode(input).p(4).fill('.') ^ 7 );

        TEST(u"   \uFFFD_\U0001F448")    (notifier, strf::join_right(7)(strf::transcode(input).p(4)) );
        TEST(u"...\uFFFD_\U0001F448~~~") (notifier, strf::join_right(10, '.')(strf::transcode(input).p(4).fill('~') < 7) );
        TEST(u"......\uFFFD_\U0001F448") (notifier, strf::join_right(10, '.')(strf::transcode(input).p(4).fill('~') < 4) );
        TEST(u"......\uFFFD_\U0001F448") (notifier, strf::join_right(10, '.')(strf::transcode(input).p(4).fill('~') < 3) );

        TEST_EQ(counter.invalid_sequences, 9);
    }
}

STRF_TEST_FUNC void unsafe_transcode()
{
    // test whether the transcode_error_notifier_c facet is applied

    {
        // trancoding UTF-32 to ISO-8859-3
        // \u02D9 is supported by ISO-8859-3, and its encoded as \xFF
        // \uAAAA is not
        const char32_t* input = U"\u02D9_\uAAAA";

        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};

        constexpr auto charset = strf::iso_8859_3_t<char>();
        TEST("\xFF_?")      (charset, notifier, strf::unsafe_transcode(input));
        TEST("   \xFF_?")   (charset, notifier, strf::unsafe_transcode(input) > 6 );
        TEST("\xFF_?...")   (charset, notifier, strf::unsafe_transcode(input).fill('.')<6 );
        TEST("...\xFF_?")   (charset, notifier, strf::unsafe_transcode(input).fill('.')>6 );
        TEST(".\xFF_?..")   (charset, notifier, strf::unsafe_transcode(input).fill('.')^6 );

        TEST("   \xFF_?")    (charset, notifier, strf::join_right(6)(strf::unsafe_transcode(input)) );
        TEST("...\xFF_?~~~") (charset, notifier, strf::join_right(9, '.')(strf::unsafe_transcode(input).fill('~')<6) );
        TEST("......\xFF_?") (charset, notifier, strf::join_right(9, '.')(strf::unsafe_transcode(input).fill('~')<3) );
        TEST("......\xFF_?") (charset, notifier, strf::join_right(9, '.')(strf::unsafe_transcode(input).fill('~')<2) );

        TEST_EQ(counter.unsupported_codepoints, 9);
    }
    {
        // trancoding UTF-32 to ISO-8859-3
        // \u02D9 is supported by ISO-8859-3, and its encoded as \xFF
        // \uAAAA is not
        const char32_t* input = U"\u02D9_\uAAAA___";

        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};

        constexpr auto charset = strf::iso_8859_3_t<char>();
        TEST("\xFF_?")      (charset, notifier, strf::unsafe_transcode(input).p(3));
        TEST("   \xFF_?")   (charset, notifier, strf::unsafe_transcode(input).p(3) > 6 );
        TEST("\xFF_?...")   (charset, notifier, strf::unsafe_transcode(input).p(3).fill('.')<6 );
        TEST("...\xFF_?")   (charset, notifier, strf::unsafe_transcode(input).p(3).fill('.')>6 );
        TEST(".\xFF_?..")   (charset, notifier, strf::unsafe_transcode(input).p(3).fill('.')^6 );

        TEST("   \xFF_?")    (charset, notifier, strf::join_right(6)(strf::unsafe_transcode(input).p(3)) );
        TEST("...\xFF_?~~~") (charset, notifier, strf::join_right(9, '.')(strf::unsafe_transcode(input).p(3).fill('~')<6) );
        TEST("......\xFF_?") (charset, notifier, strf::join_right(9, '.')(strf::unsafe_transcode(input).p(3).fill('~')<3) );
        TEST("......\xFF_?") (charset, notifier, strf::join_right(9, '.')(strf::unsafe_transcode(input).p(3).fill('~')<2) );

        TEST_EQ(counter.unsupported_codepoints, 9);
    }
}

STRF_TEST_FUNC void decode_encode()
{
    constexpr auto src_charset = strf::iso_8859_7_t<char>();
    constexpr auto dst_charset = strf::windows_1252_t<char>();

    {
        // - Codepoint 0x20AC is encode as "\xA4" in ISO-8859-16 and as "\x80" in WINDOWS-1252
        // - In ISO-8859-7, "\xC3" maps to codepoint not supported by WINDOWS-1252
        // - In ISO-8859-7, "\xD2" is an invalid byte
        const char* input = "\xA4_\xC3_\xD2";

        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};
        TEST("\x80_?_?")      (dst_charset, notifier, strf::transcode(input, src_charset));
        TEST("   \x80_?_?")   (dst_charset, notifier, strf::transcode(input, src_charset) > 8);
        TEST("\x80_?_?...")   (dst_charset, notifier, strf::transcode(input, src_charset).fill('.')<8 );
        TEST("...\x80_?_?")   (dst_charset, notifier, strf::transcode(input, src_charset).fill('.')>8 );
        TEST(".\x80_?_?..")   (dst_charset, notifier, strf::transcode(input, src_charset).fill('.')^8 );

        TEST("   \x80_?_?")    (dst_charset, notifier, strf::join_right(8)(strf::transcode(input, src_charset)) );
        TEST("...\x80_?_?~~~") (dst_charset, notifier, strf::join_right(11, '.') (strf::transcode(input, src_charset).fill('~')<8) );
        TEST("......\x80_?_?") (dst_charset, notifier, strf::join_right(11, '.') (strf::transcode(input, src_charset).fill('~')<5) );
        TEST("......\x80_?_?") (dst_charset, notifier, strf::join_right(11, '.') (strf::transcode(input, src_charset).fill('~')<4) );

        TEST_EQ(counter.unsupported_codepoints, 9);
        TEST_EQ(counter.invalid_sequences, 9);
    }
    {
        const char* input = "\xA4_\xC3_\xD2___";

        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};
        TEST("\x80_?_?")      (dst_charset, notifier, strf::transcode(input, src_charset).p(5));
        TEST("   \x80_?_?")   (dst_charset, notifier, strf::transcode(input, src_charset).p(5) > 8);
        TEST("\x80_?_?...")   (dst_charset, notifier, strf::transcode(input, src_charset).p(5).fill('.')<8 );
        TEST("...\x80_?_?")   (dst_charset, notifier, strf::transcode(input, src_charset).p(5).fill('.')>8 );
        TEST(".\x80_?_?..")   (dst_charset, notifier, strf::transcode(input, src_charset).p(5).fill('.')^8 );

        TEST("   \x80_?_?")    (dst_charset, notifier, strf::join_right(8)(strf::transcode(input, src_charset).p(5)) );
        TEST("...\x80_?_?~~~") (dst_charset, notifier, strf::join_right(11, '.') (strf::transcode(input, src_charset).p(5).fill('~')<8) );
        TEST("......\x80_?_?") (dst_charset, notifier, strf::join_right(11, '.') (strf::transcode(input, src_charset).p(5).fill('~')<5) );
        TEST("......\x80_?_?") (dst_charset, notifier, strf::join_right(11, '.') (strf::transcode(input, src_charset).p(5).fill('~')<4) );

        TEST_EQ(counter.unsupported_codepoints, 9);
        TEST_EQ(counter.invalid_sequences, 9);
    }
}

STRF_TEST_FUNC void unsafe_decode_encode()
{
    constexpr auto src_charset = strf::iso_8859_7_t<char>();
    constexpr auto dst_charset = strf::windows_1252_t<char>();
    {
        // - Codepoint 0x20AC is encode as "\xA4" in ISO-8859-16 and as "\x80" in WINDOWS-1252
        // - In ISO-8859-7, "\xC3" maps to codepoint not supported by WINDOWS-1252
        const char* input = "\xA4_\xC3";

        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};
        TEST("\x80_?")    (dst_charset, notifier, strf::unsafe_transcode(input, src_charset));
        TEST("   \x80_?") (dst_charset, notifier, strf::unsafe_transcode(input, src_charset) > 6);
        TEST("\x80_?...") (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).fill('.')<6 );
        TEST("...\x80_?") (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).fill('.')>6 );
        TEST(".\x80_?..") (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).fill('.')^6 );

        TEST("   \x80_?")      (dst_charset, notifier, strf::join_right(6)(strf::unsafe_transcode(input, src_charset)) );
        TEST("...\x80_?~~~")   (dst_charset, notifier, strf::join_right(9, '.') (strf::unsafe_transcode(input, src_charset).fill('~')<6) );
        TEST("......\x80_?")   (dst_charset, notifier, strf::join_right(9, '.') (strf::unsafe_transcode(input, src_charset).fill('~')<3) );
        TEST("......\x80_?")   (dst_charset, notifier, strf::join_right(9, '.') (strf::unsafe_transcode(input, src_charset).fill('~')<2) );

        TEST_EQ(counter.unsupported_codepoints, 9);
        TEST_EQ(counter.invalid_sequences, 0);
    }
    {
        const char* input = "\xA4_\xC3";

        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};
        TEST("\x80_?")    (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).p(3));
        TEST("   \x80_?") (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).p(3) > 6);
        TEST("\x80_?...") (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).p(3).fill('.')<6 );
        TEST("...\x80_?") (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).p(3).fill('.')>6 );
        TEST(".\x80_?..") (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).p(3).fill('.')^6 );

        TEST("   \x80_?")      (dst_charset, notifier, strf::join_right(6)(strf::unsafe_transcode(input, src_charset).p(3)) );
        TEST("...\x80_?~~~")   (dst_charset, notifier, strf::join_right(9, '.') (strf::unsafe_transcode(input, src_charset).p(3).fill('~')<6) );
        TEST("......\x80_?")   (dst_charset, notifier, strf::join_right(9, '.') (strf::unsafe_transcode(input, src_charset).p(3).fill('~')<3) );
        TEST("......\x80_?")   (dst_charset, notifier, strf::join_right(9, '.') (strf::unsafe_transcode(input, src_charset).p(3).fill('~')<2) );

        TEST_EQ(counter.unsupported_codepoints, 9);
        TEST_EQ(counter.invalid_sequences, 0);
    }
}

STRF_TEST_FUNC void bypass_transcode_because_charsets_are_statically_equal()
{
    constexpr auto charset = strf::iso_8859_7_t<char>();
    {
        const char* input = "\xD2"; // \xD2 is an invalid byte
        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};

        TEST("\xD2")    (charset, notifier, strf::transcode(input, charset));
        TEST("   \xD2") (charset, notifier, strf::transcode(input, charset) > 4);
        TEST("\xD2...") (charset, notifier, strf::transcode(input, charset).fill('.')<4 );
        TEST("...\xD2") (charset, notifier, strf::transcode(input, charset).fill('.')>4 );
        TEST(".\xD2..") (charset, notifier, strf::transcode(input, charset).fill('.')^4 );

        TEST("   \xD2")      (charset, notifier, strf::join_right(4)(strf::transcode(input, charset)) );
        TEST("...\xD2~~~")   (charset, notifier, strf::join_right(7, '.') (strf::transcode(input, charset).fill('~')<4) );
        TEST("......\xD2")   (charset, notifier, strf::join_right(7, '.') (strf::transcode(input, charset).fill('~')<1) );

        TEST_EQ(counter.unsupported_codepoints, 0);
        TEST_EQ(counter.invalid_sequences, 0);
    }
    {
        const char* input = "\xD2___"; // \xD2 is an invalid byte
        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};

        TEST("\xD2")    (charset, notifier, strf::transcode(input, charset).p(1));
        TEST("   \xD2") (charset, notifier, strf::transcode(input, charset).p(1) > 4);
        TEST("\xD2...") (charset, notifier, strf::transcode(input, charset).p(1).fill('.')<4 );
        TEST("...\xD2") (charset, notifier, strf::transcode(input, charset).p(1).fill('.')>4 );
        TEST(".\xD2..") (charset, notifier, strf::transcode(input, charset).p(1).fill('.')^4 );

        TEST("   \xD2")      (charset, notifier, strf::join_right(4)(strf::transcode(input, charset).p(1)) );
        TEST("...\xD2~~~")   (charset, notifier, strf::join_right(7, '.') (strf::transcode(input, charset).p(1).fill('~')<4) );
        TEST("......\xD2")   (charset, notifier, strf::join_right(7, '.') (strf::transcode(input, charset).p(1).fill('~')<1) );

        TEST_EQ(counter.unsupported_codepoints, 0);
        TEST_EQ(counter.invalid_sequences, 0);
    }
}

#if ! defined(__CUDACC__)

STRF_TEST_FUNC void transcode_between_dynamic_charsets()
{
    // trancoding UTF-8 to UTF-16
    const strf::dynamic_charset<char>     src_charset(strf::utf<char>);
    const strf::dynamic_charset<char16_t> dst_charset(strf::utf<char16_t>);

    {
        // "\xBF" is an invali sequence
        // "\xF0\x9F\x91\x88" encodes full-width codepoint
        const char* input = "\xBF_\xF0\x9F\x91\x88";

        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};

        TEST(u"\uFFFD_\U0001F448")      (dst_charset, notifier, strf::transcode(input, src_charset) );
        TEST(u"   \uFFFD_\U0001F448")   (dst_charset, notifier, strf::transcode(input, src_charset) > 7 );
        TEST(u"\uFFFD_\U0001F448...")   (dst_charset, notifier, strf::transcode(input, src_charset).fill('.') < 7 );
        TEST(u"...\uFFFD_\U0001F448")   (dst_charset, notifier, strf::transcode(input, src_charset).fill('.') > 7 );
        TEST(u".\uFFFD_\U0001F448..")   (dst_charset, notifier, strf::transcode(input, src_charset).fill('.') ^ 7 );

        TEST(u"   \uFFFD_\U0001F448")    (dst_charset, notifier, strf::join_right(7)(strf::transcode(input, src_charset)) );
        TEST(u"...\uFFFD_\U0001F448~~~") (dst_charset, notifier, strf::join_right(10, '.')(strf::transcode(input, src_charset).fill('~') < 7) );
        TEST(u"......\uFFFD_\U0001F448") (dst_charset, notifier, strf::join_right(10, '.')(strf::transcode(input, src_charset).fill('~') < 4) );
        TEST(u"......\uFFFD_\U0001F448") (dst_charset, notifier, strf::join_right(10, '.')(strf::transcode(input, src_charset).fill('~') < 3) );

        TEST_EQ(counter.invalid_sequences, 9);
    }
    {
        const char* input = "\xBF_\xF0\x9F\x91\x88___";
        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};

        TEST(u"\uFFFD_\U0001F448")      (dst_charset, notifier, strf::transcode(input, src_charset).p(4) );
        TEST(u"   \uFFFD_\U0001F448")   (dst_charset, notifier, strf::transcode(input, src_charset).p(4) > 7 );
        TEST(u"\uFFFD_\U0001F448...")   (dst_charset, notifier, strf::transcode(input, src_charset).p(4).fill('.') < 7 );
        TEST(u"...\uFFFD_\U0001F448")   (dst_charset, notifier, strf::transcode(input, src_charset).p(4).fill('.') > 7 );
        TEST(u".\uFFFD_\U0001F448..")   (dst_charset, notifier, strf::transcode(input, src_charset).p(4).fill('.') ^ 7 );

        TEST(u"   \uFFFD_\U0001F448")    (dst_charset, notifier, strf::join_right(7)(strf::transcode(input, src_charset).p(4)) );
        TEST(u"...\uFFFD_\U0001F448~~~") (dst_charset, notifier, strf::join_right(10, '.')(strf::transcode(input, src_charset).p(4).fill('~') < 7) );
        TEST(u"......\uFFFD_\U0001F448") (dst_charset, notifier, strf::join_right(10, '.')(strf::transcode(input, src_charset).p(4).fill('~') < 4) );
        TEST(u"......\uFFFD_\U0001F448") (dst_charset, notifier, strf::join_right(10, '.')(strf::transcode(input, src_charset).p(4).fill('~') < 3) );

        TEST_EQ(counter.invalid_sequences, 9);
    }

}

STRF_TEST_FUNC void unsafe_transcode_between_dynamic_charsets()
{
    // trancoding UTF-32 to ISO-8859-3
    const strf::dynamic_charset<wchar_t> src_charset(strf::utf<wchar_t>);
    const strf::dynamic_charset<char>    dst_charset(strf::iso_8859_3<char>);

    {
        // \u02D9 is supported by ISO-8859-3, and its encoded as \xFF
        // \uAAAA is not
        const wchar_t* input = L"\u02D9_\uAAAA";

        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};

        TEST("\xFF_?")      (dst_charset, notifier, strf::unsafe_transcode(input, src_charset));
        TEST("   \xFF_?")   (dst_charset, notifier, strf::unsafe_transcode(input, src_charset) > 6 );
        TEST("\xFF_?...")   (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).fill('.')<6 );
        TEST("...\xFF_?")   (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).fill('.')>6 );
        TEST(".\xFF_?..")   (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).fill('.')^6 );

        TEST("   \xFF_?")    (dst_charset, notifier, strf::join_right(6)(strf::unsafe_transcode(input, src_charset)) );
        TEST("...\xFF_?~~~") (dst_charset, notifier, strf::join_right(9, '.')(strf::unsafe_transcode(input, src_charset).fill('~')<6) );
        TEST("......\xFF_?") (dst_charset, notifier, strf::join_right(9, '.')(strf::unsafe_transcode(input, src_charset).fill('~')<3) );
        TEST("......\xFF_?") (dst_charset, notifier, strf::join_right(9, '.')(strf::unsafe_transcode(input, src_charset).fill('~')<2) );

        TEST_EQ(counter.unsupported_codepoints, 9);
    }
    {
        const wchar_t* input = L"\u02D9_\uAAAA___";

        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};

        TEST("\xFF_?")      (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).p(3));
        TEST("   \xFF_?")   (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).p(3) > 6 );
        TEST("\xFF_?...")   (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).p(3).fill('.')<6 );
        TEST("...\xFF_?")   (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).p(3).fill('.')>6 );
        TEST(".\xFF_?..")   (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).p(3).fill('.')^6 );

        TEST("   \xFF_?")    (dst_charset, notifier, strf::join_right(6)(strf::unsafe_transcode(input, src_charset).p(3)) );
        TEST("...\xFF_?~~~") (dst_charset, notifier, strf::join_right(9, '.')(strf::unsafe_transcode(input, src_charset).p(3).fill('~')<6) );
        TEST("......\xFF_?") (dst_charset, notifier, strf::join_right(9, '.')(strf::unsafe_transcode(input, src_charset).p(3).fill('~')<3) );
        TEST("......\xFF_?") (dst_charset, notifier, strf::join_right(9, '.')(strf::unsafe_transcode(input, src_charset).p(3).fill('~')<2) );

        TEST_EQ(counter.unsupported_codepoints, 9);
    }

}

STRF_TEST_FUNC void bypass_transcode_beetween_dynamic_charsets_because_they_are_equal()
{
    const strf::dynamic_charset<char> charset(strf::iso_8859_7<char>);
    {
        const char* input = "\xD2"; // \xD2 is an invalid byte
        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};

        TEST("\xD2")    (charset, notifier, strf::transcode(input, charset));
        TEST("   \xD2") (charset, notifier, strf::transcode(input, charset) > 4);
        TEST("\xD2...") (charset, notifier, strf::transcode(input, charset).fill('.')<4 );
        TEST("...\xD2") (charset, notifier, strf::transcode(input, charset).fill('.')>4 );
        TEST(".\xD2..") (charset, notifier, strf::transcode(input, charset).fill('.')^4 );

        TEST("   \xD2")      (charset, notifier, strf::join_right(4)(strf::transcode(input, charset)) );
        TEST("...\xD2~~~")   (charset, notifier, strf::join_right(7, '.') (strf::transcode(input, charset).fill('~')<4) );
        TEST("......\xD2")   (charset, notifier, strf::join_right(7, '.') (strf::transcode(input, charset).fill('~')<1) );

        TEST_EQ(counter.unsupported_codepoints, 0);
        TEST_EQ(counter.invalid_sequences, 0);
    }
    {
        const char* input = "\xD2___"; // \xD2 is an invalid byte
        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};

        TEST("\xD2")    (charset, notifier, strf::transcode(input, charset).p(1));
        TEST("   \xD2") (charset, notifier, strf::transcode(input, charset).p(1) > 4);
        TEST("\xD2...") (charset, notifier, strf::transcode(input, charset).p(1).fill('.')<4 );
        TEST("...\xD2") (charset, notifier, strf::transcode(input, charset).p(1).fill('.')>4 );
        TEST(".\xD2..") (charset, notifier, strf::transcode(input, charset).p(1).fill('.')^4 );

        TEST("   \xD2")      (charset, notifier, strf::join_right(4)(strf::transcode(input, charset).p(1)) );
        TEST("...\xD2~~~")   (charset, notifier, strf::join_right(7, '.') (strf::transcode(input, charset).p(1).fill('~')<4) );
        TEST("......\xD2")   (charset, notifier, strf::join_right(7, '.') (strf::transcode(input, charset).p(1).fill('~')<1) );

        TEST_EQ(counter.unsupported_codepoints, 0);
        TEST_EQ(counter.invalid_sequences, 0);
    }
}

STRF_TEST_FUNC void decode_encode_between_dynamic_charset()
{
    const strf::dynamic_charset<char> src_charset(strf::iso_8859_7<char>);
    const strf::dynamic_charset<char> dst_charset(strf::windows_1252<char>);
    {
        // - Codepoint 0x20AC is encode as "\xA4" in ISO-8859-16 and as "\x80" in WINDOWS-1252
        // - In ISO-8859-7, "\xC3" maps to codepoint not supported by WINDOWS-1252
        // - In ISO-8859-7, "\xD2" is an invalid byte
        const char* input = "\xA4_\xC3_\xD2";

        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};
        TEST("\x80_?_?")      (dst_charset, notifier, strf::transcode(input, src_charset));
        TEST("   \x80_?_?")   (dst_charset, notifier, strf::transcode(input, src_charset) > 8);
        TEST("\x80_?_?...")   (dst_charset, notifier, strf::transcode(input, src_charset).fill('.')<8 );
        TEST("...\x80_?_?")   (dst_charset, notifier, strf::transcode(input, src_charset).fill('.')>8 );
        TEST(".\x80_?_?..")   (dst_charset, notifier, strf::transcode(input, src_charset).fill('.')^8 );

        TEST("   \x80_?_?")    (dst_charset, notifier, strf::join_right(8)(strf::transcode(input, src_charset)) );
        TEST("...\x80_?_?~~~") (dst_charset, notifier, strf::join_right(11, '.') (strf::transcode(input, src_charset).fill('~')<8) );
        TEST("......\x80_?_?") (dst_charset, notifier, strf::join_right(11, '.') (strf::transcode(input, src_charset).fill('~')<5) );
        TEST("......\x80_?_?") (dst_charset, notifier, strf::join_right(11, '.') (strf::transcode(input, src_charset).fill('~')<4) );

        TEST_EQ(counter.unsupported_codepoints, 9);
        TEST_EQ(counter.invalid_sequences, 9);
    }
    {
        const char* input = "\xA4_\xC3_\xD2___";

        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};
        TEST("\x80_?_?")      (dst_charset, notifier, strf::transcode(input, src_charset).p(5));
        TEST("   \x80_?_?")   (dst_charset, notifier, strf::transcode(input, src_charset).p(5) > 8);
        TEST("\x80_?_?...")   (dst_charset, notifier, strf::transcode(input, src_charset).p(5).fill('.')<8 );
        TEST("...\x80_?_?")   (dst_charset, notifier, strf::transcode(input, src_charset).p(5).fill('.')>8 );
        TEST(".\x80_?_?..")   (dst_charset, notifier, strf::transcode(input, src_charset).p(5).fill('.')^8 );

        TEST("   \x80_?_?")    (dst_charset, notifier, strf::join_right(8)(strf::transcode(input, src_charset).p(5)) );
        TEST("...\x80_?_?~~~") (dst_charset, notifier, strf::join_right(11, '.') (strf::transcode(input, src_charset).p(5).fill('~')<8) );
        TEST("......\x80_?_?") (dst_charset, notifier, strf::join_right(11, '.') (strf::transcode(input, src_charset).p(5).fill('~')<5) );
        TEST("......\x80_?_?") (dst_charset, notifier, strf::join_right(11, '.') (strf::transcode(input, src_charset).p(5).fill('~')<4) );

        TEST_EQ(counter.unsupported_codepoints, 9);
        TEST_EQ(counter.invalid_sequences, 9);
    }

}

STRF_TEST_FUNC void unsafe_decode_encode_between_dynamic_charset()
{
    const strf::dynamic_charset<char> src_charset{strf::iso_8859_7<char>};
    const strf::dynamic_charset<char> dst_charset{strf::windows_1252<char>};

    {
        // - Codepoint 0x20AC is encode as "\xA4" in ISO-8859-16 and as "\x80" in WINDOWS-1252
        // - In ISO-8859-7, "\xC3" maps to codepoint not supported by WINDOWS-1252
        const char* input = "\xA4_\xC3";

        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};
        TEST("\x80_?")    (dst_charset, notifier, strf::unsafe_transcode(input, src_charset));
        TEST("   \x80_?") (dst_charset, notifier, strf::unsafe_transcode(input, src_charset) > 6);
        TEST("\x80_?...") (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).fill('.')<6 );
        TEST("...\x80_?") (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).fill('.')>6 );
        TEST(".\x80_?..") (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).fill('.')^6 );

        TEST("   \x80_?")      (dst_charset, notifier, strf::join_right(6)(strf::unsafe_transcode(input, src_charset)) );
        TEST("...\x80_?~~~")   (dst_charset, notifier, strf::join_right(9, '.') (strf::unsafe_transcode(input, src_charset).fill('~')<6) );
        TEST("......\x80_?")   (dst_charset, notifier, strf::join_right(9, '.') (strf::unsafe_transcode(input, src_charset).fill('~')<3) );
        TEST("......\x80_?")   (dst_charset, notifier, strf::join_right(9, '.') (strf::unsafe_transcode(input, src_charset).fill('~')<2) );

        TEST_EQ(counter.unsupported_codepoints, 9);
        TEST_EQ(counter.invalid_sequences, 0);
    }
    {
        const char* input = "\xA4_\xC3___";

        transcoding_errors_counter counter;
        strf::transcoding_error_notifier_ptr notifier{&counter};
        TEST("\x80_?")    (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).p(3));
        TEST("   \x80_?") (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).p(3) > 6);
        TEST("\x80_?...") (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).p(3).fill('.')<6 );
        TEST("...\x80_?") (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).p(3).fill('.')>6 );
        TEST(".\x80_?..") (dst_charset, notifier, strf::unsafe_transcode(input, src_charset).p(3).fill('.')^6 );

        TEST("   \x80_?")      (dst_charset, notifier, strf::join_right(6)(strf::unsafe_transcode(input, src_charset).p(3)) );
        TEST("...\x80_?~~~")   (dst_charset, notifier, strf::join_right(9, '.') (strf::unsafe_transcode(input, src_charset).p(3).fill('~')<6) );
        TEST("......\x80_?")   (dst_charset, notifier, strf::join_right(9, '.') (strf::unsafe_transcode(input, src_charset).p(3).fill('~')<3) );
        TEST("......\x80_?")   (dst_charset, notifier, strf::join_right(9, '.') (strf::unsafe_transcode(input, src_charset).p(3).fill('~')<2) );

        TEST_EQ(counter.unsupported_codepoints, 9);
        TEST_EQ(counter.invalid_sequences, 0);
    }

}

#endif // ! defined(__CUDACC__)

STRF_TEST_FUNC void test_input_string()
{
    test_basic();
    sanitize();
    transcode();
    unsafe_transcode();
    decode_encode();
    unsafe_decode_encode();
    bypass_transcode_because_charsets_are_statically_equal();

#if ! defined(__CUDACC__)
    transcode_between_dynamic_charsets();
    unsafe_transcode_between_dynamic_charsets();
    bypass_transcode_beetween_dynamic_charsets_because_they_are_equal();
    decode_encode_between_dynamic_charset();
    unsafe_decode_encode_between_dynamic_charset();
#endif // ! defined(__CUDACC__)
}

} // namespace

REGISTER_STRF_TEST(test_input_string)

