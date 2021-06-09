//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define _CRT_SECURE_NO_WARNINGS

#include <strf/locale.hpp>
#include "test_utils.hpp"

#if defined(_WIN32)
#define LOCALE_NAME(lang, region) #lang "-" #region
#else
#define LOCALE_NAME(lang, region) #lang "_" #region
#endif

void test_locale()
{
    if (setlocale(LC_NUMERIC, LOCALE_NAME(en, US))) {
        auto punct = strf::locale_numpunct();
        TEST("10,000,000,000,000,000.") .with(punct) (*!strf::fixed(1e+16));
    }
    if (setlocale(LC_NUMERIC, LOCALE_NAME(de, DE))) {
        auto punct = strf::locale_numpunct();
        TEST("10.000.000.000.000.000,") .with(punct) (*!strf::fixed(1e+16));
    }
    if (setlocale(LC_NUMERIC, LOCALE_NAME(as, IN))) {
        auto punct = strf::locale_numpunct();
        TEST("10,00,00,00,00,00,00,000.") .with(punct) (*!strf::fixed(1e+16));
    }
#if ! defined(_WIN32)
    {
        using strf::detail::make_numpunct;
        strf::digits_grouping empty_grouping{-1};
        strf::digits_grouping non_empty_grouping{1, 2, 3};
        {
            auto p = make_numpunct("unknown_encoding", ".xx", ",yy", non_empty_grouping);
            TEST_TRUE(p.decimal_point() == U'.');
            TEST_TRUE(p.thousands_sep() == U',');
            TEST_TRUE(p.grouping() == non_empty_grouping);
        }
        {
            auto p = make_numpunct("unknown_encoding", ".xx", ",yy", empty_grouping);
            TEST_TRUE(p.decimal_point() == U'.');
            TEST_TRUE(p.grouping() == empty_grouping);
        }
        {
            auto p = make_numpunct("unknown_encoding", "\xDB\xAC", "\xFF", non_empty_grouping);
            TEST_TRUE(p.decimal_point() == U'\uFFFD');
            TEST_TRUE(p.thousands_sep() == U'\u00A0');
            TEST_TRUE(p.grouping() == non_empty_grouping);
        }
        {
            auto p = make_numpunct("unknown_encoding", ".xx", "\xFF", non_empty_grouping);
            TEST_TRUE(p.decimal_point() == U'.');
            TEST_TRUE(p.thousands_sep() == U'\u00A0');
            TEST_TRUE(p.grouping() == non_empty_grouping);
        }
        {
            auto p = make_numpunct("unknown_encoding", "", "", non_empty_grouping);
            TEST_TRUE(p.decimal_point() == U'\uFFFD');
            TEST_TRUE(p.grouping() == empty_grouping);
        }

        {
            auto p = make_numpunct("ISO-8859-1", "\xB4", "\xB7", non_empty_grouping);
            TEST_TRUE(p.decimal_point() == U'\u00B4');
            TEST_TRUE(p.thousands_sep() == U'\u00B7');
            TEST_TRUE(p.grouping() == non_empty_grouping);
        }
        {
            auto p = make_numpunct("ISO-8859-3", "\xA2", "\xFF", non_empty_grouping);
            TEST_TRUE(p.decimal_point() == U'\u02D8');
            TEST_TRUE(p.thousands_sep() == U'\u02D9');
            TEST_TRUE(p.grouping() == non_empty_grouping);
        }
        {
            auto p = make_numpunct("ISO-8859-15", "\xA6", "\xA8", non_empty_grouping);
            TEST_TRUE(p.decimal_point() == U'\u0160');
            TEST_TRUE(p.thousands_sep() == U'\u0161');
            TEST_TRUE(p.grouping() == non_empty_grouping);
        }
        {
            auto p = make_numpunct("UTF-8", "\xDB\xAC", "\xD5\x9A", non_empty_grouping);
            TEST_TRUE(p.decimal_point() == U'\u06EC');
            TEST_TRUE(p.thousands_sep() == U'\u055A');
            TEST_TRUE(p.grouping() == non_empty_grouping);
        }
        {
            auto p = make_numpunct("UTF-8", ",", "\xD5\x9A", non_empty_grouping);
            TEST_TRUE(p.decimal_point() == U',');
            TEST_TRUE(p.thousands_sep() == U'\u055A');
            TEST_TRUE(p.grouping() == non_empty_grouping);
        }
        {
            auto p = make_numpunct("UTF-8", "", "", non_empty_grouping);
            TEST_TRUE(p.decimal_point() == U'\uFFFD');
            TEST_TRUE(p.grouping() == empty_grouping);
        }
        {
            auto p = make_numpunct("UTF-8", "\xFF", "\xFF", non_empty_grouping);
            TEST_TRUE(p.decimal_point() == U'\uFFFD');
            TEST_TRUE(p.grouping() == empty_grouping);
        }

    }
#endif // ! defined(_WIN32)
    {
        using strf::detail::decode_first_char_from_utf16;

        wchar_t str_10000[] = { 0xD800, 0xDC00, 0};
        wchar_t str_10FFFF[] = { 0xDBFF, 0xDFFF, 0};
        wchar_t str_invalid_1[] = { 0xD800, L'a', 0};
        wchar_t str_invalid_2[] = { 0xDC00, 0xD800, 0};
        wchar_t str_invalid_3[] = { 0xDFFF, 0xDBFF, 0};
        wchar_t str_invalid_4[] = { 0xD800, 0};

        TEST_TRUE(0xFFFF   == strf::detail::decode_first_char_from_utf16(L"\uFFFF"));
        TEST_TRUE(0x10000  == strf::detail::decode_first_char_from_utf16(str_10000));
        TEST_TRUE(0x10FFFF == strf::detail::decode_first_char_from_utf16(str_10FFFF));
        TEST_TRUE(0xFFFD == strf::detail::decode_first_char_from_utf16(str_invalid_1));
        TEST_TRUE(0xFFFD == strf::detail::decode_first_char_from_utf16(str_invalid_2));
        TEST_TRUE(0xFFFD == strf::detail::decode_first_char_from_utf16(str_invalid_3));
        TEST_TRUE(0xFFFD == strf::detail::decode_first_char_from_utf16(str_invalid_4));
        TEST_TRUE(0xFFFD == strf::detail::decode_first_char_from_utf16(L""));
    }
    {
        using strf::detail::parse_win_grouping;
        using strf::digits_grouping;
        static_assert(31 == digits_grouping::grp_max, "Need to update these test cases");
        static_assert(6 == digits_grouping::grps_count_max, "Need to update these test cases");
        TEST_TRUE(parse_win_grouping(L"")    == digits_grouping());
        TEST_TRUE(parse_win_grouping(L"0")   == digits_grouping());
        TEST_TRUE(parse_win_grouping(L"1;0") == digits_grouping(1));
        TEST_TRUE(parse_win_grouping(L"1")   == digits_grouping(1, -1));
        TEST_TRUE(parse_win_grouping(L"1;22;3;0") == digits_grouping(1, 22, 3));
        TEST_TRUE(parse_win_grouping(L"9;31")      == digits_grouping(9, 31, -1));
        TEST_TRUE(parse_win_grouping(L"1;2;3;3;3;3;0") == digits_grouping(1,2,3));
        TEST_TRUE(parse_win_grouping(L"1;2;3;3;3;3")   == digits_grouping(1,2,3,3,3,3,-1));
        TEST_TRUE(parse_win_grouping(L"1;2;3;4;5;6;0") == digits_grouping(1,2,3,4,5,6));
        TEST_TRUE(parse_win_grouping(L"1;2;3;4;5;6")   == digits_grouping(1,2,3,4,5,6,-1));
        TEST_TRUE(parse_win_grouping(L"1;2;3;4;5;6;7;0") == digits_grouping());
        TEST_TRUE(parse_win_grouping(L"1;2;3;4;5;6;7")   == digits_grouping());

        TEST_TRUE(parse_win_grouping(L"1;0;2") == digits_grouping());
        TEST_TRUE(parse_win_grouping(L"1;2a;2") == digits_grouping());
        TEST_TRUE(parse_win_grouping(L"1;11a;2") == digits_grouping());
        TEST_TRUE(parse_win_grouping(L";1;2;3") == digits_grouping());
        TEST_TRUE(parse_win_grouping(L"1;;2;3") == digits_grouping());
        TEST_TRUE(parse_win_grouping(L"9;32")      == digits_grouping());
        TEST_TRUE(parse_win_grouping(L"9;32;2;0")  == digits_grouping());
    }
}
