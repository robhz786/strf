#ifndef STRF_LOCALE_HPP
#define STRF_LOCALE_HPP

//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#if defined (STRF_FREESTANDING)
#  error "<strf/locale.hpp> is not supporded with the option STRF_FREESTANDING"
#endif

#include <strf.hpp>
#include <strf/detail/facets/numpunct.hpp>
#include <strf/detail/int_digits.hpp>
#include <cwchar>

#include <clocale>
#if defined (_WIN32)
#  include <windows.h>
#else
#  include <langinfo.h>
#  include <cstring>
#endif

namespace strf {

#if defined(STRF_OMIT_IMPL)

strf::numpunct<10> locale_numpunct();

namespace detail {

char32_t decode_first_char_from_utf16(const wchar_t* src);
char32_t decode_first_char(strf::transcode_f<char, char32_t> decode, const char* str);
strf::transcode_f<char, char32_t> get_decoder(const char* charset_name);
strf::digits_grouping parse_win_grouping(const wchar_t* str);
strf::numpunct<10> make_numpunct
    ( const char* encoding_name
    , const char* decimal_point_str
    , const char* thousands_sep_str
    , strf::digits_grouping grouping ) noexcept;

} // namespace detail

#else // defined(STRF_OMIT_IMPL)

namespace detail {

STRF_FUNC_IMPL strf::digits_grouping parse_win_grouping(const wchar_t* str)
{
    if (str[0] == L'\0') {
        return {};
    }
    strf::digits_grouping_creator creator;
    auto it = str;
    while (true) {
        if (strf::detail::not_digit(*it)) {
            return {}; // invalid input
        }
        int grp = *it - L'0';
        ++it;
        if (strf::detail::is_digit(*it)) {
            grp = grp * 10 + (*it - L'0');
            ++it;
        }
        if (grp == 0) {
            if (*it != L'\0') {
                return {}; // invalid input
            }
            return creator.finish();
        }
        creator.push_high(grp);
        if (*it == L'\0') {
            return creator.finish_no_more_sep();
        }
        if (*it != L';') {
            return {}; // invalid input
        }
        ++it;
    }
}

STRF_FUNC_IMPL char32_t decode_first_char_from_utf16(const wchar_t* src)
{
    char32_t ch0 = src[0];
    if (ch0 != '\0') {
        if (strf::detail::not_surrogate(ch0)) {
            return ch0;
        }
        if (strf::detail::is_high_surrogate(ch0)) {
            char32_t ch1 = src[1];
            if (strf::detail::is_low_surrogate(ch1)) {
                return 0x10000 + (((ch0 & 0x3FF) << 10) | (ch1 & 0x3FF));
            }
        }
    }
    return 0xFFFD;
}

#if ! defined (_WIN32)

STRF_FUNC_IMPL char32_t decode_first_char(strf::transcode_f<char, char32_t>& decode, const char* str)
{
    char32_t buff32[2] = { 0xFFFD, 0 };
    strf::u32cstr_destination dest(buff32);
    decode(dest, str, strlen(str), {}, strf::surrogate_policy::strict);
    return buff32[0];
}

STRF_FUNC_IMPL strf::transcode_f<char, char32_t> get_decoder(const char* charset_name)
{
    if (0 == strcmp(charset_name, "UTF-8")) {
        return strf::utf8_to_utf32<char, char32_t>::transcode;
    }
    else if (0 == strcmp(charset_name, "ISO-8859-1")) {
        return strf::static_transcoder
            < char, char32_t, strf::csid_iso_8859_1, strf::csid_utf32 >
            ::transcode_func();
    } else if (0 == strcmp(charset_name, "ISO-8859-3")) {
        return strf::static_transcoder
            < char, char32_t, strf::csid_iso_8859_3, strf::csid_utf32 >
            ::transcode_func();
    } else if (0 == strcmp(charset_name, "ISO-8859-15")) {
        return strf::static_transcoder
            < char, char32_t, strf::csid_iso_8859_15, strf::csid_utf32 >
            ::transcode_func();
    }
    return nullptr;
}

STRF_FUNC_IMPL strf::numpunct<10> make_numpunct
    ( const char* encoding_name
    , const char* decimal_point_str
    , const char* thousands_sep_str
    , strf::digits_grouping grouping ) noexcept
{
    strf::transcode_f<char, char32_t> decoder_func = nullptr;
    bool decoder_func_searched = false;
    strf::numpunct<10> punct(grouping);

    if (decimal_point_str[0] == '\0') {
        punct.decimal_point(0xFFFD);
    } else if (0 == (decimal_point_str[0] & 0x80)) {
        punct.decimal_point(decimal_point_str[0]);
    } else {
        decoder_func_searched = true;
        decoder_func = strf::detail::get_decoder(encoding_name);
        if (decoder_func) {
            punct.decimal_point(strf::detail::decode_first_char(decoder_func, decimal_point_str));
        } else {
            punct.decimal_point(0xFFFD);
        }
    }
    if (! grouping.empty()) {
        if (thousands_sep_str[0] == '\0') {
            punct.grouping(strf::digits_grouping());
        } else if (0 == (thousands_sep_str[0] & 0x80)){
            punct.thousands_sep(thousands_sep_str[0]);
        } else {
            if (! decoder_func_searched) {
                decoder_func = strf::detail::get_decoder(encoding_name);
            }
            if (decoder_func == nullptr) {
                punct.thousands_sep(U'\u00A0');
            } else {
                auto ch = strf::detail::decode_first_char(decoder_func, thousands_sep_str);
                if (ch == 0xFFFD) {
                    punct.grouping(strf::digits_grouping());
                } else {
                    punct.thousands_sep(ch);
                }
            }
        }
    }
    return punct;
}

#endif // ! defined(_WIN32)

} // namespace detail

STRF_FUNC_IMPL strf::numpunct<10> locale_numpunct() noexcept
{
#if defined(_WIN32)

    wchar_t str_grouping[10];
    wchar_t str_decimal_point[4];
    wchar_t str_thousands_sep[4];

    const wchar_t* locale_name = _wsetlocale(LC_NUMERIC, nullptr);
    GetLocaleInfoEx(locale_name, LOCALE_STHOUSAND, str_thousands_sep, 4);
    GetLocaleInfoEx(locale_name, LOCALE_SGROUPING, str_grouping, 10);
    GetLocaleInfoEx(locale_name, LOCALE_SDECIMAL, str_decimal_point, 4);
    auto grouping      = strf::detail::parse_win_grouping(str_grouping);
    auto decimal_point = strf::detail::decode_first_char_from_utf16(str_decimal_point);
    auto thousands_sep = strf::detail::decode_first_char_from_utf16(str_thousands_sep);
    if (thousands_sep == 0xFFFD) {
        grouping = strf::digits_grouping{};
    }
    return strf::numpunct<10>{grouping}
        .decimal_point(decimal_point)
        .thousands_sep(thousands_sep);

#else // defined(_WIN32)

    auto loc = localeconv();
    return strf::detail::make_numpunct
        ( nl_langinfo(CODESET)
        , loc->decimal_point
        , loc->thousands_sep
        , strf::digits_grouping(loc->grouping) );

#endif // defined(_WIN32)
}

#endif // defined(STRF_OMIT_IMPL)

} // namespace strf

#endif  // STRF_LOCALE_HPP

