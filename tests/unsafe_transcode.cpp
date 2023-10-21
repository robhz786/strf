﻿//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils.hpp"

#ifndef __cpp_char8_t
using char8_t = char;
#endif

#ifdef STRF_HAS_STD_STRING_VIEW
using namespace std::literals::string_view_literals;
#endif

namespace {

struct errors_counter: strf::transcoding_error_notifier {

    void STRF_HD invalid_sequence(int, const char*, const void*, std::ptrdiff_t) override {
    }

    void STRF_HD unsupported_codepoint(const char*, unsigned) override {
        ++ count;
    }
    std::ptrdiff_t count = 0;
};

template <typename CharT>
using str_view = strf::detail::simple_string_view<CharT>;

using ustr_view = str_view<char16_t>;

STRF_TEST_FUNC void test_unsafe_transcode()
{
#ifdef STRF_HAS_STD_STRING_VIEW

    {
        constexpr auto buff_size = 200;
        char8_t buff[200] = {};

        strf::unsafe_transcode<strf::utf_t, strf::utf_t>
            (u"abc\uAAAAzzz\uBBBBxxx"sv, buff, buff + buff_size);

        TEST_CSTR_EQ(buff, u8"abc\uAAAAzzz\uBBBBxxx");
    }
    {
        auto res = strf::unsafe_transcode_size <strf::utf_t, strf::utf_t<char>>(u"hello"sv, 6);

        TEST_EQ(res.ssize, 5);
    }
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        errors_counter counter;

        strf::unsafe_transcode<strf::utf_t, strf::iso_8859_3_t>
            (u"abc\uAAAAzzz\uBBBBxxx"sv, buff, buff + buff_size, &counter);

        TEST_CSTR_EQ(buff, "abc?zzz?xxx");
        TEST_EQ(counter.count, 2);
    }
    {
        auto res = strf::unsafe_transcode_size <strf::utf_t, strf::iso_8859_3_t<char>>
            (u"abc\uAAAAzzz\uBBBBxxx"sv, 12);

        TEST_EQ(res.ssize, 11);
    }

#endif // STRF_HAS_STD_STRING_VIEW
}

STRF_TEST_FUNC void test_all()
{
    test_unsafe_transcode();
}

} // namespace



REGISTER_STRF_TEST(test_all)
