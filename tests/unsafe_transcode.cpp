﻿//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils/simple_transcoding_err_notifier.hpp"

#ifndef __cpp_char8_t
#   if __GNUC__ >= 11
#       pragma GCC diagnostic ignored "-Wc++20-compat"
#   endif
using char8_t = char;
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

STRF_TEST_FUNC void test_unsafe_transcode_to_ptr()
{
    {
        constexpr auto buff_size = 200;
        char8_t buff[200] = {};
        const ustr_view input = u"abc\uAAAAzzz\uBBBBxxx";

        strf::unsafe_transcode<strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), buff, buff + buff_size);

        TEST_CSTR_EQ(buff, u8"abc\uAAAAzzz\uBBBBxxx");
    }
    {
        const ustr_view input = u"hello";
        auto res = strf::unsafe_transcode_size <strf::utf_t, strf::utf_t<char>>
            (input.begin(), input.end(), 6);

        TEST_EQ(res.ssize, 5);
    }
    {
        const ustr_view input = u"abc\uAAAAzzz\uBBBBxxx";
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        errors_counter counter;

        strf::unsafe_transcode<strf::utf_t, strf::iso_8859_3_t>
            (input.begin(), input.end(), buff, buff + buff_size, &counter);

        TEST_CSTR_EQ(buff, "abc?zzz?xxx");
        TEST_EQ(counter.count, 2);
    }
    {
        const ustr_view input = u"abc\uAAAAzzz\uBBBBxxx";;
        auto res = strf::unsafe_transcode_size <strf::utf_t, strf::iso_8859_3_t<char>>
            (input.begin(), input.end(), 12);

        TEST_EQ(res.ssize, 11);
    }
}

STRF_TEST_FUNC void test_unsafe_transcode_to_dest()
{
    // strf::unsafe_transcode just delegates to the strf::transcode
    // and strf::decode_encode that all already well covered in
    // transcode.cpp and decode_encode_to_dst.cpp tests files

    // So there isn't much to be tested here

    using stop_reason = strf::transcode_stop_reason;

    {   // when there is a direct transcoder
        const str_view<char> input("abcde"); // input with invalid sequence
        char16_t buff[50] = {};
        strf::array_destination<char16_t> dst(buff);

        auto tr_res = strf::unsafe_transcode<strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dst);
        TEST_TRUE(tr_res.stop_reason == stop_reason::completed);

        auto dst_res = dst.finish();
        const ustr_view output(buff, dst_res.ptr);
        TEST_STR_EQ(output, u"abcde");
        TEST_EQ(output.ssize(), tr_res.ssize);
    }

    {   // when there isn't a direct transcoder, so that decode_encode is called
        const ustr_view input =
            u"abcde"
            u"\u0401\u0402\u0403\u045F"
            u"\uABCD"   // an unsupported codepoints
            u"XYZWRESF";

        char buff[80] = {};
        strf::array_destination<char> dst(buff);

        const auto flags = strf::transcode_flags::stop_on_unsupported_codepoint;
        test_utils::simple_transcoding_err_notifier notifier;

        auto tr_res = strf::unsafe_transcode<strf::utf_t, strf::iso_8859_5_t>
            (input.begin(), input.end(), dst, &notifier, flags);

        TEST_TRUE(tr_res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        auto dst_res = dst.finish();
        const str_view<char> output(buff, dst_res.ptr);
        TEST_STR_EQ(output, "abcde\xA1\xA2\xA3\xFF");
        TEST_EQ(output.ssize(), tr_res.ssize);
    }
}

STRF_TEST_FUNC void test_all()
{
    test_unsafe_transcode_to_ptr();
    test_unsafe_transcode_to_dest();
}

} // namespace

REGISTER_STRF_TEST(test_all)

