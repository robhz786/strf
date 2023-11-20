//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils/simple_transcoding_err_notifier.hpp"
#include "test_utils/array_destination_with_sub_initial_space.hpp"

#ifndef __cpp_char8_t
#   if __GNUC__ >= 11
#       pragma GCC diagnostic ignored "-Wc++20-compat"
#   endif
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

STRF_TEST_FUNC void transcode_to_ptr()
{
#ifdef STRF_HAS_STD_STRING_VIEW

    {
        constexpr auto buff_size = 200;
        char8_t buff[buff_size] = {};

        strf::transcode<strf::utf_t, strf::utf_t>
            (u"abc\uAAAAzzz\uBBBBxxx"sv, buff, buff + buff_size);

        TEST_CSTR_EQ(buff, u8"abc\uAAAAzzz\uBBBBxxx");
    }
    {
        auto res = strf::transcode_size<strf::utf_t, strf::utf_t<char>>(u"hello"sv, 6);

        TEST_EQ(res.ssize, 5);
    }
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        errors_counter counter;

        strf::transcode<strf::utf_t, strf::iso_8859_3_t>
            (u"abc\uAAAAzzz\uBBBBxxx"sv, buff, buff + buff_size, &counter);

        TEST_CSTR_EQ(buff, "abc?zzz?xxx");
        TEST_EQ(counter.count, 2);
    }
    {
        auto res = strf::transcode_size <strf::utf_t, strf::iso_8859_3_t<char>>
            (u"abc\uAAAAzzz\uBBBBxxx"sv, 12);

        TEST_EQ(res.ssize, 11);
    }

#endif // STRF_HAS_STD_STRING_VIEW
}

STRF_TEST_FUNC void transcode_to_dest()
{
    using stop_reason = strf::transcode_stop_reason;

    {   // when there is a direct transcoder
        str_view<char> input("abcde\xA5zzz"); // input with invalid sequence
        char16_t buff[50] = {};
        strf::array_destination<char16_t> dst(buff);

        const auto flags = strf::transcode_flags::stop_on_invalid_sequence;
        test_utils::simple_transcoding_err_notifier notifier;

        auto tr_res = strf::transcode<strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dst, &notifier, flags);
        TEST_TRUE(tr_res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        auto dst_res = dst.finish();
        ustr_view output(buff, dst_res.ptr);
        TEST_STR_EQ(output, u"abcde");
        TEST_EQ(output.ssize(), tr_res.ssize);
    }

    {   // when there is a direct transcoder and recycle is called
        constexpr auto buff_size = 100;
        char buff[buff_size] = {};
        test_utils::array_destination_with_sub_initial_space<char> dst(buff, buff_size);
        dst.reset_with_initial_space(10);

        const ustr_view input = u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        auto tr_res = strf::transcode<strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dst);
        TEST_TRUE(tr_res.stop_reason == stop_reason::completed);

        auto output = dst.finish();
        TEST_STR_EQ(output, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");
        TEST_EQ(output.ssize(), tr_res.ssize);
    }

    {   // when there is a direct transcoder and destination is too small
        constexpr auto buff_size = 10;
        char buff[buff_size] = {};
        strf::array_destination<char> dst(buff, buff_size);
        const ustr_view input = u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

        auto tr_res = strf::transcode<strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dst);
        TEST_TRUE(tr_res.stop_reason == stop_reason::insufficient_output_space);

        auto dst_res = dst.finish();
        str_view<char> output(buff, dst_res.ptr);
        TEST_STR_EQ(output, "abcdefghij");
        TEST_EQ(output.ssize(), tr_res.ssize);
    }

    {   // when there isn't a direct transcoder, so that decode_encode is called
        ustr_view input =
            u"abcde"
            u"\u0401\u0402\u0403\u045F"
            u"\uABCD"   // an unsupported codepoints
            u"XYZWRESF";

        char buff[80] = {};
        strf::array_destination<char> dst(buff);

        const auto flags = strf::transcode_flags::stop_on_unsupported_codepoint;
        test_utils::simple_transcoding_err_notifier notifier;

        auto tr_res = strf::transcode<strf::utf_t, strf::iso_8859_5_t>
            (input.begin(), input.end(), dst, &notifier, flags);

        TEST_TRUE(tr_res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        auto dst_res = dst.finish();
        str_view<char> output(buff, dst_res.ptr);
        TEST_STR_EQ(output, "abcde\xA1\xA2\xA3\xFF");
        TEST_EQ(output.ssize(), tr_res.ssize);
    }
}

STRF_TEST_FUNC void test_all()
{
    transcode_to_ptr();
    transcode_to_dest();
}

} // namespace

REGISTER_STRF_TEST(test_all)
