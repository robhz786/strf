//  Copyright (C) (See commit logs on github.com/robhz786/strf)
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "test_utils/simple_transcoding_err_notifier.hpp"

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

using str_view = strf::detail::simple_string_view<char>;
using ustr_view = strf::detail::simple_string_view<char16_t>;

STRF_TEST_FUNC void test_decode_encode_scenarios()
{
    {   // happy scenario
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        ustr_view input = u"abcdef";
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );

        auto deres = strf::decode_encode
            <strf::utf_t, strf::utf_t>
             (input.begin(), input.end(), buff, buff + buff_size, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::completed);
        TEST_EQ(0, deres.u32dist);
        TEST_EQ(0, (deres.stale_src_ptr - input.end()));
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        str_view output(buff, deres.dst_ptr);

        TEST_EQ(output.size(), input.size());
        TEST_STR_EQ(output, "abcdef");
    }


    {   // happy scenario again,
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        ustr_view input = u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

        auto deres = strf::decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), buff, buff + buff_size, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::completed);
        TEST_EQ(0, (deres.stale_src_ptr - input.end()));
        TEST_EQ(0, deres.u32dist);
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        str_view output(buff, deres.dst_ptr);

        TEST_EQ(output.size(), input.size());
        TEST_STR_EQ(output, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");
    }

    {
        char* dst = nullptr;
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        ustr_view input = u"abcdefghijklmnopqr";

        auto deres = strf::decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dst, dst, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::insufficient_output_space);
        TEST_EQ(0, (deres.stale_src_ptr - input.begin()));
        TEST_EQ(0, deres.u32dist);
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);
    }

    {   // all input is valid, but the destination turns bad
        // during the encoding
        constexpr auto buff_size = 20;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        ustr_view input = u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

        auto deres = strf::decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), buff, buff + buff_size, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::insufficient_output_space);
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        auto cpcount_res = strf::utf_t<char16_t>::count_codepoints
            ( deres.stale_src_ptr, input.end(), deres.u32dist
            , strf::surrogate_policy::strict );

        TEST_EQ(cpcount_res.count, deres.u32dist);
        TEST_EQ((char)*cpcount_res.ptr, 'u');
        TEST_EQ(buff_size, (cpcount_res.ptr - input.begin()));

        str_view output(buff, deres.dst_ptr);
        TEST_EQ(output.size(), buff_size);
        TEST_STR_EQ(output, "abcdefghijklmnopqrst");
    }
    {   // all input is valid, but there is a codepoint that
        // is not supported by the destination encoding
        // and stop_on_unsupported_codepoint() returns true

        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        ustr_view input =
            u"abcdefghijklmnopqrstuvwxyz"
            u"\u0401\u0402\u0403\u045F"
            u"\uABCD"                   // the unsupported codepoints
            u"XYZWRESF";

        auto deres = strf::decode_encode
            <strf::utf_t, strf::iso_8859_5_t>
            (input.begin(), input.end(), buff, buff + buff_size, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::unsupported_codepoint);
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        auto cpcount_res = strf::utf_t<char16_t>::count_codepoints
            ( deres.stale_src_ptr, input.end(), deres.u32dist
            , strf::surrogate_policy::strict );

        TEST_EQ(cpcount_res.count, deres.u32dist);
        TEST_EQ((unsigned)*cpcount_res.ptr, 0xABCD);
        TEST_EQ(30, (cpcount_res.ptr - input.begin()));


        str_view output(buff, deres.dst_ptr);
        TEST_STR_EQ(output, "abcdefghijklmnopqrstuvwxyz\xA1\xA2\xA3\xFF");

    }
    {   // input has invalid sequence and stop_on_invalid_sequence flag is set
        constexpr auto buff_size = 200;
        char16_t buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        str_view input =
            "abcdefghijklmnopqrstuvwxyz"
            "\xA0\xA1\xA2\xFF"
            "\xA5"                  // the invalid sequence
            "XYZWRESF";

        auto deres = strf::decode_encode
            <strf::iso_8859_3_t, strf::utf_t>
            (input.begin(), input.end(), buff, buff + buff_size, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::invalid_sequence);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, deres.u32dist);
        TEST_EQ(30, (char*)notifier.invalid_seq - input.begin());
        TEST_EQ(30, deres.stale_src_ptr - input.begin());


        ustr_view output(buff, deres.dst_ptr);
        TEST_STR_EQ(output, u"abcdefghijklmnopqrstuvwxyz\u00A0\u0126\u02D8\u02D9");
    }
    {   // input has invalid sequence but stop_on_invalid_sequence flag is not set

        constexpr auto buff_size = 200;
        char16_t buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = strf::transcode_flags::none;
        str_view input =
            "abcdefghijklmnopqrstuvwxyz"
            "\xA0\xA1\xA2\xFF"
            "\xA5"                  // the invalid sequence
            "XYZ";

        auto deres = strf::decode_encode
            <strf::iso_8859_3_t, strf::utf_t>
            (input.begin(), input.end(), buff, buff + buff_size, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::completed);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, deres.u32dist);
        TEST_EQ(30, (char*)notifier.invalid_seq - input.begin());
        TEST_EQ(0, input.end() - deres.stale_src_ptr);


        ustr_view output(buff, deres.dst_ptr);
        TEST_STR_EQ(output, u"abcdefghijklmnopqrstuvwxyz\u00A0\u0126\u02D8\u02D9\uFFFDXYZ");
    }
    {   // The input has an invalid sequence and stop_on_invalid_sequence flag is set,
        // but it has also a codepoint that is not supported in the destinations
        // encoding and that comes before the invalid sequence.

        // hence the transcode_stop_reason shall be unsupported_codepoint

        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        str_view input =
            "abcdefghijklmnopqrstuvwxyz"
            "\xC2\xA1" // unsupported codepoint U+A1
            "AB"
            "\xE0\xA0" // the invalid sequence (missing continuation byte)
            "XYZWRESF";

        auto deres = strf::decode_encode
            <strf::utf_t, strf::iso_8859_3_t>
            (input.begin(), input.end(), buff, buff + buff_size, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::unsupported_codepoint);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(30, (char*)notifier.invalid_seq - input.begin());
        TEST_EQ(0xA1, notifier.unsupported_ch32);

        auto cpcount_res = strf::utf8_t<char>::count_codepoints
            ( deres.stale_src_ptr, input.end(), deres.u32dist
            , strf::surrogate_policy::strict );

        TEST_EQ(cpcount_res.count, deres.u32dist);
        TEST_EQ(26, (cpcount_res.ptr - input.begin()));


        str_view output(buff, deres.dst_ptr);
        TEST_STR_EQ(output, "abcdefghijklmnopqrstuvwxyz");

    }
    {   // similar as before, but now the destination has just enougth space to
        // transcode all valid sequences
        // In this case, stop_reason is invalid_sequence

        constexpr auto buff_size = 200;
        char16_t buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        str_view input =
            "abcdefghijklmnopqrstuvwxyzABCD"
            "\xE0\xA0" // the invalid sequence (missing continuation byte)
            "XYZWRESF";

        auto deres = strf::decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), buff, buff + buff_size, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::invalid_sequence);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, deres.u32dist);


        ustr_view output(buff, deres.dst_ptr);
        TEST_STR_EQ(output, u"abcdefghijklmnopqrstuvwxyzABCD");
    }
}


void  test_decode_encode_size_scenarios()
{
    using stop_reason = strf::transcode_stop_reason;

    {   // happy scenario
        ustr_view input = u"abcdef";
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );

        auto res = strf::decode_encode_size
            ( strf::utf<char16_t>, strf::utf<char>, input.begin(), input.end()
            , strf::ssize_max, flags);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_src_ptr - input.end()));
        TEST_EQ(res.ssize, input.ssize());
    }
    {   // happy scenario - but with limit equal to size
        ustr_view input = u"abcdef";
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );

        auto res = strf::decode_encode_size
            ( strf::utf<char16_t>, strf::utf<char>, input.begin(), input.end(), 6, flags);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_src_ptr - input.end()));
        TEST_EQ(res.ssize, input.ssize());
    }
    {   // with limit less than size
        ustr_view input = u"abcdef";
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );

        auto res = strf::decode_encode_size
            ( strf::utf<char16_t>, strf::utf<char>, input.begin(), input.end(), 5, flags);

        TEST_TRUE(res.stop_reason == stop_reason::insufficient_output_space);

        auto cpcount_res = strf::utf_t<char16_t>::count_codepoints
            ( res.stale_src_ptr, input.end(), res.u32dist
            , strf::surrogate_policy::strict );

        TEST_EQ(cpcount_res.count, res.u32dist);
        TEST_EQ(res.ssize, 5);
    }
    {   // all input is valid, but there is a codepoint that
        // is not supported by the destination encoding
        // and stop_on_unsupported_codepoint() returns true

        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        ustr_view input =
            u"12345678901234567890"
            u"\u0401\u0402\u0403\u045F"
            u"\uABCD"                   // the unsupported codepoints
            u"XYZWRESF";
        constexpr auto expected_size = 24;

        auto res = strf::decode_encode_size
            ( strf::utf<char16_t>, strf::iso_8859_5<char>
              , input.begin(), input.end(), expected_size, flags );

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);

        auto cpcount_res = strf::utf_t<char16_t>::count_codepoints
            ( res.stale_src_ptr, input.end(), res.u32dist
            , strf::surrogate_policy::strict );

        TEST_EQ(cpcount_res.count, res.u32dist);
        TEST_EQ((unsigned)*cpcount_res.ptr, 0xABCD);
        TEST_EQ(24, (cpcount_res.ptr - input.begin()));
        TEST_EQ(res.ssize, expected_size);
    }
    {   // Same thing, but with limit equal to expected size

        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        ustr_view input =
            u"12345678901234567890"
            u"\u0401\u0402\u0403\u045F"
            u"\uABCD"                   // the unsupported codepoints
            u"XYZWRESF";
        constexpr auto expected_size = 24;

        auto res = strf::decode_encode_size
            ( strf::utf<char16_t>, strf::iso_8859_5<char>
              , input.begin(), input.end(), expected_size, flags );

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);

        auto cpcount_res = strf::utf_t<char16_t>::count_codepoints
            ( res.stale_src_ptr, input.end(), res.u32dist
            , strf::surrogate_policy::strict );

        TEST_EQ(cpcount_res.count, res.u32dist);
        TEST_EQ((unsigned)*cpcount_res.ptr, 0xABCD);
        TEST_EQ(24, (cpcount_res.ptr - input.begin()));
        TEST_EQ(res.ssize, expected_size);
    }
    {   // input has invalid sequence and stop_on_invalid_sequence flag is set

        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        str_view input =
            "12345678901234567890"
            "\xA0\xA1\xA2\xFF"
            "\xA5"                  // the invalid sequence
            "XYZWRESF";
        constexpr auto expected_size = 24;

        auto res = strf::decode_encode_size
            ( strf::iso_8859_3<char>, strf::utf<char16_t>, input.begin(), input.end()
            , expected_size, flags);

        TEST_TRUE(res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(24, res.stale_src_ptr - input.begin());
        TEST_EQ(res.ssize, expected_size);
    }
    {   // input has invalid sequence but stop_on_invalid_sequence flag is not set

        const auto flags = strf::transcode_flags::none;
        str_view input =
            "12345678901234567890"
            "\xA0\xA1\xA2\xFF"
            "\xA5"                  // the invalid sequence
            "XYZ";
        constexpr auto expected_size = 28;

        auto res = strf::decode_encode_size
            ( strf::iso_8859_3<char>, strf::utf<char16_t>
            , input.begin(), input.end(), expected_size, flags);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, input.end() - res.stale_src_ptr);
        TEST_EQ(res.ssize, expected_size);
    }
    {   // The input has an invalid sequence and stop_on_invalid_sequence flag is set,
        // but it has also a codepoint that is not supported in the destinations
        // encoding and that comes before the invalid sequence.

        // hence the transcode_stop_reason shall be unsupported_codepoint

        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        str_view input =
            "12345678901234567890"
            "\xC2\xA1" // unsupported codepoint U+A1
            "AB"
            "\xE0\xA0" // the invalid sequence (missing continuation byte)
            "XYZWRESF";
        constexpr auto expected_size = 20;

        auto res = strf::decode_encode_size
            ( strf::utf<char>, strf::iso_8859_3<char>, input.begin(), input.end()
            , expected_size , flags);

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);

        auto cpcount_res = strf::utf8_t<char>::count_codepoints
            ( res.stale_src_ptr, input.end(), res.u32dist
            , strf::surrogate_policy::strict );

        TEST_EQ(cpcount_res.count, res.u32dist);
        TEST_EQ(20, (cpcount_res.ptr - input.begin()));

        TEST_EQ(res.ssize, expected_size);
    }
}


STRF_TEST_FUNC void test_decode_encode_overloads()
{

    constexpr auto src_enc = strf::utf8<char>;
    constexpr auto dst_enc = strf::iso_8859_5<char>;

    const str_view input =
        "abc"
        "\xD0\x81" // codepoint U+0401, which transcodes tp \xA1
        "xyz"
        "\xF1\x80\x80\xE1\x80\xC0" // invalid sequence, transcodes to "???"
        "tsw"
        "\xF0\x9F\x8E\x89" // unupported codepoint U+1F389
        "qwe";
    const auto* const src = input.data();
    const auto src_len = input.ssize();
    const auto* const src_end = src + src_len;

    const str_view expected_non_stop = "abc\xA1xyz???tsw?qwe";
    const str_view expected_stop_on_inv_seq = "abc\xA1xyz";
    const str_view expected_stop_on_unsupported_codepoint = "abc\xA1xyz???tsw";

    const auto flags_stop_inv_seq  = strf::transcode_flags::stop_on_invalid_sequence;
    const auto flags_stop_unsupported_cp = strf::transcode_flags::stop_on_unsupported_codepoint;
  //const auto flags_stop_all = flags_stop_inv_input | flags_stop_inv_output;
    const auto flags_none = strf::transcode_flags::none;

    using stop_reason = strf::transcode_stop_reason;

    // Overload 1 with flags_none
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode
            ( src_enc, dst_enc, src, src_end, buff
            , buff + buff_size, &notifier, flags_none );

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(3, notifier.invalid_sequence_calls_count);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_src_ptr - input.end()));

        str_view output(buff, res.dst_ptr);

        TEST_STR_EQ(output, expected_non_stop);
    }

    // Overload 1 with flags_stop_inv_seq
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode
            ( src_enc, dst_enc, src, src_end, buff, buff + buff_size
            , &notifier, flags_stop_inv_seq );

        TEST_TRUE(res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        str_view output(buff, res.dst_ptr);

        TEST_STR_EQ(output, expected_stop_on_inv_seq);
    }

    // Overload 1 with flags_stop_unsupported_cp
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode
            ( src_enc, dst_enc, src, src_end, buff, buff + buff_size
            , &notifier, flags_stop_unsupported_cp );

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        str_view output(buff, res.dst_ptr);

        TEST_STR_EQ(output, expected_stop_on_unsupported_codepoint);
    }


    // Overload 2 with flags_none
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode<strf::utf8_t, strf::iso_8859_5_t>
            (src, src_end, buff, buff + buff_size, &notifier, flags_none);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(3, notifier.invalid_sequence_calls_count);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_src_ptr - input.end()));

        str_view output(buff, res.dst_ptr);

        TEST_STR_EQ(output, expected_non_stop);
    }

    // Overload 2 with flags_stop_inv_seq
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode<strf::utf8_t, strf::iso_8859_5_t>
            (src, src_end, buff, buff + buff_size, &notifier, flags_stop_inv_seq);

        TEST_TRUE(res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        str_view output(buff, res.dst_ptr);

        TEST_STR_EQ(output, expected_stop_on_inv_seq);
    }

    // Overload 2 with flags_stop_unsupported_cp
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode<strf::utf8_t, strf::iso_8859_5_t>
            (src, src_end, buff, buff + buff_size, &notifier, flags_stop_unsupported_cp);

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        str_view output(buff, res.dst_ptr);

        TEST_STR_EQ(output, expected_stop_on_unsupported_codepoint);
    }



#ifdef STRF_HAS_STD_STRING_VIEW

    std::string_view src_view{src, static_cast<std::size_t>(src_end - src)};

    // Overload 1 with flags_none
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode
            (src_enc, dst_enc, src_view, buff, buff + buff_size, &notifier, flags_none);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(3, notifier.invalid_sequence_calls_count);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_src_ptr - input.end()));

        str_view output(buff, res.dst_ptr);

        TEST_STR_EQ(output, expected_non_stop);
    }

    // Overload 1 with flags_stop_inv_seq
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode
            ( src_enc, dst_enc, src_view, buff, buff + buff_size
            , &notifier, flags_stop_inv_seq);

        TEST_TRUE(res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        str_view output(buff, res.dst_ptr);

        TEST_STR_EQ(output, expected_stop_on_inv_seq);
    }

    // Overload 1 with flags_stop_unsupported_cp
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode
            (src_enc, dst_enc, src_view, buff, buff + buff_size, &notifier, flags_stop_unsupported_cp);

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        str_view output(buff, res.dst_ptr);

        TEST_STR_EQ(output, expected_stop_on_unsupported_codepoint);
    }


    // Overload 2 with flags_none
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode<strf::utf8_t, strf::iso_8859_5_t>
            (src_view, buff, buff + buff_size, &notifier, flags_none);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(3, notifier.invalid_sequence_calls_count);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_src_ptr - input.end()));

        str_view output(buff, res.dst_ptr);

        TEST_STR_EQ(output, expected_non_stop);
    }

    // Overload 2 with flags_stop_inv_seq
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode<strf::utf8_t, strf::iso_8859_5_t>
            (src_view, buff, buff + buff_size, &notifier, flags_stop_inv_seq);

        TEST_TRUE(res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        str_view output(buff, res.dst_ptr);

        TEST_STR_EQ(output, expected_stop_on_inv_seq);
    }

    // Overload 2 with flags_stop_unsupported_cp
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode<strf::utf8_t, strf::iso_8859_5_t>
            (src_view, buff, buff + buff_size, &notifier, flags_stop_unsupported_cp);

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        str_view output(buff, res.dst_ptr);

        TEST_STR_EQ(output, expected_stop_on_unsupported_codepoint);
    }

#endif // STRF_HAS_STD_STRING_VIEW
}

void test_decode_encode_size_overloads()
{
    constexpr auto src_enc = strf::utf8<char>;
    constexpr auto dst_enc = strf::iso_8859_5<char>;

    const str_view input =
        "abc"
        "\xD0\x81" // codepoint U+0401, which transcodes tp \xA1
        "xyz"
        "\xF1\x80\x80\xE1\x80\xC0" // invalid sequence, transcodes to "???"
        "tsw"
        "\xF0\x9F\x8E\x89" // unupported codepoint U+1F389
        "qwe";
    const auto* const src = input.data();
    const auto src_len = input.ssize();
    const auto* const src_end = src + src_len;

    const std::size_t expected_non_stop_size = 17; // "abc\xA1xyz???tsw?qwe";
    const std::size_t expected_stop_on_inv_seq_size = 7; // "abc\xA1xyz";
    const std::size_t expected_stop_on_unsupported_codepoint_size = 13; // "abc\xA1xyz???tsw";

    const auto flags_stop_inv_seq  = strf::transcode_flags::stop_on_invalid_sequence;
    const auto flags_stop_unsupported_cp = strf::transcode_flags::stop_on_unsupported_codepoint;
    const auto flags_none = strf::transcode_flags::none;

    using stop_reason = strf::transcode_stop_reason;

    // Overload 1 with flags_none
    {

        auto res = strf::decode_encode_size
            (src_enc, dst_enc, src, src_end, expected_non_stop_size, flags_none);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_src_ptr - input.end()));

        TEST_EQ(res.ssize, expected_non_stop_size);
    }

    // Overload 1 with flags_stop_inv_seq
    {

        auto res = strf::decode_encode_size
            ( src_enc, dst_enc, src, src_end
            , expected_stop_on_inv_seq_size, flags_stop_inv_seq);

        TEST_TRUE(res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(res.ssize, expected_stop_on_inv_seq_size);
    }

    // Overload 1 with flags_stop_unsupported_cp
    {
        auto res = strf::decode_encode_size
            ( src_enc, dst_enc, src, src_end
            , expected_stop_on_unsupported_codepoint_size
            , flags_stop_unsupported_cp);

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(res.ssize, expected_stop_on_unsupported_codepoint_size);
    }


    // Overload 2 with flags_none
    {
        auto res = strf::decode_encode_size<strf::utf8_t, strf::iso_8859_5_t<char>>
            (src, src_end, expected_non_stop_size, flags_none);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_src_ptr - input.end()));

        TEST_EQ(res.ssize, expected_non_stop_size);
    }

    // Overload 2 with flags_stop_inv_seq
    {
        auto res = strf::decode_encode_size<strf::utf8_t, strf::iso_8859_5_t<char>>
            (src, src_end, expected_stop_on_inv_seq_size, flags_stop_inv_seq);

        TEST_TRUE(res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(res.ssize, expected_stop_on_inv_seq_size);
    }

    // Overload 2 with flags_stop_unsupported_cp
    {

        auto res = strf::decode_encode_size<strf::utf8_t, strf::iso_8859_5_t<char>>
            ( src, src_end
            , expected_stop_on_unsupported_codepoint_size
            , flags_stop_unsupported_cp );

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(res.ssize, expected_stop_on_unsupported_codepoint_size);
    }



#ifdef STRF_HAS_STD_STRING_VIEW

    std::string_view src_view{src, static_cast<std::size_t>(src_end - src)};

    // Overload 1 with flags_none
    {

        auto res = strf::decode_encode_size
            (src_enc, dst_enc, src_view, expected_non_stop_size, flags_none);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_src_ptr - input.end()));
        TEST_EQ(res.ssize, expected_non_stop_size);
    }

    // Overload 1 with flags_stop_inv_seq
    {

        auto res = strf::decode_encode_size
            ( src_enc, dst_enc, src_view
            , expected_stop_on_inv_seq_size, flags_stop_inv_seq);

        TEST_TRUE(res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(res.ssize, expected_stop_on_inv_seq_size);
    }

    // Overload 1 with flags_stop_unsupported_cp
    {

        auto res = strf::decode_encode_size
            ( src_enc, dst_enc, src_view
            , expected_stop_on_unsupported_codepoint_size
            , flags_stop_unsupported_cp);

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(res.ssize, expected_stop_on_unsupported_codepoint_size);
    }


    // Overload 2 with flags_none
    {
        auto res = strf::decode_encode_size<strf::utf8_t, strf::iso_8859_5_t<char>>
            (src_view, expected_non_stop_size, flags_none);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_src_ptr - input.end()));

        TEST_EQ(res.ssize, expected_non_stop_size);
    }

    // Overload 2 with flags_stop_inv_seq
    {

        auto res = strf::decode_encode_size<strf::utf8_t, strf::iso_8859_5_t<char>>
            (src_view, expected_stop_on_inv_seq_size, flags_stop_inv_seq);

        TEST_TRUE(res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(res.ssize, expected_stop_on_inv_seq_size);
    }

    // Overload 2 with flags_stop_unsupported_cp
    {

        auto res = strf::decode_encode_size<strf::utf8_t, strf::iso_8859_5_t<char>>
            ( src_view
            , expected_stop_on_unsupported_codepoint_size
            , flags_stop_unsupported_cp );

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(res.ssize, expected_stop_on_unsupported_codepoint_size);
    }

#endif // STRF_HAS_STD_STRING_VIEW
}

STRF_TEST_FUNC void test_all()
{
    test_decode_encode_scenarios();
    test_decode_encode_size_scenarios();
    test_decode_encode_overloads();
    test_decode_encode_size_overloads();
}

} // namespace

REGISTER_STRF_TEST(test_all)


