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

STRF_TEST_FUNC void test_decode_encode_to_dst()
{
    using stop_reason = strf::transcode_stop_reason;

    {   // happy scenario
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        strf::array_destination<char> dst(buff);

        const str_view input = "abcdef";

        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );

        auto deres = strf::decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dst, &notifier, flags);

        TEST_TRUE(deres.stop_reason == stop_reason::completed);
        TEST_EQ(0, deres.u32dist);
        TEST_EQ(0, (deres.stale_src_ptr - input.end()));
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        const str_view output(buff, dst.finish().ptr);
        TEST_STR_EQ(output, input);
        TEST_EQ(output.ssize(), deres.ssize);
    }

    {   // happy scenario again, but with recycle() being called
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::array_destination_with_sub_initial_space<char> dst(buff, buff_size);
        dst.reset_with_initial_space(30);

        const str_view input = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );

        auto deres = strf::decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dst, &notifier, flags);

        TEST_TRUE(deres.stop_reason == stop_reason::completed);
        TEST_EQ(0, deres.u32dist);
        TEST_EQ(0, (deres.stale_src_ptr - input.end()));
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(1, dst.recycle_calls_count());

        const str_view output = dst.finish();
        TEST_STR_EQ(output, input);
        TEST_EQ(output.ssize(), deres.ssize);
    }

    { // When destination turns bad
        constexpr auto buff_size = 20;
        char buff[buff_size] = {};
        strf::array_destination<char> dst(buff);

        const ustr_view input = u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );

        auto deres = strf::decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dst, &notifier, flags);

        TEST_TRUE(deres.stop_reason == stop_reason::insufficient_output_space);
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        auto cpcount_res = strf::utf_t<char16_t>::count_codepoints
            ( deres.stale_src_ptr, input.end(), deres.u32dist );

        TEST_EQ(cpcount_res.count, deres.u32dist);
        TEST_EQ((char)*cpcount_res.ptr, 'u');
        TEST_EQ(buff_size, (cpcount_res.ptr - input.begin()));

        auto dst_res = dst.finish();
        TEST_TRUE(dst_res.truncated);

        const str_view output(buff, dst_res.ptr);
        TEST_EQ(output.size(), buff_size);
        TEST_EQ(output.size(), deres.ssize);
        TEST_STR_EQ(output, "abcdefghijklmnopqrst");
    }

    {   // all input is valid, but there is a codepoint that
        // is not supported by the destination encoding
        // and stop_on_unsupported_codepoint() returns true

        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        strf::array_destination<char> dst(buff);

        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        const ustr_view input =
            u"abcdefghijklmnopqrstuvwxyz"
            u"\u0401\u0402\u0403\u045F"
            u"\uABCD"                   // the unsupported codepoints
            u"XYZWRESF";

        auto deres = strf::decode_encode
            <strf::utf_t, strf::iso_8859_5_t>
            (input.begin(), input.end(), dst, &notifier, flags);

        TEST_TRUE(deres.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        auto cpcount_res = strf::utf_t<char16_t>::count_codepoints
            ( deres.stale_src_ptr, input.end(), deres.u32dist );

        TEST_EQ(cpcount_res.count, deres.u32dist);
        TEST_EQ((unsigned)*cpcount_res.ptr, 0xABCD);
        TEST_EQ(30, (cpcount_res.ptr - input.begin()));

        auto dst_res = dst.finish();
        TEST_FALSE(dst_res.truncated);

        const str_view output(buff, dst_res.ptr);
        TEST_STR_EQ(output, "abcdefghijklmnopqrstuvwxyz\xA1\xA2\xA3\xFF");
        TEST_EQ(output.ssize(), deres.ssize);
    }
    {   // input has invalid sequence and stop_on_invalid_sequence flag is set

        constexpr auto buff_size = 200;
        char16_t buff[buff_size] = {};
        strf::array_destination<char16_t> dst(buff);

        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        const str_view input =
            "abcdefghijklmnopqrstuvwxyz"
            "\xA0\xA1\xA2\xFF"
            "\xA5"                  // the invalid sequence
            "XYZWRESF";

        auto deres = strf::decode_encode
            <strf::iso_8859_3_t, strf::utf_t>
            (input.begin(), input.end(), dst, &notifier, flags);

        TEST_TRUE(deres.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, deres.u32dist);
        TEST_EQ(30, reinterpret_cast<const char*>(notifier.invalid_seq) - input.begin());
        TEST_EQ(30, deres.stale_src_ptr - input.begin());

        auto dst_res = dst.finish();
        TEST_FALSE(dst_res.truncated);

        const ustr_view output(buff, dst_res.ptr);
        TEST_STR_EQ(output, u"abcdefghijklmnopqrstuvwxyz\u00A0\u0126\u02D8\u02D9");
        TEST_EQ(output.ssize(), deres.ssize);
    }
    {   // input has invalid sequence but stop_on_invalid_sequence flag is not set

        constexpr auto buff_size = 200;
        char16_t buff[buff_size] = {};
        strf::array_destination<char16_t> dst(buff);

        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = strf::transcode_flags::none;
        const str_view input =
            "abcdefghijklmnopqrstuvwxyz"
            "\xA0\xA1\xA2\xFF"
            "\xA5"                  // the invalid sequence
            "XYZ";

        auto deres = strf::decode_encode
            <strf::iso_8859_3_t, strf::utf_t>
            (input.begin(), input.end(), dst, &notifier, flags);

        TEST_TRUE(deres.stop_reason == stop_reason::completed);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, deres.u32dist);
        TEST_EQ(30, reinterpret_cast<const char*>(notifier.invalid_seq) - input.begin());
        TEST_EQ(0, deres.stale_src_ptr - input.end());

        auto dst_res = dst.finish();
        TEST_FALSE(dst_res.truncated);

        const ustr_view output(buff, dst_res.ptr);
        TEST_STR_EQ(output, u"abcdefghijklmnopqrstuvwxyz\u00A0\u0126\u02D8\u02D9\uFFFDXYZ");
        TEST_EQ(output.ssize(), deres.ssize);
    }
    {   // The input has an invalid sequence and stop_on_invalid_sequence flag is set,
        // but it has also a codepoint that is not supported in the destinations
        // encoding and that comes before the invalid sequence.

        // hence the transcode_stop_reason shall be unsupported_codepoint

        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        strf::array_destination<char> dst(buff);
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );

        const str_view input =
            "abcdefghijklmnopqrstuvwxyz"
            "\xC2\xA1" // unsupported codepoint U+A1
            "AB"
            "\xE0\xA0" // the invalid sequence (missing continuation byte)
            "XYZWRESF";

        auto deres = strf::decode_encode
            <strf::utf_t, strf::iso_8859_3_t>
            (input.begin(), input.end(), dst, &notifier, flags);

        TEST_TRUE(deres.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(30, reinterpret_cast<const char*>(notifier.invalid_seq) - input.begin());
        TEST_EQ(0xA1, notifier.unsupported_ch32);

        auto cpcount_res = strf::utf8_t<char>::count_codepoints
            ( deres.stale_src_ptr, input.end(), deres.u32dist );

        TEST_EQ(cpcount_res.count, deres.u32dist);
        TEST_EQ(26, (cpcount_res.ptr - input.begin()));

        auto dst_res = dst.finish();
        TEST_FALSE(dst_res.truncated);
        const str_view output(buff, dst_res.ptr);
        TEST_STR_EQ(output, "abcdefghijklmnopqrstuvwxyz");
        TEST_EQ(output.ssize(), deres.ssize);
    }
    {   // There is an invalid sequence, and stop_on_invalid_sequence flag is set,
        // but now the destination has just enougth space to
        // transcode all valid sequences
        // In this case, stop_reason is invalid_sequence

        constexpr auto buff_size = 200;
        char16_t buff[buff_size] = {};
        strf::array_destination<char16_t> dst(buff);
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        const str_view input =
            "abcdefghijklmnopqrstuvwxyzABCD"
            "\xE0\xA0" // the invalid sequence (missing continuation byte)
            "XYZWRESF";

        auto deres = strf::decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dst, &notifier, flags);

        TEST_TRUE(deres.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, deres.u32dist);

        auto dst_res = dst.finish();
        TEST_FALSE(dst_res.truncated);
        const ustr_view output(buff, dst_res.ptr);
        TEST_STR_EQ(output, u"abcdefghijklmnopqrstuvwxyzABCD");
        TEST_EQ(output.ssize(), deres.ssize);
    }
    {   // Same as before, but the destination size is zero
        char16_t buff[1] = {};
        strf::array_destination<char16_t> dst(buff, 0);
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        const str_view input =
            "\xE0\xA0" // the invalid sequence (missing continuation byte)
            "XYZWRESF";

        auto deres = strf::decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dst, &notifier, flags);

        TEST_TRUE(deres.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, deres.u32dist);

        auto dst_res = dst.finish();
        TEST_FALSE(dst_res.truncated);
        TEST_TRUE(dst_res.ptr == buff)
        TEST_EQ(deres.ssize, 0);
    }
    {   // Same as before, but the destination is bad since the beginning
        // Now, the stop_reason is insufficient_output_space
        strf::discarder<char> dst;
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        const str_view input =
            "\xE0\xA0" // the invalid sequence (missing continuation byte)
            "XYZWRESF";

        auto deres = strf::decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dst, &notifier, flags);

        TEST_TRUE(deres.stop_reason == stop_reason::insufficient_output_space);
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, deres.u32dist);
    }
}

} // namespace

REGISTER_STRF_TEST(test_decode_encode_to_dst)

