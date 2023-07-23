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
        char buff[200] = {};
        strf::cstr_destination dest(buff);
        ustr_view input = u"abcdef";
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );

        auto deres = strf::decode_encode
            <strf::utf_t, strf::utf_t>
             (input.begin(), input.end(), dest, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::completed);
        TEST_EQ(0, deres.u32dist);
        TEST_EQ(0, (deres.stale_ptr - input.end()));
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        auto fres = dest.finish();
        str_view output(buff, fres.ptr);

        TEST_FALSE(fres.truncated);
        TEST_EQ(output.size(), input.size());
        TEST_STR_EQ(output, "abcdef");
    }


    {   // happy scenario again,
        // but forcing buffered_encoder<DestCharT>::recycle() to be called
        char buff[200] = {};
        strf::cstr_destination dest(buff);
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        ustr_view input = u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

        auto deres = strf::decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dest, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::completed);
        TEST_EQ(0, (deres.stale_ptr - input.end()));
        TEST_EQ(0, deres.u32dist);
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        auto fres = dest.finish();
        str_view output(buff, fres.ptr);

        TEST_EQ(output.size(), input.size());
        TEST_FALSE(fres.truncated);
        TEST_STR_EQ(output, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");
    }

    {   // destination is already bad since the begining
        strf::discarder<char> dest;
        //char tmp;
        //strf::array_destination<char> dest(&tmp, &tmp);
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        ustr_view input = u"abcdefghijklmnopqr";

        auto deres = strf::decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dest, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::bad_destination);
        TEST_EQ(0, (deres.stale_ptr - input.begin()));
        TEST_EQ(0, deres.u32dist);
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);
    }

    {   // all input is valid, but the destination turns bad
        // during the encoding
        constexpr unsigned dest_space = 20;
        char buff[dest_space] = {};
        strf::array_destination<char> dest(buff);
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        ustr_view input = u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

        auto deres = strf::decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dest, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::bad_destination);
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        auto cpcount_res = strf::utf_t<char16_t>::count_codepoints
            ( deres.stale_ptr, input.end(), deres.u32dist
            , strf::surrogate_policy::strict );

        TEST_EQ(cpcount_res.count, deres.u32dist);
        TEST_EQ((char)*cpcount_res.ptr, 'u');
        TEST_EQ(dest_space, (cpcount_res.ptr - input.begin()));

        auto fres = dest.finish();
        TEST_TRUE(fres.truncated);

        str_view output(buff, fres.ptr);
        TEST_EQ(output.size(), dest_space);
        TEST_STR_EQ(output, "abcdefghijklmnopqrst");
    }
    {   // all input is valid, but there is a codepoint that
        // is not supported by the destination encoding
        // and stop_on_unsupported_codepoint() returns true

        char buff[200] = {};
        strf::array_destination<char> dest(buff);
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
            (input.begin(), input.end(), dest, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::unsupported_codepoint);
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        auto cpcount_res = strf::utf_t<char16_t>::count_codepoints
            ( deres.stale_ptr, input.end(), deres.u32dist
            , strf::surrogate_policy::strict );

        TEST_EQ(cpcount_res.count, deres.u32dist);
        TEST_EQ((unsigned)*cpcount_res.ptr, 0xABCD);
        TEST_EQ(30, (cpcount_res.ptr - input.begin()));

        auto fres = dest.finish();
        TEST_FALSE(fres.truncated);

        str_view output(buff, fres.ptr);
        TEST_STR_EQ(output, "abcdefghijklmnopqrstuvwxyz\xA1\xA2\xA3\xFF");

    }
    {   // input has invalid sequence and stop_on_invalid_sequence flag is set

        char16_t buff[200] = {};
        strf::array_destination<char16_t> dest(buff);
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
            (input.begin(), input.end(), dest, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::invalid_sequence);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, deres.u32dist);
        TEST_EQ(30, (char*)notifier.invalid_seq - input.begin());
        TEST_EQ(30, deres.stale_ptr - input.begin());

        auto fres = dest.finish();
        TEST_FALSE(fres.truncated);

        ustr_view output(buff, fres.ptr);
        TEST_STR_EQ(output, u"abcdefghijklmnopqrstuvwxyz\u00A0\u0126\u02D8\u02D9");
    }
    {   // input has invalid sequence but stop_on_invalid_sequence flag is not set

        char16_t buff[200] = {};
        strf::array_destination<char16_t> dest(buff);
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = strf::transcode_flags::none;
        str_view input =
            "abcdefghijklmnopqrstuvwxyz"
            "\xA0\xA1\xA2\xFF"
            "\xA5"                  // the invalid sequence
            "XYZ";

        auto deres = strf::decode_encode
            <strf::iso_8859_3_t, strf::utf_t>
            (input.begin(), input.end(), dest, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::completed);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, deres.u32dist);
        TEST_EQ(30, (char*)notifier.invalid_seq - input.begin());
        TEST_EQ(0, input.end() - deres.stale_ptr);

        auto fres = dest.finish();
        TEST_FALSE(fres.truncated);

        ustr_view output(buff, fres.ptr);
        TEST_STR_EQ(output, u"abcdefghijklmnopqrstuvwxyz\u00A0\u0126\u02D8\u02D9\uFFFDXYZ");
    }
    {   // The input has an invalid sequence and stop_on_invalid_sequence flag is set,
        // but it has also a codepoint that is not supported in the destinations
        // encoding and that comes before the invalid sequence.

        // hence the transcode_stop_reason shall be unsupported_codepoint

        char buff[200] = {};
        strf::array_destination<char> dest(buff);
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
            (input.begin(), input.end(), dest, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::unsupported_codepoint);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(30, (char*)notifier.invalid_seq - input.begin());
        TEST_EQ(0xA1, notifier.unsupported_ch32);

        auto cpcount_res = strf::utf8_t<char>::count_codepoints
            ( deres.stale_ptr, input.end(), deres.u32dist
            , strf::surrogate_policy::strict );

        TEST_EQ(cpcount_res.count, deres.u32dist);
        TEST_EQ(26, (cpcount_res.ptr - input.begin()));

        auto fres = dest.finish();
        TEST_FALSE(fres.truncated);

        str_view output(buff, fres.ptr);
        TEST_STR_EQ(output, "abcdefghijklmnopqrstuvwxyz");

    }
    {   // input has invalid sequence
        // stop_on_invalid_sequence flag is set
        // but destination turns bad before last valid sequence is fully transcoded

        // hence the transcode_stop_reason shall be bad_result
        char16_t buff[200] = {};
        strf::array_destination<char16_t> dest(buff, 20);
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        str_view input =
            "abcdefghijklmnopqrstuvwxyz"
            "ABCD"
            "\xE0\xA0" // the invalid sequence (missing continuation byte)
            "XYZWRESF";

        auto deres = strf::decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dest, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::bad_destination);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        auto cpcount_res = strf::utf8_t<char>::count_codepoints
            ( deres.stale_ptr, input.end(), deres.u32dist
            , strf::surrogate_policy::strict );

        TEST_EQ(cpcount_res.count, deres.u32dist);
        TEST_EQ(20, (cpcount_res.ptr - input.begin()));


        auto fres = dest.finish();
        TEST_TRUE(fres.truncated);

        ustr_view output(buff, fres.ptr);
        TEST_STR_EQ(output, u"abcdefghijklmnopqrst");
    }

    {   // similar as before, but now the destination has just enougth space to
        // transcode all valid sequences
        // In this case, stop_reason is invalid_sequence

        char16_t buff[200] = {};
        strf::array_destination<char16_t> dest(buff, 30);
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        str_view input =
            "abcdefghijklmnopqrstuvwxyzABCD"
            "\xE0\xA0" // the invalid sequence (missing continuation byte)
            "XYZWRESF";

        auto deres = strf::decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dest, &notifier, flags);

        TEST_TRUE(deres.stop_reason == strf::transcode_stop_reason::invalid_sequence);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, deres.u32dist);

        auto fres = dest.finish();
        TEST_FALSE(fres.truncated);

        ustr_view output(buff, fres.ptr);
        TEST_STR_EQ(output, u"abcdefghijklmnopqrstuvwxyzABCD");
    }
}


void  test_decode_encode_size_scenarios()
{
    using stop_reason = strf::transcode_size_stop_reason;

    {   // happy scenario
        // char buff[200] = {};
        // strf::cstr_destination dest(buff);
        ustr_view input = u"abcdef";
        // test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );

        auto res = strf::decode_encode_size
            ( strf::utf<char16_t>, strf::utf<char>, input.begin(), input.end(), flags);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_ptr - input.end()));
        TEST_EQ(res.size, input.size());
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

        auto res = strf::decode_encode_size
            ( strf::utf<char16_t>, strf::iso_8859_5<char>
            , input.begin(), input.end(), flags );

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);

        auto cpcount_res = strf::utf_t<char16_t>::count_codepoints
            ( res.stale_ptr, input.end(), res.u32dist
            , strf::surrogate_policy::strict );

        TEST_EQ(cpcount_res.count, res.u32dist);
        TEST_EQ((unsigned)*cpcount_res.ptr, 0xABCD);
        TEST_EQ(24, (cpcount_res.ptr - input.begin()));
        TEST_EQ(res.size, 24);

    }
    {   // input has invalid sequence and stop_on_invalid_sequence flag is set

        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        str_view input =
            "12345678901234567890"
            "\xA0\xA1\xA2\xFF"
            "\xA5"                  // the invalid sequence
            "XYZWRESF";

        auto res = strf::decode_encode_size
            ( strf::iso_8859_3<char>, strf::utf<char16_t>, input.begin(), input.end(), flags);

        TEST_TRUE(res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(24, res.stale_ptr - input.begin());
        TEST_EQ(res.size, 24);
    }
    {   // input has invalid sequence but stop_on_invalid_sequence flag is not set

        const auto flags = strf::transcode_flags::none;
        str_view input =
            "12345678901234567890"
            "\xA0\xA1\xA2\xFF"
            "\xA5"                  // the invalid sequence
            "XYZ";

        auto res = strf::decode_encode_size
            ( strf::iso_8859_3<char>, strf::utf<char16_t>, input.begin(), input.end(), flags);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, input.end() - res.stale_ptr);
        TEST_EQ(res.size, 28);
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

        auto res = strf::decode_encode_size
            ( strf::utf<char>, strf::iso_8859_3<char>, input.begin(), input.end(), flags);

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);

        auto cpcount_res = strf::utf8_t<char>::count_codepoints
            ( res.stale_ptr, input.end(), res.u32dist
            , strf::surrogate_policy::strict );

        TEST_EQ(cpcount_res.count, res.u32dist);
        TEST_EQ(20, (cpcount_res.ptr - input.begin()));

        TEST_EQ(res.size, 20);
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
    const auto src_len = input.size();
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
        char buff[200] = {};
        strf::array_destination<char> dst(buff);
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode
            (src_enc, dst_enc, src, src_end, dst, &notifier, flags_none);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(3, notifier.invalid_sequence_calls_count);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_ptr - input.end()));

        auto fin_res = dst.finish();
        str_view output(buff, fin_res.ptr);

        TEST_STR_EQ(output, expected_non_stop);
    }

    // Overload 1 with flags_stop_inv_seq
    {
        char buff[200] = {};
        strf::array_destination<char> dst(buff);
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode
            (src_enc, dst_enc, src, src_end, dst, &notifier, flags_stop_inv_seq);

        TEST_TRUE(res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        auto fin_res = dst.finish();
        str_view output(buff, fin_res.ptr);

        TEST_STR_EQ(output, expected_stop_on_inv_seq);
    }

    // Overload 1 with flags_stop_unsupported_cp
    {
        char buff[200] = {};
        strf::array_destination<char> dst(buff);
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode
            (src_enc, dst_enc, src, src_end, dst, &notifier, flags_stop_unsupported_cp);

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        auto fin_res = dst.finish();
        str_view output(buff, fin_res.ptr);

        TEST_STR_EQ(output, expected_stop_on_unsupported_codepoint);
    }


    // Overload 2 with flags_none
    {
        char buff[200] = {};
        strf::array_destination<char> dst(buff);
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode<strf::utf8_t, strf::iso_8859_5_t>
            (src, src_end, dst, &notifier, flags_none);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(3, notifier.invalid_sequence_calls_count);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_ptr - input.end()));

        auto fin_res = dst.finish();
        str_view output(buff, fin_res.ptr);

        TEST_STR_EQ(output, expected_non_stop);
    }

    // Overload 2 with flags_stop_inv_seq
    {
        char buff[200] = {};
        strf::array_destination<char> dst(buff);
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode<strf::utf8_t, strf::iso_8859_5_t>
            (src, src_end, dst, &notifier, flags_stop_inv_seq);

        TEST_TRUE(res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        auto fin_res = dst.finish();
        str_view output(buff, fin_res.ptr);

        TEST_STR_EQ(output, expected_stop_on_inv_seq);
    }

    // Overload 2 with flags_stop_unsupported_cp
    {
        char buff[200] = {};
        strf::array_destination<char> dst(buff);
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode<strf::utf8_t, strf::iso_8859_5_t>
            (src, src_end, dst, &notifier, flags_stop_unsupported_cp);

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        auto fin_res = dst.finish();
        str_view output(buff, fin_res.ptr);

        TEST_STR_EQ(output, expected_stop_on_unsupported_codepoint);
    }



#ifdef STRF_HAS_STD_STRING_VIEW

    std::string_view src_view{src, static_cast<std::size_t>(src_end - src)};

    // Overload 1 with flags_none
    {
        char buff[200] = {};
        strf::array_destination<char> dst(buff);
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode
            (src_enc, dst_enc, src_view, dst, &notifier, flags_none);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(3, notifier.invalid_sequence_calls_count);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_ptr - input.end()));

        auto fin_res = dst.finish();
        str_view output(buff, fin_res.ptr);

        TEST_STR_EQ(output, expected_non_stop);
    }

    // Overload 1 with flags_stop_inv_seq
    {
        char buff[200] = {};
        strf::array_destination<char> dst(buff);
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode
            (src_enc, dst_enc, src_view, dst, &notifier, flags_stop_inv_seq);

        TEST_TRUE(res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        auto fin_res = dst.finish();
        str_view output(buff, fin_res.ptr);

        TEST_STR_EQ(output, expected_stop_on_inv_seq);
    }

    // Overload 1 with flags_stop_unsupported_cp
    {
        char buff[200] = {};
        strf::array_destination<char> dst(buff);
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode
            (src_enc, dst_enc, src_view, dst, &notifier, flags_stop_unsupported_cp);

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        auto fin_res = dst.finish();
        str_view output(buff, fin_res.ptr);

        TEST_STR_EQ(output, expected_stop_on_unsupported_codepoint);
    }


    // Overload 2 with flags_none
    {
        char buff[200] = {};
        strf::array_destination<char> dst(buff);
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode<strf::utf8_t, strf::iso_8859_5_t>
            (src_view, dst, &notifier, flags_none);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(3, notifier.invalid_sequence_calls_count);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_ptr - input.end()));

        auto fin_res = dst.finish();
        str_view output(buff, fin_res.ptr);

        TEST_STR_EQ(output, expected_non_stop);
    }

    // Overload 2 with flags_stop_inv_seq
    {
        char buff[200] = {};
        strf::array_destination<char> dst(buff);
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode<strf::utf8_t, strf::iso_8859_5_t>
            (src_view, dst, &notifier, flags_stop_inv_seq);

        TEST_TRUE(res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(1, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        auto fin_res = dst.finish();
        str_view output(buff, fin_res.ptr);

        TEST_STR_EQ(output, expected_stop_on_inv_seq);
    }

    // Overload 2 with flags_stop_unsupported_cp
    {
        char buff[200] = {};
        strf::array_destination<char> dst(buff);
        test_utils::simple_transcoding_err_notifier notifier;

        auto res = strf::decode_encode<strf::utf8_t, strf::iso_8859_5_t>
            (src_view, dst, &notifier, flags_stop_unsupported_cp);

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        auto fin_res = dst.finish();
        str_view output(buff, fin_res.ptr);

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
    const auto src_len = input.size();
    const auto* const src_end = src + src_len;

    const std::size_t expected_non_stop_size = 17; // "abc\xA1xyz???tsw?qwe";
    const std::size_t expected_stop_on_inv_seq_size = 7; // "abc\xA1xyz";
    const std::size_t expected_stop_on_unsupported_codepoint_size = 13; // "abc\xA1xyz???tsw";

    const auto flags_stop_inv_seq  = strf::transcode_flags::stop_on_invalid_sequence;
    const auto flags_stop_unsupported_cp = strf::transcode_flags::stop_on_unsupported_codepoint;
    const auto flags_none = strf::transcode_flags::none;

    using stop_reason = strf::transcode_size_stop_reason;

    // Overload 1 with flags_none
    {

        auto res = strf::decode_encode_size
            (src_enc, dst_enc, src, src_end, flags_none);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_ptr - input.end()));

        TEST_EQ(res.size, expected_non_stop_size);
    }

    // Overload 1 with flags_stop_inv_seq
    {

        auto res = strf::decode_encode_size
            (src_enc, dst_enc, src, src_end, flags_stop_inv_seq);

        TEST_TRUE(res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(res.size, expected_stop_on_inv_seq_size);
    }

    // Overload 1 with flags_stop_unsupported_cp
    {
        auto res = strf::decode_encode_size
            (src_enc, dst_enc, src, src_end, flags_stop_unsupported_cp);

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(res.size, expected_stop_on_unsupported_codepoint_size);
    }


    // Overload 2 with flags_none
    {
        auto res = strf::decode_encode_size<strf::utf8_t, strf::iso_8859_5_t<char>>
            (src, src_end, flags_none);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_ptr - input.end()));

        TEST_EQ(res.size, expected_non_stop_size);
    }

    // Overload 2 with flags_stop_inv_seq
    {
        auto res = strf::decode_encode_size<strf::utf8_t, strf::iso_8859_5_t<char>>
            (src, src_end, flags_stop_inv_seq);

        TEST_TRUE(res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(res.size, expected_stop_on_inv_seq_size);
    }

    // Overload 2 with flags_stop_unsupported_cp
    {

        auto res = strf::decode_encode_size<strf::utf8_t, strf::iso_8859_5_t<char>>
            (src, src_end, flags_stop_unsupported_cp);

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(res.size, expected_stop_on_unsupported_codepoint_size);
    }



#ifdef STRF_HAS_STD_STRING_VIEW

    std::string_view src_view{src, static_cast<std::size_t>(src_end - src)};

    // Overload 1 with flags_none
    {

        auto res = strf::decode_encode_size
            (src_enc, dst_enc, src_view, flags_none);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_ptr - input.end()));
        TEST_EQ(res.size, expected_non_stop_size);
    }

    // Overload 1 with flags_stop_inv_seq
    {

        auto res = strf::decode_encode_size
            (src_enc, dst_enc, src_view, flags_stop_inv_seq);

        TEST_TRUE(res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(res.size, expected_stop_on_inv_seq_size);
    }

    // Overload 1 with flags_stop_unsupported_cp
    {

        auto res = strf::decode_encode_size
            (src_enc, dst_enc, src_view, flags_stop_unsupported_cp);

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(res.size, expected_stop_on_unsupported_codepoint_size);
    }


    // Overload 2 with flags_none
    {

        auto res = strf::decode_encode_size<strf::utf8_t, strf::iso_8859_5_t<char>>
            (src_view, flags_none);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_ptr - input.end()));

        TEST_EQ(res.size, expected_non_stop_size);
    }

    // Overload 2 with flags_stop_inv_seq
    {

        auto res = strf::decode_encode_size<strf::utf8_t, strf::iso_8859_5_t<char>>
            (src_view, flags_stop_inv_seq);

        TEST_TRUE(res.stop_reason == stop_reason::invalid_sequence);
        TEST_EQ(res.size, expected_stop_on_inv_seq_size);
    }

    // Overload 2 with flags_stop_unsupported_cp
    {

        auto res = strf::decode_encode_size<strf::utf8_t, strf::iso_8859_5_t<char>>
            (src_view, flags_stop_unsupported_cp);

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(res.size, expected_stop_on_unsupported_codepoint_size);
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


