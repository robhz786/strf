//  Copyright (C) (See commit logs on github.com/robhz786/strf)
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

STRF_TEST_FUNC void test_unsafe_decode_encode_scenarios()
{
    using stop_reason = strf::transcode_stop_reason;

    {   // happy scenario
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        const ustr_view input = u"abcdef";
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = strf::transcode_flags::stop_on_unsupported_codepoint;

        auto deres = strf::unsafe_decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), buff, buff + buff_size, &notifier, flags);

        TEST_TRUE(deres.stop_reason == stop_reason::completed);
        TEST_EQ(0, deres.u32dist);
        TEST_EQ(0, (deres.stale_src_ptr - input.end()));
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        const str_view output(buff, deres.dst_ptr);

        TEST_EQ(output.size(), input.size());
        TEST_STR_EQ(output, "abcdef");
    }
    {   // happy scenario again,
        // but forcing buffered_encoder<DstCharT>::recycle() to be called
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = strf::transcode_flags::stop_on_unsupported_codepoint;
        const ustr_view input = u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

        auto deres = strf::unsafe_decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), buff, buff + buff_size, &notifier, flags);

        TEST_TRUE(deres.stop_reason == stop_reason::completed);
        TEST_EQ(0, (deres.stale_src_ptr - input.end()));
        TEST_EQ(0, deres.u32dist);
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        const str_view output(buff, deres.dst_ptr);

        TEST_EQ(output.size(), input.size());
        TEST_STR_EQ(output, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");
    }
    {   // destination is already bad since the begining
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = strf::transcode_flags::stop_on_unsupported_codepoint;
        const ustr_view input = u"abcdefghijklmnopqr";
        char* dst = nullptr;

        auto deres = strf::unsafe_decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dst, dst, &notifier, flags);

        TEST_TRUE(deres.stop_reason == stop_reason::insufficient_output_space);
        TEST_EQ(0, (deres.stale_src_ptr - input.begin()));
        TEST_EQ(0, deres.u32dist);
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);
    }
    {   // the destination too small
        constexpr unsigned dest_space = 20;
        char buff[dest_space] = {};
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        const ustr_view input = u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

        auto deres = strf::unsafe_decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), buff, buff + dest_space, &notifier, flags);

        TEST_TRUE(deres.stop_reason == stop_reason::insufficient_output_space);
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(0, notifier.unsupported_codepoints_calls_count);

        auto cpcount_res = strf::utf_t<char16_t>::count_codepoints
            ( deres.stale_src_ptr, input.end(), deres.u32dist );

        TEST_EQ(cpcount_res.count, deres.u32dist);
        TEST_EQ((char)*cpcount_res.ptr, 'u');
        TEST_EQ(dest_space, (cpcount_res.ptr - input.begin()));

        const str_view output(buff, deres.dst_ptr);
        TEST_EQ(output.size(), dest_space);
        TEST_STR_EQ(output, "abcdefghijklmnopqrst");
    }
    {   // there is a codepoint that
        // is not supported by the destination encoding
        // and stop_on_unsupported_codepoint flag is set

        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = strf::transcode_flags::stop_on_unsupported_codepoint;
        const ustr_view input =
            u"abcdefghijklmnopqrstuvwxyz"
            u"\u0401\u0402\u0403\u045F"
            u"\uABCD"                   // the unsupported codepoints
            u"XYZWRESF";

        auto deres = strf::unsafe_decode_encode
            <strf::utf_t, strf::iso_8859_5_t>
            (input.begin(), input.end(), buff, buff + buff_size, &notifier, flags);

        TEST_TRUE(deres.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        auto cpcount_res = strf::utf_t<char16_t>::count_codepoints
            ( deres.stale_src_ptr, input.end(), deres.u32dist );

        TEST_EQ(cpcount_res.count, deres.u32dist);
        TEST_EQ((unsigned)*cpcount_res.ptr, 0xABCD);
        TEST_EQ(30, (cpcount_res.ptr - input.begin()));

        const str_view output(buff, deres.dst_ptr);
        TEST_STR_EQ(output, "abcdefghijklmnopqrstuvwxyz\xA1\xA2\xA3\xFF");
    }
    {   // there is a codepoint that
        // is not supported by the destination encoding
        // but stop_on_unsupported_codepoint flag is set

        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;
        const auto flags = strf::transcode_flags::stop_on_invalid_sequence;
        const ustr_view input =
            u"abcdefghijklmnopqrstuvwxyz"
            u"\u0401\u0402\u0403\u045F"
            u"\uABCD"                   // the unsupported codepoints
            u"XYZ";

        auto deres = strf::unsafe_decode_encode
            <strf::utf_t, strf::iso_8859_5_t>
            (input.begin(), input.end(), buff, buff + buff_size, &notifier, flags);

        TEST_TRUE(deres.stop_reason == stop_reason::completed);
        TEST_EQ(0, deres.u32dist);
        TEST_EQ(0, (deres.stale_src_ptr - input.end()));
        TEST_EQ(0, notifier.invalid_sequence_calls_count);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        const str_view output(buff, deres.dst_ptr);
        TEST_STR_EQ(output, "abcdefghijklmnopqrstuvwxyz\xA1\xA2\xA3\xFF?XYZ");
    }
}

STRF_TEST_FUNC void test_unsafe_decode_encode_overloads()
{
    constexpr auto src_enc = strf::utf16<char16_t>;
    constexpr auto dst_enc = strf::iso_8859_5<char>;

    const ustr_view input = u"abcdef" u"\u0401\u0402" u"\uABCD" u"XYZ";
    const auto* const src = input.data();
    const auto src_len = input.size();
    const auto* const src_end = src + src_len;

    const str_view expected_full = "abcdef\xA1\xA2?XYZ";
    const str_view expected_part = "abcdef\xA1\xA2";

    const auto flags_stop = strf::transcode_flags::stop_on_unsupported_codepoint;
    const auto flags_none = strf::transcode_flags::none;

    using stop_reason = strf::transcode_stop_reason;

    // Overload 1
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto ude_res = strf::unsafe_decode_encode
            (src_enc, dst_enc, src, src_end, buff, buff + buff_size, &notifier, flags_stop);
        TEST_TRUE(ude_res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        const str_view output(buff, ude_res.dst_ptr);
        TEST_STR_EQ(output, expected_part);
    }
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto ude_res = strf::unsafe_decode_encode
            (src_enc, dst_enc, src, src_end, buff, buff + buff_size, &notifier, flags_none);
        TEST_TRUE(ude_res.stop_reason == stop_reason::completed);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        const str_view output(buff, ude_res.dst_ptr);
        TEST_STR_EQ(output, expected_full);
    }


    // Overload 2
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto ude_res = strf::unsafe_decode_encode<strf::utf16_t, strf::iso_8859_5_t>
            (src, src_end, buff, buff + buff_size, &notifier, flags_stop);
        TEST_TRUE(ude_res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        const str_view output(buff, ude_res.dst_ptr);

        TEST_STR_EQ(output, expected_part);
    }
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto ude_res = strf::unsafe_decode_encode<strf::utf16_t, strf::iso_8859_5_t>
            (src, src_end, buff, buff + buff_size, &notifier, flags_none);
        TEST_TRUE(ude_res.stop_reason == stop_reason::completed);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        const str_view output(buff, ude_res.dst_ptr);
        TEST_STR_EQ(output, expected_full);
    }

#ifdef STRF_HAS_STD_STRING_VIEW

    const std::basic_string_view<char16_t> src_sv(src, src_len);

    // Overload 1
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto ude_res = strf::unsafe_decode_encode
            (src_enc, dst_enc, src_sv, buff, buff + buff_size, &notifier, flags_stop);
        TEST_TRUE(ude_res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        const str_view output(buff, ude_res.dst_ptr);
        TEST_STR_EQ(output, expected_part);
    }
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto ude_res = strf::unsafe_decode_encode
            (src_enc, dst_enc, src_sv, buff, buff + buff_size, &notifier, flags_none);
        TEST_TRUE(ude_res.stop_reason == stop_reason::completed);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        const str_view output(buff, ude_res.dst_ptr);
        TEST_STR_EQ(output, expected_full);
    }

    // Overload 2
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto ude_res = strf::unsafe_decode_encode<strf::utf16_t, strf::iso_8859_5_t>
            (src_sv, buff, buff + buff_size, &notifier, flags_stop);
        TEST_TRUE(ude_res.stop_reason == stop_reason::unsupported_codepoint);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        const str_view output(buff, ude_res.dst_ptr);

        TEST_STR_EQ(output, expected_part);
    }
    {
        constexpr auto buff_size = 200;
        char buff[buff_size] = {};
        test_utils::simple_transcoding_err_notifier notifier;

        auto ude_res = strf::unsafe_decode_encode<strf::utf16_t, strf::iso_8859_5_t>
            (src_sv, buff, buff + buff_size, &notifier, flags_none);
        TEST_TRUE(ude_res.stop_reason == stop_reason::completed);
        TEST_EQ(1, notifier.unsupported_codepoints_calls_count);

        const str_view output(buff, ude_res.dst_ptr);
        TEST_STR_EQ(output, expected_full);
    }

#endif // STRF_HAS_STD_STRING_VIEW
}

STRF_TEST_FUNC void test_unsafe_decode_encode_size_scenarios()
{
    using stop_reason = strf::transcode_stop_reason;

    {   // happy scenario
        const ustr_view input = u"abcdef";
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );

        auto res = strf::unsafe_decode_encode_size
            ( strf::utf<char16_t>, strf::utf<char>
            , input.begin(), input.end(), strf::ssize_max, flags );

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_src_ptr - input.end()));
        TEST_EQ(res.ssize, 6);
    }
    {   // happy scenario with limit equal to expected size
        const ustr_view input = u"abcdef";
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );

        auto res = strf::unsafe_decode_encode_size
            ( strf::utf<char16_t>, strf::utf<char>
            , input.begin(), input.end(), 6, flags );

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_src_ptr - input.end()));
        TEST_EQ(res.ssize, 6);
    }
    {
        // limit less then expected size
        const ustr_view input = u"abcdef";
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );

        auto res = strf::unsafe_decode_encode_size
            ( strf::utf<char16_t>, strf::utf<char>
            , input.begin(), input.end(), 5, flags );

        TEST_TRUE(res.stop_reason == stop_reason::insufficient_output_space);
        auto cpcount_res = strf::utf_t<char16_t>::count_codepoints
            ( res.stale_src_ptr, input.end(), res.u32dist );

        TEST_EQ(cpcount_res.count, res.u32dist);
        TEST_EQ(res.ssize, 5);
    }
    {   // there is a codepoint that
        // is not supported by the destination encoding
        // and stop_on_unsupported_codepoint flag is set
        const auto flags = ( strf::transcode_flags::stop_on_invalid_sequence
                           | strf::transcode_flags::stop_on_unsupported_codepoint );
        const ustr_view input =
            u"abcdefghijklmnopqrstuvwxyz"
            u"\u0401\u0402\u0403\u045F"
            u"\uABCD"                   // the unsupported codepoints
            u"XYZWRESF";
        constexpr auto expected_size = 30;

        auto res = strf::unsafe_decode_encode_size
            ( strf::utf<char16_t>, strf::iso_8859_5<char>
            , input.begin(), input.end(), expected_size, flags);

        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);
        auto cpcount_res = strf::utf_t<char16_t>::count_codepoints
            ( res.stale_src_ptr, input.end(), res.u32dist );

        TEST_EQ(cpcount_res.count, res.u32dist);
        TEST_EQ((unsigned)*cpcount_res.ptr, 0xABCD);
        TEST_EQ(expected_size, (cpcount_res.ptr - input.begin()));
    }

    {
        // there is a codepoint that
        // is not supported by the destination encoding
        // but stop_on_unsupported_codepoint flag is NOT set

        const auto flags = strf::transcode_flags::stop_on_invalid_sequence;
        const ustr_view input =
            u"abcdefghijklmnopqrstuvwxyz"
            u"\u0401\u0402\u0403\u045F"
            u"\uABCD"                   // the unsupported codepoints
            u"XYZWRESF";
        const auto expected_size = input.ssize();

        auto res = strf::unsafe_decode_encode_size
            ( strf::utf<char16_t>, strf::iso_8859_5<char>
            , input.begin(), input.end(), expected_size, flags);

        TEST_TRUE(res.stop_reason == stop_reason::completed);
        TEST_EQ(0, res.u32dist);
        TEST_EQ(0, (res.stale_src_ptr - input.end()));
        TEST_EQ(res.ssize, expected_size);
    }
}

STRF_TEST_FUNC void test_unsafe_decode_encode_size_overloads()
{

    constexpr auto src_enc = strf::utf16<char16_t>;
    constexpr auto dst_enc = strf::iso_8859_5<char>;

    const ustr_view input = u"abcdef" u"\u0401\u0402" u"\uABCD" u"XYZ";
    const auto* const src = input.data();
    const auto src_len = input.ssize();
    const auto* const src_end = src + src_len;

    constexpr auto expected_full_size = 12; // "abcdef\xA1\xA2?XYZ";
    constexpr auto expected_part_size = 8;  // "abcdef\xA1\xA2";

    const auto flags_stop = strf::transcode_flags::stop_on_unsupported_codepoint;
    const auto flags_none = strf::transcode_flags::none;

    using stop_reason = strf::transcode_stop_reason;

    // Overload 1
    {
        const auto res = strf::unsafe_decode_encode_size
            (src_enc, dst_enc, src, src_end, expected_part_size, flags_stop);
        TEST_EQ(res.ssize, expected_part_size);
        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);

        auto cpcount_res = strf::utf_t<char16_t>::count_codepoints
            ( res.stale_src_ptr, input.end(), res.u32dist );
        TEST_EQ(res.u32dist, cpcount_res.count);
        TEST_EQ(res.u32dist, (cpcount_res.ptr - input.begin()));
        TEST_EQ((unsigned)*cpcount_res.ptr, 0xABCD);
    }
    {
        const auto res = strf::unsafe_decode_encode_size
            (src_enc, dst_enc, src, src_end, expected_full_size, flags_none);
        TEST_EQ(res.ssize, expected_full_size);
        TEST_EQ(res.u32dist, 0);
        TEST_EQ(res.u32dist, 0);
    }

    // Overload 2
    {
        const auto res = strf::unsafe_decode_encode_size
            (src_enc, dst_enc, src, src_end, expected_part_size, flags_stop);
        TEST_EQ(res.ssize, expected_part_size);
        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);

        auto cpcount_res = strf::utf_t<char16_t>::count_codepoints
            ( res.stale_src_ptr, input.end(), res.u32dist );
        TEST_EQ(res.u32dist, cpcount_res.count);
        TEST_EQ(res.u32dist, (cpcount_res.ptr - input.begin()));
        TEST_EQ((unsigned)*cpcount_res.ptr, 0xABCD);
    }
    {
        const auto res = strf::unsafe_decode_encode_size
            (src_enc, dst_enc, src, src_end, expected_full_size, flags_none);
        TEST_EQ(res.ssize, expected_full_size);
    }

#ifdef STRF_HAS_STD_STRING_VIEW

    const std::basic_string_view<char16_t> src_sv(src, src_len);

    // Overload 1
    {
        const auto res = strf::unsafe_decode_encode_size
            (src_enc, dst_enc, src_sv, expected_part_size, flags_stop);
        TEST_EQ(res.ssize, expected_part_size);
        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);

        auto cpcount_res = strf::utf_t<char16_t>::count_codepoints
            ( res.stale_src_ptr, input.end(), res.u32dist );
        TEST_EQ(res.u32dist, cpcount_res.count);
        TEST_EQ(res.u32dist, (cpcount_res.ptr - input.begin()));
        TEST_EQ((unsigned)*cpcount_res.ptr, 0xABCD);
    }
    {
        const auto res = strf::unsafe_decode_encode_size
            (src_enc, dst_enc, src_sv, expected_full_size, flags_none);
        TEST_EQ(res.ssize, expected_full_size);
    }

    // Overload 2
    {
        const auto res = strf::unsafe_decode_encode_size
            (src_enc, dst_enc, src_sv, expected_part_size, flags_stop);
        TEST_EQ(res.ssize, expected_part_size);
        TEST_TRUE(res.stop_reason == stop_reason::unsupported_codepoint);

        auto cpcount_res = strf::utf_t<char16_t>::count_codepoints
            ( res.stale_src_ptr, input.end(), res.u32dist );
        TEST_EQ(res.u32dist, cpcount_res.count);
        TEST_EQ(res.u32dist, (cpcount_res.ptr - input.begin()));
        TEST_EQ((unsigned)*cpcount_res.ptr, 0xABCD);
    }
    {
        const auto res = strf::unsafe_decode_encode_size
            (src_enc, dst_enc, src_sv, expected_full_size, flags_none);
        TEST_EQ(res.ssize, expected_full_size);
    }

#endif // defined(STRF_HAS_STD_STRING_VIEW)
}

STRF_TEST_FUNC void test_all()
{
    test_unsafe_decode_encode_scenarios();
    test_unsafe_decode_encode_overloads();
    test_unsafe_decode_encode_size_scenarios();
    test_unsafe_decode_encode_size_overloads();
}

} // namespace

REGISTER_STRF_TEST(test_all)

