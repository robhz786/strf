
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

STRF_TEST_FUNC void test_decode_encode()
{

    // to-do

    {   // Force calling buffered_encoder<DestCharT>::recycle()
        char buff[200] = {};
        strf::cstr_destination dest(buff);
        ustr_view input = u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

        strf::decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dest);

        auto res = dest.finish();

        TEST_EQ(res.ptr - buff, 52);
        TEST_FALSE(res.truncated);
        TEST_CSTR_EQ(buff, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");
    }
    {   // Force calling buffered_size_calculator<DestCharT>::recycle()
        ustr_view input = u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        auto size = strf::decode_encode_size<strf::utf_t, strf::utf_t<char>>
            (input.begin(), input.end());

        TEST_EQ(size, 52);
    }
    {   // When destination is too small
        char8_t buff[200] = {};
        strf::basic_cstr_destination<char8_t> dest(buff, 10);
        ustr_view input = u"abcd\U00010000\U0010FFFF";

        strf::decode_encode
            <strf::utf_t, strf::utf_t>
            (input.begin(), input.end(), dest);
        auto res = dest.finish();
        TEST_EQ(res.ptr - buff, 8);
        TEST_TRUE(res.truncated);
    }
    {
        ustr_view input = u"abc\uAAAAzzz\uBBBBxxx";
        auto size = strf::decode_encode_size <strf::utf_t, strf::iso_8859_3_t<char>>
            (input.begin(), input.end());

        TEST_EQ(size, input.size());
    }

#ifdef STRF_HAS_STD_STRING_VIEW

    {
        char8_t buff[200] = {};
        strf::basic_cstr_destination<char8_t> dest(buff);
        strf::decode_encode
            <strf::utf_t, strf::utf_t>
            (u"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF"sv, dest);
        auto res = dest.finish();

        TEST_EQ(res.ptr - buff, 18);
        TEST_FALSE(res.truncated)
        TEST_CSTR_EQ(buff, u8"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF");
    }
    {
        auto size = strf::decode_encode_size
            <strf::utf_t, strf::utf_t<char8_t>>
            (u"ab\u0080\u0800\uD7FF\U00010000\U0010FFFF"sv);

        TEST_EQ(size, 18);
    }
    {
        char buff[200] = {};
        strf::cstr_destination dest(buff);
        errors_counter counter;

        strf::decode_encode <strf::utf_t, strf::iso_8859_3_t>
            (u"abc\uAAAAzzz\uBBBBxxx"sv, dest, &counter);

        auto res = dest.finish();

        TEST_EQ(res.ptr - buff, 11);
        TEST_FALSE(res.truncated)
        TEST_CSTR_EQ(buff, "abc?zzz?xxx");
        TEST_EQ(counter.count, 2);
    }
    {
        auto size = strf::decode_encode_size <strf::utf_t, strf::iso_8859_3_t<char>>
            (u"abc\uAAAAzzz\uBBBBxxx"sv);

        TEST_EQ(size, 11);
    }

#endif // STRF_HAS_STD_STRING_VIEW

}

STRF_TEST_FUNC void test_transcode()
{
#ifdef STRF_HAS_STD_STRING_VIEW

    {
        char8_t buff[200] = {};
        strf::basic_cstr_destination<char8_t> dest(buff);

        strf::do_transcode<strf::utf_t, strf::utf_t>
            (u"abc\uAAAAzzz\uBBBBxxx"sv, dest);

        TEST_CSTR_EQ(buff, u8"abc\uAAAAzzz\uBBBBxxx");
    }
    {
        auto size = strf::transcode_size <strf::utf_t, strf::utf_t<char>>(u"hello"sv);

        TEST_EQ(size, 5);
    }
    {
        char buff[200] = {};
        strf::cstr_destination dest(buff);
        errors_counter counter;

        strf::do_transcode<strf::utf_t, strf::iso_8859_3_t>
            (u"abc\uAAAAzzz\uBBBBxxx"sv, dest, &counter);

        TEST_CSTR_EQ(buff, "abc?zzz?xxx");
        TEST_EQ(counter.count, 2);
    }
    {
        auto size = strf::transcode_size <strf::utf_t, strf::iso_8859_3_t<char>>
            (u"abc\uAAAAzzz\uBBBBxxx"sv);

        TEST_EQ(size, 11);
    }

#endif // STRF_HAS_STD_STRING_VIEW
}

} // namespace


STRF_TEST_FUNC void test_do_transcode()
{
    test_decode_encode();
    test_transcode();
}

REGISTER_STRF_TEST(test_do_transcode)
