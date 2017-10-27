#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"

#define TEST(EXPECTED) make_tester((EXPECTED), __FILE__, __LINE__)

#define TEST_RF(EXPECTED, RF) make_tester((EXPECTED), __FILE__, __LINE__, RF)

#define TEST_ERR(EXPECTED, ERR) make_tester((EXPECTED), __FILE__, __LINE__, ERR)

#define TEST_ERR_RF(EXPECTED, ERR, RF) make_tester((EXPECTED), __FILE__, __LINE__, ERR, RF)

namespace strf = boost::stringify::v0;

int main()
{
    {
        TEST(u"--\u0080--\uD7FF--\uE000--\uFFFF--\U00100000--\U0010FFFF") =
            {
                u"--", (char32_t)0x0080,
                u"--", (char32_t)0xD7FF,
                u"--", (char32_t)0xE000,
                u"--", (char32_t)0xFFFF,
                u"--", (char32_t)0x100000,
                u"--", (char32_t)0x10FFFF,
            };

    }

    {   // defaul error handling: replace codepoints, by '\uFFFD'
        TEST_RF(u"--\uFFFD--\uFFFD--\uFFFD--\uFFFD--\uFFFD\uFFFD\uFFFD--", 1.5) =
            {
                u"--", (char32_t)0xD800,
                u"--", (char32_t)0xDBFF,
                u"--", (char32_t)0xDC00,
                u"--", (char32_t)0xDFFF,
                u"--", {(char32_t)0x110000, {"", 3}},
                u"--", {(char32_t)0x110000, {"", 0}}
            };
    }

    {   // replace invalid codepoints by '?'

        auto err_func = [](auto& ow, auto count) -> bool { return ow.repeat(count, u'x'); };
        auto facet = strf::make_u16encoder<char16_t>(err_func);

        TEST_RF(u"------x------x------x------x------xxx------", 1.5)
            .with(facet) =
            {
                u"------", (char32_t)0xD800,
                u"------", (char32_t)0xDBFF,
                u"------", (char32_t)0xDC00,
                u"------", (char32_t)0xDFFF,
                u"------", {(char32_t)0x110000, {"", 3}},
                u"------", {(char32_t)0x110000, {"", 0}}
            };
    }

    {   // emit error code on invalid codepoints
        auto err_func = strf::from_utf32_set_error_code<char16_t>;
        auto facet = strf::make_u16encoder<char16_t>(err_func);
        auto expected_error = std::make_error_code(std::errc::illegal_byte_sequence);

        TEST_ERR(u"------", expected_error)
            .with(facet)
            = { u"------", (char32_t)0x110000 };
    }

    {  // throw exception on invalid codepoints
        auto err_func = strf::from_utf32_throw<char16_t>;
        auto facet = strf::make_u16encoder<char16_t>(err_func);

        std::exception_ptr eptr;
        try
        {
            auto rstr = strf::make_u16string .with(facet) = { (char32_t)0x110000 };
        }
        catch(...)
        {
            eptr = std::current_exception();
        }
        BOOST_TEST(eptr);
    }

    {   // tolerate surrogates
        auto facet = strf::make_u16encoders().tolerate_surrogates();

        char16_t sample [] =
            {
                u'-', (char16_t)0xD800,
                u'-', (char16_t)0xDBFF,
                u'-', (char16_t)0xDC00,
                u'-', (char16_t)0xDFFF
            };

        TEST(sample) .with(facet) = {sample};
    }

#if defined(_WIN32) && ! defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)

    {   // defaul error handling: replace codepoints, by '\uFFFD'
        TEST_RF(L"--\uFFFD--\uFFFD--\uFFFD--\uFFFD--\uFFFD\uFFFD\uFFFD--", 1.5) =
            {
                L"--", (char32_t)0xD800,
                L"--", (char32_t)0xDBFF,
                L"--", (char32_t)0xDC00,
                L"--", (char32_t)0xDFFF,
                L"--", {(char32_t)0x110000, {"", 3}},
                L"--", {(char32_t)0x110000, {"", 0}}
            };
    }

    {   // replace invalid codepoints by '?'

        auto err_func = [](auto& ow, auto count){ ow.repeat(count, u'x'); };
        auto facet = strf::make_u16encoder<wchar_t>(err_func);

        TEST_RF(L"------x------x------x------x------xxx------", 1.5)
            .with(facet) =
            {
                L"------", (char32_t)0xD800,
                L"------", (char32_t)0xDBFF,
                L"------", (char32_t)0xDC00,
                L"------", (char32_t)0xDFFF,
                L"------", {(char32_t)0x110000, {"", 3}},
                L"------", {(char32_t)0x110000, {"", 0}}
            };
    }

    {   // emit error code on invalid codepoints
        auto err_func = strf::from_utf32_set_error_code<wchar_t>;
        auto facet = strf::make_u16encoder<wchar_t>(err_func);
        auto expected_error = std::make_error_code(std::errc::illegal_byte_sequence);

        TEST_ERR(L"------", expected_error)
            .with(facet)
            = { L"------", (char32_t)0x110000 };
    }

    {  // throw exception on invalid codepoints
        auto err_func = strf::from_utf32_throw<wchar_t>;
        auto facet = strf::make_u16encoder<wchar_t>(err_func);

        std::exception_ptr eptr;
        try
        {
            auto rstr = strf::make_u16string .with(facet) = { (char32_t)0x110000 };
        }
        catch(...)
        {
            eptr = std::current_exception();
        }
        BOOST_TEST(eptr);
    }

    {   // tolerate surrogates
        auto facet = strf::make_u16encoder().tolerate_surrogates();

        wchar_t sample [] =
            {
                u'-', (wchar_t)0xD800,
                u'-', (wchar_t)0xDBFF,
                u'-', (wchar_t)0xDC00,
                u'-', (wchar_t)0xDFFF
            };

        TEST(sample) .with(facet) = {sample};
    }

#endif


    return report_errors();
}
