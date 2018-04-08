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
        TEST(u"--\u0080--\uD7FF--\uE000--\uFFFF--\U00100000--\U0010FFFF") .exception
            (
                u"--", (char32_t)0x0080,
                u"--", (char32_t)0xD7FF,
                u"--", (char32_t)0xE000,
                u"--", (char32_t)0xFFFF,
                u"--", (char32_t)0x100000,
                u"--", (char32_t)0x10FFFF
            );

    }

    {   // defaul error handling: replace codepoints, by '\uFFFD'
        TEST_RF(u"--\uFFFD--\uFFFD--\uFFFD--\uFFFD--\uFFFD\uFFFD\uFFFD--", 1.5) .exception
            (
                u"--", (char32_t)0xD800,
                u"--", (char32_t)0xDBFF,
                u"--", (char32_t)0xDC00,
                u"--", (char32_t)0xDFFF,
                u"--", strf::multi((char32_t)0x110000, 3),
                u"--", strf::multi((char32_t)0x110000, 0)
            );
    }

    {   // replace invalid codepoints by '?'

        auto err_func = [](auto& ow, auto count) -> bool { return ow.put(count, u'x'); };
        auto facet = strf::make_u16encoder<char16_t>(err_func);

        TEST_RF(u"------x------x------x------x------xxx------", 1.5)
            .facets(facet) .exception
            (
                u"------", (char32_t)0xD800,
                u"------", (char32_t)0xDBFF,
                u"------", (char32_t)0xDC00,
                u"------", (char32_t)0xDFFF,
                u"------", strf::multi((char32_t)0x110000, 3),
                u"------", strf::multi((char32_t)0x110000, 0)
            );
    }

    {   // emit error code on invalid codepoints
        auto err_func = strf::from_utf32_set_error_code<char16_t>;
        auto facet = strf::make_u16encoder<char16_t>(err_func);
        auto expected_error = std::make_error_code(std::errc::illegal_byte_sequence);

        TEST_ERR(u"------", expected_error)
            .facets(facet)
            .exception(u"------", (char32_t)0x110000);
    }

    {  // throw exception on invalid codepoints
        auto err_func = strf::from_utf32_throw<char16_t>;
        auto facet = strf::make_u16encoder<char16_t>(err_func);

        std::exception_ptr eptr;
        try
        {
            auto rstr = strf::make_u16string .facets(facet) .error_code((char32_t)0x110000);
        }
        catch(...)
        {
            eptr = std::current_exception();
        }
        BOOST_TEST(eptr);
    }

    {   // tolerate surrogates
        auto facet = strf::make_u16encoder<char16_t>(true);

        char16_t sample [] =
            {
                u'-', (char16_t)0xD800,
                u'-', (char16_t)0xDBFF,
                u'-', (char16_t)0xDC00,
                u'-', (char16_t)0xDFFF,
                u'\0'
            };

        TEST(sample) .facets(facet) .exception (sample);
    }

#if defined(_WIN32) && ! defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)

    {   // defaul error handling: replace codepoints, by '\uFFFD'
        TEST_RF(L"--\uFFFD--\uFFFD--\uFFFD--\uFFFD--\uFFFD\uFFFD\uFFFD--", 1.5) .exception
            (
                L"--", (char32_t)0xD800,
                L"--", (char32_t)0xDBFF,
                L"--", (char32_t)0xDC00,
                L"--", (char32_t)0xDFFF,
                L"--", strf::multi((char32_t)0x110000, 3),
                L"--", strf::multi((char32_t)0x110000, 0)
            );
    }

    {   // replace invalid codepoints by '?'

        auto err_func = [](auto& ow, auto count) -> bool { return ow.put(count, u'x'); };
        auto facet = strf::make_u16encoder<wchar_t>(err_func);

        TEST_RF(L"------x------x------x------x------xxx------", 1.5)
            .facets(facet) .exception
            (
                L"------", (char32_t)0xD800,
                L"------", (char32_t)0xDBFF,
                L"------", (char32_t)0xDC00,
                L"------", (char32_t)0xDFFF,
                L"------", strf::multi((char32_t)0x110000, 3),
                L"------", strf::multi((char32_t)0x110000, 0)
            );
    }

    {   // emit error code on invalid codepoints
        auto err_func = strf::from_utf32_set_error_code<wchar_t>;
        auto facet = strf::make_u16encoder<wchar_t>(err_func);
        auto expected_error = std::make_error_code(std::errc::illegal_byte_sequence);

        TEST_ERR(L"------", expected_error)
            .facets(facet)
            .exception ( L"------", (char32_t)0x110000 );
    }

    {  // throw exception on invalid codepoints
        auto err_func = strf::from_utf32_throw<wchar_t>;
        auto facet = strf::make_u16encoder<wchar_t>(err_func);

        std::exception_ptr eptr;
        try
        {
            auto rstr = strf::make_wstring .facets(facet) .exception((char32_t)0x110000);
        }
        catch(...)
        {
            eptr = std::current_exception();
        }
        BOOST_TEST(eptr);
    }

    {   // tolerate surrogates
        auto facet = strf::make_u16encoder<wchar_t>().tolerate_surrogates();

        const wchar_t expected [] =
        {
            L'-', (wchar_t)0xD800,
            L'-', (wchar_t)0xDBFF,
            L'-', (wchar_t)0xDC00,
            L'-', (wchar_t)0xDFFF,
            L'\0'
        };

        const char32_t sample [] =
        {
            U'-', (char32_t)0xD800,
            U'-', (char32_t)0xDBFF,
            U'-', (char32_t)0xDC00,
            U'-', (char32_t)0xDFFF,
            U'\0'
        };

        TEST(expected) .facets(facet) .exception(sample);
    }

#endif


    return report_errors() || boost::report_errors();
}
