#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include "test_utils.hpp"

#define TEST(EXPECTED) make_tester((EXPECTED), __FILE__, __LINE__)

#define TEST_RF(EXPECTED, RF) make_tester((EXPECTED), __FILE__, __LINE__, RF)

#define TEST_ERR(EXPECTED, ERR) make_tester((EXPECTED), __FILE__, __LINE__, ERR)

#define TEST_ERR_RF(EXPECTED, ERR, RF) make_tester((EXPECTED), __FILE__, __LINE__, ERR, RF)

namespace strf = boost::stringify::v0;

bool emit_illegal_byte_sequence(strf::u32output& rec)
{
    rec.set_error(std::make_error_code(std::errc::illegal_byte_sequence));
    return false;
}


void char16_tests()
{
    // basic sample
    TEST(U"--\u0080--\uD7FF--\uE000--\uFFFF--\U00100000--\U0010FFFF") .exception
        (
            u"--\u0080--\uD7FF--\uE000--\uFFFF"
            u"--\U00100000--\U0010FFFF"
        );

    TEST(U"--\u0080--\uD7FF--\uE000--\uFFFF--\U00100000--\U0010FFFF")
        .facets(strf::lax_u16decoder<char16_t>{}) .exception
        (
            u"--\u0080--\uD7FF--\uE000--\uFFFF"
            u"--\U00100000--\U0010FFFF"
        );


    const char16_t sample_with_alone_surrogates[] =
        {
            u' ', 0xD800,
            u' ', 0xD800,
            u' ', 0xDBFF,
            u' ', 0xDC00,
            u' ', 0xDFFF,
            u' ', 0x0
        };


    // defaul error handling: replace invalid codepoints, by '\uFFFD'
    TEST(U" \uFFFD \uFFFD \uFFFD \uFFFD \uFFFD ") .exception
        (sample_with_alone_surrogates);

    // allowing alone surrogates
    {
        auto result = strf::make_u32string
            .facets(strf::lax_u16decoder<char16_t>{}) .exception
            (sample_with_alone_surrogates);

        BOOST_TEST(result[1] == 0xD800);
        BOOST_TEST(result[3] == 0xD800);
        BOOST_TEST(result[5] == 0xDBFF);
        BOOST_TEST(result[7] == 0xDC00);
        BOOST_TEST(result[9] == 0xDFFF);
    }

    //customize error handling function
    {
        auto errcond = std::errc::illegal_byte_sequence;
        auto err = std::make_error_code(errcond);
        auto err_hndl_func = [=](strf::u32output& out) -> bool
            {
                out.set_error(err);
                return false;
            };

        TEST_ERR(U" ", err)
            .facets(strf::make_u16decoder<char16_t>(err_hndl_func))
            .exception(sample_with_alone_surrogates);
    }

    {   // emit error code on invalid sequece
        auto expected_error = std::make_error_code(std::errc::illegal_byte_sequence);

        TEST_ERR(U"blah ", expected_error)
            .facets(strf::make_u16decoder<char16_t>(emit_illegal_byte_sequence))
            .exception( u"blah", sample_with_alone_surrogates, u"blah");
    }

}


void wchar_tests()
{
#if defined(_WIN32) && ! defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)

    // basic sample
    TEST(U"--\u0080--\uD7FF--\uE000--\uFFFF--\U00100000--\U0010FFFF") .exception
        (L"--\u0080--\uD7FF--\uE000--\uFFFF--\U00100000--\U0010FFFF");

    // basic sample
    TEST(U"--\u0080--\uD7FF--\uE000--\uFFFF--\U00100000--\U0010FFFF")
        .facets(strf::make_u16decoder<wchar_t>()) .exception
        (L"--\u0080--\uD7FF--\uE000--\uFFFF--\U00100000--\U0010FFFF");


    const wchar_t sample_with_alone_surrogates[] =
    {
        L' ', 0xD800,
        L' ', 0xD800,
        L' ', 0xDBFF,
        L' ', 0xDC00,
        L' ', 0xDFFF,
        L' ', L'\0'
    };


    // defaul error handling: replace invalid codepoints, by '\uFFFD'
    TEST(U" \uFFFD \uFFFD \uFFFD \uFFFD \uFFFD ")
        .exception(sample_with_alone_surrogates);

    // allowing alone surrogates
    {
        auto result = strf::make_u32string
            .facets(strf::lax_u16decoder<wchar_t>{})
            .exception(sample_with_alone_surrogates);

        BOOST_TEST(result[1] == 0xD800);
        BOOST_TEST(result[3] == 0xD800);
        BOOST_TEST(result[5] == 0xDBFF);
        BOOST_TEST(result[7] == 0xDC00);
        BOOST_TEST(result[9] == 0xDFFF);
    }

    //customize error handling function
    {
        auto errcond = std::errc::illegal_byte_sequence;
        auto err = std::make_error_code(errcond);
        auto err_hndl_func = [=](strf::u32output& out) -> bool
        {
            out.set_error(err);
            return false;
        };

        TEST_ERR(U" ", err)
            .facets(strf::make_u16decoder<wchar_t>(err_hndl_func))
            .error_code(sample_with_alone_surrogates);
    }

#endif // defined(_WIN32) && ! defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)

}


int main()
{
    char16_tests();
    wchar_tests();

    return report_errors() || boost::report_errors();
}
