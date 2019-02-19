//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include "test_utils.hpp"
#include "error_code_emitter_arg.hpp"
#include "exception_thrower_arg.hpp"
#include <boost/stringify.hpp>

namespace strf = boost::stringify::v0;

template <typename CharT>
void basic_ec_test()
{
    CharT output[100];
    std::fill(output, output + 100, CharT{'-'});
    std::basic_string<CharT> expected(50, CharT{'*'});
    auto ec = strf::ec_write(output) (strf::multi(CharT{'*'}, 50));

    BOOST_TEST(std::error_code{} == ec);
    BOOST_TEST(expected == output);
}

template <typename CharT>
void ec_test_array_too_small()
{
    CharT buff[3] = { 'a', 'a', 0 };
    auto ec = strf::ec_write(buff) ((CharT)'1', (CharT)'2', (CharT)'3');

    BOOST_TEST(ec == std::errc::result_out_of_range);
    BOOST_TEST(buff[0] == '1');
    BOOST_TEST(buff[1] == '2');
    BOOST_TEST(buff[2] == '\0');
}

template <typename CharT>
void ec_test_informed_size_too_small()
{
    CharT buff[100] = { 'a', 'a', 0 };
    auto ec = strf::ec_write(buff, 3) ((CharT)'1', (CharT)'2', (CharT)'3');

    BOOST_TEST(ec == std::errc::result_out_of_range);
    BOOST_TEST(buff[0] == '1');
    BOOST_TEST(buff[1] == '2');
    BOOST_TEST(buff[2] == '\0');
}

template <typename CharT>
void ec_test_informed_end_too_close()
{
    CharT buff[100] = { 'a', 'a', 0 };
    auto ec = strf::ec_write(buff, &buff[3]) ((CharT)'1', (CharT)'2', (CharT)'3');

    BOOST_TEST(ec == std::errc::result_out_of_range);
    BOOST_TEST(buff[0] == '1');
    BOOST_TEST(buff[1] == '2');
    BOOST_TEST(buff[2] == '\0');
}

#if ! defined(BOOST_NO_EXCEPTION)

template <typename CharT>
void basic_test()
{
    CharT output[100];
    std::fill(output, output + 100, CharT{'-'});
    std::basic_string<CharT> expected(50, CharT{'*'});
    auto len = strf::write(output) (strf::multi(CharT{'*'}, 50));

    BOOST_TEST(expected.length() == len);
    BOOST_TEST(expected == output);
}

template <typename CharT>
void test_array_too_small()
{
    CharT buff[3] = { 'a', 'a', 0 };
    std::error_code ec;
    try
    {
        strf::write(buff) ( 1234567 );
    }
    catch(strf::stringify_error& x)
    {
        ec = x.code();
    }

    BOOST_TEST(ec == std::errc::result_out_of_range);
    BOOST_TEST(buff[0] == 0);
}

template <typename CharT>
void test_informed_size_too_small()
{
    CharT buff[100] = { 'a', 'a', 0 };
    std::error_code ec;
    try
    {
        strf::write(buff, 3) ( 1234567 );
    }
    catch(strf::stringify_error& x)
    {
        ec = x.code();
    }

    BOOST_TEST(ec == std::errc::result_out_of_range);
    BOOST_TEST(buff[0] == 0);
}

template <typename CharT>
void test_informed_end_too_close()
{
    CharT buff[100] = { 'a', 'a', 0 };
    std::error_code ec;
    try
    {
        strf::write(buff, &buff[3]) ( 1234567 );
    }
    catch(strf::stringify_error& x)
    {
        ec = x.code();
    }

    BOOST_TEST(ec == std::errc::result_out_of_range);
    BOOST_TEST(buff[0] == 0);
}

#endif // defined(BOOST_NO_EXCEPTION)

int main()
{
    basic_ec_test<char>();
    basic_ec_test<char16_t>();
    basic_ec_test<char32_t>();
    basic_ec_test<wchar_t>();

    ec_test_array_too_small<char>();
    ec_test_array_too_small<char16_t>();
    ec_test_array_too_small<char32_t>();
    ec_test_array_too_small<wchar_t>();

    ec_test_informed_size_too_small<char>();
    ec_test_informed_size_too_small<char16_t>();
    ec_test_informed_size_too_small<char32_t>();
    ec_test_informed_size_too_small<wchar_t>();

    ec_test_informed_end_too_close<char>();
    ec_test_informed_end_too_close<char16_t>();
    ec_test_informed_end_too_close<char32_t>();
    ec_test_informed_end_too_close<wchar_t>();

    {  // Test ec_char_ptr_writer::set_error

        char16_t result[200] = u"-----------------------------";
        auto ec = strf::ec_write(result)
            (u"abcd", error_code_emitter_arg, u"lkjlj");

        BOOST_TEST(ec == std::errc::invalid_argument);
        BOOST_TEST(std::u16string(u"abcd") == result);
    }

    {  // When exception is thrown

        char16_t result[200] = u"-----------------------------";
        try
        {
            (void) strf::ec_write(result) (u"abcd", exception_thrower_arg, u"lkjlj");
        }
        catch(...)
        {
        }

        BOOST_TEST(std::u16string(u"abcd") == result);
    }

#if ! defined(BOOST_NO_EXCEPTION)

    basic_test<char>();
    basic_test<char16_t>();
    basic_test<char32_t>();
    basic_test<wchar_t>();

    test_array_too_small<char>();
    test_array_too_small<char16_t>();
    test_array_too_small<char32_t>();
    test_array_too_small<wchar_t>();

    test_informed_size_too_small<char>();
    test_informed_size_too_small<char16_t>();
    test_informed_size_too_small<char32_t>();
    test_informed_size_too_small<wchar_t>();

    test_informed_end_too_close<char>();
    test_informed_end_too_close<char16_t>();
    test_informed_end_too_close<char32_t>();
    test_informed_end_too_close<wchar_t>();

    {
        char buff[10];
        auto ec = strf::ec_write(buff) ("01234567890123456789");
        BOOST_TEST(ec == std::errc::result_out_of_range);
        BOOST_TEST_CSTR_EQ(buff, "012345678");
    }
    {
        char buff[10];
        auto ec = strf::ec_write(buff) (strf::left(0, 20, '.'));
        BOOST_TEST(ec == std::errc::result_out_of_range);
        BOOST_TEST_CSTR_EQ(buff, "0........");
    }

    {   // Test char_ptr_writer::set_error

        char16_t result[200] = u"-----------------------------";

        std::error_code ec;
        try
        {
            strf::write(result) (u"abcd", error_code_emitter_arg, u"lkjlj");
        }
        catch(strf::stringify_error& x)
        {
            ec = x.code();
        }

        BOOST_TEST(result[0] == u'\0');
        BOOST_TEST(ec == std::errc::invalid_argument);
    }

    {  // When exception is thrown

        char16_t result[200] = u"-----------------------------";
        try
        {
            strf::write(result) (u"abcd", exception_thrower_arg, u"lkjlj");
        }
        catch(...)
        {
        }

        BOOST_TEST(result[0] == u'\0');
    }


#endif // defined(BOOST_NO_EXCEPTION)

    return boost::report_errors();
}
