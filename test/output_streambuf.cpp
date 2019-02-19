//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include "test_utils.hpp"
#include "error_code_emitter_arg.hpp"
#include "exception_thrower_arg.hpp"
#include "streambuf_that_fails_on_overflow.hpp"
#include <boost/stringify.hpp>
#include <sstream>

namespace strf = boost::stringify::v0;

template <typename CharT>
void ec_basic_test()
{
    std::basic_ostringstream<CharT> result;
    std::streamsize result_length = 1000;
    std::basic_string<CharT> expected(50, CharT{'*'});

    auto ec = strf::ec_write(result.rdbuf(), &result_length)(strf::multi(CharT{'*'}, 50));

    BOOST_TEST(ec == std::error_code{});
    BOOST_TEST(static_cast<std::streamsize>(expected.length()) == result_length);
    BOOST_TEST(expected == result.str());
}

#if ! defined(BOOST_NO_EXCEPTION)

template <typename CharT>
void basic_test()
{
    std::basic_ostringstream<CharT> result;
    std::streamsize result_length = 1000;
    std::basic_string<CharT> expected(50, CharT{'*'});
    std::error_code ec;

    try
    {
        strf::write(result.rdbuf(), &result_length)(strf::multi(CharT{'*'}, 50));
    }
    catch(...)
    {
        ec = std::make_error_code(std::errc::not_supported);
    }

    BOOST_TEST(ec == std::error_code{});
    BOOST_TEST(static_cast<std::streamsize>(expected.length()) == result_length);
    BOOST_TEST(expected == result.str());
}

#endif // ! defined(BOOST_NO_EXCEPTION)

int main()
{
    ec_basic_test<char>();
    ec_basic_test<char16_t>();
    ec_basic_test<char32_t>();
    ec_basic_test<wchar_t>();

    {   // When set_error is called

        std::ostringstream result;
        std::streamsize result_length = 1000;

        auto ec = strf::ec_write(result.rdbuf(), &result_length)
            ("abcd", error_code_emitter_arg, "lkjlj");

        BOOST_TEST(ec == std::errc::invalid_argument);
        BOOST_TEST(result.str() == "abcd");
        BOOST_TEST(result_length == 4);
    }

    {   // When exception is thrown

        std::ostringstream result;
        std::streamsize result_length = 1000;

        try
        {
            (void) strf::ec_write(result.rdbuf(), &result_length)
                ("abcd", exception_thrower_arg, "lkjlj");
        }
        catch(...)
        {
        }

        BOOST_TEST(result.str() == "abcd");
        BOOST_TEST(result_length == 4);
    }

    {   // When streambuf::xputn() fails

        streambuf_that_fails_on_overflow<10> result;
        std::streamsize result_length = 1000;

        auto ec = strf::ec_write(result, &result_length)
            (strf::multi('a', 6), "ABCDEF", 'b');

        BOOST_TEST(ec == std::errc::io_error);
        BOOST_TEST(result_length == 10);
        BOOST_TEST(result.str() == "aaaaaaABCD");
    }

#if ! defined(BOOST_NO_EXCEPTION)

    basic_test<char>();
    basic_test<char16_t>();
    basic_test<char32_t>();
    basic_test<wchar_t>();

    {   // When set_error is called

        std::ostringstream result;
        std::streamsize result_length = 1000;
        std::error_code ec;

        try
        {
            strf::write(result.rdbuf(), &result_length)
                ("abcd", error_code_emitter_arg, "lkjlj");
        }
        catch(strf::stringify_error& x)
        {
            ec = x.code();
        }

        BOOST_TEST(ec == std::errc::invalid_argument);
        BOOST_TEST(result.str() == "abcd");
        BOOST_TEST(result_length == 4);
    }

    {   // When exception is thrown

        std::ostringstream result;
        std::streamsize result_length = 1000;

        try
        {
            (void) strf::write(result.rdbuf(), &result_length)
                ("abcd", exception_thrower_arg, "lkjlj");
        }
        catch(...)
        {
        }

        BOOST_TEST(result.str() == "abcd");
        BOOST_TEST(result_length == 4);
    }

    {   // When streambuf::xputn() fails

        streambuf_that_fails_on_overflow<10> result;
        std::streamsize result_length = 1000;
        std::error_code ec;

        try
        {
            strf::write(result, &result_length)
                (strf::multi('a', 6), "ABCDEF", 'b');
        }
        catch(strf::stringify_error& x)
        {
            ec = x.code();
        }

        BOOST_TEST(ec == std::errc::io_error);
        BOOST_TEST(result_length == 10);
        BOOST_TEST(result.str() == "aaaaaaABCD");
    }

#endif // ! defined(BOOST_NO_EXCEPTION)

    return boost::report_errors();
}
