//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <boost/detail/lightweight_test.hpp>
#include "test_utils.hpp"
#include "error_code_emitter_arg.hpp"
#include "exception_thrower_arg.hpp"
#include "streambuf_that_fails_on_overflow.hpp"
#include <boost/stringify.hpp>
#include <sstream>

namespace strf = boost::stringify::v0;

template <typename CharT>
void basic_test()
{
    std::basic_ostringstream<CharT> result;
    std::size_t result_length = 1000;
    std::basic_string<CharT> expected;

    auto x = use_all_writing_function_of_output_writer
        ( strf::format(result.rdbuf(), &result_length)
        , expected );

    BOOST_TEST(x);
    BOOST_TEST(expected.length() == result_length);
    BOOST_TEST(expected == result.str());
}


int main()
{
    basic_test<char>();
    basic_test<char16_t>();
    basic_test<char32_t>();
    basic_test<wchar_t>();

    {   // When count is nullptr
        std::basic_ostringstream<char> result;
        std::basic_string<char> expected;

        auto x = use_all_writing_function_of_output_writer
            ( strf::format(result.rdbuf(), nullptr)
            , expected );

        BOOST_TEST(x);
        BOOST_TEST(expected == result.str());
    }

    {   // When set_error is called

        std::ostringstream result;
        std::size_t result_length = 1000;

        auto x = strf::format(result.rdbuf(), &result_length)
            ("abcd", error_code_emitter_arg, "lkjlj");

        BOOST_TEST(!x);
        BOOST_TEST(x.error() == std::errc::invalid_argument);
        BOOST_TEST(result.str() == "abcd");
        BOOST_TEST(result_length == 4);
    }

    {   // When exception is thrown

        std::ostringstream result;
        std::size_t result_length = 1000;

        try
        {
            (void) strf::format(result.rdbuf(), &result_length)
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
        std::size_t result_length = 1000;

        auto x = strf::format(result, &result_length)
            (strf::multi('a', 6), "ABCDEF", 'b');

        BOOST_TEST(!x && x.error() == std::errc::io_error);
        BOOST_TEST(result_length == 10);
        BOOST_TEST(result.str() == "aaaaaaABCD");
    }

    {   // When streambuf::putc() fails

        streambuf_that_fails_on_overflow<10> result;
        std::size_t result_length = 1000;

        auto x = strf::format(result, &result_length)
            ("ABCDEF", strf::multi('a', 6), "ABCDEF");

        BOOST_TEST(!x && x.error() == std::errc::io_error);
        BOOST_TEST(result_length == 10);
        BOOST_TEST(result.str() == "ABCDEFaaaa");
    }

    return report_errors() || boost::report_errors();
}
