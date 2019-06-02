//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include "test_utils.hpp"
#include "exception_thrower_arg.hpp"
#include "streambuf_that_fails_on_overflow.hpp"
#include <boost/stringify.hpp>
#include <sstream>

namespace strf = boost::stringify::v0;

template <typename CharT>
void basic_test()
{
    std::basic_ostringstream<CharT> result;
    std::streamsize result_length = 1000;
    std::basic_string<CharT> expected(50, CharT{'*'});

    strf::write(result.rdbuf(), &result_length)(strf::multi(CharT{'*'}, 50));

    BOOST_TEST(static_cast<std::streamsize>(expected.length()) == result_length);
    BOOST_TEST(expected == result.str());
}

int main()
{

    basic_test<char>();
    basic_test<char16_t>();
    basic_test<char32_t>();
    basic_test<wchar_t>();

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
        bool failed = false;

        try
        {
            strf::write(result, &result_length)
                (strf::multi('a', 6), "ABCDEF", 'b');
        }
        catch(std::runtime_error& x)
        {
            failed = true;
        }

        BOOST_TEST(failed);
        BOOST_TEST(result_length == 10);
        BOOST_TEST(result.str() == "aaaaaaABCD");
    }

    return boost::report_errors();
}
