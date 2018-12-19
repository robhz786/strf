//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <boost/detail/lightweight_test.hpp>
#include "test_utils.hpp"
#include "error_code_emitter_arg.hpp"
#include "exception_thrower_arg.hpp"
#include <boost/stringify.hpp>

#include <iostream>

namespace strf = boost::stringify::v0;


template <typename CharT>
void basic_assign_test()
{
    std::basic_string<CharT> output(CharT{'-'}, 10);
    std::basic_string<CharT> expected(50, CharT{'*'});

    auto x = strf::assign(output) (strf::multi(CharT{'*'}, 50));

    BOOST_TEST(x);
    BOOST_TEST(*x == 50);
    BOOST_TEST(expected == output);
}


template <typename CharT>
void basic_append_test()
{
    std::basic_string<CharT> output(CharT{'-'}, 10);
    std::basic_string<CharT> expected
        = output
        + std::basic_string<CharT>(50, CharT{'*'});

    auto x = strf::append(output) (strf::multi(CharT{'*'}, 50));

    BOOST_TEST(x);
    BOOST_TEST(*x == 50);
    BOOST_TEST(expected == output);
}

template <typename CharT>
void basic_make_test()
{
    std::basic_string<CharT> expected(50, CharT{'*'});

    auto x = strf::to_basic_string<CharT> (strf::multi(CharT{'*'}, 50));

    BOOST_TEST(x);
    BOOST_TEST(*x == expected);
}

int main()
{
    basic_assign_test<char>();
    basic_assign_test<char16_t>();
    basic_assign_test<char32_t>();
    basic_assign_test<wchar_t>();

    basic_append_test<char>();
    basic_append_test<char16_t>();
    basic_append_test<char32_t>();
    basic_append_test<wchar_t>();

    basic_make_test<char>();
    basic_make_test<char16_t>();
    basic_make_test<char32_t>();
    basic_make_test<wchar_t>();

    {   // When set_error is called during to_string

        auto result = strf::to_string
            ("abcd", error_code_emitter_arg, "lkjlj");

        BOOST_TEST(!result);
        BOOST_TEST(!result && result.error() == std::errc::invalid_argument);
    }

    {   // When set_error is called during assign

        std::string result = "bla";

        auto x = strf::assign(result)
            ("abcd", error_code_emitter_arg, "lkjlj");

        BOOST_TEST(!x && x.error() == std::errc::invalid_argument);
        BOOST_TEST(result == "");
    }

    {   // When set_error is called during append

        std::string result = "bla";

        auto x = strf::append(result)
            ( "abcd", error_code_emitter_arg, "lkjlj" );

        BOOST_TEST(!x && x.error() == std::errc::invalid_argument);
        BOOST_TEST(result == "bla");
    }


    {   // When exception is thrown in assign

        std::string result = "bla";

        try
        {
            (void) strf::assign(result) ("abcd", exception_thrower_arg, "lkjlj");
        }
        catch(...)
        {
        }

        BOOST_TEST(result == "");
    }

    {   // When exception is thrown in append

        std::string result = "bla";

        try
        {
            (void) strf::append(result) ( "abcd", exception_thrower_arg, "lkjlj");
        }
        catch(...)
        {
        }

        BOOST_TEST(result == "bla");
    }

    return report_errors() || boost::report_errors();
}
