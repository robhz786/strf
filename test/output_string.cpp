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

#if !defined(BOOST_NO_EXCEPTIONS)

template <typename CharT>
void basic_assign_test()
{
    std::basic_string<CharT> output(CharT{'-'}, 10);
    std::basic_string<CharT> expected(50, CharT{'*'});

    strf::assign(output) (strf::multi(CharT{'*'}, 50));

    BOOST_TEST(expected == output);
}

template <typename CharT>
void basic_append_test()
{
    std::basic_string<CharT> output(CharT{'-'}, 10);
    std::basic_string<CharT> expected
        = output
        + std::basic_string<CharT>(50, CharT{'*'});

    strf::append(output) (strf::multi(CharT{'*'}, 50));

    BOOST_TEST(expected == output);
}

template <typename CharT>
void basic_to_string_test()
{
    std::basic_string<CharT> expected(50, CharT{'*'});

    auto result = strf::to_basic_string<CharT> (strf::multi(CharT{'*'}, 50));

    BOOST_TEST(result == expected);
}

#endif // !defined(BOOST_NO_EXCEPTIONS)

template <typename CharT>
void basic_ec_assign_test()
{
    std::basic_string<CharT> output(CharT{'-'}, 10);
    std::basic_string<CharT> expected(50, CharT{'*'});

    auto ec = strf::ec_assign(output) (strf::multi(CharT{'*'}, 50));

    BOOST_TEST(std::error_code{} == ec);
    BOOST_TEST(expected == output);
}

template <typename CharT>
void basic_ec_append_test()
{
    std::basic_string<CharT> output(CharT{'-'}, 10);
    std::basic_string<CharT> expected
        = output
        + std::basic_string<CharT>(50, CharT{'*'});

    auto ec = strf::ec_append(output) (strf::multi(CharT{'*'}, 50));

    BOOST_TEST(std::error_code{} == ec);
    BOOST_TEST(expected == output);
}

int main()
{

    basic_ec_assign_test<char>();
    basic_ec_assign_test<char16_t>();
    basic_ec_assign_test<char32_t>();
    basic_ec_assign_test<wchar_t>();
    basic_ec_append_test<char>();
    basic_ec_append_test<char16_t>();
    basic_ec_append_test<char32_t>();
    basic_ec_append_test<wchar_t>();

#if !defined(BOOST_NO_EXCEPTIONS)

    basic_assign_test<char>();
    basic_assign_test<char16_t>();
    basic_assign_test<char32_t>();
    basic_assign_test<wchar_t>();

    basic_append_test<char>();
    basic_append_test<char16_t>();
    basic_append_test<char32_t>();
    basic_append_test<wchar_t>();

    basic_to_string_test<char>();
    basic_to_string_test<char16_t>();
    basic_to_string_test<char32_t>();
    basic_to_string_test<wchar_t>();

    {   // When set_error is called during to_string

        std::error_code ec;
        try
        {
            auto str = strf::to_string
                ("abcd", error_code_emitter_arg, "lkjlj");
        }
        catch(strf::stringify_error& x)
        {
            ec = x.code();
        }

        BOOST_TEST(ec == std::errc::invalid_argument);
    }
    {   // When set_error is called during assign
        std::error_code ec;
        try
        {
            std::string s;
            strf::assign(s) ("abcd", error_code_emitter_arg, "lkjlj");
        }
        catch(strf::stringify_error& x)
        {
            ec = x.code();
        }
        BOOST_TEST(ec == std::errc::invalid_argument);
    }
    {   // When set_error is called during append
        std::error_code ec;
        try
        {
            std::string s;
            strf::assign(s) ("abcd", error_code_emitter_arg, "lkjlj");
        }
        catch(strf::stringify_error& x)
        {
            ec = x.code();
        }
        BOOST_TEST(ec == std::errc::invalid_argument);
    }
    {   // When exception is thrown in assign

        std::string result = "bla";
        try
        {
            strf::assign(result) ("abcd", exception_thrower_arg, "lkjlj");
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
            strf::append(result) ( "abcd", exception_thrower_arg, "lkjlj");
        }
        catch(...)
        {
        }
        BOOST_TEST(result == "bla");
    }

#endif // !defined(BOOST_NO_EXCEPTIONS)

    {   // When set_error is called during ec_assign

        std::string result = "bla";
        auto ec = strf::ec_assign(result)
            ( "abcd", error_code_emitter_arg, "lkjlj" );

        BOOST_TEST(ec == std::errc::invalid_argument);
        BOOST_TEST(result == "abcd");
    }
    {   // When set_error is called during ec_append

        std::string result = "bla";
        auto ec = strf::ec_append(result)
            ( "abcd", error_code_emitter_arg, "lkjlj" );

        BOOST_TEST(ec == std::errc::invalid_argument);
        BOOST_TEST(result == "blaabcd");
    }
    {   // When exception is thrown in ec_assign
        std::string result = "bla";
        try
        {
            (void) strf::ec_assign(result) ("abcd", exception_thrower_arg, "lkjlj");
        }
        catch(...)
        {
        }
        BOOST_TEST(result == "");
    }
    {   // When exception is thrown in ec_append

        std::string result = "bla";
        try
        {
            (void) strf::ec_append(result) ( "abcd", exception_thrower_arg, "lkjlj");
        }
        catch(...)
        {
        }
        BOOST_TEST(result == "bla");
    }
    return report_errors() || boost::report_errors();
}
