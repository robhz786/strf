//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

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

    auto size = strf::assign(output) (strf::multi(CharT{'*'}, 50));

    BOOST_TEST(expected == output);
    BOOST_TEST(size == expected.size());
}

template <typename CharT>
void basic_append_test()
{
    std::basic_string<CharT> output(CharT{'-'}, 10);
    std::basic_string<CharT> expected
        = output
        + std::basic_string<CharT>(50, CharT{'*'});

    auto size = strf::append(output) (strf::multi(CharT{'*'}, 50));

    BOOST_TEST(expected == output);
    BOOST_TEST(size == 50);
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
    std::size_t size = 0;

    auto ec = strf::ec_assign(output, &size) (strf::multi(CharT{'*'}, 50));

    BOOST_TEST(std::error_code{} == ec);
    BOOST_TEST(expected == output);
    BOOST_TEST(size == 50);
}

template <typename CharT>
void basic_ec_append_test()
{
    std::basic_string<CharT> output(CharT{'-'}, 10);
    std::basic_string<CharT> expected
        = output
        + std::basic_string<CharT>(50, CharT{'*'});
    std::size_t size = 0;

    auto ec = strf::ec_append(output, &size) (strf::multi(CharT{'*'}, 50));

    BOOST_TEST(std::error_code{} == ec);
    BOOST_TEST(expected == output);
    BOOST_TEST(size == 50);
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

    {   // assign reserve
        std::string result;
        auto ec = strf::ec_assign(result).reserve(2000) ("aaa");
        BOOST_ASSERT(ec == std::error_code{});
        BOOST_ASSERT(result.capacity() >= 2000);
    }
    {   // append reserve
        std::string result(500, 'x');
        auto ec = strf::ec_append(result).reserve(1000) ("aaa");
        BOOST_ASSERT(ec == std::error_code{});
        BOOST_ASSERT(result.capacity() >= 1500);
    }

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

    {   // to_string reserve
        auto result = strf::to_string.reserve(2000) ("aaa");
        BOOST_ASSERT(result.capacity() >= 2000);
    }
    {   // assign reserve
        std::string result;
        auto size = strf::assign(result).reserve(2000) ("aaa");
        BOOST_ASSERT(result.capacity() >= 2000);
        BOOST_ASSERT(size == 3);
    }
    {   // append reserve
        std::string result(500, 'x');
        auto size = strf::append(result).reserve(1000) ("aaa");
        BOOST_ASSERT(result.capacity() >= 1500);
        BOOST_ASSERT(size == 3);
    }

    {
        std::string big_string(500, 'x');
        auto str = strf::to_string(big_string);
        BOOST_TEST(str == big_string);
    }

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
    return boost::report_errors();
}
