//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include "test_utils.hpp"
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

    basic_to_string_test<char>();
    basic_to_string_test<char16_t>();
    basic_to_string_test<char32_t>();
    basic_to_string_test<wchar_t>();

    {   // to_string reserve
        auto result = strf::to_string.reserve(2000) ("aaa");
        BOOST_TEST(result.capacity() >= 2000);
    }
    {   // assign reserve
        std::string result;
        auto size = strf::assign(result).reserve(2000) ("aaa");
        BOOST_TEST(result.capacity() >= 2000);
        BOOST_TEST(size == 3);
    }
    {   // append reserve
        std::string result(500, 'x');
        auto size = strf::append(result).reserve(1000) ("aaa");
        BOOST_TEST(result.capacity() >= 1500);
        BOOST_TEST(size == 3);
    }

    {
        std::string big_string(500, 'x');
        auto str = strf::to_string(big_string);
        BOOST_TEST(str == big_string);
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

        BOOST_TEST(result == "abcd");
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
        BOOST_TEST(result == "blaabcd");
    }

    return boost::report_errors();
}
