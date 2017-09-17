//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <boost/detail/lightweight_test.hpp>
#include "test_utils.hpp"
#include "error_code_emitter_arg.hpp"
#include "exception_thrower_arg.hpp"
#include <boost/stringify.hpp>

namespace strf = boost::stringify::v0;


template <typename CharT>
void basic_assign_test()
{
    std::basic_string<CharT> result('a', 10);;
    std::basic_string<CharT> expected;

    std::error_code err = use_all_writing_function_of_output_writer
        ( strf::assign_to(result)
        , expected );

    BOOST_TEST(!err);
    BOOST_TEST(expected == result);
}


template <typename CharT>
void basic_append_test()
{
    std::basic_string<CharT> result('a', 10);;
    std::basic_string<CharT> expected = result;
    std::basic_string<CharT> expected_append;

    std::error_code err = use_all_writing_function_of_output_writer
        ( strf::append_to(result)
        , expected_append );

    expected += expected_append;

    BOOST_TEST(!err);
    BOOST_TEST(expected == result);
}

template <typename CharT>
void basic_make_test()
{
    std::basic_string<CharT> expected;

    auto result = use_all_writing_function_of_output_writer
        ( strf::make_basic_string<CharT>
        , expected );

    BOOST_TEST(result);
    BOOST_TEST(result && expected == result.value());
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


   //TODO

    {   // When set_error is called during make_string

        auto result = strf::make_string
            [{ "abcd", error_code_emitter_arg, "lkjlj" }];

        BOOST_TEST(!result);
        BOOST_TEST(!result && result.error() == std::errc::invalid_argument);
    }

    {   // When set_error is called during assign_to

        std::string result = "bla";

        std::error_code ec = strf::assign_to(result)
            [{"abcd", error_code_emitter_arg, "lkjlj"}];

        BOOST_TEST(ec == std::errc::invalid_argument);
        BOOST_TEST(result == "");
    }

    {   // When set_error is called during append_to

        std::string result = "bla";

        std::error_code ec = strf::append_to(result)
            [{ "abcd", error_code_emitter_arg, "lkjlj" }];

        BOOST_TEST(ec == std::errc::invalid_argument);
        BOOST_TEST(result == "bla");
    }


    {   // When exception is thrown in assign_to

        std::string result = "bla";

        try
        {
            strf::assign_to(result)
                [{"abcd", exception_thrower_arg, "lkjlj"}];
        }
        catch(...)
        {
        }

        BOOST_TEST(result == "");
    }

    {   // When exception is thrown in append_to

        std::string result = "bla";

        try
        {
            strf::append_to(result)
                [{ "abcd", exception_thrower_arg, "lkjlj" }];
        }
        catch(...)
        {
        }

        BOOST_TEST(result == "bla");
    }

    return boost::report_errors();
}
