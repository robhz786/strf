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
void basic_test()
{
    CharT output[100];
    std::fill(output, output + 100, CharT{'-'});
    std::basic_string<CharT> expected(50, CharT{'*'});
    auto x = strf::write(output) (strf::multi(CharT{'*'}, 50));

    BOOST_TEST(x);
    BOOST_TEST(expected.length() == x.value());
    BOOST_TEST(expected == output);
}


template <typename CharT>
void test_array_too_small()
{
    CharT buff[3] = { 'a', 'a', 0 };
    auto x = strf::write(buff) ( 1234567 );

    BOOST_TEST(buff[0] == 0);
    BOOST_TEST(!x);
    BOOST_TEST(x.error() == std::errc::result_out_of_range);
}

template <typename CharT>
void test_informed_size_too_small()
{
    CharT buff[100] = { 'a', 'a', 0 };
    auto x = strf::write(buff, 3) ( 1234567 );

    BOOST_TEST(buff[0] == 0);
    BOOST_TEST(! x);
    BOOST_TEST(x.error() == std::errc::result_out_of_range);
}

template <typename CharT>
void test_informed_end_too_close()
{
    CharT buff[100] = { 'a', 'a', 0 };
    auto x = strf::write(buff, &buff[3]) ( 1234567 );

    BOOST_TEST(buff[0] == 0);
    BOOST_TEST(! x);
    BOOST_TEST(x.error() == std::errc::result_out_of_range);
}

int main()
{
    basic_test<char>();
    basic_test<char16_t>();
    basic_test<char32_t>();
    basic_test<wchar_t>();

    {   // Test char_ptr_writer::set_error
        //
        // When set_error(some_err) is called, some_err is returned at the end

        char16_t result[200] = u"-----------------------------";

        auto x = strf::write(result)
            (u"abcd", error_code_emitter_arg, u"lkjlj");

        BOOST_TEST(result[0] == u'\0');
        BOOST_TEST(! x);
        BOOST_TEST(x.error() == std::errc::invalid_argument);
    }

    {  // When exception is thrown

        char16_t result[200] = u"-----------------------------";
        try
        {
            (void) strf::write(result) (u"abcd", exception_thrower_arg, u"lkjlj");
        }
        catch(...)
        {
        }

        BOOST_TEST(result[0] == u'\0');
    }

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

    int rc = report_errors() || boost::report_errors();
    return rc;
}
