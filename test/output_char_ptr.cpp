//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include "test_utils.hpp"
#include "exception_thrower_arg.hpp"
#include <boost/stringify.hpp>

namespace strf = boost::stringify::v0;

#if ! defined(BOOST_NO_EXCEPTION)

template <typename CharT>
void basic_test()
{
    CharT output[100];
    std::fill(output, output + 100, CharT{'-'});
    std::basic_string<CharT> expected(50, CharT{'*'});
    auto r = strf::write(output) (strf::multi(CharT{'*'}, 50));

    BOOST_TEST_EQ(expected.length(), r.ptr - output);
    BOOST_TEST(!r.truncated);
    BOOST_TEST(expected == output);
}

template <typename CharT>
void test_array_too_small()
{
    CharT a = 'a';
    CharT b = 'b';
    CharT c = 'c';    
    CharT buff[3] = { 'x', 'y', 'z' };
    auto r = strf::write(buff)(a, b, c);
    BOOST_TEST_EQ(r.ptr, &buff[2]);
    BOOST_TEST(r.truncated);
    BOOST_TEST_EQ(buff[0], a);
    BOOST_TEST_EQ(buff[1], b);
    BOOST_TEST_EQ(buff[2], 0);
}

template <typename CharT>
void test_informed_size_too_small()
{
    CharT a = 'a';
    CharT b = 'b';
    CharT c = 'c';    
    CharT buff[3] = { 'x', 'y', 'z' };
    auto r = strf::write(buff, 3) (a, b, c);
    BOOST_TEST_EQ(r.ptr, &buff[2]);
    BOOST_TEST(r.truncated);
    BOOST_TEST_EQ(buff[0], a);
    BOOST_TEST_EQ(buff[1], b);
    BOOST_TEST_EQ(buff[2], 0);
}

template <typename CharT>
void test_informed_end_too_close()
{
    CharT a = 'a';
    CharT b = 'b';
    CharT c = 'c';
    CharT buff[3] = { 'x', 'y', 'z' };
    auto r = strf::write(buff, &buff[3]) (a, b, c);
    BOOST_TEST_EQ(r.ptr, &buff[2]);
    BOOST_TEST(r.truncated);
    BOOST_TEST_EQ(buff[0], a);
    BOOST_TEST_EQ(buff[1], b);
    BOOST_TEST_EQ(buff[2], 0);
}

#endif // defined(BOOST_NO_EXCEPTION)

int main()
{
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

    {  // When exception is thrown

        char result[200] = "-----------------------------";
        try
        {
            (void) strf::write(result) ("abcd", exception_thrower_arg, "lkjlj");
        }
        catch(...)
        {
        }

        BOOST_TEST(result[4] == '\0');
        BOOST_TEST_CSTR_EQ(result, "abcd");
    }

    return boost::report_errors();
}
