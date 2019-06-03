//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <boost/stringify.hpp>
#include <boost/config.hpp>
#include "test_utils.hpp"
#include "exception_thrower_arg.hpp"
#include <climits>

namespace strf = boost::stringify::v0;

std::wstring read_wfile(std::FILE* file)
{
    std::wstring result;
    wint_t ch = fgetwc(file);
    while(ch != WEOF)
    {
        result += static_cast<wchar_t>(ch);
        ch = fgetwc(file);
    };

    return result;
}


template <typename CharT>
std::basic_string<CharT> read_file(std::FILE* file)
{
    constexpr std::size_t buff_size = 500;
    CharT buff[buff_size];
    std::basic_string<CharT> result;
    std::size_t read_size = 0;
    do
    {
        read_size = std::fread(buff, sizeof(buff[0]), buff_size, file);
        result.append(buff, read_size);
    }
    while(read_size == buff_size);

    return result;
}


template <typename CharT>
std::basic_string<CharT> read_file(const char* filename)
{
    std::basic_string<CharT> result;

    std::FILE* file = nullptr;
    try
    {
        file = filename == nullptr ? nullptr : std::fopen(filename, "r");
        if(file != nullptr)
        {
            result = read_file<CharT>(file);
        }
    }
    catch(...)
    {
    }
    if (file != nullptr)
    {
        fclose(file);
    }

    return result;
}

#if ! defined(BOOST_NO_EXCEPTION)

template <typename CharT>
void basic_test__narrow()
{
    std::size_t result_length = 1000;
    std::basic_string<CharT> expected(50, CharT{'*'});
    std::basic_string<CharT> result;
    std::FILE* file = nullptr;

    try
    {
        file = std::tmpfile();
        strf::write<CharT>(file, &result_length)(strf::multi(CharT{'*'}, 50));

        std::fflush(file);
        std::rewind(file);
        result = read_file<CharT>(file);
    }
    catch(...)
    {
    }

    if(file != nullptr)
    {
        std::fclose(file);
    }

    BOOST_TEST(expected.length() == result_length);
    BOOST_TEST(expected == result);
}


template <typename CharT>
void exception_thrown_test__narrow()
{
    std::size_t result_length = 1000;
    std::basic_string<CharT> expected;
    expected.push_back('a');
    expected.push_back('b');
    expected.push_back('c');

    std::basic_string<CharT> result;
    if(std::FILE* file = std::tmpfile())
    {
        try
        {
            strf::write<CharT>(file, &result_length)
                ( CharT{'a'}, CharT{'b'}, CharT{'c'}
                , exception_thrower_arg
                , CharT{'x'}, CharT{'y'}, CharT{'z'} );

        }
        catch(...)
        {
        }
        std::fflush(file);
        std::rewind(file);
        result = read_file<CharT>(file);
        std::fclose(file);
    }
    BOOST_TEST(expected.length() == result_length);
    BOOST_TEST(expected == result);
}


void basic_test__wide()
{
    std::size_t result_length = 1000;
    std::wstring expected = L"abcdyyyyz";
    std::wstring result;
    std::FILE* file = nullptr;
    try
    {
        file = std::tmpfile();
        strf::wwrite(file, &result_length)
            (L"abcd", strf::multi(L'x', 0), strf::multi(L'y', 4), L'z');
        std::fflush(file);
        std::rewind(file);
        result = read_wfile(file);
    }
    catch(...)
    {
    }

    if(file != nullptr)
    {
        std::fclose(file);
    }

    BOOST_TEST(expected.length() == result_length);
    BOOST_TEST(expected == result);
}

void exception_thrown_test__wide()
{
    std::size_t result_length = 1000;
    std::wstring expected = L"abc";
    std::wstring result;
    if (std::FILE* file = std::tmpfile())
    {
        try
        {
            strf::wwrite(file, &result_length)
                (L"abc", exception_thrower_arg, L"xyz");

        }
        catch(...)
        {
        }
        std::fflush(file);
        std::rewind(file);
        result = read_wfile(file);
    }
    BOOST_TEST(expected.length() == result_length);
    BOOST_TEST(expected == result);
}

#endif // ! defined(BOOST_NO_EXCEPTION)

int main()
{

#if ! defined(BOOST_NO_EXCEPTION)

    basic_test__narrow<char>();
    basic_test__narrow<char16_t>();
    basic_test__narrow<char32_t>();
    basic_test__narrow<wchar_t>();
    basic_test__wide();

    exception_thrown_test__narrow<char>();
    exception_thrown_test__narrow<char16_t>();
    exception_thrown_test__narrow<char32_t>();
    exception_thrown_test__narrow<wchar_t>();
    exception_thrown_test__wide();

#endif // ! defined(BOOST_NO_EXCEPTION)

    return boost::report_errors();
}
