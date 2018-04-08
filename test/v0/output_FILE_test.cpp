//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#define  _CRT_SECURE_NO_WARNINGS

#include <boost/detail/lightweight_test.hpp>
#include <boost/stringify.hpp>
#include <boost/config.hpp>
#include "test_utils.hpp"
#include "error_code_emitter_arg.hpp"
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


template <typename CharT>
void basic_test__narrow()
{
    std::size_t result_length = 1000;
    std::basic_string<CharT> expected;
    std::basic_string<CharT> result;
    std::error_code err = std::make_error_code(std::errc::operation_canceled);
    std::FILE* file = nullptr;

    try
    {
        file = std::tmpfile();
        err = use_all_writing_function_of_output_writer
            ( strf::format<CharT>(file, &result_length)
            , expected );

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

    BOOST_TEST(!err);
    BOOST_TEST(expected.length() == result_length);
    BOOST_TEST(expected == result);
}


template <typename CharT>
void error_code_test__narrow()
{
    std::size_t result_length = 1000;
    std::basic_string<CharT> expected;
    expected.push_back('a');
    expected.push_back('b');
    expected.push_back('c');

    std::basic_string<CharT> result;
    std::error_code err {};
    std::FILE* file = nullptr;

    try
    {
        file = std::tmpfile();
        err = strf::format<CharT>(file, &result_length)
            .error_code(U'a', U'b', U'c', error_code_emitter_arg, U'x', U'y', U'z');

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

    BOOST_TEST(err == std::errc::invalid_argument);
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
            strf::format<CharT>(file, &result_length)
                .exception(U'a', U'b', U'c', exception_thrower_arg, U'x', U'y', U'z');

        }
        catch(...)
        {
        }
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
    std::error_code err = std::make_error_code(std::errc::operation_canceled);
    std::FILE* file = nullptr;

    try
    {
        file = std::tmpfile();
        err = strf::wformat(file, &result_length)
            .error_code(L"abcd", strf::multi(L'x', 0), strf::multi(L'y', 4), L'z');
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

    BOOST_TEST(!err);
    BOOST_TEST(expected.length() == result_length);
    BOOST_TEST(expected == result);
}


void error_code_test__wide()
{
    std::size_t result_length = 1000;
    std::wstring expected = L"abc";
    std::wstring result;
    std::error_code err {};
    std::FILE* file = nullptr;

    try
    {
        file = std::tmpfile();
        err = strf::wformat(file, &result_length)
            .error_code(L"abc", error_code_emitter_arg, L"xyz");

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

    BOOST_TEST(err == std::errc::invalid_argument);
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
            strf::wformat(file, &result_length)
                .exception(L"abc", exception_thrower_arg, L"xyz");

        }
        catch(...)
        {
        }
        std::rewind(file);
        result = read_wfile(file);
    }
    BOOST_TEST(expected.length() == result_length);
    BOOST_TEST(expected == result);
}



int main()
{
    basic_test__narrow<char>();
    basic_test__narrow<char16_t>();
    basic_test__narrow<char32_t>();
    basic_test__narrow<wchar_t>();
    basic_test__wide();

    error_code_test__narrow<char>();
    error_code_test__narrow<char16_t>();
    error_code_test__narrow<char32_t>();
    error_code_test__narrow<wchar_t>();
    error_code_test__wide();

    exception_thrown_test__narrow<char>();
    exception_thrown_test__narrow<char16_t>();
    exception_thrown_test__narrow<char32_t>();
    exception_thrown_test__narrow<wchar_t>();
    exception_thrown_test__wide();


    const std::size_t buff_size = strf::detail::narrow_file_writer<char>::m_buff_size;
    BOOST_ASSERT(buff_size <= (INT_MAX));
    const int buff_size_int = static_cast<int>(buff_size);

    {   // testing narrow_file_writer::put(char_type)

        std::string result;
        std::string str(buff_size - 2, 'x');

        {
            FILE* file = std::tmpfile();
            strf::format(file) .exception( str, 'a', 'b', 'c', 'd', 'e', 'f');
            std::rewind(file);
            result = read_file<char>(file);
        }

        BOOST_TEST(result == str + "abcdef");

    }

    {   // testing narrow_file_writer::put(const char_type* str, std::size count)

        std::string str_a(buff_size / 3, 'a');
        std::string str_b(buff_size / 3, 'b');
        std::string str_c(buff_size / 2, 'c');
        std::string str_d(buff_size * 2, 'd');
        std::string expected = str_a + str_b + str_c + str_d;

        std::string result;
        {
            FILE* file = std::tmpfile();
            strf::format(file) .exception(str_a, str_b, str_c, str_d);
            std::rewind(file);
            result = read_file<char>(file);
        }

        BOOST_TEST(expected == result);
    }


    {   // testing narrow_file_writer::put

        std::string result;
        {
            FILE* file = std::tmpfile();
            strf::format(file) .exception(strf::multi(U'\u0800', buff_size_int * 2));
            std::rewind(file);
            result = read_file<char>(file);
            std::fclose(file);
        }

        std::string expected = strf::make_string.exception(strf::multi(U'\u0800', buff_size_int * 2));
        BOOST_TEST(expected == result);
    }


    return report_errors() || boost::report_errors();
}
