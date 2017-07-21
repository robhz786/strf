#ifndef STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED
#define STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <boost/stringify.hpp>
#include <cctype>

template <typename CharT>
struct to_upper_char_traits : public std::char_traits<CharT>
{
    static CharT*
    copy(CharT* to, const CharT* from, std::size_t n)
    {
        CharT* it = to;
        while(n--)
            *it++ = std::toupper(*from++);
        return to;
    }

    static void
    assign(CharT& c1, const CharT& c2)
    {
        c1 = std::toupper(c2);
    }

    static CharT*
    assign(CharT* dest, std::size_t n, CharT a)
    {
        std::fill_n(dest, n, std::toupper(a));
        return dest;
    }
};


template <typename CharT>
struct weird_char_traits : public std::char_traits<CharT>
{
    static CharT*
    copy(CharT* to, const CharT* from, std::size_t n)
    {
        CharT* it = to;
        while(n--)
        {
            assign(*it++, *from++);
        }
        return to;
    }

    static void
    assign(CharT& c1, const CharT& c2)
    {
        if (c2 == CharT())
        {
            c1 = c2;
        }
        else
        {
            c1 = c2 | ( 1 << (sizeof(CharT) * 8 - 1));
        }
    }

    static CharT*
    assign(CharT* dest, std::size_t n, CharT a)
    {
        CharT b;
        assign(b, a);
        std::fill_n(dest, n, b);
        return dest;
    }
};



static int global_errors_count = 0;

int report_errors()
{
    if (global_errors_count)
    {
        std::cout << global_errors_count << " tests failed\n";
    }
    else
    {
        std::cout << "No errors found\n";
    }

    return global_errors_count;
}


void print(const char* label, const std::u16string&)
{
    std::cout << label << ": (unable to print) \n";
}

void print(const char* label, const std::u32string&)
{
    std::cout << label << ": (unable to print) \n";
}


void print(const char* label, const std::string& str)
{
    std::cout << label << ": \"" << str << "\"\n";
}

void print(const char* label, const std::wstring& str)
{
    std::cout << label << ": \"";
    std::wcout << str;
    std::cout  << "\"\n";
}


template <typename CharT>
class input_tester
{
public:
    input_tester
        ( std::basic_string<CharT> expected
        , std::string src_filename
        , int src_line
        )
        : m_expected(std::move(expected))
        , m_reserved_size(0)
        , m_src_filename(std::move(src_filename))
        , m_src_line(src_line)
    {
    }

    using char_type = CharT;

    void put(char_type character)
    {
        m_result.push_back(character);
    }

    void put(char_type character, std::size_t repetitions)
    {
        m_result.append(repetitions, character);
    }

    void put(const char_type* str, std::size_t count)
    {
        m_result.append(str, count);
    }

    std::basic_string<CharT> finish()
    {
        if (m_expected != m_result || m_reserved_size != 1 + m_result.length())
        {
            std::cout << m_src_filename << ":" << m_src_line << ":" << " error: \n";
            ++global_errors_count;
        }
        if (m_expected != m_result)
        {
            print("expected", m_expected);
            print("obtained", m_result);
        }
        if(m_reserved_size != 1 + m_result.length())
        {
            std::cout << "reserved size  :" <<  m_reserved_size << "\n";
            std::cout << "necessary size :" <<  m_result.length() + 1 << "\n";
        }

        return std::move(m_result);
    }

    void reserve(std::size_t size)
    {
        m_reserved_size = size;
        m_result.reserve(size);
    }


private:

    std::basic_string<CharT> m_result;
    std::basic_string<CharT> m_expected;
    std::size_t m_reserved_size;
    std::string m_src_filename;
    int m_src_line;
};


template<typename CharT>
auto make_tester(const CharT* expected, const char* filename, int line)
{
    using writer = input_tester<CharT>;
    return std::move(boost::stringify::v0::make_args_handler
                     <writer, const CharT*, const char*, int>
                     (expected, filename, line));
}


#define TEST(EXPECTED) make_tester((EXPECTED), __FILE__, __LINE__)

#endif


