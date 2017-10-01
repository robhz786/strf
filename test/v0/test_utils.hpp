#ifndef STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED
#define STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <boost/stringify.hpp>
#include <cctype>


template <typename W>
decltype(auto) use_all_writing_function_of_output_writer(W&& w, std::string& expected)
{
    expected =
        u8" abcd xyyabb\u00a1\u00a2\u00a2\u2080\u2081\u2081"
        u8"\U00010000\U00010001\U00010001";

    return w
        [{
            " abcd ", 'x', {'y', {"", 2}}, {'z', {"", 0}},
            U'a', {U'b', {"", 2}}, {U'c', {"", 0}},
            U'\u00a1', {U'\u00a2', {"", 2}}, {U'\u00a3', {"", 0}},
            U'\u2080', {U'\u2081', {"", 2}}, {U'\u2082', {"", 0}},
            U'\U00010000', {U'\U00010001', {"", 2}}, {U'\U00010002', {"", 0}},
        }];
}

template <typename W>
decltype(auto) use_all_writing_function_of_output_writer(W&& w, std::u16string& expected)
{
    expected =
        u" abcd xyyabb\u0080\u0081\u0081\u0800\u0801\u0801"
        u"\U00010000\U00010001\U00010001";

    return w
        [{
                u" abcd ", u'x', {u'y', {"", 2}}, {u'z', {"", 0}},
                U'a', {U'b', {"", 2}}, {U'c', {"", 0}},
                U'\u0080', {U'\u0081', {"", 2}}, {U'\u0082', {"", 0}},
                U'\u0800', {U'\u0801', {"", 2}}, {U'\u0802', {"", 0}},
                U'\U00010000', {U'\U00010001', {"", 2}}, {U'\U00010002', {"", 0}},
        }];

}

template <typename W>
decltype(auto) use_all_writing_function_of_output_writer(W&& w, std::u32string& expected)
{
    expected =
        U" abcd xyyabb\u0080\u0081\u0081\u0800\u0801\u0801"
        U"\U00010000\U00010001\U00010001";

    return w
        [{
               U" abcd ", U'x', {U'y', {"", 2}}, {U'z', {"", 0}},
               U'a', {U'b', {"", 2}}, {U'c', {"", 0}},
               U'\u0080', {U'\u0081', {"", 2}}, {U'\u0082', {"", 0}},
               U'\u0800', {U'\u0801', {"", 2}}, {U'\u0802', {"", 0}},
               U'\U00010000', {U'\U00010001', {"", 2}}, {U'\U00010002', {"", 0}},
        }];

}


template <typename W>
decltype(auto) use_all_writing_function_of_output_writer(W&& w, std::wstring& expected)
{
    expected =
        L" abcd xyyabb\u00a1\u00a2\u00a2\u0800\u0801\u0801"
        L"\U00010000\U00010001\U00010001";

    return w
        [{
               L" abcd ", L'x', {L'y', {"", 2}}, {L'z', {"", 0}},
               U'a', {U'b', {"", 2}}, {U'c', {"", 0}},
               L'\u00a1', {L'\u00a2', {"", 2}}, {L'\u00a3', {"", 0}},
               L'\u0800', {L'\u0801', {"", 2}}, {L'\u0802', {"", 0}},
               U'\U00010000', {U'\U00010001', {"", 2}}, {U'\U00010002', {"", 0}}
        }];

}

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
class input_tester: public boost::stringify::v0::output_writer<CharT>
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

    void set_error(std::error_code) override
    {
        std::string err_msg = "[*** error code ***]";
        for(auto it = err_msg.begin(); it != err_msg.end(); ++it)
        {
            m_result.push_back(static_cast<CharT>(*it));
        }
    }

    virtual bool good() const override
    {
        return true;
    }

    void put(const char_type* str, std::size_t count) override
    {
        m_result.append(str, count);
    }

    void put(char_type character) override
    {
        m_result.push_back(character);
    }

    void repeat(std::size_t count, char_type ch) override
    {
        m_result.append(count, ch);
    }

    void repeat(std::size_t count, char_type ch1, char_type ch2) override
    {
        for(; count > 0; --count)
        {
            m_result.push_back(ch1);
            m_result.push_back(ch2);
        }
    }

    void repeat(std::size_t count, char_type ch1, char_type ch2, char_type ch3) override
    {
        for(; count > 0; --count)
        {
            m_result.push_back(ch1);
            m_result.push_back(ch2);
            m_result.push_back(ch3);
        }
    }

    void repeat(std::size_t count, char_type ch1, char_type ch2, char_type ch3, char_type ch4) override
    {
        for(; count > 0; --count)
        {
            m_result.push_back(ch1);
            m_result.push_back(ch2);
            m_result.push_back(ch3);
            m_result.push_back(ch4);
        }
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


