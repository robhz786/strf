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
    namespace strf = boost::stringify::v0;

    expected =
        u8" abcd xyyabb\u00a1\u00a2\u00a2\u2080\u2081\u2081"
        u8"\U00010000\U00010001\U00010001";

    return w.error_code
        (
            " abcd ", 'x', strf::multi('y', 2), strf::multi('z', 0),
            U'a', strf::multi(U'b', 2), strf::multi(U'c', 0),
            U'\u00a1', strf::multi(U'\u00a2', 2), strf::multi(U'\u00a3', 0),
            U'\u2080', strf::multi(U'\u2081', 2), strf::multi(U'\u2082', 0),
            U'\U00010000', strf::multi(U'\U00010001', 2), strf::multi(U'\U00010002', 0)
        );
}

template <typename W>
decltype(auto) use_all_writing_function_of_output_writer(W&& w, std::u16string& expected)
{
    namespace strf = boost::stringify::v0;

    expected =
        u" abcd xyyabb\u0080\u0081\u0081\u0800\u0801\u0801"
        u"\U00010000\U00010001\U00010001";

    return w.error_code
        (
            u" abcd ", u'x', strf::multi(u'y', 2), strf::multi(u'z', 0),
            U'a', strf::multi(U'b', 2), strf::multi(U'c', 0),
            U'\u0080', strf::multi(U'\u0081', 2), strf::multi(U'\u0082', 0),
            U'\u0800', strf::multi(U'\u0801', 2), strf::multi(U'\u0802', 0),
            U'\U00010000', strf::multi(U'\U00010001', 2), strf::multi(U'\U00010002', 0)
        );

}

template <typename W>
decltype(auto) use_all_writing_function_of_output_writer(W&& w, std::u32string& expected)
{
    namespace strf = boost::stringify::v0;

    expected =
        U" abcd xyyabb\u0080\u0081\u0081\u0800\u0801\u0801"
        U"\U00010000\U00010001\U00010001";

    return w.error_code
        (
            U" abcd ", U'x', strf::multi(U'y', 2), strf::multi(U'z', 0),
            U'a', strf::multi(U'b', 2), strf::multi(U'c', 0),
            U'\u0080', strf::multi(U'\u0081', 2), strf::multi(U'\u0082', 0),
            U'\u0800', strf::multi(U'\u0801', 2), strf::multi(U'\u0802', 0),
            U'\U00010000', strf::multi(U'\U00010001', 2), strf::multi(U'\U00010002', 0)
        );

}


template <typename W>
decltype(auto) use_all_writing_function_of_output_writer(W&& w, std::wstring& expected)
{
    namespace strf = boost::stringify::v0;

    expected =
        L" abcd xyyabb\u00a1\u00a2\u00a2\u0800\u0801\u0801"
        L"\U00010000\U00010001\U00010001";

    return w.error_code
        (
            L" abcd ", L'x', strf::multi(L'y', 2), strf::multi(L'z', 0),
            U'a', strf::multi(U'b', 2), strf::multi(U'c', 0),
            L'\u00a1', strf::multi(L'\u00a2', 2), strf::multi(L'\u00a3', 0),
            L'\u0800', strf::multi(L'\u0801', 2), strf::multi(L'\u0802', 0),
            U'\U00010000', strf::multi(U'\U00010001', 2), strf::multi(U'\U00010002', 0)
        );

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


void print(const char* label, const std::u16string& str)
{
    std::cout << label << "\n";
    for(auto it = str.begin(); it != str.end(); ++it)
    {
        printf("%4x ", (unsigned)*it);
    }
    std::cout << "\n";
}

void print(const char* label, const std::u32string& str)
{
    std::cout << label << "\n";
    for(auto it = str.begin(); it != str.end(); ++it)
    {
        printf("%8x ", (unsigned)*it);
    }
    std::cout << "\n";
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
        , std::error_code expected_error
        , std::string src_filename
        , int src_line
        , double reserve_factor
        )
        : m_expected(std::move(expected))
        , m_expected_error(expected_error)
        , m_reserved_size(0)
        , m_src_filename(std::move(src_filename))
        , m_src_line(src_line)
        , m_reserve_factor(reserve_factor)
    {
    }

    using char_type = CharT;

    void set_error(std::error_code err) override
    {
        m_obtained_error = err;
    }

    virtual bool good() const override
    {
        return ! m_obtained_error;
    }

    bool put(const char_type* str, std::size_t count) override
    {
        if (good())
        {
            m_result.append(str, count);
            return true;
        }
        return false;
    }

    bool put(char_type character) override
    {
        if (good())
        {
            m_result.push_back(character);
            return true;
        }
        return false;
    }

    bool repeat(std::size_t count, char_type ch) override
    {
        if (good())
        {
            m_result.append(count, ch);
            return true;
        }
        return false;
    }

    bool repeat(std::size_t count, char_type ch1, char_type ch2) override
    {
        if (good())
        {
            for(; count > 0; --count)
            {
                m_result.push_back(ch1);
                m_result.push_back(ch2);
            }
            return true;
        }
        return false;
    }

    bool repeat(std::size_t count, char_type ch1, char_type ch2, char_type ch3) override
    {
        if (good())
        {
            for(; count > 0; --count)
            {
                m_result.push_back(ch1);
                m_result.push_back(ch2);
                m_result.push_back(ch3);
            }
            return true;
        }
        return false;
    }

    bool repeat(std::size_t count, char_type ch1, char_type ch2, char_type ch3, char_type ch4) override
    {
        if (good())
        {
            for(; count > 0; --count)
            {
                m_result.push_back(ch1);
                m_result.push_back(ch2);
                m_result.push_back(ch3);
                m_result.push_back(ch4);
            }
            return true;
        }
        return false;
    }

    std::error_code finish_error_code()
    {
        if (m_expected_error != m_obtained_error || m_expected != m_result || wrongly_reserved())
        {
            std::cout << m_src_filename << ":" << m_src_line << ":" << " error: \n";
            ++global_errors_count;
        }
        if (m_expected_error != m_obtained_error)
        {
            print("expected error_code", m_expected_error.message());
            print("obtained error_code", m_obtained_error.message());
        }
        if (m_expected != m_result)
        {
            print("expected", m_expected);
            print("obtained", m_result);
        }
        if(wrongly_reserved())
        {
            std::cout << "reserved size  :" <<  m_reserved_size << "\n";
            std::cout << "necessary size :" <<  m_result.length() << "\n";
        }

        return {};
    }

    void finish_exception()
    {
        auto err = finish_error_code();
        if(err)
        {
            throw std::system_error(err);
        }
    }

    void reserve(std::size_t size)
    {
        m_reserved_size = size;
        m_result.reserve(size);
    }


private:

    bool wrongly_reserved() const
    {
        return
            ( ! m_obtained_error
              && (m_reserved_size < m_result.length() || too_much_reserved()));
    }

    bool too_much_reserved() const
    {
        return
            static_cast<double>(m_reserved_size) /
            static_cast<double>(m_result.length())
            > m_reserve_factor;
    }


    std::basic_string<CharT> m_result;
    std::basic_string<CharT> m_expected;
    std::error_code m_expected_error;
    std::error_code m_obtained_error;
    std::size_t m_reserved_size;
    std::string m_src_filename;
    int m_src_line;
    double m_reserve_factor;
};


template<typename CharT>
auto make_tester
    ( const CharT* expected
    , const char* filename
    , int line
    , std::error_code err = {}
    , double reserve_factor = 1.0
    )
{
    using writer = input_tester<CharT>;
    return boost::stringify::v0::make_args_handler
        <writer, const CharT*,std::error_code, const char*, int>
        (expected, err, filename, line, reserve_factor);
}

template<typename CharT>
auto make_tester
    ( const CharT* expected
    , const char* filename
    , int line
    , double reserve_factor
    )
{
    using writer = input_tester<CharT>;
    return boost::stringify::v0::make_args_handler
        <writer, const CharT*,std::error_code, const char*, int>
        (expected, {}, filename, line, reserve_factor);
}


#define TEST(EXPECTED) make_tester((EXPECTED), __FILE__, __LINE__)

#endif


