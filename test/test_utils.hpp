#ifndef STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED
#define STRINGIFY_TEST_TEST_UTILS_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <boost/stringify.hpp>
#include <cctype>

// template <typename CharT>
// struct to_upper_char_traits : public std::char_traits<CharT>
// {
//     static CharT*
//     copy(CharT* to, const CharT* from, std::size_t n)
//     {
//         CharT* it = to;
//         while(n--)
//             *it++ = std::toupper(*from++);
//         return to;
//     }

//     static void
//     assign(CharT& c1, const CharT& c2)
//     {
//         c1 = std::toupper(c2);
//     }

//     static CharT*
//     assign(CharT* dest, std::size_t n, CharT a)
//     {
//         std::fill_n(dest, n, std::toupper(a));
//         return dest;
//     }
// };


// template <typename CharT>
// struct weird_char_traits : public std::char_traits<CharT>
// {
//     static CharT*
//     copy(CharT* to, const CharT* from, std::size_t n)
//     {
//         CharT* it = to;
//         while(n--)
//         {
//             assign(*it++, *from++);
//         }
//         return to;
//     }

//     static void
//     assign(CharT& c1, const CharT& c2)
//     {
//         if (c2 == CharT())
//         {
//             c1 = c2;
//         }
//         else
//         {
//             c1 = c2 | ( 1 << (sizeof(CharT) * 8 - 1));
//         }
//     }

//     static CharT*
//     assign(CharT* dest, std::size_t n, CharT a)
//     {
//         CharT b;
//         assign(b, a);
//         std::fill_n(dest, n, b);
//         return dest;
//     }
// };

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
struct input_tester_buffer
{
    input_tester_buffer(typename std::basic_string<CharT>::size_type size)
        : buffer(size, static_cast<CharT>('#'))
    {
    }

    std::basic_string<CharT> buffer;
};


template <typename CharOut>
class input_tester
    : private input_tester_buffer<CharOut>
    , public boost::stringify::v0::buffer_recycler<CharOut>
{
    using  input_tester_buffer<CharOut>::buffer;
public:

    input_tester
        ( std::basic_string<CharOut> expected
        , std::error_code expected_error
        , std::string src_filename
        , int src_line
        , double reserve_factor
        , std::size_t buffer_size = boost::stringify::v0::min_buff_size
        )
        : input_tester_buffer<CharOut>{buffer_size}
        , m_expected(std::move(expected))
        , m_expected_error(expected_error)
        , m_reserved_size(0)
        , m_src_filename(std::move(src_filename))
        , m_src_line(src_line)
        , m_reserve_factor(reserve_factor)
    {
    }

    ~input_tester()
    {
    }

    using char_type = CharOut;

    boost::stringify::v0::expected_buff_it<CharOut> start()
    {
        return { boost::stringify::v0::in_place_t{}
               , boost::stringify::v0::buff_it<CharOut>{m_buff, m_buff_end}};            
    }
    boost::stringify::v0::expected_buff_it<CharOut> recycle(CharOut* it)
    {
        m_result.append(m_buff_begin, it);
        return { boost::stringify::v0::in_place_t{}
               , boost::stringify::v0::buff_it<CharOut>{m_buff, m_buff_end}};
    }
    boost::stringify::v0::expected<void, std::error_code> finish(CharOut* it)
    {
        m_result.append(m_buff_begin, it);
        if (m_expected != m_result)
        {
            print("expected", m_expected);
            print("obtained", m_result);
            ++global_errors_count;
        }
        if(wrongly_reserved())
        {
            std::cout << "reserved size  :" <<  m_reserved_size << "\n";
            std::cout << "necessary size :" <<  m_result.length() << "\n";
            ++global_errors_count;
        }
        return {};        
    }

    void reserve(std::size_t size)
    {
        m_reserved_size = size;
        m_result.reserve(size);
    }

private:

    bool wrongly_reserved() const
    {
        return (m_reserved_size < m_result.length() || too_much_reserved());
    }

    bool too_much_reserved() const
    {
        return
            static_cast<double>(m_reserved_size) /
            static_cast<double>(m_result.length())
            > m_reserve_factor;
    }

    std::basic_string<CharOut> m_result;
    std::basic_string<CharOut> m_expected;
    std::error_code m_expected_error;
    std::size_t m_reserved_size;
    std::string m_src_filename;
    int m_src_line;
    double m_reserve_factor;

    CharOut m_buff[200];
    CharOut* m_buff_begin = m_buff;
    CharOut* m_buff_end = m_buff_begin + sizeof(m_buff) / sizeof(m_buff[0]);
};


//template<typename CharT>
//auto make_tester
//    ( const CharT* expected
//    , const char* filename
//    , int line
//    , std::error_code err = {}
//    , double reserve_factor = 1.0
//    )
//{
//    using writer = input_tester<CharT>;
//    return boost::stringify::v0::make_destination
//        <writer, const CharT*, std::error_code, const char*, int>
//        (expected, err, filename, line, reserve_factor);
//}

//template<typename CharT>
//auto make_tester
//    ( const CharT* expected
//    , const char* filename
//    , int line
//    , double reserve_factor
//    )
//{
//    using writer = input_tester<CharT>;
//    return boost::stringify::v0::make_destination
//        <writer, const CharT*, std::error_code, const char*, int, double, std::size_t>
//        (expected, {}, filename, line, reserve_factor, buffer_size);
//}


template<typename CharT>
auto make_tester
    ( const CharT* expected
    , const char* filename
    , int line
    , std::error_code err
    , double reserve_factor
    , std::size_t buffer_size)
{
    using writer = input_tester<CharT>;
    return boost::stringify::v0::make_destination
        <writer, const CharT*, std::error_code, const char*, int, double, std::size_t>
        (expected, err, filename, line, reserve_factor, buffer_size);
}

template<typename CharT>
auto make_tester
    ( const std::basic_string<CharT>& expected
    , const char* filename
    , int line
    , std::error_code err
    , double reserve_factor
    , std::size_t buffer_size )
{
    using writer = input_tester<CharT>;
    return boost::stringify::v0::make_destination
        <writer, const std::basic_string<CharT>&, std::error_code, const char*, int, double, std::size_t>
        (expected, err, filename, line, reserve_factor, buffer_size);
}

#define TEST(EXPECTED) (void)make_tester((EXPECTED), __FILE__, __LINE__, std::error_code(), 1.0, 60)

#define TEST_RF(EXPECTED, RF) (void)make_tester((EXPECTED), __FILE__, __LINE__, std::error_code(), (RF), 60)

#define TEST_ERR(EXPECTED, ERR) (void)make_tester((EXPECTED), __FILE__, __LINE__, (ERR), 1.0, 60)

#define TEST_ERR_RF(EXPECTED, ERR, RF) (void)make_tester((EXPECTED), __FILE__, __LINE__, (ERR), (RF), 60)

#define BUFFERED_TEST(SIZE, EXPECTED) (void)make_tester((EXPECTED), __FILE__, __LINE__, std::error_code(), 1.0, (SIZE))

#define BUFFERED_TEST_RF(SIZE, EXPECTED, RF) (void)make_tester((EXPECTED), __FILE__, __LINE__, std::error_code(), (RF), (SIZE))

#define BUFFERED_TEST_ERR(SIZE, EXPECTED, ERR) (void)make_tester((EXPECTED), __FILE__, __LINE__, (ERR), 1.0, (SIZE))

#define BUFFERED_TEST_ERR_RF(SIZE, EXPECTED, ERR, RF) (void)make_tester((EXPECTED), __FILE__, __LINE__, (ERR), (RF), (SIZE))


#endif


