#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_FILE_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_FILE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#include <cstdio>

namespace boost {
namespace stringify {
inline namespace v0 {

struct FILE_result
{
    std::size_t count;
    bool success;
};

namespace detail {

class narrow_file_writer
{

public:

    using char_type = char;

    narrow_file_writer(std::FILE* file) : m_file(file)
    {
    }

    void put(char_type ch)
    {
        if(std::fputc(ch, m_file) == EOF)
        {
            m_success = false;
        }
        else
        {
            ++m_count;
        }
    }

    void put(char_type ch, std::size_t repetitions)
    {
        while(repetitions-- > 0)
        {
            put(ch);
        }
    }

    void put(const char_type* str, std::size_t count)
    {
        std::size_t count_inc = std::fwrite(str, 1, count, m_file);
        m_success &= (count == count_inc);
        m_count += count_inc;
    }

    boost::stringify::v0::FILE_result finish()
    {
        std::fflush(m_file);
        return {m_count, m_success};
    }

private:

    std::FILE* m_file;
    std::size_t m_count = 0;
    bool m_success = true;
};


class wide_file_writer
{

public:

    using char_type = wchar_t;

    wide_file_writer(std::FILE* file) : m_file(file)
    {
    }

    void put(char_type ch)
    {
        if(std::fputwc(ch, m_file) == WEOF)
        {
            m_success = false;
        }
        else
        {
            ++m_count;
        }
    }

    void put(char_type ch, std::size_t repetitions)
    {
        while(repetitions-- > 0)
        {
            put(ch);
        }
    }

    void put(const char_type* str, std::size_t count)
    {
        for(;count > 0; --count)
        {
            put(*str);
            ++str;
        }
    }

    boost::stringify::v0::FILE_result finish()
    {
        std::fflush(m_file);
        return {m_count, m_success};
    }

private:

    std::FILE* m_file;
    std::size_t m_count = 0;
    bool m_success = true;

};

} // namespace detail


auto write_to(std::FILE* destination)
{
    using writer = boost::stringify::v0::detail::narrow_file_writer;
    return boost::stringify::v0::make_args_handler<writer, std::FILE*>(destination);
}

auto wwrite_to(std::FILE* destination)
{
    using writer = boost::stringify::v0::detail::wide_file_writer;
    return boost::stringify::v0::make_args_handler<writer, std::FILE*>(destination);
}

} // inline namespace v0
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_V0_OUTPUT_TYPES_FILE_HPP

