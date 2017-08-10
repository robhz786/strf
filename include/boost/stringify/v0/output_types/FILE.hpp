#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_FILE_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_FILE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <cstdio>
#include <boost/stringify/v0/output_writer.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

struct FILE_result
{
    std::size_t count;
    bool success;
};

namespace detail {

class narrow_file_writer: public output_writer<char>
{

public:

    using char_type = char;

    narrow_file_writer(std::FILE* file) : m_file(file)
    {
    }

    void put(const char_type* str, std::size_t count) override;

    void put(char_type ch) override;

    void repeat
        ( char_type ch1
        , std::size_t count
        ) override;

    void repeat
        ( char_type ch1
        , char_type ch2
        , std::size_t count
        ) override;

    void repeat
        ( char_type ch1
        , char_type ch2
        , char_type ch3
        , std::size_t count
        ) override;

    void repeat
        ( char_type ch1
        , char_type ch2
        , char_type ch3
        , char_type ch4
        , std::size_t count
        ) override;

    boost::stringify::v0::FILE_result finish();

private:

    void do_put(char_type ch);

    std::FILE* m_file;
    std::size_t m_count = 0;
    bool m_success = true;
};


class wide_file_writer: public output_writer<wchar_t>
{

public:

    using char_type = wchar_t;

    wide_file_writer(std::FILE* file) : m_file(file)
    {
    }

    void put(const char_type* str, std::size_t count) override;

    void put(char_type ch) override;

    void repeat
        ( char_type ch
        , std::size_t count
        ) override;

    void repeat
        ( char_type ch1
        , char_type ch2
        , std::size_t count
        ) override;

    void repeat
        ( char_type ch1
        , char_type ch2
        , char_type ch3
        , std::size_t count
        ) override;

    void repeat
        ( char_type ch1
        , char_type ch2
        , char_type ch3
        , char_type ch4
        , std::size_t count
        ) override;

    FILE_result finish();

private:

    void do_put(char_type ch);

    std::FILE* m_file;
    std::size_t m_count = 0;
    bool m_success = true;

};


#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_INLINE void narrow_file_writer::put
    ( const char_type* str
    , std::size_t count
    )
{
    std::size_t count_inc = std::fwrite(str, 1, count, m_file);
    m_success &= (count == count_inc);
    m_count += count_inc;
}

BOOST_STRINGIFY_INLINE void narrow_file_writer::put(char_type ch)
{
    do_put(ch);
}

BOOST_STRINGIFY_INLINE void narrow_file_writer::repeat
    ( char_type ch
    , std::size_t count
    )
{
    for(;count > 0; --count)
    {
        do_put(ch);
    }
}

BOOST_STRINGIFY_INLINE void narrow_file_writer::repeat
    ( char_type ch1
    , char_type ch2
    , std::size_t count
    )
{
    for(;count > 0; --count)
    {
        do_put(ch1);
        do_put(ch2);
    }
}

BOOST_STRINGIFY_INLINE void narrow_file_writer::repeat
    ( char_type ch1
    , char_type ch2
    , char_type ch3
    , std::size_t count
    )
{
    for(;count > 0; --count)
    {
        do_put(ch1);
        do_put(ch2);
        do_put(ch3);
    }
}

BOOST_STRINGIFY_INLINE void narrow_file_writer::repeat
    ( char_type ch1
    , char_type ch2
    , char_type ch3
    , char_type ch4
    , std::size_t count
    )
{
    for(;count > 0; --count)
    {
        do_put(ch1);
        do_put(ch2);
        do_put(ch3);
        do_put(ch4);
    }
}

BOOST_STRINGIFY_INLINE FILE_result narrow_file_writer::finish()
{
    std::fflush(m_file);
    return {m_count, m_success};
}

BOOST_STRINGIFY_INLINE void narrow_file_writer::do_put(char_type ch)
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

BOOST_STRINGIFY_INLINE void wide_file_writer::put
    ( const char_type* str
    , std::size_t count
    )
{
    for(;count > 0; --count)
    {
        put(*str);
        ++str;
    }
}

BOOST_STRINGIFY_INLINE void wide_file_writer::put(char_type ch)
{
    do_put(ch);
}

BOOST_STRINGIFY_INLINE void wide_file_writer::repeat(char_type ch, std::size_t count)
{
    for(;count > 0; --count)
    {
        do_put(ch);
    }
}

BOOST_STRINGIFY_INLINE void wide_file_writer::repeat
    ( char_type ch1
    , char_type ch2
    , std::size_t count
    )
{
    for(;count > 0; --count)
    {
        do_put(ch1);
        do_put(ch2);
    }
}

BOOST_STRINGIFY_INLINE void wide_file_writer::repeat
    ( char_type ch1
    , char_type ch2
    , char_type ch3
    , std::size_t count
    )
{
    for(;count > 0; --count)
    {
        do_put(ch1);
        do_put(ch2);
        do_put(ch3);
    }
}

BOOST_STRINGIFY_INLINE void wide_file_writer::repeat
    ( char_type ch1
    , char_type ch2
    , char_type ch3
    , char_type ch4
    , std::size_t count
    )
{
    for(;count > 0; --count)
    {
        do_put(ch1);
        do_put(ch2);
        do_put(ch3);
        do_put(ch4);
    }
}

BOOST_STRINGIFY_INLINE FILE_result wide_file_writer::finish()
{
    std::fflush(m_file);
    return {m_count, m_success};
}

BOOST_STRINGIFY_INLINE void wide_file_writer::do_put(char_type ch)
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

#endif //! defined(BOOST_STRINGIFY_OMIT_IMPL)

} // namespace detail

inline auto write_to(std::FILE* destination)
{
    using writer = boost::stringify::v0::detail::narrow_file_writer;
    return boost::stringify::v0::make_args_handler<writer, std::FILE*>(destination);
}

inline auto wwrite_to(std::FILE* destination)
{
    using writer = boost::stringify::v0::detail::wide_file_writer;
    return boost::stringify::v0::make_args_handler<writer, std::FILE*>(destination);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_OUTPUT_TYPES_FILE_HPP

