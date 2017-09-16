#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_FILE_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_FILE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <cstdio>
#include <boost/stringify/v0/output_writer.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

class narrow_file_writer: public output_writer<char>
{

public:

    using char_type = char;

    narrow_file_writer(std::FILE* file, std::size_t* count)
        : m_file(file)
        , m_count(count)
    {
        if(m_count != nullptr)
        {
            *m_count = 0;
        }
    }

    void set_error(std::error_code err) override;

    bool good() const override;

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

    std::error_code finish();

private:

    bool do_put(char_type ch);

    std::FILE* m_file;
    std::size_t* m_count = nullptr;
    std::error_code m_err;
};


class wide_file_writer: public output_writer<wchar_t>
{

public:

    using char_type = wchar_t;

    wide_file_writer(std::FILE* file, std::size_t* count)
        : m_file(file)
        , m_count(count)
    {
        if(m_count != nullptr)
        {
            *m_count = 0;
        }
    }

    void set_error(std::error_code err) override;

    bool good() const override;

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

    std::error_code finish();

private:

    bool do_put(char_type ch);

    std::FILE* m_file;
    std::size_t* m_count = nullptr;
    std::error_code m_err;

};


#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_INLINE void narrow_file_writer::set_error(std::error_code err)
{
    if(err && !m_err)
    {
        m_err = err;
    }
}

BOOST_STRINGIFY_INLINE bool narrow_file_writer::good() const
{
    return ! m_err;
}

BOOST_STRINGIFY_INLINE void narrow_file_writer::put
    ( const char_type* str
    , std::size_t count
    )
{
    if ( ! m_err )
    {
        std::size_t count_inc = std::fwrite(str, 1, count, m_file);
        if(m_count != nullptr)
        {
            *m_count += count_inc;
        }
        if (count != count_inc)
        {
            m_err = std::error_code{errno, std::generic_category()};
        }
    }
}

BOOST_STRINGIFY_INLINE void narrow_file_writer::put(char_type ch)
{
    if ( ! m_err)
    {
        do_put(ch);
    }
}

BOOST_STRINGIFY_INLINE void narrow_file_writer::repeat
    ( char_type ch
    , std::size_t count
    )
{
    if( ! m_err)
    {
        while(count > 0 && do_put(ch))
        {
            --count;
        }
    }
}

BOOST_STRINGIFY_INLINE void narrow_file_writer::repeat
    ( char_type ch1
    , char_type ch2
    , std::size_t count
    )
{
    if( ! m_err)
    {
        while(count > 0 && do_put(ch1) && do_put(ch2))
        {
            --count;
        }
    }
}

BOOST_STRINGIFY_INLINE void narrow_file_writer::repeat
    ( char_type ch1
    , char_type ch2
    , char_type ch3
    , std::size_t count
    )
{
    if( ! m_err)
    {
        while(count > 0 && do_put(ch1) && do_put(ch2) && do_put(ch3))
        {
            --count;
        }
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
    if( ! m_err)
    {
        while
            ( count > 0
           && do_put(ch1)
           && do_put(ch2)
           && do_put(ch3)
           && do_put(ch4)
            )
        {
            --count;
        }
    }    
}

BOOST_STRINGIFY_INLINE std::error_code narrow_file_writer::finish()
{
    return m_err;
}

BOOST_STRINGIFY_INLINE bool narrow_file_writer::do_put(char_type ch)
{
    if(std::fputc(ch, m_file) == EOF)
    {
        m_err = std::error_code{errno, std::generic_category()};
        return false;
    }
    else
    {
        if(m_count != nullptr)
        {
            ++ *m_count;
        }
        return true;
    }
}

BOOST_STRINGIFY_INLINE void wide_file_writer::set_error(std::error_code err)
{
    if(err && !m_err)
    {
        m_err = err;
    }
}

BOOST_STRINGIFY_INLINE bool wide_file_writer::good() const
{
    return ! m_err;
}


BOOST_STRINGIFY_INLINE void wide_file_writer::put
    ( const char_type* str
    , std::size_t count
    )
{
    if ( ! m_err)
    {
        while(count > 0 && do_put(*str))
        {
            --count;
            ++str;
        }
    }
}

BOOST_STRINGIFY_INLINE void wide_file_writer::put(char_type ch)
{
    if ( ! m_err)
    {
        do_put(ch);
    }
}

BOOST_STRINGIFY_INLINE void wide_file_writer::repeat(char_type ch, std::size_t count)
{
    if( ! m_err)
    {
        while(count > 0 && do_put(ch))
        {
            --count;
        }
    }
}

BOOST_STRINGIFY_INLINE void wide_file_writer::repeat
    ( char_type ch1
    , char_type ch2
    , std::size_t count
    )
{
    if( ! m_err)
    {
        while(count > 0 && do_put(ch1) && do_put(ch2))
        {
            --count;
        }
    }
}

BOOST_STRINGIFY_INLINE void wide_file_writer::repeat
    ( char_type ch1
    , char_type ch2
    , char_type ch3
    , std::size_t count
    )
{
    if( ! m_err)
    {
        while(count > 0 && do_put(ch1) && do_put(ch2) && do_put(ch3))
        {
            --count;
        }
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
    if( ! m_err)
    {
        while
            ( count > 0
           && do_put(ch1)
           && do_put(ch2)
           && do_put(ch3)
           && do_put(ch4)
            )
        {
            --count;
        }
    }
}

BOOST_STRINGIFY_INLINE std::error_code wide_file_writer::finish()
{
    return m_err;
}

BOOST_STRINGIFY_INLINE bool wide_file_writer::do_put(char_type ch)
{
    if(std::fputwc(ch, m_file) == WEOF) 
    //if(std::fwrite(&ch, sizeof(char_type), 1, m_file) != sizeof(char_type))
    {
        m_err = std::error_code{errno, std::generic_category()};
        return false;
    }
    else
    {
        if(m_count != nullptr)
        {
            ++ *m_count;
        }
        return true;
    }
}

#endif //! defined(BOOST_STRINGIFY_OMIT_IMPL)

} // namespace detail

inline auto write_to(std::FILE* destination, std::size_t* count = nullptr)
{
    using writer = stringify::v0::detail::narrow_file_writer;
    return stringify::v0::make_args_handler<writer>(destination, count);
}

inline auto wwrite_to(std::FILE* destination, std::size_t* count = nullptr)
{
    using writer = boost::stringify::v0::detail::wide_file_writer;
    return boost::stringify::v0::make_args_handler<writer>(destination, count);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_OUTPUT_TYPES_FILE_HPP

