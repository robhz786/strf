#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_FILE_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_FILE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <cstdio>
#include <cstring>
#include <boost/stringify/v0/output_writer.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename CharT>
class narrow_file_writer: public output_writer<CharT>
{

public:

    using char_type = CharT;

    narrow_file_writer(std::FILE* file, std::size_t* count)
        : m_file(file)
        , m_count(count)
    {
        if (m_count != nullptr)
        {
            *m_count = 0;
        }
    }

    ~narrow_file_writer()
    {
        if (m_buff_pos > 0)
        {
            flush();
        }
    }

    void set_error(std::error_code err) override
    {
        if(err && !m_err)
        {
            m_err = err;
        }
    }

    bool good() const override
    {
        return ! m_err;
    }

    bool put(const char_type* str, std::size_t count) override
    {
        if (m_err)
        {
            return false;
        }
        std::size_t remaining_buff_capacity = m_buff_size - m_buff_pos;
        if(remaining_buff_capacity < count)
        {
            if ( ! flush())
            {
                return false;
            }
            remaining_buff_capacity = m_buff_size;
        }
        if(remaining_buff_capacity < count)
        {
            return write(str, count);
        }
        std::memcpy(&m_buff[m_buff_pos], str, count * sizeof(char_type));
        m_buff_pos += count;
        return true;
    }

    bool put(char_type ch) override
    {
        if (m_err || (m_buff_pos == m_buff_size && ! flush()))
        {
            return false;
        }
        m_buff[m_buff_pos] = ch;
        ++m_buff_pos;
        return true;
    }

    bool repeat
        ( std::size_t count
        , char_type ch
        ) override
    {
        return count == 1
            ? put(ch)
            : do_repeat<1>(count, {{ch}});
    }

    bool repeat
        ( std::size_t count
        , char_type ch1
        , char_type ch2
        ) override
    {
        return count == 1
            ? put_sequence(char_array<2>{{ch1, ch2}})
            : do_repeat<2>(count, {{ch1, ch2}});
    }

    bool repeat
        ( std::size_t count
        , char_type ch1
        , char_type ch2
        , char_type ch3
        ) override
    {
        return count == 1
            ? put_sequence<3>({{ch1, ch2, ch3}})
            : do_repeat<3>(count, {{ch1, ch2, ch3}});
    }

    bool repeat
        ( std::size_t count
        , char_type ch1
        , char_type ch2
        , char_type ch3
        , char_type ch4
        ) override
    {
        return count == 1
            ? put_sequence<4>({{ch1, ch2, ch3, ch4}})
            : do_repeat<4>(count, {{ch1, ch2, ch3, ch4}});
    }

    std::error_code finish_error_code()
    {
        if (m_buff_pos > 0)
        {
            flush();
        }
        return m_err;
    }

    void finish_exception()
    {
        if (m_buff_pos > 0)
        {
            flush();
        }
        if(m_err)
        {
            throw std::system_error(m_err);
        }
    }

private:

    template <std::size_t N>
    struct char_array
    {
        char_type arr[N];
    };

    template <std::size_t N>
    bool put_sequence(char_array<N> obj)
    {
        if(m_err || (m_buff_pos >= m_buff_size - N && ! flush()))
        {
            return false;
        }
        *reinterpret_cast<char_array<N>*>(&m_buff[m_buff_pos]) = obj;
        m_buff_pos += N;
        return true;
    }

    template <std::size_t N>
    bool do_repeat(std::size_t count, char_array<N> obj)
    {
        if(m_err)
        {
            return false;
        }
        while (count > 0)
        {
            std::size_t remaining_buff_capacity = (m_buff_size - m_buff_pos) / N;
            if(remaining_buff_capacity == 0)
            {
                if ( ! flush())
                {
                    return false;
                }
                remaining_buff_capacity = m_buff_size / N;
            }
            std::size_t sub_count = std::min(count, remaining_buff_capacity);
            auto* buff_head = reinterpret_cast<char_array<N>*>(&m_buff[m_buff_pos]);
            std::fill_n(buff_head, sub_count, obj);
            m_buff_pos += sub_count * N;
            count -= sub_count;
        }
        return true;
    }

    bool flush()
    {
        std::size_t pos = m_buff_pos;
        m_buff_pos = 0;
        return write(m_buff, pos);
    }

    bool write(const char_type* str, std::size_t count)
    {
        auto count_inc = std::fwrite(str, sizeof(char_type), count, m_file);

        if (m_count != nullptr)
        {
            *m_count += count_inc;
        }
        if (count != count_inc)
        {
            m_err = std::error_code{errno, std::generic_category()};
            return false;
        }
        return true;
    }

public:

    constexpr static std::size_t m_buff_size = 60;

private:

    std::size_t m_buff_pos = 0;
    char_type m_buff[m_buff_size];

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
        if (m_count != nullptr)
        {
            *m_count = 0;
        }
    }

    void set_error(std::error_code err) override;

    bool good() const override;

    bool put(const char_type* str, std::size_t count) override;

    bool put(char_type ch) override;

    bool repeat
        ( std::size_t count
        , char_type ch
        ) override;

    bool repeat
        ( std::size_t count
        , char_type ch1
        , char_type ch2
        ) override;

    bool repeat
        ( std::size_t count
        , char_type ch1
        , char_type ch2
        , char_type ch3
        ) override;

    bool repeat
        ( std::size_t count
        , char_type ch1
        , char_type ch2
        , char_type ch3
        , char_type ch4
        ) override;

    std::error_code finish_error_code();

    void finish_exception();

private:

    bool do_put(char_type ch);

    std::FILE* m_file;
    std::size_t* m_count = nullptr;
    std::error_code m_err;

};

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<wchar_t>;

#endif

#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

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


BOOST_STRINGIFY_INLINE bool wide_file_writer::put
    ( const char_type* str
    , std::size_t count
    )
{
    if ( ! m_err)
    {
        for (; count != 0; --count, ++str)
        {
            if(!do_put(*str))
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

BOOST_STRINGIFY_INLINE bool wide_file_writer::put(char_type ch)
{
    return ! m_err && do_put(ch);
}

BOOST_STRINGIFY_INLINE bool wide_file_writer::repeat(std::size_t count, char_type ch)
{
        if( ! m_err)
        {
            for(; count > 0; --count)
            {
                if (!do_put(ch))
                {
                    return false;
                }
            }
            return true;
        }
        return false;
}

BOOST_STRINGIFY_INLINE bool wide_file_writer::repeat
    ( std::size_t count
    , char_type ch1
    , char_type ch2
    )
{
    if( ! m_err)
    {
        for(; count > 0; --count)
        {
            if(!do_put(ch1) || !do_put(ch2))
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

BOOST_STRINGIFY_INLINE bool wide_file_writer::repeat
    ( std::size_t count
    , char_type ch1
    , char_type ch2
    , char_type ch3
    )
{
    if( ! m_err)
    {
        for(; count > 0; --count)
        {
            if(!do_put(ch1) || !do_put(ch2) || !do_put(ch3))
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

BOOST_STRINGIFY_INLINE bool wide_file_writer::repeat
    ( std::size_t count
    , char_type ch1
    , char_type ch2
    , char_type ch3
    , char_type ch4
    )
{
    if( ! m_err)
    {
        for(; count > 0; --count)
        {
            if(!do_put(ch1) || !do_put(ch2) || !do_put(ch3) || !do_put(ch4))
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

BOOST_STRINGIFY_INLINE std::error_code wide_file_writer::finish_error_code()
{
    return m_err;
}

BOOST_STRINGIFY_INLINE void wide_file_writer::finish_exception()
{
    if(m_err)
    {
        throw std::system_error(m_err);
    }
}

BOOST_STRINGIFY_INLINE bool wide_file_writer::do_put(char_type ch)
{
    if(std::fputwc(ch, m_file) == WEOF)
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

template <typename CharT = char>
inline auto format(std::FILE* destination, std::size_t* count = nullptr)
{
    using writer = stringify::v0::detail::narrow_file_writer<CharT>;
    return stringify::v0::make_args_handler<writer>(destination, count);
}

inline auto wformat(std::FILE* destination, std::size_t* count = nullptr)
{
    using writer = boost::stringify::v0::detail::wide_file_writer;
    return stringify::v0::make_args_handler<writer>(destination, count);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_OUTPUT_TYPES_FILE_HPP

