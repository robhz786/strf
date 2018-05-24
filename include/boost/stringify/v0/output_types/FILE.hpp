#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_FILE_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_FILE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <cstdio>
#include <cstring>
#include <boost/stringify/v0/syntax.hpp>
#include <boost/stringify/v0/expected.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT>
class buffered_writer: public stringify::v0::output_writer<CharT>
{
public:

    using char_type = CharT;

    buffered_writer
        ( stringify::v0::output_writer_init<CharT> init
        , CharT* buffer
        , std::size_t buffer_size )
        : stringify::v0::output_writer<CharT>(init)
        , m_buff(buffer)
        , m_buff_size(buffer_size)
    {
    }

    ~buffered_writer()
    {
        BOOST_ASSERT_MSG(m_buff == m_it || ! m_good,
                         "you probably forgot to call flush() or discard() in the"
                         " destructor of a class deriving from buffered_writer");
    }

    void set_error(std::error_code err) override
    {
        if(m_err == std::error_code{})
        {
            flush();
            m_err = err;
            m_good = false;
        }
    }

    bool good() const override
    {
        return m_good;
    }

    bool put(stringify::v0::source<char_type>& src) final
    {
        if (m_good)
        {
            do
            {
                m_it = src.get(m_it, m_end);
            }
            while(src.more() && do_flush());
            if( ! m_good)
            {
                // set_error has been called. Force flush
                if(m_it != m_buff)
                {
                    do_put(m_buff, m_it - m_buff);
                    m_it = m_buff;
                }
            }
            else
            {
                m_good = src.success();
            }
        }
        return m_good;
    }

    bool put(const char_type* str, std::size_t count) final
    {
        if(m_good)
        {
            char_type* it_adv = m_it + count;
            if(it_adv <= m_end)
            {
                 std::char_traits<char_type>::copy(m_it, str, count);
                 m_it = it_adv;
            }
            else
            {
                m_good = flush() && do_put(str, count);
            }
        }
        return m_good;
    }

    bool put(char_type ch) final
    {
        if(m_good && (m_it != m_end || do_flush()))
        {
            *m_it = ch;
            ++m_it;
            return true;
        }
        return false;
    }

    bool put(std::size_t count, char_type ch) final
    {
        if (m_good)
        {
            do
            {
                std::size_t available = m_end - m_it;
                std::size_t sub_count = std::min(count, available);
                std::char_traits<char_type>::assign(m_it, sub_count, ch);
                m_it += sub_count;
                count -= sub_count;
            }
            while(count > 0 && flush());
        }
        return m_good;
    }

    stringify::v0::expected<void, std::error_code> finish()
    {
        flush();
        if(m_good)
        {
            return {};
        }
        else if (m_err == std::error_code{})
        {
            // this seems to me to be the closest to "unknown error"
            m_err = std::make_error_code(std::errc::operation_canceled);
        }
        return {stringify::v0::unexpect_t{}, m_err};
    }

    void finish_exception()
    {
        flush();
        if(m_err)
        {
            throw std::system_error(m_err);
        }
    }

    bool flush()
    {
        return m_it == m_buff || do_flush();
    }

    void discard()
    {
        m_it = m_buff;
    }

protected:

    virtual bool do_put(const CharT* str, std::size_t count) = 0;

private:

    bool do_flush()
    {
        std::size_t count = m_it - m_buff;
        m_it = m_buff;
        if (m_good)
        {
            m_good = do_put(m_buff, count);
        }
        return m_good;
    }

    std::error_code m_err;
    char_type* m_buff;
    std::size_t m_buff_size;
    char_type* m_it = m_buff;
    char_type* const m_end = m_buff + m_buff_size;
    std::size_t m_count = 0;
    bool m_good = true;

};

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class buffered_writer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class buffered_writer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class buffered_writer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class buffered_writer<wchar_t>;

#endif

namespace detail {

template <typename CharT>
class narrow_file_writer final: public stringify::v0::buffered_writer<CharT>
{
public:
    constexpr static std::size_t buff_size = 60;

private:
    CharT buff[buff_size];

public:

    using char_type = CharT;

    narrow_file_writer
        ( stringify::v0::output_writer_init<CharT> init
        , std::FILE* file
        , std::size_t* count
        )
        : stringify::v0::buffered_writer<CharT>{init, buff, buff_size}
        , m_file(file)
        , m_count(count)
    {
        if (m_count != nullptr)
        {
            *m_count = 0;
        }
    }

    ~narrow_file_writer()
    {
        this->flush();
    }

protected:

    bool do_put(const CharT* str, std::size_t count) override
    {
        auto count_inc = std::fwrite(str, sizeof(char_type), count, m_file);

        if (m_count != nullptr)
        {
            *m_count += count_inc;
        }
        if (count != count_inc)
        {
            this->set_error(std::error_code{errno, std::generic_category()});
            return false;
        }
        return true;
    }

    std::FILE* m_file;
    std::size_t* m_count = nullptr;
};





class wide_file_writer final: public stringify::v0::buffered_writer<wchar_t>
{
    constexpr static std::size_t buff_size = 60;
    wchar_t buff[buff_size];

public:

    using char_type = wchar_t;

    wide_file_writer
        ( stringify::v0::output_writer_init<wchar_t> init
        , std::FILE* file
        , std::size_t* count
        );

    ~wide_file_writer();

protected:

    bool do_put(const wchar_t* str, std::size_t count) override;

    std::FILE* m_file;
    std::size_t* m_count = nullptr;
};

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class narrow_file_writer<wchar_t>;

#endif

#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_INLINE wide_file_writer::wide_file_writer
    ( stringify::v0::output_writer_init<wchar_t> init
    , std::FILE* file
    , std::size_t* count
    )
    : stringify::v0::buffered_writer<wchar_t>{init, buff, buff_size}
    , m_file(file)
    , m_count(count)
{
    if (m_count != nullptr)
    {
        *m_count = 0;
    }
}

BOOST_STRINGIFY_INLINE wide_file_writer::~wide_file_writer()
{
    this->flush();
}

BOOST_STRINGIFY_INLINE bool wide_file_writer::do_put(const wchar_t* str, std::size_t count)
{
    std::size_t i = 0;
    bool good = true;
    for( ; i < count && good; ++i, ++str)
    {
        auto ret = std::fputwc(*str, m_file);
        if(ret == WEOF)
        {
            this->set_error(std::error_code{errno, std::generic_category()});
            good = false;;
        }
    }
    if (m_count != nullptr)
    {
        *m_count += i;
    }
    return good;
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

