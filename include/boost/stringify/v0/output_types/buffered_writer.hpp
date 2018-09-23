#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_BUFFERED_WRITER_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_BUFFERED_WRITER_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/expected.hpp>
#include <boost/stringify/v0/basic_types.hpp>

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
        // BOOST_ASSERT(buffer_size >= stringify::v0::min_buff_size);
    }

    ~buffered_writer()
    {
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

    bool put(stringify::v0::piecemeal_input<char_type>& src) final
    {
        if (!m_good)
        {
            return false;
        }

        do
        {
            m_it = src.get_some(m_it, m_end);
        }
        while(src.more() && do_flush());

        if(m_it != m_buff)
        {
            m_good = do_put(m_buff, m_it - m_buff);
            m_it = m_buff;
        }
        if(! src.success())
        {
            BOOST_ASSERT(m_err == std::error_code{});
            m_err = src.get_error();
            m_good = false;
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


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_OUTPUT_TYPES_BUFFERED_WRITER_HPP

