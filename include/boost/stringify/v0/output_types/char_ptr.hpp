#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_CHAR_PTR_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_CHAR_PTR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <boost/stringify/v0/syntax.hpp>
#include <boost/stringify/v0/expected.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail{

template<typename CharT>
class char_ptr_writer: public output_writer<CharT>
{
    using Traits = std::char_traits<CharT>;

public:

    using char_type = CharT;

    char_ptr_writer() = delete;

    char_ptr_writer
        ( stringify::v0::output_writer_init<CharT> init
        , CharT* destination
        , CharT* end
        )
        : stringify::v0::output_writer<CharT>{init}
        , m_begin{destination}
        , m_it{destination}
        , m_end{end}
    {
        if(m_end < m_begin)
        {
            set_overflow_error();
        }
    }

    ~char_ptr_writer()
    {
        if (! m_finished) // means that an exception has been thrown
        {
            if(m_begin != m_end)
            {
                Traits::assign(*m_begin, CharT());
            }
        }
    }

    void set_error(std::error_code err) override
    {
        if (m_good)
        {
            m_err = err;
            m_good = false;
            if (m_begin != m_end)
            {
                Traits::assign(*m_begin, CharT());
            }
            // prevent any further writting:
            m_end = m_begin;
            m_it = m_begin;
        }
    }

    bool good() const override
    {
        return ! m_err;
    }


    bool put(stringify::v0::piecemeal_writer<char_type>& src) override
    {
        m_it = src.write(m_it, m_end);
        if(src.more())
        {
            set_overflow_error();
            return false;
        }
        if(src.success())
        {
            return true;
        }
        set_error(src.get_error());
        return false;
    };

    
    // bool put32(char32_t ch) override
    // {
    //     auto it = this->encode(m_it, m_end, ch);
    //     if(it != nullptr && it != m_end + 1)
    //     {
    //         m_it = it;
    //         return true;
    //     }
    //     if (it == m_end + 1)
    //     {
    //         set_overflow_error();
    //         return false;
    //     }
    //     return this->signal_encoding_error();
    // }

    // bool put32(std::size_t count, char32_t ch) override
    // {
    //     auto res = this->encode(m_it, m_end, count, ch);
    //     if (res.dest_it == nullptr)
    //     {
    //         return this->signal_encoding_error();
    //     }
    //     if(res.count == count)
    //     {
    //         m_it = res.dest_it;
    //         return true;
    //     }
    //     set_overflow_error();
    //     return false;
    // }

    bool put(const CharT* str, std::size_t count) override
    {
        if (m_it + count >= m_end)
        {
            set_overflow_error();
            return false;
        }
        Traits::copy(m_it, str, count);
        m_it += count;
        return true;
    }

    bool put(CharT ch) override
    {
        if (m_it + 1 == m_end)
        {
            set_overflow_error();
            return false;
        }
        Traits::assign(*m_it, ch);
        ++m_it;
        return true;
     }

    bool put(std::size_t count, CharT ch) override
    {
        if (m_it + count >= m_end)
        {
            set_overflow_error();
            return false;
        }
        Traits::assign(m_it, count, ch);
        m_it += count;
        return true;
    }

    stringify::v0::expected<std::size_t, std::error_code>
    finish() noexcept
    {
        do_finish();
        if (m_good)
        {
            return {boost::stringify::v0::in_place_t{}, m_it - m_begin};
        }
        else
        {
            return {boost::stringify::v0::unexpect_t{}, m_err};
        }
    }

    void finish_exception()
    {
        do_finish();
        if(m_err != std::error_code{})
        {
            throw std::system_error(m_err);
        }
    }

private:

    void do_finish() noexcept
    {
        if ( ! m_finished)
        {
            if(m_begin != m_end)
            {
                if(m_it == m_end)
                {
                    m_err = std::make_error_code(std::errc::result_out_of_range);
                    m_good = false;
                    m_it = m_begin;
                }
                Traits::assign(*m_it, CharT());
            }
        }
        m_finished = true;
    }

    void set_overflow_error()
    {
        set_error(std::make_error_code(std::errc::result_out_of_range));
    }

    CharT* m_begin;
    CharT* m_it;
    CharT* m_end;
    std::error_code m_err;
    bool m_good = true;
    bool m_finished = false;
};

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_ptr_writer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_ptr_writer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_ptr_writer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_ptr_writer<wchar_t>;

#endif


} // namespace detail


template<std::size_t N>
auto write(char (&destination)[N])
{
    using writer = stringify::v0::detail::char_ptr_writer<char>;
    return stringify::v0::make_destination<writer, char*>
        (destination, destination + N);
}

template<std::size_t N>
auto write(char16_t (&destination)[N])
{
    using writer = stringify::v0::detail::char_ptr_writer<char16_t>;
    return stringify::v0::make_destination<writer, char16_t*>
        (destination, destination + N);
}

template<std::size_t N>
auto write(char32_t (&destination)[N])
{
    using writer = stringify::v0::detail::char_ptr_writer<char32_t>;
    return stringify::v0::make_destination<writer, char32_t*>
        (destination, destination + N);
}

template<std::size_t N>
auto write(wchar_t (&destination)[N])
{
    using writer = stringify::v0::detail::char_ptr_writer<wchar_t>;
    return stringify::v0::make_destination<writer, wchar_t*>
        (destination, destination + N);
}

inline auto write(char* destination, char* end)
{
    using writer = stringify::v0::detail::char_ptr_writer<char>;
    return stringify::v0::make_destination<writer, char*, char*>
        (destination, end);
}

inline auto write(char16_t* destination, char16_t* end)
{
    using writer = stringify::v0::detail::char_ptr_writer<char16_t>;
    return stringify::v0::make_destination<writer, char16_t*, char16_t*>
        (destination, end);
}

inline auto write(char32_t* destination, char32_t* end)
{
    using writer = stringify::v0::detail::char_ptr_writer<char32_t>;
    return stringify::v0::make_destination<writer, char32_t*, char32_t*>
        (destination, end);
}

inline auto write(wchar_t* destination, wchar_t* end)
{
    using writer = stringify::v0::detail::char_ptr_writer<wchar_t>;
    return stringify::v0::make_destination<writer, wchar_t*, wchar_t*>
        (destination, end);
}

inline auto write(char* destination, std::size_t count)
{
    using writer = stringify::v0::detail::char_ptr_writer<char>;
    return stringify::v0::make_destination<writer, char*, char*>
        (destination, destination + count);
}

inline auto write(char16_t* destination, std::size_t count)
{
    using writer = stringify::v0::detail::char_ptr_writer<char16_t>;
    return stringify::v0::make_destination<writer, char16_t*, char16_t*>
        (destination, destination + count);
}

inline auto write(char32_t* destination, std::size_t count)
{
    using writer = stringify::v0::detail::char_ptr_writer<char32_t>;
    return stringify::v0::make_destination<writer, char32_t*, char32_t*>
        (destination, destination + count);
}

inline auto write(wchar_t* destination, std::size_t count)
{
    using writer = stringify::v0::detail::char_ptr_writer<wchar_t>;
    return stringify::v0::make_destination<writer, wchar_t*, wchar_t*>
        (destination, destination + count);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_OUTPUT_TYPES_CHAR_PTR_HPP */

