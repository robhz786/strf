#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_CHAR_PTR_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_CHAR_PTR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <boost/stringify/v0/syntax.hpp>
#include <boost/stringify/v0/output_writer.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail{

template<typename CharT, typename Traits>
class char_ptr_writer: public output_writer<CharT>
{
public:

    typedef CharT char_type;

    char_ptr_writer() = delete;

    ~char_ptr_writer()
    {
        if (! m_finished) // means that an exception has been thrown
        {
            if(m_begin != m_end)
            {
                Traits::assign(*m_begin, CharT());
            }
            if(m_out_count != nullptr)
            {
                *m_out_count = 0;
            }
        }
    }

    char_ptr_writer
        ( CharT* destination
        , CharT* end
        , std::size_t* out_count
        )
        : m_begin{destination}
        , m_it{destination}
        , m_end{end}
        , m_out_count{out_count}
    {
        if(m_end < m_begin)
        {
            set_overflow_error();
        }
    }

    void set_error(std::error_code err) override
    {
        if (err && ! m_err)
        {
            m_err = err;
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
        if (m_it + 1 >= m_end)
        {
            set_overflow_error();
            return false;
        }
        Traits::assign(*m_it, ch);
        ++m_it;
        return true;
     }

    bool repeat(std::size_t count, CharT ch) override
    {
        if (m_it + count >= m_end)
        {
            set_overflow_error();
            return false;
        }

        if (count == 1)
        {
            Traits::assign(*m_it, ch);
            ++m_it;
        }
        else
        {
            Traits::assign(m_it, count, ch);
            m_it += count;
        }
        return true;
    }

    bool repeat(std::size_t count, CharT ch1, CharT ch2) override
    {
        if (m_it + 2 * count >= m_end)
        {
            set_overflow_error();
            return false;
        }
        while(count > 0)
        {
            Traits::assign(m_it[0], ch1);
            Traits::assign(m_it[1], ch2);
            m_it += 2;
            --count;
        }
        return true;
    }

    bool repeat(std::size_t count, CharT ch1, CharT ch2, CharT ch3) override
    {
        if (m_it + 3 * count >= m_end)
        {
            set_overflow_error();
            return false;
        }
        while(count > 0)
        {
            Traits::assign(m_it[0], ch1);
            Traits::assign(m_it[1], ch2);
            Traits::assign(m_it[2], ch3);
            m_it += 3;
            --count;
        }
        return true;
    }

    bool repeat
        ( std::size_t count
        , CharT ch1
        , CharT ch2
        , CharT ch3
        , CharT ch4

        ) override
    {
        if (m_it + 4 * count >= m_end)
        {
            set_overflow_error();
            return false;
        }
        while (count > 0)
        {
            Traits::assign(m_it[0], ch1);
            Traits::assign(m_it[1], ch2);
            Traits::assign(m_it[2], ch3);
            Traits::assign(m_it[3], ch4);
            m_it += 4;
            --count;
        }
        return true;
    }

    std::error_code finish_error_code() noexcept
    {
        do_finish();
        return m_err;
    }

    void finish_exception()
    {
        do_finish();
        if(m_err)
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
                BOOST_ASSERT(m_it != m_end);
                Traits::assign(*m_it, CharT());
            }
            if(m_out_count != nullptr)
            {
                *m_out_count = (m_it - m_begin);
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
    std::size_t* m_out_count = nullptr;
    std::error_code m_err;
    bool m_finished = false;
};

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class char_ptr_writer<char, std::char_traits<char>>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class char_ptr_writer<char16_t, std::char_traits<char16_t>>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class char_ptr_writer<char32_t, std::char_traits<char32_t>>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class char_ptr_writer<wchar_t, std::char_traits<wchar_t>>;

#endif


} // namespace detail


template<typename CharTraits = std::char_traits<char>, std::size_t N>
auto format(char (&destination)[N], std::size_t* out_count = nullptr)
{
    using writer = stringify::v0::detail::char_ptr_writer<char, CharTraits>;
    return stringify::v0::make_args_handler<writer, char*>
        (destination, destination + N, out_count);
}

template<typename CharTraits = std::char_traits<char16_t>, std::size_t N>
auto format(char16_t (&destination)[N], std::size_t* out_count = nullptr)
{
    using writer = stringify::v0::detail::char_ptr_writer<char16_t, CharTraits>;
    return stringify::v0::make_args_handler<writer, char16_t*>
        (destination, destination + N, out_count);
}

template<typename CharTraits = std::char_traits<char32_t>, std::size_t N>
auto format(char32_t (&destination)[N], std::size_t* out_count = nullptr)
{
    using writer = stringify::v0::detail::char_ptr_writer<char32_t, CharTraits>;
    return stringify::v0::make_args_handler<writer, char32_t*>
        (destination, destination + N, out_count);
}

template<typename CharTraits = std::char_traits<wchar_t>, std::size_t N>
auto format(wchar_t (&destination)[N], std::size_t* out_count = nullptr)
{
    using writer = stringify::v0::detail::char_ptr_writer<wchar_t, CharTraits>;
    return stringify::v0::make_args_handler<writer, wchar_t*>
        (destination, destination + N, out_count);
}

template<typename CharTraits = std::char_traits<char> >
auto format
    ( char* destination
    , char* end
    , std::size_t* out_count = nullptr
    )
{
    using writer = stringify::v0::detail::char_ptr_writer<char, CharTraits>;
    return stringify::v0::make_args_handler<writer, char*, char*>
        (destination, end, out_count);
}

template<typename CharTraits = std::char_traits<char16_t> >
auto format
    ( char16_t* destination
    , char16_t* end
    , std::size_t* out_count = nullptr
    )
{
    using writer = stringify::v0::detail::char_ptr_writer<char16_t, CharTraits>;
    return stringify::v0::make_args_handler<writer, char16_t*, char16_t*>
        (destination, end, out_count);
}

template<typename CharTraits = std::char_traits<char32_t> >
auto format
    ( char32_t* destination
    , char32_t* end
    , std::size_t* out_count = nullptr
    )
{
    using writer = stringify::v0::detail::char_ptr_writer<char32_t, CharTraits>;
    return stringify::v0::make_args_handler<writer, char32_t*, char32_t*>
        (destination, end, out_count);
}

template<typename CharTraits = std::char_traits<wchar_t> >
auto format
    ( wchar_t* destination
    , wchar_t* end
    , std::size_t* out_count = nullptr
    )
{
    using writer = stringify::v0::detail::char_ptr_writer<wchar_t, CharTraits>;
    return stringify::v0::make_args_handler<writer, wchar_t*, wchar_t*>
        (destination, end, out_count);
}

template<typename CharTraits = std::char_traits<char> >
auto format
    ( char* destination
    , std::size_t count
    , std::size_t* out_count = nullptr
    )
{
    using writer = stringify::v0::detail::char_ptr_writer<char, CharTraits>;
    return stringify::v0::make_args_handler<writer, char*, char*>
        (destination, destination + count, out_count);
}

template<typename CharTraits = std::char_traits<char16_t> >
auto format
    ( char16_t* destination
    , std::size_t count
    , std::size_t* out_count = nullptr
    )
{
    using writer = stringify::v0::detail::char_ptr_writer<char16_t, CharTraits>;
    return stringify::v0::make_args_handler<writer, char16_t*, char16_t*>
        (destination, destination + count, out_count);
}

template<typename CharTraits = std::char_traits<char32_t> >
auto format
    ( char32_t* destination
    , std::size_t count
    , std::size_t* out_count = nullptr
    )
{
    using writer = stringify::v0::detail::char_ptr_writer<char32_t, CharTraits>;
    return stringify::v0::make_args_handler<writer, char32_t*, char32_t*>
        (destination, destination + count, out_count);
}

template<typename CharTraits = std::char_traits<wchar_t> >
auto format
    ( wchar_t* destination
    , std::size_t count
    , std::size_t* out_count = nullptr)
{
    using writer = stringify::v0::detail::char_ptr_writer<wchar_t, CharTraits>;
    return stringify::v0::make_args_handler<writer, wchar_t*, wchar_t*>
        (destination, destination + count, out_count);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_OUTPUT_TYPES_CHAR_PTR_HPP */

