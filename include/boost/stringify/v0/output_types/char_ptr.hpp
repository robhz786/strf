#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_CHAR_PTR_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_CHAR_PTR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <boost/stringify/v0/syntax.hpp>

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
        , std::size_t* out_count
        )
        : stringify::v0::output_writer<CharT>{init}
        , m_begin{destination}
        , m_it{destination}
        , m_end{end}
        , m_out_count{out_count}
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
            if(m_out_count != nullptr)
            {
                *m_out_count = 0;
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


    bool put(stringify::v0::source<char_type>& src) override
    {
        m_it = src.get(m_it, m_end);
        if(src.more())
        {
            set_overflow_error();
            return false;
        }
        return src.success();
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

    std::error_code finish_error_code() noexcept
    {
        do_finish();
        return m_err;
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
auto format(char (&destination)[N], std::size_t* out_count = nullptr)
{
    using writer = stringify::v0::detail::char_ptr_writer<char>;
    return stringify::v0::make_args_handler<writer, char*>
        (destination, destination + N, out_count);
}

template<std::size_t N>
auto format(char16_t (&destination)[N], std::size_t* out_count = nullptr)
{
    using writer = stringify::v0::detail::char_ptr_writer<char16_t>;
    return stringify::v0::make_args_handler<writer, char16_t*>
        (destination, destination + N, out_count);
}

template<std::size_t N>
auto format(char32_t (&destination)[N], std::size_t* out_count = nullptr)
{
    using writer = stringify::v0::detail::char_ptr_writer<char32_t>;
    return stringify::v0::make_args_handler<writer, char32_t*>
        (destination, destination + N, out_count);
}

template<std::size_t N>
auto format(wchar_t (&destination)[N], std::size_t* out_count = nullptr)
{
    using writer = stringify::v0::detail::char_ptr_writer<wchar_t>;
    return stringify::v0::make_args_handler<writer, wchar_t*>
        (destination, destination + N, out_count);
}

inline auto  format
    ( char* destination
    , char* end
    , std::size_t* out_count = nullptr
    )
{
    using writer = stringify::v0::detail::char_ptr_writer<char>;
    return stringify::v0::make_args_handler<writer, char*, char*>
        (destination, end, out_count);
}

inline auto format
    ( char16_t* destination
    , char16_t* end
    , std::size_t* out_count = nullptr
    )
{
    using writer = stringify::v0::detail::char_ptr_writer<char16_t>;
    return stringify::v0::make_args_handler<writer, char16_t*, char16_t*>
        (destination, end, out_count);
}

inline auto format
    ( char32_t* destination
    , char32_t* end
    , std::size_t* out_count = nullptr
    )
{
    using writer = stringify::v0::detail::char_ptr_writer<char32_t>;
    return stringify::v0::make_args_handler<writer, char32_t*, char32_t*>
        (destination, end, out_count);
}

inline auto format
    ( wchar_t* destination
    , wchar_t* end
    , std::size_t* out_count = nullptr
    )
{
    using writer = stringify::v0::detail::char_ptr_writer<wchar_t>;
    return stringify::v0::make_args_handler<writer, wchar_t*, wchar_t*>
        (destination, end, out_count);
}

inline auto format
    ( char* destination
    , std::size_t count
    , std::size_t* out_count = nullptr
    )
{
    using writer = stringify::v0::detail::char_ptr_writer<char>;
    return stringify::v0::make_args_handler<writer, char*, char*>
        (destination, destination + count, out_count);
}

inline auto format
    ( char16_t* destination
    , std::size_t count
    , std::size_t* out_count = nullptr
    )
{
    using writer = stringify::v0::detail::char_ptr_writer<char16_t>;
    return stringify::v0::make_args_handler<writer, char16_t*, char16_t*>
        (destination, destination + count, out_count);
}

inline auto format
    ( char32_t* destination
    , std::size_t count
    , std::size_t* out_count = nullptr
    )
{
    using writer = stringify::v0::detail::char_ptr_writer<char32_t>;
    return stringify::v0::make_args_handler<writer, char32_t*, char32_t*>
        (destination, destination + count, out_count);
}

inline auto format
    ( wchar_t* destination
    , std::size_t count
    , std::size_t* out_count = nullptr)
{
    using writer = stringify::v0::detail::char_ptr_writer<wchar_t>;
    return stringify::v0::make_args_handler<writer, wchar_t*, wchar_t*>
        (destination, destination + count, out_count);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_OUTPUT_TYPES_CHAR_PTR_HPP */

