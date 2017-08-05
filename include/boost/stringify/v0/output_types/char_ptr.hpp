#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_CHAR_PTR_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_CHAR_PTR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <boost/stringify/v0/args_handler.hpp>
#include <boost/stringify/v0/output_writer.hpp>

namespace boost{
namespace stringify{
inline namespace v0 {

struct char_ptr_result
{
    std::size_t length;
    bool success;
};

namespace detail{

template<typename CharT, typename Traits>
class char_ptr_writer: public output_writer<CharT>
{
public:

    typedef CharT char_type;

    char_ptr_writer(const char_ptr_writer&) = default;

    explicit char_ptr_writer(CharT* out)
        : m_out(out)
    {
    }

    void put(const CharT* str, std::size_t count) override
    {
        Traits::copy(m_out, str, count);
        m_out += count;
    }

    void put(CharT character) override
    {
        Traits::assign(*m_out++, character);
    }

    void repeat(CharT character, std::size_t count) override
    {
        Traits::assign(m_out, count, character);
        m_out += count;
    }

    void repeat(CharT ch1, CharT ch2, std::size_t count) override
    {
        std::size_t n = 2 * count;
        for(std::size_t i=0; i < n; i += 2)
        {
            Traits::assign(m_out[i],   ch1);
            Traits::assign(m_out[i+1], ch2);
        }
        m_out += n;
    }

    void repeat(CharT ch1, CharT ch2, CharT ch3, std::size_t count) override
    {
        std::size_t n = 3 * count;
        for(std::size_t i=0; i < n; i += 3)
        {
            Traits::assign(m_out[i],   ch1);
            Traits::assign(m_out[i + 1], ch2);
            Traits::assign(m_out[i + 2], ch3);
        }
        m_out += n;
    }

    void repeat(CharT ch1, CharT ch2, CharT ch3, CharT ch4, std::size_t count) override
    {
        std::size_t n = 4 * count;
        for(std::size_t i=0; i < n; i += 4)
        {
            Traits::assign(m_out[i],   ch1);
            Traits::assign(m_out[i + 1], ch2);
            Traits::assign(m_out[i + 2], ch3);
            Traits::assign(m_out[i + 3], ch4);
        }
        m_out += n;
    }

    CharT* finish() noexcept
    {
        Traits::assign(*m_out, CharT());
        return m_out;
    }

private:

    CharT* m_out;
};


template<typename CharT, typename Traits>
class limited_char_ptr_writer: public output_writer<CharT>
{
public:

    typedef CharT char_type;

    limited_char_ptr_writer(const limited_char_ptr_writer&) = default;

    explicit limited_char_ptr_writer(CharT* destination, CharT* end)
        : m_begin(destination)
        , m_it(destination)
        , m_end(end)
    {
    }

    void put(const CharT* str, std::size_t count) override
    {
        count = std::min(count, static_cast<std::size_t>(m_end - m_it));
        Traits::copy(m_it, str, count);
        m_it += count;
    }

    void put(CharT character) override
    {
        if (m_it != m_end)
        {
            Traits::assign(*m_it, character);
            ++m_it;
        }
    }

    void repeat(CharT character, std::size_t count) override
    {
        count = std::min(count, static_cast<std::size_t>(m_end - m_it));
        Traits::assign(m_it, count, character);
        m_it += count;
    }
    
    void repeat(CharT ch1, CharT ch2, std::size_t count) override
    {
        for(;count > 0 && m_it != m_end; --count)
        {
            Traits::assign(*m_it, ch1);
            if(++m_it != m_end)
            {
                Traits::assign(*m_it, ch2);
            }
        }
    }

    void repeat(CharT ch1, CharT ch2, CharT ch3, std::size_t count) override
    {
        for(;count > 0 && m_it != m_end; --count)
        {
            Traits::assign(*m_it, ch1);
            if(++m_it == m_end)
            {
                break;
            }
            Traits::assign(*m_it, ch2);
            if(++m_it == m_end)
            {
                break;
            }
            Traits::assign(*m_it, ch3);
        }
    }

    void repeat(CharT ch1, CharT ch2, CharT ch3, CharT ch4, std::size_t count) override
    {
        for(;count > 0 && m_it != m_end; --count)
        {
            Traits::assign(*m_it, ch1);
            if(++m_it == m_end)
            {
                break;
            }
            Traits::assign(*m_it, ch2);
            if(++m_it == m_end)
            {
                break;
            }
            Traits::assign(*m_it, ch3);
            if(++m_it == m_end)
            {
                break;
            }
            Traits::assign(*m_it, ch4);
        }
    }

    boost::stringify::v0::char_ptr_result finish() noexcept
    {
        if(m_begin == m_end)
        {
            return {0, false};
        }
        if(m_it == m_end)
        {
            -- m_it;
            Traits::assign(*m_it, CharT());
            return {static_cast<std::size_t>(m_it - m_begin), false};
        }
        Traits::assign(*m_it, CharT());
        return {static_cast<std::size_t>(m_it - m_begin), true};
    }

private:

    CharT* m_begin;
    CharT* m_it;
    CharT* m_end;
};



} // namespace detail


template<typename CharTraits = std::char_traits<char> >
auto write_to(char* destination)
{
    using writer = boost::stringify::v0::detail::char_ptr_writer<char, CharTraits>;
    return boost::stringify::v0::make_args_handler<writer, char*>(destination);
}

template<typename CharTraits = std::char_traits<char16_t> >
auto write_to(char16_t* destination)
{
    using writer = boost::stringify::v0::detail::char_ptr_writer<char16_t, CharTraits>;
    return boost::stringify::v0::make_args_handler<writer, char16_t*>(destination);
}

template<typename CharTraits = std::char_traits<char32_t> >
auto write_to(char32_t* destination)
{
    using writer = boost::stringify::v0::detail::char_ptr_writer<char32_t, CharTraits>;
    return boost::stringify::v0::make_args_handler<writer, char32_t*>(destination);
}

template<typename CharTraits = std::char_traits<wchar_t> >
auto write_to(wchar_t* destination)
{
    using writer = boost::stringify::v0::detail::char_ptr_writer<wchar_t, CharTraits>;
    return boost::stringify::v0::make_args_handler<writer, wchar_t*>(destination);
}

template<typename CharTraits = std::char_traits<char> >
auto write_to(char* destination, char* end)
{
    using writer = boost::stringify::v0::detail::limited_char_ptr_writer<char, CharTraits>;
    return boost::stringify::v0::make_args_handler<writer, char*, char*>
        (destination, end);
}

template<typename CharTraits = std::char_traits<char16_t> >
auto write_to(char16_t* destination, char16_t* end)
{
    using writer = boost::stringify::v0::detail::limited_char_ptr_writer<char16_t, CharTraits>;
    return boost::stringify::v0::make_args_handler<writer, char16_t*, char16_t*>
        (destination, end);
}

template<typename CharTraits = std::char_traits<char32_t> >
auto write_to(char32_t* destination, char32_t* end)
{
    using writer = boost::stringify::v0::detail::limited_char_ptr_writer<char32_t, CharTraits>;
    return boost::stringify::v0::make_args_handler<writer, char32_t*, char32_t*>
        (destination, end);
}

template<typename CharTraits = std::char_traits<wchar_t> >
auto write_to(wchar_t* destination, wchar_t* end)
{
    using writer = boost::stringify::v0::detail::limited_char_ptr_writer<wchar_t, CharTraits>;
    return boost::stringify::v0::make_args_handler<writer, wchar_t*, wchar_t*>
        (destination, end);
}

template<typename CharTraits = std::char_traits<char> >
auto write_to(char* destination, std::size_t count)
{
    using writer = boost::stringify::v0::detail::limited_char_ptr_writer<char, CharTraits>;
    return boost::stringify::v0::make_args_handler<writer, char*, char*>
        (destination, destination + count);
}

template<typename CharTraits = std::char_traits<char16_t> >
auto write_to(char16_t* destination, std::size_t count)
{
    using writer = boost::stringify::v0::detail::limited_char_ptr_writer<char16_t, CharTraits>;
    return boost::stringify::v0::make_args_handler<writer, char16_t*, char16_t*>
        (destination, destination + count);
}

template<typename CharTraits = std::char_traits<char32_t> >
auto write_to(char32_t* destination, std::size_t count)
{
    using writer = boost::stringify::v0::detail::limited_char_ptr_writer<char32_t, CharTraits>;
    return boost::stringify::v0::make_args_handler<writer, char32_t*, char32_t*>
        (destination, destination + count);
}

template<typename CharTraits = std::char_traits<wchar_t> >
auto write_to(wchar_t* destination, std::size_t count)
{
    using writer = boost::stringify::v0::detail::limited_char_ptr_writer<wchar_t, CharTraits>;
    return boost::stringify::v0::make_args_handler<writer, wchar_t*, wchar_t*>
        (destination, destination + count);
}


} // inline namespace v0
} // namespace stringify
} // namespace boost

#endif  /* BOOST_STRINGIFY_V0_OUTPUT_TYPES_CHAR_PTR_HPP */

