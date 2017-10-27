#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STREAMBUF_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STREAMBUF_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <streambuf>
#include <boost/stringify/v0/output_writer.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename CharT, typename Traits>
class std_streambuf_writer: public output_writer<CharT>
{
public:

    using char_type = CharT;

    explicit std_streambuf_writer
        ( std::basic_streambuf<CharT, Traits>& out
        , std::size_t* count
        )
        : m_out(out)
        , m_count(count)
    {
        if(m_count)
        {
            *m_count = 0;
        }
    }

    bool good() const override
    {
        return ! m_err;
    }

    void set_error(std::error_code err) override
    {
        if (err && ! m_err)
        {
            m_err = err;
        }
    }

    bool put(const CharT* str, std::size_t ucount) override
    {
        if( ! m_err)
        {
            std::streamsize count = ucount;
            auto count_inc = m_out.sputn(str, count);
            if(m_count != nullptr && count_inc > 0)
            {
                *m_count += static_cast<std::size_t>(count_inc);
            }
            if (count_inc != count)
            {
                m_err = std::make_error_code(std::errc::io_error);
            }
        }
        return false;
    }

    bool put(CharT ch) override
    {
        return ! m_err && do_put(ch);
    }

    bool repeat(std::size_t count, CharT ch) override
    {
        if( ! m_err)
        {
            for(; count > 0; --count)
            {
                if(!do_put(ch))
                {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    bool repeat(std::size_t count, CharT ch1, CharT ch2) override
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

    bool repeat(std::size_t count, CharT ch1, CharT ch2, CharT ch3) override
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

    bool repeat
        ( std::size_t count
        , CharT ch1
        , CharT ch2
        , CharT ch3
        , CharT ch4
        ) override
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

    std::error_code finish() noexcept
    {
        return m_err;
    }

private:

    bool do_put(CharT character)
    {
        if(Traits::eq_int_type(m_out.sputc(character), Traits::eof()))
        {
            m_err = std::make_error_code(std::errc::io_error);
            return false;
        }
        if(m_count != nullptr)
        {
            ++ *m_count;
        }
        return true;
    }


    std::basic_streambuf<CharT, Traits>& m_out;
    std::size_t * m_count = 0;
    std::error_code m_err;

};

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class std_streambuf_writer<char, std::char_traits<char>>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class std_streambuf_writer<char16_t, std::char_traits<char16_t>>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class std_streambuf_writer<char32_t, std::char_traits<char32_t>>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class std_streambuf_writer<wchar_t, std::char_traits<wchar_t>>;

#endif

} // namespace detail


template<typename CharT, typename Traits = std::char_traits<CharT> >
auto write_to
    ( std::basic_streambuf<CharT, Traits>& dest
    , std::size_t* count = nullptr
    )
{
    using intput_type = std::basic_streambuf<CharT, Traits>&;
    using writer = stringify::v0::detail::std_streambuf_writer<CharT, Traits>;
    return stringify::v0::make_args_handler<writer, intput_type>(dest, count);
}


template<typename CharT, typename Traits = std::char_traits<CharT> >
auto write_to
    ( std::basic_streambuf<CharT, Traits>* dest
    , std::size_t* count = nullptr
    )
{
    using intput_type = std::basic_streambuf<CharT, Traits>&;
    using writer = stringify::v0::detail::std_streambuf_writer<CharT, Traits>;
    return stringify::v0::make_args_handler<writer, intput_type>(*dest, count);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STREAMBUF_HPP

