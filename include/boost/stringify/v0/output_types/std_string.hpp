#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STRING_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/syntax.hpp>
#include <system_error>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename StringType>
class string_appender: public output_writer<typename StringType::value_type>
{
public:
    typedef typename StringType::value_type char_type;

    string_appender(StringType& out)
        : m_out(&out)
        , m_initial_length(out.length())
    {
    }

    ~string_appender()
    {
        if (! m_finished && m_out != nullptr)
        {
            m_out->resize(m_initial_length);
        }
    }

    void set_error(std::error_code err) override
    {
        if (err && ! m_err)
        {
            m_err = err;
            if (m_out != nullptr)
            {
                m_out->resize(m_initial_length);
                m_out = nullptr;
            }
        }
    }

    bool good() const override
    {
        return ! m_err;
    }


    bool put(const char_type* str, std::size_t count) override
    {
        if(m_out != nullptr)
        {
            m_out->append(str, count);
            return true;
        }
        return false;
    }

    bool put(char_type character) override
    {
        if(m_out != nullptr)
        {
            m_out->push_back(character);
            return true;
        }
        return false;
    }

    bool repeat(std::size_t count, char_type character) override
    {
        if(m_out != nullptr)
        {
            m_out->append(count, character);
            return true;
        }
        return false;
    }

    bool repeat
        ( std::size_t count
        , char_type ch1
        , char_type ch2
        ) override
    {
        if(m_out != nullptr)
        {
            for(; count > 0; --count)
            {
                m_out->push_back(ch1);
                m_out->push_back(ch2);
            }
            return true;
        }
        return false;
    }

    bool repeat
        ( std::size_t count
        , char_type ch1
        , char_type ch2
        , char_type ch3
        ) override
    {
        if(m_out != nullptr)
        {
            for(; count > 0; --count)
            {
                m_out->push_back(ch1);
                m_out->push_back(ch2);
                m_out->push_back(ch3);
            }
            return true;
        }
        return false;
    }

    bool repeat
        ( std::size_t count
        , char_type ch1
        , char_type ch2
        , char_type ch3
        , char_type ch4
        ) override
    {
        if(m_out != nullptr)
        {
            for(; count > 0; --count)
            {
                m_out->push_back(ch1);
                m_out->push_back(ch2);
                m_out->push_back(ch3);
                m_out->push_back(ch4);
            }
            return true;
        }
        return false;
    }

    std::error_code finish_error_code()
    {
        m_finished = true;
        return m_err;
    }

    void finish_exception()
    {
        m_finished = true;
        if(m_err)
        {
            throw std::system_error(m_err);
        }
    }

    void reserve(std::size_t size)
    {
        if(m_out != nullptr)
        {
            m_out->reserve(m_out->length() + size);
        }
    }

private:

    StringType* m_out = nullptr;
    std::size_t m_initial_length = 0;
    std::error_code m_err;
    bool m_finished = false;
};

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_appender<std::string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_appender<std::u16string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_appender<std::u32string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_appender<std::wstring>;

#endif

} // namespace detail


template <typename CharT, typename Traits, typename Allocator>
auto append(std::basic_string<CharT, Traits, Allocator>& str)
{
    using string_type = std::basic_string<CharT, Traits, Allocator>;
    using writer = boost::stringify::v0::detail::string_appender<string_type>;
    return boost::stringify::v0::make_args_handler<writer, string_type&>(str);
}


template <typename CharT, typename Traits, typename Allocator>
auto assign(std::basic_string<CharT, Traits, Allocator>& str)
{
    using string_type = std::basic_string<CharT, Traits, Allocator>;
    str.clear();
    using writer = boost::stringify::v0::detail::string_appender<string_type>;
    return boost::stringify::v0::make_args_handler<writer, string_type&>(str);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STRING_HPP

