#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STRING_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/output_writer.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename StringType>
class string_appender: public output_writer<typename StringType::value_type>
{
public:
    typedef typename StringType::value_type char_type;

    string_appender(StringType& out)
        : m_out(out)
    {
    }

    string_appender(const string_appender& r)
        : m_out(r.m_out)
    {
    }

    void put(const char_type* str, std::size_t count) override
    {
        m_out.append(str, count);
    }

    void put(char_type character) override
    {
        m_out.push_back(character);
    }

    void repeat(char_type character, std::size_t repetitions) override
    {
        m_out.append(repetitions, character);
    }

    void repeat
        ( char_type ch1
        , char_type ch2
        , std::size_t count
        ) override
    {
        for(; count > 0; --count)
        {
            m_out.push_back(ch1);
            m_out.push_back(ch2);
        }
    }

    void repeat
        ( char_type ch1
        , char_type ch2
        , char_type ch3
        , std::size_t count
        ) override
    {
        for(; count > 0; --count)
        {
            m_out.push_back(ch1);
            m_out.push_back(ch2);
            m_out.push_back(ch3);
        }
    }

    void repeat
        ( char_type ch1
        , char_type ch2
        , char_type ch3
        , char_type ch4
        , std::size_t count
        ) override
    {
        for(; count > 0; --count)
        {
            m_out.push_back(ch1);
            m_out.push_back(ch2);
            m_out.push_back(ch3);
            m_out.push_back(ch4);
        }
    }

    StringType& finish()
    {
        return m_out;
    }

    void reserve(std::size_t size)
    {
        m_out.reserve(m_out.capacity() + size);
    }

private:

    StringType& m_out;
};

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_appender<std::string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_appender<std::u16string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_appender<std::u32string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_appender<std::wstring>;

#endif 

} // namespace detail


template <typename CharT, typename Traits, typename Allocator>
auto append_to(std::basic_string<CharT, Traits, Allocator>& str)
{
    using string_type = std::basic_string<CharT, Traits, Allocator>;
    using writer = boost::stringify::v0::detail::string_appender<string_type>;
    return boost::stringify::v0::make_args_handler<writer, string_type&>(str);
}


template <typename CharT, typename Traits, typename Allocator>
auto assign_to(std::basic_string<CharT, Traits, Allocator>& str)
{
    using string_type = std::basic_string<CharT, Traits, Allocator>;
    str.clear();
    using writer = boost::stringify::v0::detail::string_appender<string_type>;
    return boost::stringify::v0::make_args_handler<writer, string_type&>(str);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STRING_HPP

