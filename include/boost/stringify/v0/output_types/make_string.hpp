#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_MAKE_STRING_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_MAKE_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/output_writer.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {

template <typename StringType>
class string_maker: public output_writer<typename StringType::value_type>
{
public:

    typedef typename StringType::value_type char_type;

    string_maker() = default;

    string_maker(const string_maker&) = delete;

    string_maker(string_maker&&) = default;

    void put(const char_type* str, std::size_t count) override
    {
        m_out.append(str, count);
    }

    void put(char_type ch) override
    {
        m_out.push_back(ch);
    }

    void repeat(char_type ch, std::size_t count) override
    {
        m_out.append(count, ch);
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

    StringType finish()
    {
        return std::move(m_out);
    }

    void reserve(std::size_t size)
    {
        m_out.reserve(m_out.capacity() + size);
    }

private:

    StringType m_out;
};

} // namespace detail


template
    < typename CharT
    , typename Traits = std::char_traits<CharT>
    , typename Allocator = std::allocator<CharT>
    >
constexpr auto make_basic_string
= boost::stringify::v0::make_args_handler
    <boost::stringify::v0::detail::string_maker
         <std::basic_string<CharT, Traits, Allocator>>>();


constexpr auto make_string
= boost::stringify::v0::make_args_handler
    <boost::stringify::v0::detail::string_maker<std::string>>();

constexpr auto make_u16string
= boost::stringify::v0::make_args_handler
    <boost::stringify::v0::detail::string_maker<std::u16string>>();

constexpr auto make_u32string
= boost::stringify::v0::make_args_handler
    <boost::stringify::v0::detail::string_maker<std::u32string>>();

constexpr auto make_wstring
= boost::stringify::v0::make_args_handler
    <boost::stringify::v0::detail::string_maker<std::wstring>>();


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_OUTPUT_TYPES_MAKE_STRING_HPP

