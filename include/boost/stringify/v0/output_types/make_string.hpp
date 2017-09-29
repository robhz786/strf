#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_MAKE_STRING_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_MAKE_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <system_error>
#include <boost/stringify/v0/output_writer.hpp>
#include <boost/stringify/v0/detail/expected.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT, typename Traits>
using expected_basic_string = stringify::v0::detail::expected
    < std::basic_string<CharT, Traits>
    , std::error_code
    >;

using expected_string = stringify::v0::detail::expected
    < std::string
    , std::error_code
    >;

using expected_u16string = stringify::v0::detail::expected
    < std::u16string
    , std::error_code
    >;

using expected_u32string = stringify::v0::detail::expected
    < std::u32string
    , std::error_code
    >;

using expected_wstring = stringify::v0::detail::expected
    < std::wstring
    , std::error_code
    >;

namespace detail {

template <typename StringType>
class string_maker: public output_writer<typename StringType::value_type>
{
public:

    using char_type = typename StringType::value_type;

    string_maker()
    {
    }

    void set_error(std::error_code err) override
    {
        if(err && ! m_err)
        {
            m_err = err;
        }
    }

    bool good() const override
    {
        return ! m_err;
    }

    void put(const char_type* str, std::size_t count) override
    {
        if( ! m_err)
        {
            m_out.append(str, count);
        }
    }

    void put(char_type ch) override
    {
        if( ! m_err)
        {
            m_out.push_back(ch);
        }
    }

    void repeat(std::size_t count, char_type ch) override
    {
        if( ! m_err)
        {
            m_out.append(count, ch);
        }
    }

    void repeat
        ( std::size_t count
        , char_type ch1
        , char_type ch2
        
        ) override
    {
        if( ! m_err)
        {
            for(; count > 0; --count)
            {
                m_out.push_back(ch1);
                m_out.push_back(ch2);
            }
        }
    }

    void repeat
        ( std::size_t count
        , char_type ch1
        , char_type ch2
        , char_type ch3
        
        ) override
    {
        if( ! m_err)
        {
            for(; count > 0; --count)
            {
                m_out.push_back(ch1);
                m_out.push_back(ch2);
                m_out.push_back(ch3);
            }
        }
    }

    void repeat
        ( std::size_t count
        , char_type ch1
        , char_type ch2
        , char_type ch3
        , char_type ch4
        
        ) override
    {
        if( ! m_err)
        {
            for(; count > 0; --count)
            {
                m_out.push_back(ch1);
                m_out.push_back(ch2);
                m_out.push_back(ch3);
                m_out.push_back(ch4);
            }
        }
    }

    stringify::v0::detail::expected<StringType, std::error_code> finish()
    {
        if (m_err)
        {
            return m_err;
        }
        return std::move(m_out);
    }

    void reserve(std::size_t size)
    {
        m_out.reserve(m_out.capacity() + size);
    }

private:

    StringType m_out;
    std::error_code m_err;
};


#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_maker<std::string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_maker<std::u16string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_maker<std::u32string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_maker<std::wstring>;

#endif

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

