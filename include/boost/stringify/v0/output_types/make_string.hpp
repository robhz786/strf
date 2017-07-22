#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_MAKE_STRING_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_MAKE_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/output_writer.hpp>

namespace boost {
namespace stringify {
inline namespace v0 {
namespace detail {

template <typename StringType>
class string_maker: public output_writer<typename StringType::value_type>
{
public:

    typedef typename StringType::value_type char_type;
    
    string_maker() = default;

    string_maker(const string_maker&) = delete;

    string_maker(string_maker&&) = default;
    
    void put(char_type character) override
    {
        m_out.push_back(character);
    }

    void repeat(char_type character, std::size_t repetitions) override
    {
        m_out.append(repetitions, character);
    }

    void put(const char_type* str, std::size_t count) override
    {
        m_out.append(str, count);
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



} // inline namespace v0
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_V0_OUTPUT_TYPES_MAKE_STRING_HPP

