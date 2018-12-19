#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STRING_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <system_error>
#include <boost/stringify/v0/basic_types.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename StringType>
class string_appender final
    : public stringify::v0::buffer_recycler<typename StringType::value_type>
{
    constexpr static std::size_t _buffer_size = stringify::v0::min_buff_size;
    typename StringType::value_type _buffer[_buffer_size];

public:

    typedef typename StringType::value_type char_type;

    string_appender ( StringType& out )
        : _out(out)
        , _initial_length(out.length())
    {
    }

    ~string_appender()
    {
        if( ! _finished)
        {
            _out.resize(_initial_length);
        }
    }

    void reserve(std::size_t size)
    {
        _out.reserve(_out.length() + size);
    }

    stringify::v0::expected_output_buffer<char_type> start() noexcept
    {
        return { stringify::v0::in_place_t{}
               , stringify::v0::output_buffer<char_type>
                   { _buffer, _buffer + _buffer_size } };
    }

    stringify::v0::expected_output_buffer<char_type> recycle(char_type* it) override
    {
        BOOST_ASSERT(_buffer <= it && it <= _buffer + _buffer_size);
        _out.append(_buffer, it);
        return start();
    }

    stringify::v0::expected<std::size_t, std::error_code> finish(char_type *it)
    {
        BOOST_ASSERT(_buffer <= it && it <= _buffer + _buffer_size);
        _finished = true;
        _out.append(_buffer, it);
        return { boost::stringify::v0::in_place_t{}, _out.size() - _initial_length };
    }


private:

    StringType& _out;
    std::size_t _initial_length = 0;
    bool _finished = false;
};


template <typename StringType>
class string_maker final
    : public stringify::v0::buffer_recycler<typename StringType::value_type>
{
    constexpr static std::size_t _buffer_size = stringify::v0::min_buff_size;
    typename StringType::value_type _buffer[_buffer_size];

public:

    using char_type = typename StringType::value_type;

    string_maker()
    {
    }

    ~string_maker()
    {
    }

    stringify::v0::expected_output_buffer<char_type> start() noexcept
    {
        return { stringify::v0::in_place_t{}
               , stringify::v0::output_buffer<char_type>
                   { _buffer, _buffer + _buffer_size } };
    }

    stringify::v0::expected_output_buffer<char_type> recycle(char_type* it) override
    {
        BOOST_ASSERT(_buffer <= it && it <= _buffer + _buffer_size);
        _out.append(_buffer, it);
        return start();
    }

    stringify::v0::expected<StringType, std::error_code> finish(char_type *it)
    {
        BOOST_ASSERT(_buffer <= it && it <= _buffer + _buffer_size);
        _out.append(_buffer, it);
        return {boost::stringify::v0::in_place_t{}, std::move(_out)};
    }

    void reserve(std::size_t size)
    {
        _out.reserve(_out.size() + size);
    }

private:

    StringType _out;
};

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_appender<std::string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_appender<std::u16string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_appender<std::u32string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_appender<std::wstring>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_maker<std::string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_maker<std::u16string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_maker<std::u32string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_maker<std::wstring>;

#endif

} // namespace detail

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class expected<std::string, std::error_code>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class expected<std::u16string, std::error_code>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class expected<std::u32string, std::error_code>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class expected<std::wstring, std::error_code>;

#endif

template <typename CharT, typename Traits, typename Allocator>
auto append(std::basic_string<CharT, Traits, Allocator>& str)
{
    using string_type = std::basic_string<CharT, Traits, Allocator>;
    using writer = boost::stringify::v0::detail::string_appender<string_type>;
    return boost::stringify::v0::make_destination<writer, string_type&>(str);
}


template <typename CharT, typename Traits, typename Allocator>
auto assign(std::basic_string<CharT, Traits, Allocator>& str)
{
    using string_type = std::basic_string<CharT, Traits, Allocator>;
    str.clear();
    using writer = boost::stringify::v0::detail::string_appender<string_type>;
    return boost::stringify::v0::make_destination<writer, string_type&>(str);
}

template
    < typename CharT
    , typename Traits = std::char_traits<CharT>
    , typename Allocator = std::allocator<CharT> >
constexpr auto to_basic_string
= boost::stringify::v0::make_destination
    <boost::stringify::v0::detail::string_maker
         <std::basic_string<CharT, Traits, Allocator>>>();

constexpr auto to_string
= boost::stringify::v0::make_destination
    <boost::stringify::v0::detail::string_maker<std::string>>();

constexpr auto to_u16string
= boost::stringify::v0::make_destination
    <boost::stringify::v0::detail::string_maker<std::u16string>>();

constexpr auto to_u32string
= boost::stringify::v0::make_destination
    <boost::stringify::v0::detail::string_maker<std::u32string>>();

constexpr auto to_wstring
= boost::stringify::v0::make_destination
    <boost::stringify::v0::detail::string_maker<std::wstring>>();

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STRING_HPP

