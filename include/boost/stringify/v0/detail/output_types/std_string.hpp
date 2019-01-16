#ifndef BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STRING_HPP
#define BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <system_error>
#include <boost/stringify/v0/make_destination.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename StringType>
class string_appender final
    : public stringify::v0::output_buffer<typename StringType::value_type>
{
    constexpr static std::size_t _buff_size = stringify::v0::min_buff_size;
    typename StringType::value_type _buff[_buff_size];

public:

    typedef typename StringType::value_type char_type;

    string_appender ( StringType& out )
        : output_buffer<char_type>{_buff, _buff + _buff_size}
        , _out(out)
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

    bool recycle() override;

    stringify::v0::expected<std::size_t, std::error_code> finish();

private:

    StringType& _out;
    std::size_t _initial_length = 0;
    bool _finished = false;
};

template <typename StringType>
bool string_appender<StringType>::recycle()
{
    auto pos = this->pos();
    BOOST_ASSERT(_buff <= pos && pos <= _buff + _buff_size);
    _out.append(_buff, pos);
    this->reset(_buff, _buff + _buff_size);
    return true;
}

template <typename StringType>
stringify::v0::expected<std::size_t, std::error_code>
string_appender<StringType>::finish()
{
    auto pos = this->pos();
    _finished = true;
    if ( ! this->has_error() )
    {
        _out.append(_buff, pos);
        return { boost::stringify::v0::in_place_t{}
               , _out.size() - _initial_length };
    }
    _out.resize(_initial_length);
    return { boost::stringify::v0::unexpect_t{}, this->get_error() };
}


template <typename StringType>
class string_maker final
    : public stringify::v0::output_buffer<typename StringType::value_type>
{
    constexpr static std::size_t _buff_size = stringify::v0::min_buff_size;
    typename StringType::value_type _buff[_buff_size];

public:

    using char_type = typename StringType::value_type;

    string_maker()
        : output_buffer<char_type>{_buff, _buff + _buff_size}
    {
    }

    ~string_maker()
    {
    }

    bool recycle() override;

    stringify::v0::expected<StringType, std::error_code> finish();

    void reserve(std::size_t size)
    {
        _out.reserve(_out.size() + size);
    }

private:

    StringType _out;
};

template <typename StringType>
bool string_maker<StringType>::recycle()
{
    auto pos = this->pos();
    BOOST_ASSERT(_buff <= pos && pos <= _buff + _buff_size);
    _out.append(_buff, pos);
    this->reset(_buff, _buff + _buff_size);
    return true;
}

template <typename StringType>
stringify::v0::expected<StringType, std::error_code>
inline string_maker<StringType>::finish()
{
    auto pos = this->pos();
    BOOST_ASSERT(_buff <= pos && pos <= _buff + _buff_size);
    if ( ! this->has_error())
    {
        _out.append(_buff, pos);
        return {boost::stringify::v0::in_place_t{}, std::move(_out)};
    }
    return { stringify::v0::unexpect_t{}, this->get_error() };
}


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

#endif  // BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STRING_HPP

