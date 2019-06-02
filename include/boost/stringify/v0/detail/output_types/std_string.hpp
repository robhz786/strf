#ifndef BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STRING_HPP
#define BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <system_error>
#include <boost/stringify/v0/dispatcher.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

#if !defined(BOOST_NO_EXCEPTIONS)

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
    }

    void reserve(std::size_t size)
    {
        _out.reserve(_out.length() + size);
    }

    void recycle() override;

    std::size_t finish();

private:

    StringType& _out;
    std::size_t _initial_length = 0;
};

template <typename StringType>
void string_appender<StringType>::recycle()
{
    auto pos = this->pos();
    BOOST_ASSERT(_buff <= pos && pos <= _buff + _buff_size);
    _out.append(_buff, pos);
    this->set_pos(_buff);
}

template <typename StringType>
std::size_t string_appender<StringType>::finish()
{
    auto pos = this->pos();
    BOOST_ASSERT(_buff <= pos && pos <= _buff + _buff_size);
    if (pos != _buff)
    {
        _out.append(_buff, pos);
    }
    return _out.size() - _initial_length;
}

template <typename StringType>
class string_maker final
    : public stringify::v0::output_buffer<typename StringType::value_type>
{
    constexpr static std::size_t _buff_size = stringify::v0::min_buff_size;
    typename StringType::value_type _buff[_buff_size];

public:

    using char_type = typename StringType::value_type;

    string_maker();

    ~string_maker();

#if defined(__GNUC__) && (__GNUC__ < 7)

    string_maker(const string_maker&)
        : output_buffer<char_type>{_buff, _buff + _buff_size}
    {
        BOOST_ASSERT(false);
    }

#endif

    void recycle() override;

    StringType finish();

    void reserve(std::size_t size)
    {
        _out.reserve(_out.size() + size);
    }

private:

    StringType _out;
};

template <typename StringType>
inline string_maker<StringType>::string_maker()
    : output_buffer<char_type>{_buff, _buff + _buff_size}
{
}

template <typename StringType>
inline string_maker<StringType>::~string_maker()
{
}

template <typename StringType>
void string_maker<StringType>::recycle()
{
    auto pos = this->pos();
    BOOST_ASSERT(_buff <= pos && pos <= _buff + _buff_size);
    _out.append(_buff, pos);
    this->set_pos(_buff);
}

template <typename StringType>
inline StringType string_maker<StringType>::finish()
{
    auto pos = this->pos();
    BOOST_ASSERT(_buff <= pos && pos <= _buff + _buff_size);
    _out.append(_buff, pos);
    return std::move(_out);
}

#if defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_appender<std::basic_string<char8_t>>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_maker<std::basic_string<char8_t>>;
#endif

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

template <typename CharT, typename Traits, typename Allocator>
auto append(std::basic_string<CharT, Traits, Allocator>& str)
{
    using str_type = std::basic_string<CharT, Traits, Allocator>;
    using writer = boost::stringify::v0::detail::string_appender<str_type>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer
                                    , str_type& >
        (str);
}


template <typename CharT, typename Traits, typename Allocator>
auto assign(std::basic_string<CharT, Traits, Allocator>& str)
{
    str.clear();
    return append(str);
}

template< typename CharT
        , typename Traits = std::char_traits<CharT>
        , typename Allocator = std::allocator<CharT> >
constexpr boost::stringify::v0::dispatcher
    < stringify::v0::facets_pack<>
    , stringify::v0::detail::string_maker
          < std::basic_string<CharT, Traits, Allocator >>>
    to_basic_string{};

#if defined(__cpp_char8_t)

constexpr boost::stringify::v0::dispatcher
    < stringify::v0::facets_pack<>
    , stringify::v0::detail::string_maker<std::basic_string<char8_t>> >
    to_u8string{};

#endif

constexpr boost::stringify::v0::dispatcher
    < stringify::v0::facets_pack<>
    , stringify::v0::detail::string_maker<std::string> >
    to_string{};

constexpr boost::stringify::v0::dispatcher
    < stringify::v0::facets_pack<>
    , stringify::v0::detail::string_maker<std::u16string> >
    to_u16string{};

constexpr boost::stringify::v0::dispatcher
    < stringify::v0::facets_pack<>
    , stringify::v0::detail::string_maker<std::u32string> >
    to_u32string{};

constexpr boost::stringify::v0::dispatcher
    < stringify::v0::facets_pack<>
    , stringify::v0::detail::string_maker<std::wstring> >
    to_wstring{};

#endif // !defined(BOOST_NO_EXCEPTIONS)

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STRING_HPP

