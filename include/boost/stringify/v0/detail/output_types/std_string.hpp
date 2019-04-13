#ifndef BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STRING_HPP
#define BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <system_error>
#include <boost/stringify/v0/dispatcher.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename StringType>
class ec_string_appender final
    : public stringify::v0::output_buffer<typename StringType::value_type>
{
    constexpr static std::size_t _buff_size = stringify::v0::min_buff_size;
    typename StringType::value_type _buff[_buff_size];

public:

    typedef typename StringType::value_type char_type;

    ec_string_appender(StringType& out, std::size_t* count_ptr)
        : output_buffer<char_type>{_buff, _buff + _buff_size}
        , _out(out)
        , _initial_length(out.length())
        , _count_ptr(count_ptr)
    {
    }

    ~ec_string_appender()
    {
        if( ! _finished)
        {
            _out.resize(_initial_length);
        }
        else if (_count_ptr != nullptr)
        {
            * _count_ptr = _out.size() - _initial_length;
        }
    }

    void reserve(std::size_t size)
    {
        _out.reserve(_out.length() + size);
    }

    bool recycle() override;

    stringify::v0::nodiscard_error_code finish();

protected:

    void on_error() override;

private:

    StringType& _out;
    std::size_t _initial_length = 0;
    std::size_t* _count_ptr;
    bool _finished = false;
};

template <typename StringType>
bool ec_string_appender<StringType>::recycle()
{
    auto pos = this->pos();
    BOOST_ASSERT(_buff <= pos && pos <= _buff + _buff_size);
    _out.append(_buff, pos);
    this->set_pos(_buff);
    return true;
}

template <typename StringType>
void ec_string_appender<StringType>::on_error()
{
    recycle();
}

template <typename StringType>
inline
stringify::v0::nodiscard_error_code ec_string_appender<StringType>::finish()
{
    _finished = true;
    if ( ! this->has_error())
    {
        auto pos = this->pos();
        BOOST_ASSERT(_buff <= pos && pos <= _buff + _buff_size);
        _out.append(_buff, pos);
        return {};
    }
    return this->get_error();
}

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class ec_string_appender<std::string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class ec_string_appender<std::u16string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class ec_string_appender<std::u32string>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class ec_string_appender<std::wstring>;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

} // namespace detail

template <typename CharT, typename Traits, typename Allocator>
inline auto ec_append
    ( std::basic_string<CharT, Traits, Allocator>& str
    , std::size_t* count_ptr = nullptr )
{
    using str_type = std::basic_string<CharT, Traits, Allocator>;
    using writer = boost::stringify::v0::detail::ec_string_appender<str_type>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer, str_type&, std::size_t* >
        (stringify::v0::pack(), str, count_ptr);
}


template <typename CharT, typename Traits, typename Allocator>
inline auto ec_assign
    ( std::basic_string<CharT, Traits, Allocator>& str
    , std::size_t* count_ptr = nullptr )
{
    str.clear();
    return ec_append(str, count_ptr);
}

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

    std::size_t finish();

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
    this->set_pos(_buff);
    return true;
}

template <typename StringType>
std::size_t string_appender<StringType>::finish()
{
    if ( ! this->has_error() )
    {
        auto pos = this->pos();
        BOOST_ASSERT(_buff <= pos && pos <= _buff + _buff_size);
        if (pos != _buff)
        {
            _out.append(_buff, pos);
        }
        _finished = true;
        return _out.size() - _initial_length;
    }
    else
    {
        _out.resize(_initial_length);
        throw stringify::v0::stringify_error(this->get_error());
    }
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

    bool recycle() override;

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
bool string_maker<StringType>::recycle()
{
    auto pos = this->pos();
    BOOST_ASSERT(_buff <= pos && pos <= _buff + _buff_size);
    _out.append(_buff, pos);
    this->set_pos(_buff);
    return true;
}

template <typename StringType>
inline StringType string_maker<StringType>::finish()
{
    if (this->has_error())
    {
        throw stringify::v0::stringify_error(this->get_error());
    }
    auto pos = this->pos();
    BOOST_ASSERT(_buff <= pos && pos <= _buff + _buff_size);
    _out.append(_buff, pos);
    return std::move(_out);
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

template <typename CharT, typename Traits, typename Allocator>
auto append(std::basic_string<CharT, Traits, Allocator>& str)
{
    using str_type = std::basic_string<CharT, Traits, Allocator>;
    using writer = boost::stringify::v0::detail::string_appender<str_type>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer
                                    , str_type& >
        (stringify::v0::pack(), str);
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
    to_basic_string{stringify::v0::pack()};

constexpr boost::stringify::v0::dispatcher
    < stringify::v0::facets_pack<>
    , stringify::v0::detail::string_maker<std::string> >
    to_string{stringify::v0::pack()};

constexpr boost::stringify::v0::dispatcher
    < stringify::v0::facets_pack<>
    , stringify::v0::detail::string_maker<std::u16string> >
    to_u16string{stringify::v0::pack()};

constexpr boost::stringify::v0::dispatcher
    < stringify::v0::facets_pack<>
    , stringify::v0::detail::string_maker<std::u32string> >
    to_u32string{stringify::v0::pack()};

constexpr boost::stringify::v0::dispatcher
    < stringify::v0::facets_pack<>
    , stringify::v0::detail::string_maker<std::wstring> >
    to_wstring{stringify::v0::pack()};

#endif // !defined(BOOST_NO_EXCEPTIONS)

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STRING_HPP

