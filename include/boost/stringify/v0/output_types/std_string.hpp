#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STRING_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <string>
#include <system_error>
#include <boost/stringify/v0/output_types/buffered_writer.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename StringType>
class string_appender: public buffered_writer<typename StringType::value_type>
{
    constexpr static std::size_t buffer_size = stringify::v0::min_buff_size;
    typename StringType::value_type buffer[buffer_size];

    using buff_writter = buffered_writer<typename StringType::value_type>;

public:

    typedef typename StringType::value_type char_type;

    string_appender
        ( stringify::v0::output_writer_init<char_type> init
        , StringType& out
        )
        : stringify::v0::buffered_writer<char_type>{init, buffer, buffer_size}
        , m_out(&out)
        , m_initial_length(out.length())
    {
    }

    ~string_appender()
    {
        if(m_out != nullptr)
        {
            m_out->resize(m_initial_length);
        }
        m_out = nullptr;
        buff_writter::discard();
    }

    void reserve(std::size_t size)
    {
        if(m_out != nullptr)
        {
            m_out->reserve(m_out->length() + size);
        }
    }

    void set_error(std::error_code ec) override
    {
        if(m_out != nullptr)
        {
            m_out->resize(m_initial_length);
            m_out = nullptr;
        }
        buff_writter::set_error(ec);
    }

    auto finish()
    {
        buff_writter::flush();
        m_out = nullptr;
        return buff_writter::finish();
    }

    void finish_exception()
    {
        buff_writter::flush();
        m_out = nullptr;
        buff_writter::finish_exception();
    }

protected:

    bool do_put(const char_type* str, std::size_t count) override
    {
        if (m_out != nullptr)
        {
            m_out->append(str, count);
        }
        return true;
    }

private:

    StringType* m_out = nullptr;
    std::size_t m_initial_length = 0;
    std::error_code m_err;
    bool m_finished = false;
};


template <typename StringType>
class string_maker: public buffered_writer<typename StringType::value_type>
{
    constexpr static std::size_t buffer_size = stringify::v0::min_buff_size;
    typename StringType::value_type buffer[buffer_size];

    using buff_writter = buffered_writer<typename StringType::value_type>;

public:

    using char_type = typename StringType::value_type;

    string_maker(stringify::v0::output_writer_init<char_type> init)
        : buff_writter{init, buffer, buffer_size}
    {
    }

    ~string_maker()
    {
    }

    stringify::v0::expected<StringType, std::error_code> finish()
    {
        buff_writter::flush();
        auto x = buff_writter::finish();
        if (x)
        {
            return {boost::stringify::v0::in_place_t{}, std::move(m_out)};
        }
        return {boost::stringify::v0::unexpect_t{}, x.error()};
    }

    StringType finish_exception()
    {
        buff_writter::finish_exception();
        return std::move(m_out);
    }

    void reserve(std::size_t size)
    {
        m_out.reserve(m_out.size() + size);
    }

protected:

    bool do_put(const char_type* str, std::size_t count) override
    {
        m_out.append(str, count);
        return true;
    }

private:

    StringType m_out;
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
    , typename Allocator = std::allocator<CharT>
    >
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

