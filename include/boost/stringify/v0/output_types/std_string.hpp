#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STRING_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STRING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/output_types/FILE.hpp>
#include <system_error>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename StringType>
class string_appender: public buffered_writer<typename StringType::value_type>
{
    constexpr static std::size_t buffer_size = 60;
    typename StringType::value_type buffer[buffer_size];

    using parent = buffered_writer<typename StringType::value_type>;

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
        this->discard();
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
        parent::set_error(ec);
    }

    auto finish()
    {
        this->flush();
        m_out = nullptr;
        return parent::finish();
    }

    void finish_exception()
    {
        this->flush();
        m_out = nullptr;
        parent::finish_exception();
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

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STRING_HPP

