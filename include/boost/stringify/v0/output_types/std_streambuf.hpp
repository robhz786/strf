#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STREAMBUF_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STREAMBUF_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <streambuf>
#include <boost/stringify/v0/output_types/buffered_writer.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename CharT, typename Traits>
class std_streambuf_writer: public buffered_writer<CharT>
{
    constexpr static std::size_t buffer_size = stringify::v0::min_buff_size;
    CharT buffer[40];

public:

    using char_type = CharT;

    std_streambuf_writer
        ( stringify::v0::output_writer_init<CharT> init
        , std::basic_streambuf<CharT, Traits>& out
        , std::size_t* count )
        : stringify::v0::buffered_writer<CharT>{init, buffer, buffer_size}
        , m_out(out)
        , m_count(count)
    {
        if (m_count != nullptr)
        {
            *m_count = 0;
        }
    }

    ~std_streambuf_writer()
    {
        this->flush();
    }

protected:

    bool do_put(const CharT* str, std::size_t count) override
    {
        auto count_inc = m_out.sputn(str, count);

        if (m_count != nullptr && count_inc > 0)
        {
            *m_count += static_cast<std::size_t>(count_inc);
        }
        if (static_cast<std::streamsize>(count) != count_inc)
        {
            this->set_error(std::make_error_code(std::errc::io_error));
            return false;
        }
        return true;
    }

    std::basic_streambuf<CharT, Traits>& m_out;
    std::size_t* m_count = nullptr;

};


#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class std_streambuf_writer<char, std::char_traits<char>>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class std_streambuf_writer<char16_t, std::char_traits<char16_t>>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class std_streambuf_writer<char32_t, std::char_traits<char32_t>>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class std_streambuf_writer<wchar_t, std::char_traits<wchar_t>>;

#endif

} // namespace detail


template<typename CharT, typename Traits = std::char_traits<CharT> >
auto write
    ( std::basic_streambuf<CharT, Traits>& dest
    , std::size_t* count = nullptr )
{
    using intput_type = std::basic_streambuf<CharT, Traits>&;
    using writer = stringify::v0::detail::std_streambuf_writer<CharT, Traits>;
    return stringify::v0::make_destination<writer, intput_type>(dest, count);
}


template<typename CharT, typename Traits = std::char_traits<CharT> >
auto write
    ( std::basic_streambuf<CharT, Traits>* dest
    , std::size_t* count = nullptr )
{
    using intput_type = std::basic_streambuf<CharT, Traits>&;
    using writer = stringify::v0::detail::std_streambuf_writer<CharT, Traits>;
    return stringify::v0::make_destination<writer, intput_type>(*dest, count);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STREAMBUF_HPP

