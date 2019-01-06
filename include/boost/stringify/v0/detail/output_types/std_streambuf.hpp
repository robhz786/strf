#ifndef BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STREAMBUF_HPP
#define BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STREAMBUF_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <streambuf>
#include <boost/stringify/v0/make_destination.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

namespace detail {

template <typename CharT, typename Traits>
class std_streambuf_writer final: public stringify::v0::buffer_recycler<CharT>
{
    constexpr static std::size_t _buff_size = stringify::v0::min_buff_size;
    CharT _buff[_buff_size];

public:

    using char_type = CharT;

    std_streambuf_writer
        ( std::basic_streambuf<CharT, Traits>& out
        , std::size_t* count )
        : m_out(out)
        , m_count(count)
    {
        if (m_count != nullptr)
        {
            *m_count = 0;
        }
    }

    stringify::v0::output_buffer<CharT> start() noexcept
    {
        return  { _buff, _buff + _buff_size };
    }

    bool recycle(stringify::v0::output_buffer<CharT>& buff) override;

    stringify::v0::expected<void, std::error_code> finish(CharT* it)
    {
        if ( ! this->has_error() && do_put(it))
        {
            return {};
        }
        return { stringify::v0::unexpect_t{}, this->get_error() };
    }

private:

    bool do_put(const CharT* end);

    std::basic_streambuf<CharT, Traits>& m_out;
    std::size_t* m_count = nullptr;

};

template <typename CharT, typename Traits>
bool std_streambuf_writer<CharT, Traits>::recycle
    ( stringify::v0::output_buffer<CharT>& buff )
{
    if (do_put(buff.it))
    {
        buff = start();
        return true;
    }
    return false;
}


template <typename CharT, typename Traits>
bool std_streambuf_writer<CharT, Traits>::do_put(const CharT* end)
{
    std::size_t count = end - _buff;
    auto count_inc = m_out.sputn(_buff, count);

    if (m_count != nullptr && count_inc > 0)
    {
        *m_count += static_cast<std::size_t>(count_inc);
    }
    if (static_cast<std::streamsize>(count) == count_inc)
    {
        return true;
    }
    this->set_error(std::make_error_code(std::errc::io_error));
    return false;
}


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

#endif  // BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STREAMBUF_HPP

