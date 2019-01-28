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
class std_streambuf_writer final: public stringify::v0::output_buffer<CharT>
{
    constexpr static std::size_t _buff_size = stringify::v0::min_buff_size;
    CharT _buff[_buff_size];

public:

    using char_type = CharT;

    std_streambuf_writer
        ( std::basic_streambuf<CharT, Traits>& out
        , std::size_t* count )
        : stringify::v0::output_buffer<CharT>{_buff, _buff + _buff_size}
        , _out(out)
        , _count(count)
    {
        if (_count != nullptr)
        {
            *_count = 0;
        }
    }

    bool recycle() override;

    stringify::v0::expected<void, std::error_code> finish()
    {
        if ( ! this->has_error() && recycle())
        {
            return {};
        }
        return { stringify::v0::unexpect_t{}, this->get_error() };
    }

private:

    std::basic_streambuf<CharT, Traits>& _out;
    std::size_t* _count = nullptr;
};

template <typename CharT, typename Traits>
bool std_streambuf_writer<CharT, Traits>::recycle()
{
    auto end = this->pos();
    std::size_t count = end - _buff;
    auto count_inc = _out.sputn(_buff, count);

    if (_count != nullptr && count_inc > 0)
    {
        *_count += static_cast<std::size_t>(count_inc);
    }
    if (static_cast<std::streamsize>(count) == count_inc)
    {
        this->set_pos(_buff);
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

