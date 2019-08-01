#ifndef BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STREAMBUF_HPP
#define BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STREAMBUF_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <streambuf>
#include <boost/stringify/v0/dispatcher.hpp>

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
        , std::streamsize* count_ptr )
        : stringify::v0::output_buffer<CharT>{_buff, _buff + _buff_size}
        , _out(out)
        , _count_ptr(count_ptr)
    {
    }

    ~std_streambuf_writer()
    {
        if (_count_ptr != nullptr)
        {
            *_count_ptr = _count;
        }
    }

    void recycle() override;

    std::streamsize finish()
    {
        if (this->pos() != _buff)
        {
           recycle();
        }
        return _count;
    }

private:

    std::basic_streambuf<CharT, Traits>& _out;
    std::streamsize _count = 0;
    std::streamsize* _count_ptr = nullptr;
};

template <typename CharT, typename Traits>
void std_streambuf_writer<CharT, Traits>::recycle()
{
    std::streamsize count = this->pos() - _buff;
    this->set_pos(_buff);
    auto count_inc = _out.sputn(_buff, count);
    _count += count_inc;

    if (count != count_inc)
    {
        throw std::runtime_error("Boost.Stringify: std::basic_streambuf::sputn failed");
    }
}


#if defined(BOOST_STRINGIFY_SEPARATE_COMPILATION)

#if defined(__cpp_char8_t)
BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class std_streambuf_writer<char8_t, std::char_traits<char8_t>>;
#endif

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
inline auto write
    ( std::basic_streambuf<CharT, Traits>& dest
    , std::streamsize* count = nullptr )
{
    using writer = stringify::v0::detail::std_streambuf_writer<CharT, Traits>;
    return stringify::v0::dispatcher< stringify::v0::facets_pack<>
                                    , writer
                                    , std::basic_streambuf<CharT, Traits>&
                                    , std::streamsize* >
        (dest, count );
}


template<typename CharT, typename Traits = std::char_traits<CharT> >
inline auto write
    ( std::basic_streambuf<CharT, Traits>* dest
    , std::streamsize* count = nullptr )
{
    return stringify::v0::write(*dest, count);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STREAMBUF_HPP

