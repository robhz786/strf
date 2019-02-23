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
class ec_std_streambuf_writer final: public stringify::v0::output_buffer<CharT>
{
    constexpr static std::size_t _buff_size = stringify::v0::min_buff_size;
    CharT _buff[_buff_size];

public:

    using char_type = CharT;

    ec_std_streambuf_writer
        ( std::basic_streambuf<CharT, Traits>& out
        , std::streamsize* count_ptr )
        : stringify::v0::output_buffer<CharT>{_buff, _buff + _buff_size}
        , _out(out)
        , _count_ptr(count_ptr)
    {
    }

    ~ec_std_streambuf_writer()
    {
        if (_count_ptr != nullptr)
        {
            *_count_ptr = _count;
        }
    }

    bool recycle() override;

    stringify::v0::nodiscard_error_code finish()
    {
        if ( ! this->has_error() && this->size() != 0)
        {
            recycle();
        }
        return this->get_error();
    }

protected:

    void on_error() override;

private:

    std::basic_streambuf<CharT, Traits>& _out;
    std::streamsize _count = 0;
    std::streamsize* _count_ptr = nullptr;
};

template <typename CharT, typename Traits>
bool ec_std_streambuf_writer<CharT, Traits>::recycle()
{
    std::streamsize count = this->pos() - _buff;
    this->set_pos(_buff);
    auto count_inc = _out.sputn(_buff, count);

    _count += count_inc;
    if (count == count_inc)
    {
        return true;
    }
    this->set_error(std::errc::io_error);
    return false;
}

template <typename CharT, typename Traits>
void ec_std_streambuf_writer<CharT, Traits>::on_error()
{
    std::streamsize count = this->pos() - _buff;
    this->set_pos(_buff);
    _count += _out.sputn(_buff, count);
}

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class ec_std_streambuf_writer<char, std::char_traits<char>>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class ec_std_streambuf_writer<char16_t, std::char_traits<char16_t>>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class ec_std_streambuf_writer<char32_t, std::char_traits<char32_t>>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE
class ec_std_streambuf_writer<wchar_t, std::char_traits<wchar_t>>;

#endif

} // namespace detail


template<typename CharT, typename Traits = std::char_traits<CharT> >
auto ec_write
    ( std::basic_streambuf<CharT, Traits>& dest
    , std::streamsize* count = nullptr )
{
    using intput_type = std::basic_streambuf<CharT, Traits>&;
    using writer = stringify::v0::detail::ec_std_streambuf_writer<CharT, Traits>;
    return stringify::v0::make_destination<writer, intput_type>(dest, count);
}


template<typename CharT, typename Traits = std::char_traits<CharT> >
auto ec_write
    ( std::basic_streambuf<CharT, Traits>* dest
    , std::streamsize* count = nullptr )
{
    using intput_type = std::basic_streambuf<CharT, Traits>&;
    using writer = stringify::v0::detail::ec_std_streambuf_writer<CharT, Traits>;
    return stringify::v0::make_destination<writer, intput_type>(*dest, count);
}



#if ! defined(BOOST_NO_EXCEPTION)

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

    bool recycle() override;

    std::streamsize finish()
    {
        if (this->has_error() || (this->size() != 0 && ! recycle()))
        {
            throw stringify::v0::stringify_error(this->get_error());
        }
        return _count;
    }

protected:

    void on_error() override;

private:

    std::basic_streambuf<CharT, Traits>& _out;
    std::streamsize _count = 0;
    std::streamsize* _count_ptr = nullptr;
};

template <typename CharT, typename Traits>
bool std_streambuf_writer<CharT, Traits>::recycle()
{
    std::streamsize count = this->pos() - _buff;
    this->set_pos(_buff);
    auto count_inc = _out.sputn(_buff, count);

    _count += count_inc;
    if (count == count_inc)
    {
        return true;
    }
    this->set_error(std::errc::io_error);
    return false;
}

template <typename CharT, typename Traits>
void std_streambuf_writer<CharT, Traits>::on_error()
{
    std::streamsize count = this->pos() - _buff;
    this->set_pos(_buff);
    _count += _out.sputn(_buff, count);
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
    , std::streamsize* count = nullptr )
{
    using intput_type = std::basic_streambuf<CharT, Traits>&;
    using writer = stringify::v0::detail::std_streambuf_writer<CharT, Traits>;
    return stringify::v0::make_destination<writer, intput_type>(dest, count);
}


template<typename CharT, typename Traits = std::char_traits<CharT> >
auto write
    ( std::basic_streambuf<CharT, Traits>* dest
    , std::streamsize* count = nullptr )
{
    using intput_type = std::basic_streambuf<CharT, Traits>&;
    using writer = stringify::v0::detail::std_streambuf_writer<CharT, Traits>;
    return stringify::v0::make_destination<writer, intput_type>(*dest, count);
}

#endif // ! defined(BOOST_NO_EXCEPTION)

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_DETAIL_OUTPUT_TYPES_STD_STREAMBUF_HPP

