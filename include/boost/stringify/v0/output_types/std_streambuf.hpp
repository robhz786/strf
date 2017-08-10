#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STREAMBUF_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STREAMBUF_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <streambuf>
#include <boost/stringify/v0/output_writer.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

struct streambuf_result
{
    std::streamsize count;
    bool success;
};

namespace detail {

template <typename CharT, typename Traits>
class std_streambuf_writer: public output_writer<CharT>
{
public:

    using char_type = CharT;

    explicit std_streambuf_writer(std::basic_streambuf<CharT, Traits>& out)
        : m_out(out)
    {
    }

    void put(const CharT* str, std::size_t ucount) override
    {
        std::streamsize count = ucount;
        auto count_inc = m_out.sputn(str, count);
        m_success &= (count_inc == count);
        m_count += count_inc;
    }

    void put(CharT ch) override
    {
        do_put(ch);
    }

    void repeat(CharT ch, std::size_t count) override
    {
        for(; count > 0; --count)
        {
            do_put(ch);
        }
    }

    void repeat(CharT ch1, CharT ch2, std::size_t count) override
    {
        for(; count > 0; --count)
        {
            do_put(ch1);
            do_put(ch2);
        }
    }

    void repeat(CharT ch1, CharT ch2, CharT ch3, std::size_t count) override
    {
        for(; count > 0; --count)
        {
            do_put(ch1);
            do_put(ch2);
            do_put(ch3);
        }
    }

    void repeat(CharT ch1, CharT ch2, CharT ch3, CharT ch4, std::size_t count) override
    {
        for(; count > 0; --count)
        {
            do_put(ch1);
            do_put(ch2);
            do_put(ch3);
            do_put(ch4);
        }
    }

    boost::stringify::v0::streambuf_result finish() noexcept
    {
        return {m_count, m_success};
    }

private:

    void do_put(CharT character)
    {
        if(m_out.sputc(character) == Traits::eof())
        {
            m_success = false;
        }
        else
        {
            ++m_count;
        }
    }


    std::basic_streambuf<CharT, Traits>& m_out;
    std::streamsize m_count = 0;
    bool m_success = true;

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
auto write_to(std::basic_streambuf<CharT, Traits>& destination)
{
    using intput_type = std::basic_streambuf<CharT, Traits>&;
    using writer = boost::stringify::v0::detail::std_streambuf_writer<CharT, Traits>;
    return boost::stringify::v0::make_args_handler<writer, intput_type>(destination);
}


template<typename CharT, typename Traits = std::char_traits<CharT> >
auto write_to(std::basic_streambuf<CharT, Traits>* destination)
{
    using intput_type = std::basic_streambuf<CharT, Traits>&;
    using writer = boost::stringify::v0::detail::std_streambuf_writer<CharT, Traits>;
    return boost::stringify::v0::make_args_handler<writer, intput_type>(*destination);
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STREAMBUF_HPP

