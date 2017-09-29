#ifndef BOOST_STRINGIFY_TEST_V0_STREAMBUF_THAT_FAILS_ON_OVERFLOW_HPP
#define BOOST_STRINGIFY_TEST_V0_STREAMBUF_THAT_FAILS_ON_OVERFLOW_HPP

#include <streambuf>
#include <array>
#include <string>

template
    < std::size_t SIZE
    , typename CharT = char
    , typename Traits = std::char_traits<CharT> >
class streambuf_that_fails_on_overflow
    : public std::basic_streambuf<CharT, Traits>
{

public:

    using Base = std::basic_streambuf<CharT>;
    using char_type = typename Base::char_type;
    using int_type = typename Base::int_type;

    streambuf_that_fails_on_overflow()
    {
        Base::setp(m_buffer.begin(), m_buffer.end());
    }

    int_type overflow(int_type) override
    {
        return Traits::eof();
    }

    std::basic_string<CharT> str()
    {
        return {Base::pbase(), Base::pptr()};
    }

private:

    std::array<char_type, SIZE> m_buffer = {};
};

#endif
