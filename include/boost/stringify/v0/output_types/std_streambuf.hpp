#ifndef BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STREAMBUF_HPP
#define BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STREAMBUF_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <streambuf>

namespace boost {
namespace stringify {
inline namespace v0 {

struct streambuf_result
{
    std::streamsize count;
    bool success;
};
       
namespace detail {

template <typename CharT, typename Traits>
class std_streambuf_writer
{
public:

    using char_type = CharT;

    explicit std_streambuf_writer(std::basic_streambuf<CharT, Traits>& out)
        : m_out(out)
    {
    }

    void put(CharT character) noexcept
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

    void put(CharT character, std::size_t repetitions) noexcept
    {
        for(;repetitions > 0; --repetitions)
        {
            put(character);
        }
    }

    void put(const CharT* str, std::size_t ucount) noexcept
    {
        std::streamsize count = ucount;
        auto count_inc = m_out.sputn(str, count);
        m_success &= (count_inc == count);
        m_count += count_inc;
    }

    boost::stringify::v0::streambuf_result finish() noexcept
    {
        return {m_count, m_success};
    }

private:

    std::basic_streambuf<CharT, Traits>& m_out;
    std::streamsize m_count = 0;
    bool m_success = true;
    
};

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


} // inline namespace v0
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_V0_OUTPUT_TYPES_STD_STREAMBUF_HPP

