#ifndef BOOST_STRINGIFY_WRITEF_CHAR_PTR_HPP
#define BOOST_STRINGIFY_WRITEF_CHAR_PTR_HPP

#include <string>
#include <boost/stringify/writef_helper.hpp>
#include <boost/stringify/ftuple.hpp>

namespace boost{
namespace stringify{
namespace detail{

template<typename CharT, typename Traits>
class char_ptr_writer
{
public:

    typedef CharT char_type;

    char_ptr_writer(const char_ptr_writer&) = default;
    
    char_ptr_writer(CharT* out)
        : m_out(out)
    {
    }

    void put(CharT character) noexcept
    {
        Traits::assign(*m_out++, character);
    }

    void put(CharT character, std::size_t repetitions) noexcept
    {
        Traits::assign(m_out, repetitions, character);
        m_out += repetitions;
    }

    void put(const CharT* str, std::size_t count) noexcept
    {
        Traits::copy(m_out, str, count);
        m_out += count;
    }
    
    CharT* finish() noexcept
    {
        Traits::assign(*m_out, CharT());
        return m_out;
    }

    // bool set_pos(CharT* pos) noexcept
    // {
    //     m_out = pos;
    //     return true;
    // }

    // CharT* get_pos() noexcept
    // {
    //     return m_out;
    // }
    
    // void rput(CharT character) noexcept
    // {
    //     Traits::assign(*--m_out, character);
    // }

private:

    CharT* m_out;
};

} // namespace detail


template<typename CharT, typename CharTraits = std::char_traits<CharT> >
inline boost::stringify::writef_helper
    < boost::stringify::detail::char_ptr_writer<CharT, CharTraits> >
writef(CharT* output)
{
    return output;
}
    

} // namespace stringify
} // namespace boost    

#endif  /* BOOST_STRINGIFY_WRITEF_CHAR_PTR_HPP */

