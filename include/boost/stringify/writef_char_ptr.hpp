#ifndef BOOST_STRINGIFY_WRITEF_CHAR_PTR_HPP
#define BOOST_STRINGIFY_WRITEF_CHAR_PTR_HPP

#include <string>
#include <boost/stringify/writef_helper.hpp>
#include <boost/stringify/formater_tuple.hpp>

namespace boost{
namespace stringify{
namespace detail{

template<typename charT, typename traits>
class char_ptr_writer
{
public:

    typedef charT char_type;

    char_ptr_writer(const char_ptr_writer&) = default;
    
    char_ptr_writer(charT* out)
        : m_out(out)
    {
    }

    void put(charT character) noexcept
    {
        traits::assign(*m_out++, character);
    }

    void put(charT character, std::size_t repetitions) noexcept
    {
        traits::assign(m_out, repetitions, character);
        m_out += repetitions;
    }

    void put(const charT* str, std::size_t count) noexcept
    {
        traits::copy(m_out, str, count);
        m_out += count;
    }
    
    charT* finish() noexcept
    {
        traits::assign(*m_out, charT());
        return m_out;
    }

    bool set_pos(charT* pos) noexcept
    {
        m_out = pos;
        return true;
    }

    charT* get_pos() noexcept
    {
        return m_out;
    }
    
    void rput(charT character) noexcept
    {
        traits::assign(*--m_out, character);
    }

private:

    charT* m_out;
};

} // namespace detail


template<typename charT, typename charTraits = std::char_traits<charT> >
inline boost::stringify::writef_helper
    < boost::stringify::detail::char_ptr_writer<charT, charTraits> >
writef(charT* output)
{
    return output;
}
    

} // namespace stringify
} // namespace boost    

#endif  /* BOOST_STRINGIFY_WRITEF_CHAR_PTR_HPP */

