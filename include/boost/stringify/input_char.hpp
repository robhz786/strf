#ifndef BOOST_STRINGIFY_INPUT_CHAR_HPP_INCLUDED
#define BOOST_STRINGIFY_INPUT_CHAR_HPP_INCLUDED

#include <boost/stringify/input_base.hpp>
#include <type_traits>

namespace boost
{
namespace stringify
{

template <typename charT, typename traits, typename Formating>
class input_char: public boost::stringify::input_base<charT, Formating>
{
public:
    
    input_char() noexcept
        : character()
    {
    }

    input_char(charT _character) noexcept
        : character(_character)
    {
    }

    void set(charT _character) noexcept
    {
        character = _character;
    }

    virtual std::size_t length(const Formating&) const noexcept
    {
        return 1;
    }

    virtual charT* write_without_termination_char(charT* out, const Formating&)
        const noexcept
    {
        traits::assign(*out, character);
        return out + 1;
    }

    // virtual void write
    //     ( boost::stringify::simple_ostream<charT>& out
    //     , const Formating&
    //     ) const
    // {
    //     out.put(character);
    // }
    
private:
    
    charT character;
};


template <typename charT, typename traits, typename Formating>
boost::stringify::input_char<charT, traits, Formating>
argf(charT c) noexcept
{
    return c;
}


template <typename charT, typename traits, typename Formating>
typename std::enable_if
    < std::is_same<charT, wchar_t>::value && sizeof(charT) == sizeof(char32_t)
    , boost::stringify::input_char<charT, traits, Formating>
    > :: type
argf(char32_t c) noexcept
{
    return c;
}

} // namespace stringify
} // namespace boost

#endif



