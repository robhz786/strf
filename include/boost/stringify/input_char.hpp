#ifndef BOOST_STRINGIFY_INPUT_CHAR_HPP_INCLUDED
#define BOOST_STRINGIFY_INPUT_CHAR_HPP_INCLUDED

#include <boost/stringify/input_char32.hpp>
#include <boost/stringify/type_traits.hpp>

namespace boost {
namespace stringify {
namespace detail {

template <typename CharT, typename Output, typename Formatting>
class char_stringificator
    : public boost::stringify::input_base<CharT, Output, Formatting>
{
    typedef boost::stringify::input_base<CharT, Output, Formatting> base;
    
public:
    
    char_stringificator() noexcept
        : m_char()
    {
    }

    char_stringificator(CharT _character) noexcept
        : m_char(_character)
    {
    }

    void set(CharT _character) noexcept
    {
        m_char = _character;
    }

    virtual std::size_t length(const Formatting&) const noexcept override
    {
        return 1;
    }
    
    void write
        ( Output& out
        , const Formatting& fmt
        ) const noexcept(base::noexcept_output) override
    {
        out.put(m_char);
    }
    
private:
    
    CharT m_char;
};

template <typename CharIn>
struct char_input_traits
{
    
private:

    template <typename CharOut>
    struct helper
    {
        static_assert(sizeof(CharIn) == sizeof(CharOut), "");
        
        template <typename Output, typename Formatting>
        using stringificator
        = boost::stringify::detail::char_stringificator
            <CharOut, Output, Formatting>;
    };
    
public:
    
    template <typename CharT, typename Output, typename Formatting>
    using stringificator
    = typename helper<CharT>::template stringificator<Output, Formatting>;
};

} //namepace detail


boost::stringify::detail::char_input_traits<char>
boost_stringify_input_traits_of(char);

boost::stringify::detail::char_input_traits<char16_t>
boost_stringify_input_traits_of(char16_t);

boost::stringify::detail::char_input_traits<wchar_t>
boost_stringify_input_traits_of(wchar_t);


} // namespace stringify
} // namespace boost

#endif



