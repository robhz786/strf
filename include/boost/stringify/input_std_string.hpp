#ifndef BOOST_STRINGIFY_INPUT_STD_STRING_HPP_INCLUDED
#define BOOST_STRINGIFY_INPUT_STD_STRING_HPP_INCLUDED

#include <string>
#include <boost/stringify/input_base.hpp>

namespace boost {
namespace stringify{
namespace detail {

template <class CharT, typename Traits, typename Output, class Formatting>
class std_string_stringificator
    : public boost::stringify::input_base<CharT, Output, Formatting>
{
    typedef boost::stringify::input_base<CharT, Output, Formatting> base;

public:

    const std::basic_string<CharT, Traits>* str;

    std_string_stringificator() noexcept
        : str(0)
    {
    }

    std_string_stringificator(const std::basic_string<CharT, Traits>& _str) noexcept
        : str(&_str)
    {
    }

    void set(const std::basic_string<CharT, Traits>& _str) noexcept
    {
        str = &_str;
    }

    virtual std::size_t length(const Formatting&) const noexcept override
    {
        return str ? str->length() : 0;
    }

    void write
        ( Output& out
        , const Formatting&
        ) const noexcept(base::noexcept_output) override
    {
        if(str)
        {
            out.put(str->c_str(), str->length());
        }
    }
};

template <typename CharIn, typename CharTraits>
struct std_string_input_traits
{
private:

    template <typename CharOut>
    struct helper
    {
        static_assert(sizeof(CharIn) == sizeof(CharOut), "");

        template <typename Output, typename Formatting>
        using stringificator
        = boost::stringify::detail::std_string_stringificator
            <CharOut, CharTraits, Output, Formatting>;
    };

public:

    template <typename CharT, typename Output, typename Formatting>
    using stringificator
    = typename helper<CharT>::template stingificator<Output, Formatting>;
};

} // namespace detail


template <typename CharT, typename CharTraits>
boost::stringify::detail::std_string_input_traits<CharT, CharTraits>
boost_stringify_input_traits_of(const std::basic_string<CharT, CharTraits>&);


} // namespace stringify
} // namespace boost


#endif
