#ifndef BOOST_STRINGIFY_INPUT_CHAR_PTR
#define BOOST_STRINGIFY_INPUT_CHAR_PTR

#include <algorithm>
#include <limits>
#include <boost/stringify/stringifier.hpp>

namespace boost {
namespace stringify {
namespace detail {

template<typename CharT, typename Output, typename Formatting>
struct char_ptr_stringifier
    : boost::stringify::stringifier<CharT, Output, Formatting>
{
    typedef const CharT* input_type;

    typedef boost::stringify::stringifier<CharT, Output, Formatting> base;
    
    typedef
        boost::stringify::width_accumulator<Formatting, input_type, CharT>
        width_accumulator_type;
    
public:
   
    char_ptr_stringifier(const CharT* str) noexcept
        : m_str(str)
        , m_len(std::char_traits<CharT>::length(str))
    {
    }

    virtual std::size_t length(const Formatting& fmt) const noexcept override
    {
        return m_len + fill_length(fmt);
    }

    void write
        ( Output& out
        , const Formatting& fmt
        ) const noexcept(base::noexcept_output) override
    {
        auto fw = fill_width(fmt);
        if (fw > 0)
        {
            boost::stringify::write_fill<CharT, input_type>(fw, out, fmt);
        }

        if(m_str)
        {
            out.put(m_str, m_len);
        }
    }


private:
    const CharT* m_str;
    mutable std::size_t m_len;

    std::size_t fill_length(const Formatting& fmt) const noexcept
    {
        auto fw = fill_width(fmt);
        if (fw > 0)
        {
            return boost::stringify::fill_length<CharT, input_type>(fw, fmt);
        }
        return 0;
    }

    boost::stringify::width_t fill_width(const Formatting& fmt) const noexcept
    {
        auto total_width = boost::stringify::get_width<input_type>(fmt);
        if(total_width > 0)
        {
            width_accumulator_type acc;
            if(acc.add(m_str, m_len, total_width))
            {
                boost::stringify::width_t nonfill_width = acc.result();
                if (nonfill_width < total_width)
                {
                    return total_width - nonfill_width;
                }
            }
        }
        return 0;
    }
};


template <typename CharIn>
struct char_ptr_input_traits
{
private:

    template <typename CharOut>
    struct helper
    {
        static_assert(sizeof(CharIn) == sizeof(CharOut), "");
        
        template <typename Output, typename Formatting>
        using stringifier
        = boost::stringify::detail::char_ptr_stringifier
            <CharOut, Output, Formatting>;
    };

public:
    
    template <typename CharT, typename Output, typename Formatting>
    using stringifier
    = typename helper<CharT>::template stringifier<Output, Formatting>;
};

} // namespace detail

boost::stringify::detail::char_ptr_input_traits<char>
boost_stringify_input_traits_of(const char*);

boost::stringify::detail::char_ptr_input_traits<char16_t>
boost_stringify_input_traits_of(const char16_t*);

boost::stringify::detail::char_ptr_input_traits<char32_t>
boost_stringify_input_traits_of(const char32_t*);

boost::stringify::detail::char_ptr_input_traits<wchar_t>
boost_stringify_input_traits_of(const wchar_t*);

} // namespace stringify
} // namespace boost



#endif  /* BOOST_STRINGIFY_INPUT_CHAR_PTR */

