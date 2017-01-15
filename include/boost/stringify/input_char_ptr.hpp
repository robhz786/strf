#ifndef BOOST_STRINGIFY_INPUT_CHAR_PTR
#define BOOST_STRINGIFY_INPUT_CHAR_PTR

#include <algorithm>
#include <limits>
#include <boost/stringify/stringifier.hpp>
#include <boost/stringify/custom_alignment.hpp>
#include <boost/stringify/custom_width_calculator.hpp>


namespace boost {
namespace stringify {
namespace detail {


struct string_arg_formatting
{
    typedef boost::stringify::width_t width_t;

    typedef
    boost::stringify::detail::char_flags<'<', '>', '='>
    char_flags_type;

    constexpr string_arg_formatting(const char* flags, int width = -1)
        : m_flags(flags)
        , m_width(width)
    {
    }

    constexpr string_arg_formatting(int width)
        : m_width(width)
    {
    }
    
    template <typename inputType, typename FTuple>
    width_t get_width(const FTuple& fmt) const
    {
        if (m_width >= 0)
        {
            return m_width;
        }
        return boost::stringify::get_width<inputType>(fmt);
    }

    char_flags_type m_flags;
    width_t m_width;
};


template<typename CharT, typename Output, typename Formatting>
struct char_ptr_stringifier
    : boost::stringify::stringifier<CharT, Output, Formatting>
{
public:

    typedef const CharT* input_type;
    typedef CharT char_type;
    typedef Output output_type;
    typedef Formatting ftuple_type;
    
private:
    
    typedef boost::stringify::width_t width_t;
    
    typedef boost::stringify::stringifier<CharT, Output, Formatting> base;

public:
    
    typedef boost::stringify::detail::string_arg_formatting arg_format_type;
    
    char_ptr_stringifier
        ( const Formatting& fmt
        , const CharT* str
        , arg_format_type argf
        ) noexcept
        : m_fmt(fmt)
        , m_str(str)
        , m_len(std::char_traits<CharT>::length(str))
        , m_padding_width(padding_width(argf.get_width<input_type>(fmt)))
        , m_alignment(boost::stringify::get_alignment<input_type>(fmt, argf.m_flags))
    {
    }
    
    char_ptr_stringifier(const Formatting& fmt, const CharT* str) noexcept
        : m_fmt(fmt)
        , m_str(str)
        , m_len(std::char_traits<CharT>::length(str))
        , m_padding_width
          (padding_width(boost::stringify::get_width<input_type>(fmt)))
        , m_alignment(boost::stringify::get_alignment<input_type>(fmt))
    {
    }

    virtual std::size_t length() const noexcept override
    {
        if (m_padding_width > 0)
        {
            return m_len + 
                boost::stringify::fill_length<CharT, input_type>
                (m_padding_width, m_fmt);
        }
        return m_len;
    }

    void write(Output& out) const noexcept(base::noexcept_output) override
    {
        if (m_padding_width > 0)
        {
            if(m_alignment == boost::stringify::alignment::left)
            {
                out.put(m_str, m_len);
                write_fill(out);
            }
            else
            {
                write_fill(out);
                out.put(m_str, m_len);
            }
        }
        else
        {
            out.put(m_str, m_len);
        }
    }


private:

    const Formatting& m_fmt;
    const CharT* m_str;
    const std::size_t m_len;
    const width_t m_padding_width;
    boost::stringify::alignment m_alignment;
    
    void write_fill(Output& out) const noexcept(base::noexcept_output)
    {
        boost::stringify::write_fill<CharT, input_type>
                (m_padding_width, out, m_fmt);
    }
    
    width_t padding_width(width_t total_width) const noexcept
    {
        return
            boost::stringify::get_width_calculator<input_type>(m_fmt)
            .remaining_width(total_width, m_str, m_len);
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

