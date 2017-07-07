#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR_PTR
#define BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR_PTR

#include <algorithm>
#include <limits>
#include <boost/stringify/v0/conventional_argf_reader.hpp>
#include <boost/stringify/v0/facets/width_calculator.hpp>


namespace boost {
namespace stringify {
inline namespace v0 {
namespace detail {

struct string_argf
{
    using char_flags_type = boost::stringify::v0::char_flags<'<', '>', '='>;
    
    constexpr string_argf(int w): width(w) {}
    constexpr string_argf(const char* f): flags(f) {}
    constexpr string_argf(int w, const char* f): width(w), flags(f) {}
    constexpr string_argf(const string_argf&) = default;
    
    int width = -1;
    char_flags_type flags;
};


template<typename CharT, typename FTuple>
class char_ptr_stringifier
{

public:

    using char_type  = CharT;
    using input_type = const CharT*;
    using output_type = boost::stringify::v0::output_writer<CharT>;
    using ftuple_type = FTuple;
    using second_arg = boost::stringify::v0::detail::string_argf;

private:

    using argf_reader = boost::stringify::v0::conventional_argf_reader<input_type>;
    using width_t = boost::stringify::v0::width_t;
    using width_tag = boost::stringify::v0::width_tag;
    using alignment_tag = boost::stringify::v0::alignment_tag;

public:
    
    char_ptr_stringifier
        ( const FTuple& ft
        , const char_type* str
        , second_arg argf
        ) noexcept
        : m_fmt(ft)
        , m_str(str)
        , m_len(std::char_traits<char_type>::length(str))
        , m_total_width(argf_reader::get_width(argf, ft))
        , m_padding_width(padding_width())
        , m_alignment(argf_reader::get_alignment(argf, ft))
    {
    }

    char_ptr_stringifier(const FTuple& fmt, const char_type* str) noexcept
        : m_fmt(fmt)
        , m_str(str)
        , m_len(std::char_traits<char_type>::length(str))
        , m_total_width(get_facet<width_tag>().width())
        , m_padding_width(padding_width())
        , m_alignment(get_facet<alignment_tag>().value())
    {
    }

    ~char_ptr_stringifier()
    {
    }

    std::size_t length() const
    {
        if (m_padding_width > 0)
        {
            return m_len +
                boost::stringify::v0::fill_length<char_type, input_type>
                (m_padding_width, m_fmt);
        }
        return m_len;
    }

    void write(output_type& out) const
    {
        if (m_padding_width > 0)
        {
            if(m_alignment == boost::stringify::v0::alignment::left)
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


    int remaining_width(int w) const
    {
        if(m_total_width > w)
        {
            return 0;
        }
        if(m_padding_width > 0)
        {
            return w - m_total_width;
        }
        return
            boost::stringify::v0::get_width_calculator<input_type>(m_fmt)
            .remaining_width(w, m_str, m_len);
    }
    

private:

    const FTuple& m_fmt;
    const char_type* m_str;
    const std::size_t m_len;
    const width_t m_total_width;
    const width_t m_padding_width;
    boost::stringify::v0::alignment m_alignment;

    template <typename FacetCategory>
    decltype(auto) get_facet() const noexcept
    {
        return boost::stringify::v0::get_facet<FacetCategory, input_type>(m_fmt);
    }

    void write_fill(output_type& out) const
    {
        boost::stringify::v0::write_fill<char_type, input_type>
                (m_padding_width, out, m_fmt);
    }

    width_t padding_width() const
    {
        return
            boost::stringify::v0::get_width_calculator<input_type>(m_fmt)
            .remaining_width(m_total_width, m_str, m_len);
    }
};


template <typename CharIn>
struct char_ptr_input_traits
{
private:

    template <typename CharOut, typename FTuple>
    struct helper
    {
        static_assert(sizeof(CharIn) == sizeof(CharOut), "");

        using stringifier = boost::stringify::v0::detail::char_ptr_stringifier
            <CharOut, FTuple>;
    };

public:

    template <typename CharOut, typename FTuple>
    using stringifier = typename helper<CharOut, FTuple>::stringifier;
};

} // namespace detail

boost::stringify::v0::detail::char_ptr_input_traits<char>
boost_stringify_input_traits_of(const char*);

boost::stringify::v0::detail::char_ptr_input_traits<char16_t>
boost_stringify_input_traits_of(const char16_t*);

boost::stringify::v0::detail::char_ptr_input_traits<char32_t>
boost_stringify_input_traits_of(const char32_t*);

boost::stringify::v0::detail::char_ptr_input_traits<wchar_t>
boost_stringify_input_traits_of(const wchar_t*);

} // inline namespace v0
} // namespace stringify
} // namespace boost



#endif  /* BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR_PTR */

