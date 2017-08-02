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


template<typename CharT>
class char_ptr_stringifier
{

public:

    using char_type  = CharT;
    using input_type = const CharT*;
    using output_type = boost::stringify::v0::output_writer<CharT>;
    using second_arg = boost::stringify::v0::detail::string_argf;

private:

    using cv_tag = boost::stringify::v0::conversion_from_utf32_tag<CharT>;
    using wcalc_tag = boost::stringify::v0::width_calculator_tag;
    using argf_reader = boost::stringify::v0::conventional_argf_reader<input_type>;

public:

    template <typename FTuple>
    char_ptr_stringifier
        ( const FTuple& ft
        , const char_type* str
        , second_arg argf
        ) noexcept
        : m_str(str)
        , m_len(std::char_traits<char_type>::length(str))
        , m_conv(get_facet<cv_tag>(ft))
        , m_fillchar(get_facet<fill_tag>(ft).fill_char())
        , m_total_width(argf_reader::get_width(argf, ft))
        , m_alignment(argf_reader::get_alignment(argf, ft))
    {
        init(get_facet<wcalc_tag>(ft));
    }

    template <typename FTuple>
    char_ptr_stringifier(const FTuple& ft, const char_type* str) noexcept
        : m_str(str)
        , m_len(std::char_traits<char_type>::length(str))
        , m_conv(get_facet<cv_tag>(ft))
        , m_fillchar(get_facet<fill_tag>(ft).fill_char())
        , m_total_width(get_facet<width_tag>(ft).width())
        , m_alignment(get_facet<alignment_tag>(ft).value())
    {
        init(get_facet<wcalc_tag>(ft));
    }

    ~char_ptr_stringifier()
    {
    }

    std::size_t length() const
    {
        if (m_fillcount > 0)
        {
            return m_len + m_conv.length(m_fillchar) * m_fillcount;
        }
        return m_len;
    }

    void write(output_type& out) const
    {
        if (m_fillcount > 0)
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
        return w > m_total_width ? w - m_total_width : 0;
    }

private:

    const char_type* m_str;
    const std::size_t m_len;
    const boost::stringify::v0::conversion_from_utf32<CharT>& m_conv;
    const char32_t m_fillchar;
    width_t m_fillcount = 0;
    width_t m_total_width;
    boost::stringify::v0::alignment m_alignment;

    template <typename Category, typename FTuple>
    const auto& get_facet(const FTuple& ft) const
    {
        return ft.template get_facet<Category, input_type>();
    }

    template <typename WidthCalculator>
    void init(const WidthCalculator& wcalc)
    {
        int content_width = wcalc.width_of(m_str, m_len);
        if(content_width < m_total_width)
        {
            auto fillchar_width = wcalc.width_of(m_fillchar);
            m_fillcount = (m_total_width - content_width) / fillchar_width;
        }
        else
        {
            m_total_width = content_width;
        }
    }

    void write_fill(output_type& out) const
    {
        m_conv.write(out, m_fillchar, m_fillcount);
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

        using stringifier = boost::stringify::v0::detail::char_ptr_stringifier
            <CharOut>;
    };

public:

    template <typename CharOut, typename FTuple>
    using stringifier = typename helper<CharOut>::stringifier;
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

