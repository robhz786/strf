#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_STRING
#define BOOST_STRINGIFY_V0_INPUT_TYPES_STRING

#include <algorithm>
#include <limits>
#include <boost/stringify/v0/conventional_argf_reader.hpp>
#include <boost/stringify/v0/facets/width_calculator.hpp>
#include <boost/stringify/v0/stringifier.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {

struct string_argf
{
    using char_flags_type = stringify::v0::char_flags<'<', '>', '='>;

    constexpr string_argf(int w): width(w) {}
    constexpr string_argf(const char* f): flags(f) {}
    constexpr string_argf(int w, const char* f): width(w), flags(f) {}
    constexpr string_argf(const string_argf&) = default;

    int width = -1;
    char_flags_type flags;
};


template<typename CharT>
class string_stringifier: public stringifier<CharT>
{

public:

    using second_arg = stringify::v0::detail::string_argf;

private:

    using input_type = const CharT*;
    using writer_type = stringify::v0::output_writer<CharT>;
    using from32_tag = stringify::v0::conversion_from_utf32_tag<CharT>;
    using to32_tag = stringify::v0::conversion_to_utf32_tag<CharT>;
    using wcalc_tag = stringify::v0::width_calculator_tag;
    using argf_reader = stringify::v0::conventional_argf_reader<input_type>;

public:

    template <typename FTuple>
    string_stringifier
        ( const FTuple& ft
        , const CharT* str
        , const std::size_t len
        , const second_arg& argf
        ) noexcept
        : m_str(str)
        , m_len(len)
        , m_from32cv(get_facet<from32_tag>(ft))
        , m_to32cv(get_facet<to32_tag>(ft))
        , m_wcalc(get_facet<wcalc_tag>(ft))
        , m_fillchar(get_facet<fill_tag>(ft).fill_char())
        , m_width(argf_reader::get_width(argf, ft))
        , m_alignment(argf_reader::get_alignment(argf, ft))
    {
        if(m_width > 0)
        {
            determinate_fill();
        }
    }

    template <typename FTuple>
    string_stringifier
        (const FTuple& ft
        , const CharT* str
        , const std::size_t len
        ) noexcept
        : m_str(str)
        , m_len(len)
        , m_from32cv(get_facet<from32_tag>(ft))
        , m_to32cv(get_facet<to32_tag>(ft))
        , m_wcalc(get_facet<wcalc_tag>(ft))
        , m_fillchar(get_facet<fill_tag>(ft).fill_char())
        , m_width(get_facet<width_tag>(ft).width())
        , m_alignment(get_facet<alignment_tag>(ft).value())
    {
        if(m_width > 0)
        {
            determinate_fill();
        }
    }

    template <typename FTuple, typename StringType>
    string_stringifier(const FTuple& ft, const StringType& str) noexcept
        : string_stringifier(ft, &str[0], length(str))
    {
    }

    template <typename FTuple, typename StringType>
    string_stringifier
        ( const FTuple& ft
        , const StringType& str
        , const second_arg& argf
        ) noexcept
        : string_stringifier(ft, &str[0], length(str), argf)
    {
    }

    ~string_stringifier()
    {
    }

    std::size_t length() const override
    {
        if (m_fillcount > 0)
        {
            return m_len + m_from32cv.length(m_fillchar) * m_fillcount;
        }
        return m_len;
    }

    void write(writer_type& out) const override
    {
        if (m_fillcount > 0)
        {
            if(m_alignment == stringify::v0::alignment::left)
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

    int remaining_width(int w) const override
    {
        if(m_fillcount > 0)
        {
            return w > m_width ? w - m_width : 0;
        }
        return m_wcalc.remaining_width(w, m_str, m_len, m_to32cv);
    }

private:

    const CharT* m_str;
    const std::size_t m_len;
    const stringify::v0::conversion_from_utf32<CharT>& m_from32cv;
    const stringify::v0::conversion_to_utf32<CharT>& m_to32cv;
    const stringify::v0::width_calculator& m_wcalc;
    const char32_t m_fillchar;
    width_t m_fillcount = 0;
    width_t m_width;
    stringify::v0::alignment m_alignment;

    template <typename StringType>
    static std::size_t length(const StringType& str)
    {
        return str.length();
    }

    static std::size_t length(const CharT* str)
    {
        return std::char_traits<CharT>::length(str);
    }

    template <typename Category, typename FTuple>
    const auto& get_facet(const FTuple& ft) const
    {
        return ft.template get_facet<Category, input_type>();
    }

    void determinate_fill()
    {
        int fillwidth = m_wcalc.remaining_width(m_width, m_str, m_len, m_to32cv);
        if (fillwidth > 0)
        {
            m_fillcount = fillwidth / m_wcalc.width_of(m_fillchar);
        }
    }

    void write_fill(writer_type& out) const
    {
        m_from32cv.write(out, m_fillchar, m_fillcount);
    }
};


template <typename CharIn>
struct string_input_traits
{
private:

    template <typename CharOut>
    struct helper
    {
        static_assert(sizeof(CharIn) == sizeof(CharOut), "");

        using stringifier = stringify::v0::detail::string_stringifier
            <CharOut>;
    };

public:

    template <typename CharOut, typename FTuple>
    using stringifier = typename helper<CharOut>::stringifier;
};

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_stringifier<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_stringifier<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_stringifier<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_stringifier<wchar_t>;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

} // namespace detail

stringify::v0::detail::string_input_traits<char>
boost_stringify_input_traits_of(const char*);

stringify::v0::detail::string_input_traits<char16_t>
boost_stringify_input_traits_of(const char16_t*);

stringify::v0::detail::string_input_traits<char32_t>
boost_stringify_input_traits_of(const char32_t*);

stringify::v0::detail::string_input_traits<wchar_t>
boost_stringify_input_traits_of(const wchar_t*);

template<typename CharT, typename CharTraits>
stringify::v0::detail::string_input_traits<CharT>
boost_stringify_input_traits_of(const std::basic_string<CharT, CharTraits>& str);

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

template<typename CharT, typename CharTraits>
stringify::v0::detail::string_input_traits<CharT>
boost_stringify_input_traits_of(const std::basic_string_view<CharT, CharTraits>& str);

#endif //defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR_PTR */

