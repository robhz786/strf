#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_STRING
#define BOOST_STRINGIFY_V0_INPUT_TYPES_STRING

#include <algorithm>
#include <limits>
#include <boost/stringify/v0/conventional_argf_reader.hpp>
#include <boost/stringify/v0/facets/width_calculator.hpp>
#include <boost/stringify/v0/formatter.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

struct string_input_tag_base
{
};

template <typename CharT>
struct string_input_tag: string_input_tag_base
{
};


namespace detail {

struct string_argf
{
    using char_flags_type = stringify::v0::char_flags<'<', '>', '=', '^'>;

    constexpr string_argf(int w): width(w) {}
    constexpr string_argf(const char* f): flags(f) {}
    constexpr string_argf(int w, const char* f): width(w), flags(f) {}
    constexpr string_argf(const string_argf&) = default;

    int width = -1;
    char_flags_type flags;
};


template <typename CharOut>
class length_accumulator: public stringify::v0::u32output
{
public:

    length_accumulator
        ( const stringify::v0::encoder<CharOut>& encoder )
        : m_encoder(encoder)
    {
    }

    bool put(char32_t ch) override
    {
        m_length += m_encoder.length(ch);
        return true;
    }

    void set_error(std::error_code) override
    {
    }

    std::size_t get_length() const
    {
        return m_length;
    }

private:

    const stringify::v0::encoder<CharOut>& m_encoder;
    std::size_t m_length = 0;
};



template <typename CharOut>
class u32writer: public stringify::v0::u32output
{
public:
    u32writer(
        const stringify::v0::encoder<CharOut>& encoder,
        stringify::v0::output_writer<CharOut>& destination
    )
        : m_encoder(encoder)
        , m_destination(destination)
    {
    }

    bool put(char32_t ch) override
    {
        return m_encoder.encode(m_destination, 1, ch);
    }

    void set_error(std::error_code err) override
    {
        m_destination.set_error(err);
    }

private:

    const stringify::v0::encoder<CharOut>& m_encoder;
    stringify::v0::output_writer<CharOut>& m_destination;
};


template <typename CharIn, typename CharOut>
class string_writer_decode_encode
{
    using input_tag = stringify::v0::string_input_tag<CharIn>;
    using writer_type = stringify::v0::output_writer<CharOut>;
    using encoder_tag = stringify::v0::encoder_tag<CharOut>;
    using decoder_tag = stringify::v0::decoder_tag<CharIn>;
    using wcalc_tag = stringify::v0::width_calculator_tag;

    template <typename Category, typename FTuple>
    const auto& get_facet(const FTuple& ft) const
    {
        return ft.template get_facet<Category, input_tag>();
    }

public:

    template <typename FTuple>
    string_writer_decode_encode
        ( const FTuple& ft
        , const CharIn* begin
        , const CharIn* end
        ) noexcept
        : m_begin(begin)
        , m_end(end)
        , m_decoder(get_facet<decoder_tag>(ft))
        , m_encoder(get_facet<encoder_tag>(ft))
        , m_wcalc(get_facet<wcalc_tag>(ft))
    {
    }

    void write_str(stringify::v0::output_writer<CharOut>& dest) const
    {
        stringify::v0::detail::u32writer<CharOut> c32w{m_encoder, dest};
        m_decoder.decode(c32w, m_begin, m_end);
    }

    auto remaining_width(int w) const
    {
        return m_wcalc.remaining_width(w, m_begin, m_end, m_decoder);
    }

    std::size_t length() const
    {
        stringify::v0::detail::length_accumulator<CharOut> acc{m_encoder};
        m_decoder.decode(acc, m_begin, m_end);
        return acc.get_length();
    }

private:

    const CharIn* m_begin;
    const CharIn* m_end;
    const stringify::v0::decoder<CharIn>& m_decoder;
    const stringify::v0::encoder<CharOut>& m_encoder;
    const stringify::v0::width_calculator& m_wcalc;
};


class string_writer_from32_to32
{
    using CharOut = char32_t;
    using CharIn = char32_t;
    using input_tag = stringify::v0::string_input_tag<CharIn>;
    using writer_type = stringify::v0::output_writer<CharOut>;
    using wcalc_tag = stringify::v0::width_calculator_tag;

    template <typename Category, typename FTuple>
    const auto& get_facet(const FTuple& ft) const
    {
        return ft.template get_facet<Category, input_tag>();
    }

public:

    template <typename FTuple>
    string_writer_from32_to32
        ( const FTuple& ft
        , const CharIn* begin
        , const CharIn* end
        ) noexcept
        : m_begin(begin)
        , m_end(end)
        , m_wcalc(get_facet<wcalc_tag>(ft))
    {
    }

    void write_str(stringify::v0::output_writer<CharOut>& dest) const
    {
        dest.put(m_begin, m_end - m_begin);
    }

    auto remaining_width(int w) const
    {
        return m_wcalc.remaining_width(w, m_begin, m_end);
    }

    std::size_t length() const
    {
        return m_end - m_begin;
    }

private:

    const CharIn* m_begin;
    const CharIn* m_end;
    const stringify::v0::width_calculator& m_wcalc;
};



template <typename CharIn, typename CharOut>
class string_writer_reinterpret
{
    using input_tag = stringify::v0::string_input_tag<CharIn>;
    using writer_type = stringify::v0::output_writer<CharOut>;
    using decoder_tag = stringify::v0::decoder_tag<CharIn>;
    using wcalc_tag = stringify::v0::width_calculator_tag;

    template <typename Category, typename FTuple>
    const auto& get_facet(const FTuple& ft) const
    {
        return ft.template get_facet<Category, input_tag>();
    }

public:

    template <typename FTuple>
    string_writer_reinterpret
        ( const FTuple& ft
        , const CharIn* begin
        , const CharIn* end
        ) noexcept
        : m_begin(begin)
        , m_end(end)
        , m_decoder(get_facet<decoder_tag>(ft))
        , m_wcalc(get_facet<wcalc_tag>(ft))
    {
    }

    void write_str(stringify::v0::output_writer<CharOut>& dest) const
    {
        dest.put(reinterpret_cast<const CharOut*>(m_begin), m_end - m_begin);
    }

    auto remaining_width(int w) const
    {
        return m_wcalc.remaining_width(w, m_begin, m_end, m_decoder);
    }

    std::size_t length() const
    {
        return m_end - m_begin;
    }

private:

    const CharIn* m_begin;
    const CharIn* m_end;
    const stringify::v0::decoder<CharIn>& m_decoder;
    const stringify::v0::width_calculator& m_wcalc;
};



template <typename CharOut>
class string_writer_from32
{
    using CharIn = char32_t;

    using input_tag = stringify::v0::string_input_tag<CharIn>;
    using writer_type = stringify::v0::output_writer<CharOut>;
    using encoder_tag = stringify::v0::encoder_tag<CharOut>;
    using wcalc_tag = stringify::v0::width_calculator_tag;

    template <typename Category, typename FTuple>
    const auto& get_facet(const FTuple& ft) const
    {
        return ft.template get_facet<Category, input_tag>();
    }

public:

    template <typename FTuple>
    string_writer_from32
        ( const FTuple& ft
        , const CharIn* begin
        , const CharIn* end
        ) noexcept
        : m_begin(begin)
        , m_end(end)
        , m_encoder(get_facet<encoder_tag>(ft))
        , m_wcalc(get_facet<wcalc_tag>(ft))
    {
    }

    void write_str(stringify::v0::output_writer<CharOut>& dest) const
    {
        for(auto it = m_begin; it < m_end; ++it)
        {
            if( ! m_encoder.encode(dest, 1, *it))
            {
                break;
            }
        }
    }

    int remaining_width(int w) const
    {
        return m_wcalc.remaining_width(w, m_begin, m_end);
    }

    std::size_t length() const
    {
        std::size_t len = 0;
        for(auto it = m_begin; it != m_end; ++it)
        {
            len += m_encoder.length(*it);
        }
        return len;
    }

private:

    const CharIn* m_begin;
    const CharIn* m_end;
    const stringify::v0::encoder<CharOut>& m_encoder;
    const stringify::v0::width_calculator& m_wcalc;
};


template <typename CharIn, typename CharOut>
struct string_writer_helper
{
    using type
    = stringify::v0::detail::string_writer_decode_encode<CharIn, CharOut>;
};

template <>
struct string_writer_helper<char32_t, char32_t>
{
    using type = stringify::v0::detail::string_writer_from32_to32;
};

template <typename CharT>
struct string_writer_helper<char32_t, CharT>
{
    using type = stringify::v0::detail::string_writer_from32<CharT>;
};

template <typename CharT>
struct string_writer_helper<CharT, CharT>
{
    using type = stringify::v0::detail::string_writer_reinterpret<CharT, CharT>;
};


#if ! defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)

template <>
struct string_writer_helper<stringify::v0::detail::wchar_equivalent, wchar_t>
{
    using type = stringify::v0::detail::string_writer_reinterpret
        < stringify::v0::detail::wchar_equivalent
        , wchar_t
        >;
};

template <>
struct string_writer_helper<wchar_t, stringify::v0::detail::wchar_equivalent>
{
    using type = stringify::v0::detail::string_writer_reinterpret
        < wchar_t
        , stringify::v0::detail::wchar_equivalent
        >;
};

#endif //  ! defined(BOOST_STRINGIFY_DONT_ASSUME_WCHAR_ENCODING)

template <typename CharIn, typename CharOut>
using string_writer
= typename stringify::v0::detail::string_writer_helper<CharIn, CharOut>::type;


template<typename CharIn, typename CharOut>
class string_formatter: public formatter<CharOut>
{

public:

    using second_arg = stringify::v0::detail::string_argf;

private:

    using input_tag = stringify::v0::string_input_tag<CharIn>;
    using writer_type = stringify::v0::output_writer<CharOut>;
    using encoder_tag = stringify::v0::encoder_tag<CharOut>;
    using argf_reader = stringify::v0::conventional_argf_reader<input_tag>;

public:

    template <typename FTuple>
    string_formatter
        ( const FTuple& ft
        , const CharIn* begin
        , const CharIn* end
        , const second_arg& argf
        ) noexcept
        : m_str(ft, begin, end)
        , m_encoder(get_facet<encoder_tag>(ft))
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
    string_formatter
        ( const FTuple& ft
        , const CharIn* begin
        , const CharIn* end
        ) noexcept
        : m_str(ft, begin, end)
        , m_encoder(get_facet<encoder_tag>(ft))
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
    string_formatter(const FTuple& ft, const StringType& str) noexcept
        : string_formatter(ft, &str[0], end_of(str))
    {
    }

    template <typename FTuple, typename StringType>
    string_formatter
        ( const FTuple& ft
        , const StringType& str
        , const second_arg& argf
        ) noexcept
        : string_formatter(ft, &str[0], end_of(str), argf)
    {
    }

    ~string_formatter()
    {
    }

    std::size_t length() const override
    {
        std::size_t len = m_str.length();
        if (m_fillcount > 0)
        {
            len += m_encoder.length(m_fillchar) * m_fillcount;
        }
        return len;
    }

    void write(writer_type& out) const override
    {
        if (m_fillcount > 0)
        {
            if(m_alignment == stringify::v0::alignment::left)
            {
                m_str.write_str(out);
                write_fill(out, m_fillcount);
            }
            else if (m_alignment == stringify::v0::alignment::center)
            {
                int halfcount = m_fillcount / 2;
                write_fill(out, halfcount);
                m_str.write_str(out);
                write_fill(out, m_fillcount - halfcount);
            }
            else
            {
                write_fill(out, m_fillcount);
                m_str.write_str(out);
            }
        }
        else
        {
            m_str.write_str(out);
        }
    }

    int remaining_width(int w) const override
    {
        if(m_fillcount > 0)
        {
            return w > m_width ? w - m_width : 0;
        }
        return m_str.remaining_width(w);
    }

private:

    const stringify::v0::detail::string_writer<CharIn, CharOut> m_str;
    const stringify::v0::encoder<CharOut>& m_encoder;
    const char32_t m_fillchar;
    width_t m_fillcount = 0;
    width_t m_width;
    stringify::v0::alignment m_alignment;

    template <typename StringType>
    static const auto* end_of(const StringType& str)
    {
        return &str[0] + str.length();
    }

    static const CharIn* end_of(const CharIn* str)
    {
        return str + std::char_traits<CharIn>::length(str);
    }

    template <typename Category, typename FTuple>
    const auto& get_facet(const FTuple& ft) const
    {
        return ft.template get_facet<Category, input_tag>();
    }

    void determinate_fill()
    {
        m_fillcount = m_str.remaining_width(m_width);
    }

    void write_fill(writer_type& out, int count ) const
    {
        m_encoder.encode(out, count, m_fillchar);
    }
};


template <typename CharIn>
struct string_input_traits
{
    template <typename CharOut, typename FTuple>
    using formatter
    = stringify::v0::detail::string_formatter<CharIn, CharOut>;
};


#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_formatter<char, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_formatter<char, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_formatter<char, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_formatter<char, wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_formatter<char16_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_formatter<char16_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_formatter<char16_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_formatter<char16_t, wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_formatter<char32_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_formatter<char32_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_formatter<char32_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_formatter<char32_t, wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_formatter<wchar_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_formatter<wchar_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_formatter<wchar_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_formatter<wchar_t, wchar_t>;

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

