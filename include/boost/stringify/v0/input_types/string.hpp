#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_STRING
#define BOOST_STRINGIFY_V0_INPUT_TYPES_STRING

#include <algorithm>
#include <limits>
#include <boost/stringify/v0/basic_types.hpp>
#include <boost/stringify/v0/align_formatting.hpp>
#include <boost/stringify/v0/ftuple.hpp>
#include <boost/stringify/v0/facets/encoder.hpp>
#include <boost/stringify/v0/facets/decoder.hpp>
#include <boost/stringify/v0/facets/width_calculator.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

struct string_input_tag_base
{
};

template <typename CharT>
struct string_input_tag: string_input_tag_base
{
};

template <typename CharT>
struct is_string_of
{
    template <typename T>
    using fn = std::is_base_of<string_input_tag<CharT>, T>;
};

template <typename T>
using is_string = std::is_base_of<string_input_tag_base, T>;


template <typename CharT>
class string_with_formatting
    : public stringify::v0::align_formatting<string_with_formatting<CharT> >
{
public:

    template <typename T>
    using fmt_tmpl = stringify::v0::align_formatting<T>;

    using fmt_type = fmt_tmpl<string_with_formatting>;

    constexpr string_with_formatting
        ( const CharT* str
        , std::size_t len
        ) noexcept
        : m_str(str)
        , m_end(str + len)
    {
    }

    constexpr string_with_formatting
        ( const CharT* str
        , std::size_t len
        , const fmt_type& fmt
        ) noexcept
        : fmt_type(fmt)
        , m_str(str)
        , m_end(str + len)
    {
    }

    constexpr string_with_formatting
        ( const CharT* str
        ) noexcept
        : m_str(str)
        , m_end(str + std::char_traits<CharT>::length(str))
    {
    }

    constexpr string_with_formatting(const string_with_formatting&) = default;

    void value(const CharT* str)
    {
        m_str = str;
        m_end = std::char_traits<CharT>::length(str);
    }

    template <class StringType>
    void value(const StringType& str)
    {
        m_str = &str[0];
        m_end = m_str + str.length();
    }

    constexpr const CharT* begin() const
    {
        return m_str;
    }
    constexpr const CharT* end() const
    {
        return m_end;
    }

    void operator%(int) const = delete;

private:

    const CharT* m_str;
    const CharT* m_end;
};


namespace detail {

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
class encoder_adapter: public stringify::v0::u32output
{
public:
    encoder_adapter
        ( const stringify::v0::encoder<CharOut>& encoder
        , stringify::v0::output_writer<CharOut>& destination
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

template <typename CharOut>
class u32writer: public stringify::v0::u32output
{
public:

    u32writer(stringify::v0::output_writer<CharOut>& destination)
        : m_destination(destination)
    {
    }

    bool put(char32_t ch) override
    {
        return m_destination.put(static_cast<CharOut>(ch));
    }

    void set_error(std::error_code err) override
    {
        m_destination.set_error(err);
    }

private:

    stringify::v0::output_writer<CharOut>& m_destination;
};

template <typename CharIn, typename CharOut>
class string_writer
{
    using input_tag = stringify::v0::string_input_tag<CharIn>;

public:

    string_writer
        ( const CharIn* begin
        , const CharIn* end
        , const stringify::v0::width_calculator& wcalc
        , const stringify::v0::decoder<CharIn>& decoder
        , const stringify::v0::encoder<CharOut>& encoder
//        , bool assume_wchar_encoding
        ) noexcept
        : m_begin(begin)
        , m_end(end)
        , m_wcalc(wcalc)
        , m_decoder(decoder)
        , m_encoder(encoder)
//        , m_assume_wchar_encoding(assume_wchar_encoding)
    {
    }

    void write_str(stringify::v0::output_writer<CharOut>& dest) const
    {
        if (shall_skip_encoder_and_decoder())
        {
            dest.put(reinterpret_cast<const CharOut*>(m_begin), m_end - m_begin);
        }
        else if(shall_skip_decoder())
        {
            for(auto it = m_begin; it < m_end; ++it)
            {
                if( ! m_encoder.encode(dest, 1, *it))
                {
                    return;
                }
            }
        }
        else if(shall_skip_encoder())
        {
            stringify::v0::detail::u32writer<CharOut> writer{dest};
            m_decoder.decode(writer, m_begin, m_end);
        }
        else
        {
            //decode and encode
            stringify::v0::detail::encoder_adapter<CharOut> c32w{m_encoder, dest};
            m_decoder.decode(c32w, m_begin, m_end);
        }
    }

    auto remaining_width(int w) const
    {
        if (shall_skip_encoder())
        {
            auto begin = reinterpret_cast<const char32_t*>(m_begin);
            auto end   = reinterpret_cast<const char32_t*>(m_end);
            return m_wcalc.remaining_width(w, begin, end);
        }
        return m_wcalc.remaining_width(w, m_begin, m_end, m_decoder);
    }

    std::size_t length() const
    {

        if(shall_skip_encoder_and_decoder())
        {
            return m_end - m_begin;
        }
        else if(shall_skip_decoder())
        {
            std::size_t len = 0;
            for(auto it = m_begin; it != m_end; ++it)
            {
                len += m_encoder.length(*it);
            }
            return len;
        }
        else if(shall_skip_encoder())
        {
            stringify::v0::u32encoder<char32_t> dummy_encoder;
            stringify::v0::detail::length_accumulator<char32_t> acc{dummy_encoder};
            m_decoder.decode(acc, m_begin, m_end);
            return acc.get_length();
        }
        else
        {
            stringify::v0::detail::length_accumulator<CharOut> acc{m_encoder};
            m_decoder.decode(acc, m_begin, m_end);
            return acc.get_length();
        }
    }

private:

    const CharIn* m_begin;
    const CharIn* m_end;

public:

    const stringify::v0::width_calculator& m_wcalc;
    const stringify::v0::decoder<CharIn>& m_decoder;
    const stringify::v0::encoder<CharOut>& m_encoder;

private:

    constexpr static bool m_assume_wchar_encoding = true;

    constexpr bool shall_skip_encoder_and_decoder() const
    {
        return
            std::is_same<CharIn, CharOut>::value
            || (m_assume_wchar_encoding && sizeof(CharIn) == sizeof(CharOut));
    }

    constexpr bool shall_skip_decoder() const
    {
        return std::is_same<CharIn, char32_t>::value
            || (m_assume_wchar_encoding && sizeof(CharIn) == 4);
    }

    constexpr bool shall_skip_encoder() const
    {
        return std::is_same<CharOut, char32_t>::value
            || (m_assume_wchar_encoding && sizeof(CharOut) == 4);
    }
};

} // namespace detail


template<typename CharIn, typename CharOut>
class string_printer: public printer<CharOut>
{
private:

    using input_tag = stringify::v0::string_input_tag<CharIn>;
    using writer_type = stringify::v0::output_writer<CharOut>;

public:

    template <typename FTuple>
    string_printer
        ( const FTuple& ft
        , const CharIn* str
        , std::size_t len
        ) noexcept
        : string_printer
            ( stringify::v0::string_with_formatting<CharIn>{str, len}
            , get_facet<stringify::v0::decoder_category<CharIn>>(ft)
            , get_facet<stringify::v0::encoder_category<CharOut>>(ft)
            , get_facet<stringify::v0::width_calculator_category>(ft)
            )
    {
    }

    template <typename FTuple>
    string_printer
        ( const FTuple& ft
        , const stringify::v0::string_with_formatting<CharIn>& input
        ) noexcept
        : string_printer
            ( input
            , get_facet<stringify::v0::decoder_category<CharIn>>(ft)
            , get_facet<stringify::v0::encoder_category<CharOut>>(ft)
            , get_facet<stringify::v0::width_calculator_category>(ft)
            )
    {
    }

    string_printer
        ( const stringify::v0::string_with_formatting<CharIn>& input
        , const stringify::v0::decoder<CharIn>& decoder
        , const stringify::v0::encoder<CharOut>& encoder
        , const stringify::v0::width_calculator& wcalc
        ) noexcept;

    string_printer
        ( const CharIn* begin
        , const CharIn* end
        , const stringify::v0::decoder<CharIn>& decoder
        , const stringify::v0::encoder<CharOut>& encoder
        , const stringify::v0::width_calculator& wcalc
        ) noexcept;

    ~string_printer();

    std::size_t length() const override;

    void write(writer_type& out) const override;

    int remaining_width(int w) const override;

private:

    const stringify::v0::detail::string_writer<CharIn, CharOut> m_str;
    const typename stringify::v0::string_with_formatting<CharIn>::fmt_type m_fmt;
    const int m_fillcount = 0;

    template <typename Category, typename FTuple>
    const auto& get_facet(const FTuple& ft) const
    {
        return ft.template get_facet<Category, input_tag>();
    }

    void write_fill(writer_type& out, int count ) const
    {
        m_str.m_encoder.encode(out, count, m_fmt.fill());
    }
};

template<typename CharIn, typename CharOut>
string_printer<CharIn, CharOut>::string_printer
    ( const stringify::v0::string_with_formatting<CharIn>& input
    , const stringify::v0::decoder<CharIn>& decoder
    , const stringify::v0::encoder<CharOut>& encoder
    , const stringify::v0::width_calculator& wcalc
    ) noexcept
    : m_str
          ( input.begin()
          , input.end()
          , wcalc
          , decoder
          , encoder
          )
    , m_fmt(input)
    , m_fillcount(input.width() > 0 ? m_str.remaining_width(input.width()) : 0)
{
}

template<typename CharIn, typename CharOut>
string_printer<CharIn, CharOut>::string_printer
    ( const CharIn* begin
    , const CharIn* end
    , const stringify::v0::decoder<CharIn>& decoder
    , const stringify::v0::encoder<CharOut>& encoder
    , const stringify::v0::width_calculator& wcalc
    ) noexcept
    : m_str(begin, end, wcalc, decoder, encoder)
    , m_fmt()
{
}

template<typename CharIn, typename CharOut>
string_printer<CharIn, CharOut>::~string_printer()
{
}


template<typename CharIn, typename CharOut>
std::size_t string_printer<CharIn, CharOut>::length() const
{
    std::size_t len = m_str.length();
    if (m_fillcount > 0)
    {
        len += m_str.m_encoder.length(m_fmt.fill()) * m_fillcount;
    }
    return len;
}


template<typename CharIn, typename CharOut>
void string_printer<CharIn, CharOut>::write(writer_type& out) const
{
    if (m_fillcount > 0)
    {
        switch(m_fmt.alignment())
        {
            case stringify::v0::alignment::left:
            {
                m_str.write_str(out);
                write_fill(out, m_fillcount);
                break;
            }
            case stringify::v0::alignment::center:
            {
                int halfcount = m_fillcount / 2;
                write_fill(out, halfcount);
                m_str.write_str(out);
                write_fill(out, m_fillcount - halfcount);
                break;
            }
            default:
            {
                write_fill(out, m_fillcount);
                m_str.write_str(out);
            }
        }
    }
    else
    {
        m_str.write_str(out);
    }
}


template<typename CharIn, typename CharOut>
int string_printer<CharIn, CharOut>::remaining_width(int w) const
{
    return m_str.remaining_width(w) - m_fillcount;
}


#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char, wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char16_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char16_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char16_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char16_t, wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char32_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char32_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char32_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<char32_t, wchar_t>;

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<wchar_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<wchar_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<wchar_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class string_printer<wchar_t, wchar_t>;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

// namespace detail {
// struct string_input_traits
// {
//     template
//         < typename CharOut
//         , typename FTuple
//         , typename CharIn
//         >
//     static inline stringify::v0::string_printer<CharIn, CharOut>
//     make_printer(const FTuple& ft, const CharIn* str)
//     {
//         return {ft, str, std::char_traits<CharIn>::length(str)};
//     }

//     template
//         < typename CharOut
//         , typename FTuple
//         , typename String
//         , typename CharIn = typename String::value_type
//         >
//     static inline stringify::v0::string_printer<CharIn, CharOut>
//     make_printer(const FTuple& ft, const String& str)
//     {
//         return {ft, str.data(), str.size()};
//     }

//     template <typename CharOut, typename FTuple, typename CharIn>
//     static inline stringify::v0::string_printer<CharIn, CharOut>
//     make_printer
//        ( const FTuple& ft
//        , stringify::v0::string_with_formatting<CharIn> x
//        )
//     {
//         return {ft, x};
//     }

//     template <typename String, typename CharIn = typename String::value_type>
//     static inline stringify::v0::string_with_formatting<CharIn> fmt(const String& str)
//     {
//         return {str.data(), str.size()};
//     }

//     template <typename CharIn>
//     static inline stringify::v0::string_with_formatting<CharIn> fmt(const CharIn* str)
//     {
//         return {str, std::char_traits<CharIn>::length(str)};
//     }
// };
// }

// stringify::v0::detail::string_input_traits stringify_get_input_traits(const char*);
// stringify::v0::detail::string_input_traits stringify_get_input_traits(const wchar_t*);
// stringify::v0::detail::string_input_traits stringify_get_input_traits(const char16_t*);
// stringify::v0::detail::string_input_traits stringify_get_input_traits(const char32_t*);

// template <typename CharIn, typename Traits, typename Allocator>
// stringify::v0::detail::string_input_traits stringify_get_input_traits
//     ( const std::basic_string<CharIn, Traits, Allocator>& );

// #if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

// template <typename CharIn, typename Traits>
// stringify::v0::detail::string_input_traits stringify_get_input_traits
//     ( const std::basic_string_view<CharIn, Traits>& );

// #endif // defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

// stringify::v0::detail::string_input_traits
// stringify_get_input_traits(const stringify::v0::string_with_formatting<char>&);

// stringify::v0::detail::string_input_traits
// stringify_get_input_traits(const stringify::v0::string_with_formatting<wchar_t>&);

// stringify::v0::detail::string_input_traits
// stringify_get_input_traits(const stringify::v0::string_with_formatting<char16_t>&);

// stringify::v0::detail::string_input_traits
// stringify_get_input_traits(const stringify::v0::string_with_formatting<char32_t>&);

template
    < typename CharOut
    , typename FTuple
    , typename CharIn
    , typename Traits
    , typename Allocator
    >
inline stringify::v0::string_printer<CharIn, CharOut>
stringify_make_printer
   ( const FTuple& ft
   , const std::basic_string<CharIn, Traits, Allocator>& str
   )
{
    return {ft, str.data(), str.size()};
}

template <typename CharOut, typename FTuple>
inline stringify::v0::string_printer<char, CharOut>
stringify_make_printer
   ( const FTuple& ft
   , const char* str
   )
{
    return {ft, str, std::char_traits<char>::length(str)};
}

template <typename CharOut, typename FTuple>
inline stringify::v0::string_printer<wchar_t, CharOut>
stringify_make_printer
   ( const FTuple& ft
   , const wchar_t* str
   )
{
    return {ft, str, std::char_traits<wchar_t>::length(str)};
}

template <typename CharOut, typename FTuple>
inline stringify::v0::string_printer<char16_t, CharOut>
stringify_make_printer
   ( const FTuple& ft
   , const char16_t* str
   )
{
    return {ft, str, std::char_traits<char16_t>::length(str)};
}

template <typename CharOut, typename FTuple>
inline stringify::v0::string_printer<char32_t, CharOut>
stringify_make_printer
   ( const FTuple& ft
   , const char32_t* str
   )
{
    return {ft, str, std::char_traits<char32_t>::length(str)};
}

template <typename CharT, typename Traits>
inline stringify::v0::string_with_formatting<CharT>
stringify_fmt(const std::basic_string<CharT, Traits>& str)
{
    return {str.data(), str.size()};
}

inline stringify::v0::string_with_formatting<char>
stringify_fmt(const char* str)
{
    return {str, std::char_traits<char>::length(str)};
}
inline stringify::v0::string_with_formatting<wchar_t>
stringify_fmt(const wchar_t* str)
{
    return {str, std::char_traits<wchar_t>::length(str)};
}
inline stringify::v0::string_with_formatting<char16_t>
stringify_fmt(const char16_t* str)
{
    return {str, std::char_traits<char16_t>::length(str)};
}
inline stringify::v0::string_with_formatting<char32_t>
stringify_fmt(const char32_t* str)
{
    return {str, std::char_traits<char32_t>::length(str)};
}

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

template <typename CharOut, typename FTuple, typename CharIn, typename Traits>
inline stringify::v0::string_printer<CharIn, CharOut>
stringify_make_printer
   ( const FTuple& ft
   , const std::basic_string_view<CharIn, Traits>& str
   )
{
    return {ft, str.data(), str.size()};
}

template <typename CharT, typename Traits>
inline stringify::v0::string_with_formatting<CharT>
stringify_fmt(const std::basic_string_view<CharT, Traits>& str)
{
    return {str.data(), str.size()};
}

#endif

template <typename CharOut, typename FTuple, typename CharIn>
inline stringify::v0::string_printer<CharIn, CharOut>
stringify_make_printer
    ( const FTuple& ft
    , const stringify::v0::string_with_formatting<CharIn>& ch
    )
{
    return {ft, ch};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR_PTR */

