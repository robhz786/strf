#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_STRING
#define BOOST_STRINGIFY_V0_INPUT_TYPES_STRING

#include <algorithm>
#include <limits>
#include <boost/stringify/v0/facets_pack.hpp>
#include <boost/stringify/v0/facets/width_calculator.hpp>
#include <boost/utility/string_view.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT>
struct is_string_of
{
    template <typename T>
    using fn = std::is_base_of<string_input_tag<CharT>, T>;
};

template <typename T>
using is_string = std::is_base_of<string_input_tag_base, T>;


template <typename CharIn, typename T>
class string_formatting: public stringify::v0::align_formatting<T>
{
    using child_type = T;

public:

    template <typename, typename>
    friend class string_formatting;

    template <typename U>
    using fmt_other = stringify::v0::string_formatting<CharIn, U>;

    constexpr string_formatting() = default;

    constexpr string_formatting(const string_formatting&) = default;

    template <typename U>
    string_formatting(const string_formatting<CharIn, U>& u)
        : stringify::v0::align_formatting<T>(u)
        , m_encoding_info(u.m_encoding_info)
        , m_sani(u.m_sani)
    {
    }

    ~string_formatting() = default;

    constexpr child_type& sani(bool s = true) &
    {
        m_sani = s;
        return *this;
    }
    constexpr child_type&& sani(bool s = true) &&
    {
        m_sani = s;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& encoding(stringify::v0::encoding<CharIn> eid) &
    {
        m_encoding_info = *eid.info();
        return *this;
    }
    constexpr child_type&& encoding(stringify::v0::encoding<CharIn> eid) &&
    {
        m_encoding_info = & eid.info();
        return static_cast<child_type&&>(*this);
    }
    constexpr bool get_sani() const
    {
        return m_sani;
    }
    bool has_encoding() const
    {
        return  m_encoding_info != nullptr;
    }
    stringify::v0::encoding<CharIn> encoding() const
    {
        BOOST_ASSERT(has_encoding());
        return *m_encoding_info;
    }

private:

    const stringify::v0::encoding_info<CharIn>* m_encoding_info = nullptr;
    bool m_sani = false;
};


template <typename CharIn>
class string_with_formatting
    : public stringify::v0::string_formatting<CharIn, string_with_formatting<CharIn> >
{
public:

    using fmt_type = stringify::v0::string_formatting<CharIn, string_with_formatting>;

    constexpr string_with_formatting(const CharIn* str) noexcept
        : m_begin(str)
        , m_end(str + std::char_traits<CharIn>::length(str))
    {
    }

    constexpr string_with_formatting(const CharIn* str, std::size_t len) noexcept
        : m_begin(str)
        , m_end(str + len)
    {
    }

    template <typename Traits>
    string_with_formatting(const std::basic_string<CharIn, Traits>& str)
        : m_begin(str.data())
        , m_end(m_begin + str.size())
    {
    }

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

    template <typename Traits>
    string_with_formatting(const std::basic_string_view<CharIn, Traits>& str)
        : m_begin(str.begin())
        , m_end(str.end())
    {
    }

#endif //defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

    template <typename U>
    constexpr string_with_formatting
        ( const CharIn* str
        , const stringify::v0::string_formatting<CharIn, U>& u )
        noexcept
        : fmt_type(u)
        , m_begin(str)
        , m_end(str + std::char_traits<CharIn>::length(str))
    {
    }

    template <typename Traits, typename U>
    string_with_formatting
        ( const std::basic_string<CharIn, Traits>& str
        , const stringify::v0::string_formatting<CharIn, U>& u )
        : fmt_type(u)
        , m_begin(str.data())
        , m_end(m_begin + str.size())
    {
    }

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

    template <typename Traits, typename U>
    string_with_formatting
        ( const std::basic_string_view<CharIn, Traits>& str
        , const stringify::v0::string_formatting<CharIn, U>& u )
        : fmt_type(u)
        , m_begin(str.begin())
        , m_end(str.end())
    {
    }

#endif // defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

    constexpr string_with_formatting(const string_with_formatting&) = default;

    void value(const CharIn* str)
    {
        m_begin = str;
        m_end = std::char_traits<CharIn>::length(str);
    }

    template <class StringType>
    void value(const StringType& str)
    {
        m_begin = &str[0];
        m_end = m_begin + str.length();
    }

    constexpr const CharIn* begin() const
    {
        return m_begin;
    }
    constexpr const CharIn* end() const
    {
        return m_end;
    }

    void operator%(int) const = delete;

private:

    const CharIn* m_begin;
    const CharIn* m_end;
};

template<typename CharIn, typename CharOut>
class simple_string_printer: public stringify::v0::printer<CharOut>
{
    using input_tag = stringify::v0::string_input_tag<CharIn>;
    using writer_type = stringify::v0::output_writer<CharOut>;

public:

    template <typename FPack>
    simple_string_printer
        ( stringify::v0::output_writer<CharOut>& out
        , const FPack& ft
        , const CharIn* str
        , std::size_t len
        ) noexcept
        : simple_string_printer
            ( out
            , str
            , str + len
            , get_facet<stringify::v0::encoding_category<CharIn>>(ft)
            , get_facet<stringify::v0::width_calculator_category>(ft)
            )
    {
    }

    simple_string_printer
        ( stringify::v0::output_writer<CharOut>& out
        , const CharIn* begin
        , const CharIn* end
        , const stringify::v0::encoding<CharIn>& input_enc
        , const stringify::v0::width_calculator& wcalc
        ) noexcept
        : m_begin(begin)
        , m_end(end)
        , m_sw(out, input_enc, false)
        , m_decoder(input_enc.decoder())
        , m_wcalc(wcalc)
    {
    }

    ~simple_string_printer() = default;

    std::size_t length() const override;

    void write() const override;

    int remaining_width(int w) const override;

private:

    const CharIn* m_begin;
    const CharIn* m_end;
    const stringify::v0::string_writer<CharIn, CharOut> m_sw;
    const stringify::v0::decoder<CharIn>& m_decoder;
    const stringify::v0::width_calculator m_wcalc;

    template <typename Category, typename FPack>
    const auto& get_facet(const FPack& ft) const
    {
        return ft.template get_facet<Category, input_tag>();
    }
};

template<typename CharIn, typename CharOut>
std::size_t simple_string_printer<CharIn, CharOut>::length() const
{
    return m_sw.length(m_begin, m_end);
}

template<typename CharIn, typename CharOut>
void simple_string_printer<CharIn, CharOut>::write() const
{
    m_sw.write(m_begin, m_end);
}

template<typename CharIn, typename CharOut>
int simple_string_printer<CharIn, CharOut>::remaining_width(int w) const
{
    return m_wcalc.remaining_width
        ( w
        , m_begin
        , m_end
        , m_decoder
        , m_sw.on_error()
        , m_sw.allow_surrogates() );
}

template <typename CharT>
class simple_string_printer<CharT, CharT>: public stringify::v0::printer<CharT>
{
    using CharIn = CharT;
    using CharOut = CharT;
    using input_tag = stringify::v0::string_input_tag<CharIn>;
    using writer_type = stringify::v0::output_writer<CharOut>;

public:

    template <typename FPack>
    simple_string_printer
        ( stringify::v0::output_writer<CharOut>& out
        , const FPack& ft
        , const CharIn* str
        , std::size_t len
        ) noexcept
        : m_out(out)
        , m_str(str)
        , m_len(len)
        , m_wcalc(get_facet<stringify::v0::width_calculator_category>(ft))
    {
    }

    ~simple_string_printer() = default;

    std::size_t length() const override;

    void write() const override;

    int remaining_width(int w) const override;

private:

    stringify::v0::output_writer<CharOut>& m_out;
    const CharIn* m_str;
    const std::size_t m_len;
    const stringify::v0::width_calculator m_wcalc;

    template <typename Category, typename FPack>
    const auto& get_facet(const FPack& ft) const
    {
        return ft.template get_facet<Category, input_tag>();
    }
};

template<typename CharT>
std::size_t simple_string_printer<CharT, CharT>::length() const
{
    return m_len;
}

template<typename CharT>
void simple_string_printer<CharT, CharT>::write() const
{
    m_out.put(m_str, m_len);
}

template<typename CharT>
int simple_string_printer<CharT, CharT>::remaining_width(int w) const
{
    return m_wcalc.remaining_width
        ( w
        , m_str
        , m_str + m_len
        , m_out.encoding().decoder()
        , m_out.on_error()
        , m_out.allow_surrogates() );
}

template<typename CharIn, typename CharOut>
class string_printer: public printer<CharOut>
{
private:

    using input_tag = stringify::v0::string_input_tag<CharIn>;
    using writer_type = stringify::v0::output_writer<CharOut>;

public:

    template <typename FPack>
    string_printer
        ( stringify::v0::output_writer<CharOut>& out
        , const FPack& ft
        , const stringify::v0::string_with_formatting<CharIn>& input
        ) noexcept
        : string_printer
            ( out
            , input
            , get_facet<stringify::v0::encoding_category<CharIn>>(ft)
            , get_facet<stringify::v0::width_calculator_category>(ft)
            )
    {
    }

    string_printer
        ( stringify::v0::output_writer<CharOut>& out
        , const stringify::v0::string_with_formatting<CharIn>& input
        , const stringify::v0::encoding<CharIn> input_enc
        , const stringify::v0::width_calculator& wcalc
        ) noexcept;

    ~string_printer();

    std::size_t length() const override;

    void write() const override;

    int remaining_width(int w) const override;

private:

    const stringify::v0::string_with_formatting<CharIn> m_fmt;
    const stringify::v0::string_writer<CharIn, CharOut> m_sw;
    const stringify::v0::decoder<CharIn>& m_decoder;
    const stringify::v0::width_calculator m_wcalc;
    const int m_fillcount = 0;

    template <typename Category, typename FPack>
    const auto& get_facet(const FPack& ft) const
    {
        return ft.template get_facet<Category, input_tag>();
    }

    void write_string() const
    {
        m_sw.write(m_fmt.begin(), m_fmt.end());
    }

    void write_fill(int count) const
    {
        m_sw.put32(count, m_fmt.fill());
    }
};

template<typename CharIn, typename CharOut>
string_printer<CharIn, CharOut>::string_printer
    ( stringify::v0::output_writer<CharOut>& out
    , const stringify::v0::string_with_formatting<CharIn>& input
    , const stringify::v0::encoding<CharIn> input_enc
    , const stringify::v0::width_calculator& wcalc
    ) noexcept
    : m_fmt(input)
    , m_sw
        ( out
        , input.has_encoding() ? input.encoding() : input_enc
        , input.get_sani() )
    , m_decoder(input_enc.decoder())
    , m_wcalc(wcalc)
    , m_fillcount
        ( input.width() > 0
        ? wcalc.remaining_width
            ( input.width()
            , input.begin()
            , input.end()
            , input_enc.decoder()
            , out.on_error()
            , out.allow_surrogates() )
        : 0 )
{
}

template<typename CharIn, typename CharOut>
string_printer<CharIn, CharOut>::~string_printer()
{
}


template<typename CharIn, typename CharOut>
std::size_t string_printer<CharIn, CharOut>::length() const
{
    std::size_t len = m_sw.length(m_fmt.begin(), m_fmt.end());

    if (m_fillcount > 0)
    {
        len += m_fillcount * m_sw.required_size(m_fmt.fill());
    }
    return len;
}


template<typename CharIn, typename CharOut>
void string_printer<CharIn, CharOut>::write() const
{
    if (m_fillcount > 0)
    {
        switch(m_fmt.alignment())
        {
            case stringify::v0::alignment::left:
            {
                write_string();
                write_fill(m_fillcount);
                break;
            }
            case stringify::v0::alignment::center:
            {
                int halfcount = m_fillcount / 2;
                write_fill(halfcount);
                write_string();
                write_fill(m_fillcount - halfcount);
                break;
            }
            default:
            {
                write_fill(m_fillcount);
                write_string();
            }
        }
    }
    else
    {
        write_string();
    }
}


template<typename CharIn, typename CharOut>
int string_printer<CharIn, CharOut>::remaining_width(int w) const
{
    if (w > m_fillcount)
    {
        return m_wcalc.remaining_width
            ( w - m_fillcount
            , m_fmt.begin()
            , m_fmt.end()
            , m_decoder
            , m_sw.on_error()
            , m_sw.allow_surrogates() );
    }
    return 0;
}


#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char16_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char16_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char16_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char16_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char32_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char32_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char32_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<char32_t, wchar_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<wchar_t, char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<wchar_t, char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<wchar_t, char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class simple_string_printer<wchar_t, wchar_t>;

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
//         , typename FPack
//         , typename CharIn
//         >
//     static inline stringify::v0::string_printer<CharIn, CharOut>
//     make_printer(const FPack& ft, const CharIn* str)
//     {
//         return {ft, str, std::char_traits<CharIn>::length(str)};
//     }

//     template
//         < typename CharOut
//         , typename FPack
//         , typename String
//         , typename CharIn = typename String::value_type
//         >
//     static inline stringify::v0::string_printer<CharIn, CharOut>
//     make_printer(const FPack& ft, const String& str)
//     {
//         return {ft, str.data(), str.size()};
//     }

//     template <typename CharOut, typename FPack, typename CharIn>
//     static inline stringify::v0::string_printer<CharIn, CharOut>
//     make_printer
//        ( const FPack& ft
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
    , typename FPack
    , typename CharIn
    , typename Traits
    , typename Allocator
    >
inline stringify::v0::simple_string_printer<CharIn, CharOut>
stringify_make_printer
   ( stringify::v0::output_writer<CharOut>& out
   , const FPack& ft
   , const std::basic_string<CharIn, Traits, Allocator>& str
   )
{
    return {out, ft, str.data(), str.size()};
}

template
    < typename CharOut
    , typename FPack
    , typename CharIn
    , typename Traits
    >
inline stringify::v0::simple_string_printer<CharIn, CharOut>
stringify_make_printer
   ( stringify::v0::output_writer<CharOut>& out
   , const FPack& ft
   , const boost::basic_string_view<CharIn, Traits>& str
   )
{
    return {out, ft, str.data(), str.size()};
}

template <typename CharOut, typename FPack>
inline stringify::v0::simple_string_printer<char, CharOut>
stringify_make_printer
   ( stringify::v0::output_writer<CharOut>& out
   , const FPack& ft
   , const char* str
   )
{
    return {out, ft, str, std::char_traits<char>::length(str)};
}

template <typename CharOut, typename FPack>
inline stringify::v0::simple_string_printer<wchar_t, CharOut>
stringify_make_printer
   ( stringify::v0::output_writer<CharOut>& out
   , const FPack& ft
   , const wchar_t* str
   )
{
    return {out, ft, str, std::char_traits<wchar_t>::length(str)};
}

template <typename CharOut, typename FPack>
inline stringify::v0::simple_string_printer<char16_t, CharOut>
stringify_make_printer
   ( stringify::v0::output_writer<CharOut>& out
   , const FPack& ft
   , const char16_t* str
   )
{
    return {out, ft, str, std::char_traits<char16_t>::length(str)};
}

template <typename CharOut, typename FPack>
inline stringify::v0::simple_string_printer<char32_t, CharOut>
stringify_make_printer
   ( stringify::v0::output_writer<CharOut>& out
   , const FPack& ft
   , const char32_t* str
   )
{
    return {out, ft, str, std::char_traits<char32_t>::length(str)};
}


template <typename CharIn, typename Traits>
inline stringify::v0::string_with_formatting<CharIn>
stringify_fmt(const std::basic_string<CharIn, Traits>& str)
{
    return {str};
}

template <typename CharIn, typename Traits>
inline stringify::v0::string_with_formatting<CharIn>
stringify_fmt(const boost::basic_string_view<CharIn, Traits>& str)
{
    return {str.data(), str.size()};
}

inline stringify::v0::string_with_formatting<char>
stringify_fmt(const char* str)
{
    return {str};
}
inline stringify::v0::string_with_formatting<wchar_t>
stringify_fmt(const wchar_t* str)
{
    return {str};
}
inline stringify::v0::string_with_formatting<char16_t>
stringify_fmt(const char16_t* str)
{
    return {str};
}
inline stringify::v0::string_with_formatting<char32_t>
stringify_fmt(const char32_t* str)
{
    return {str};
}

#if defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

template <typename CharOut, typename FPack, typename CharIn, typename Traits>
inline stringify::v0::simple_string_printer<CharIn, CharOut>
stringify_make_printer
   ( stringify::v0::output_writer<CharOut>& out
   , const FPack& ft
   , const std::basic_string_view<CharIn, Traits>& str
   )
{
    return {out, ft, str.data(), str.size()};
}

template <typename CharIn, typename Traits>
inline stringify::v0::string_with_formatting<CharIn>
stringify_fmt(const std::basic_string_view<CharIn, Traits>& str)
{
    return {str};
}

#endif //defined(BOOST_STRINGIFY_HAS_STD_STRING_VIEW)

template <typename CharOut, typename FPack, typename CharIn>
inline stringify::v0::string_printer<CharIn, CharOut>
stringify_make_printer
    ( stringify::v0::output_writer<CharOut>& out
    , const FPack& ft
    , const stringify::v0::string_with_formatting<CharIn>& ch
    )
{
    return {out, ft, ch};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR_PTR */

