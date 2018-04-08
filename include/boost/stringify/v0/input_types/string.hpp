#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_STRING
#define BOOST_STRINGIFY_V0_INPUT_TYPES_STRING

#include <algorithm>
#include <limits>
#include <boost/stringify/v0/ftuple.hpp>
#include <boost/stringify/v0/facets/width_calculator.hpp>

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

    using child_type = typename std::conditional
        < std::is_same<T, void>::value
        , string_formatting<CharIn, void>
        , T
        > :: type;

public:

    template <typename, typename>
    friend class string_formatting;

    template <typename U>
    using other = stringify::v0::string_formatting<CharIn, U>;

    constexpr string_formatting() = default;

    constexpr string_formatting(const string_formatting&) = default;

    template <typename U>
    constexpr child_type& format_as(const string_formatting<CharIn, U>& other) &
    {
        m_sani = other.m_sani;
        return align_formatting<T>::format_as(other);
    }

    template <typename U>
    constexpr child_type&& format_as(const string_formatting<CharIn, U>& other) &&
    {
        return static_cast<child_type&&>(format_as(other));
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
    constexpr child_type& encoding(stringify::v0::encoding_id<CharIn> eid) &
    {
        m_encoding_info = *eid.info();
        return *this;
    }
    constexpr child_type&& encoding(stringify::v0::encoding_id<CharIn> eid) &&
    {
        m_encoding_info = *eid.info();
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
    stringify::v0::encoding_id<CharIn> encoding() const
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

    template <typename T>
    using other = stringify::v0::string_formatting<CharIn, T>;

    using fmt_type = other<string_with_formatting>;

    constexpr string_with_formatting
        ( const CharIn* begin
        , const CharIn* end
        ) noexcept
        : m_str(begin)
        , m_end(end)
    {
    }

    constexpr string_with_formatting
        ( const CharIn* str
        , std::size_t len
        ) noexcept
        : m_str(str)
        , m_end(str + len)
    {
    }

    constexpr string_with_formatting
        ( const CharIn* str
        , std::size_t len
        , const fmt_type& fmt
        ) noexcept
        : fmt_type(fmt)
        , m_str(str)
        , m_end(str + len)
    {
    }

    constexpr string_with_formatting
        ( const CharIn* str
        ) noexcept
        : m_str(str)
        , m_end(str + std::char_traits<CharIn>::length(str))
    {
    }

    constexpr string_with_formatting(const string_with_formatting&) = default;

    void value(const CharIn* str)
    {
        m_str = str;
        m_end = std::char_traits<CharIn>::length(str);
    }

    template <class StringType>
    void value(const StringType& str)
    {
        m_str = &str[0];
        m_end = m_str + str.length();
    }

    constexpr const CharIn* begin() const
    {
        return m_str;
    }
    constexpr const CharIn* end() const
    {
        return m_end;
    }

    void operator%(int) const = delete;

private:

    const CharIn* m_str;
    const CharIn* m_end;
};

template<typename CharIn, typename CharOut>
class simple_string_printer: public stringify::v0::printer<CharOut>
{
    using input_tag = stringify::v0::string_input_tag<CharIn>;
    using writer_type = stringify::v0::output_writer<CharOut>;

public:

    template <typename FTuple>
    simple_string_printer
        ( stringify::v0::output_writer<CharOut>& out
        , const FTuple& ft
        , const CharIn* str
        , std::size_t len
        ) noexcept
        : simple_string_printer
            ( out
            , str
            , str + len
            , get_facet<stringify::v0::input_encoding_category<CharIn>>(ft)
            , get_facet<stringify::v0::width_calculator_category>(ft)
            )
    {
    }

    simple_string_printer
        ( stringify::v0::output_writer<CharOut>& out
        , const CharIn* begin
        , const CharIn* end
        , const stringify::v0::input_encoding<CharIn> input_enc
          , const stringify::v0::width_calculator& wcalc
        ) noexcept
        : m_begin(begin)
        , m_end(end)
        , m_sw(out, input_enc.id, false)
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

    template <typename Category, typename FTuple>
    const auto& get_facet(const FTuple& ft) const
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
        , m_sw.keep_surrogates() );
}



template<typename CharIn, typename CharOut>
class string_printer: public printer<CharOut>
{
private:

    using input_tag = stringify::v0::string_input_tag<CharIn>;
    using writer_type = stringify::v0::output_writer<CharOut>;

public:

    template <typename FTuple>
    string_printer
        ( stringify::v0::output_writer<CharOut>& out
        , const FTuple& ft
        , const stringify::v0::string_with_formatting<CharIn>& input
        ) noexcept
        : string_printer
            ( out
            , input
            , get_facet<stringify::v0::input_encoding_category<CharIn>>(ft)
            , get_facet<stringify::v0::width_calculator_category>(ft)
            )
    {
    }

    string_printer
        ( stringify::v0::output_writer<CharOut>& out
        , const stringify::v0::string_with_formatting<CharIn>& input
        , const stringify::v0::input_encoding<CharIn> input_enc
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

    template <typename Category, typename FTuple>
    const auto& get_facet(const FTuple& ft) const
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
    , const stringify::v0::input_encoding<CharIn> input_enc
    , const stringify::v0::width_calculator& wcalc
    ) noexcept
    : m_fmt(input)
    , m_sw
        ( out
        , input.has_encoding() ? input.encoding() : input_enc.id
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
            , out.keep_surrogates() )
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
            , m_sw.keep_surrogates() );
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
inline stringify::v0::simple_string_printer<CharIn, CharOut>
stringify_make_printer
   ( stringify::v0::output_writer<CharOut>& out
   , const FTuple& ft
   , const std::basic_string<CharIn, Traits, Allocator>& str
   )
{
    return {out, ft, str.data(), str.size()};
}

template <typename CharOut, typename FTuple>
inline stringify::v0::simple_string_printer<char, CharOut>
stringify_make_printer
   ( stringify::v0::output_writer<CharOut>& out
   , const FTuple& ft
   , const char* str
   )
{
    return {out, ft, str, std::char_traits<char>::length(str)};
}

template <typename CharOut, typename FTuple>
inline stringify::v0::simple_string_printer<wchar_t, CharOut>
stringify_make_printer
   ( stringify::v0::output_writer<CharOut>& out
   , const FTuple& ft
   , const wchar_t* str
   )
{
    return {out, ft, str, std::char_traits<wchar_t>::length(str)};
}

template <typename CharOut, typename FTuple>
inline stringify::v0::simple_string_printer<char16_t, CharOut>
stringify_make_printer
   ( stringify::v0::output_writer<CharOut>& out
   , const FTuple& ft
   , const char16_t* str
   )
{
    return {out, ft, str, std::char_traits<char16_t>::length(str)};
}

template <typename CharOut, typename FTuple>
inline stringify::v0::simple_string_printer<char32_t, CharOut>
stringify_make_printer
   ( stringify::v0::output_writer<CharOut>& out
   , const FTuple& ft
   , const char32_t* str
   )
{
    return {out, ft, str, std::char_traits<char32_t>::length(str)};
}

template <typename CharIn, typename Traits>
inline stringify::v0::string_with_formatting<CharIn>
stringify_fmt(const std::basic_string<CharIn, Traits>& str)
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
inline stringify::v0::simple_string_printer<CharIn, CharOut>
stringify_make_printer
   ( stringify::v0::output_writer<CharOut>& out
   , const FTuple& ft
   , const std::basic_string_view<CharIn, Traits>& str
   )
{
    return {out, ft, str.data(), str.size()};
}

template <typename CharIn, typename Traits>
inline stringify::v0::string_with_formatting<CharIn>
stringify_fmt(const std::basic_string_view<CharIn, Traits>& str)
{
    return {str.data(), str.size()};
}

#endif

template <typename CharOut, typename FTuple, typename CharIn>
inline stringify::v0::string_printer<CharIn, CharOut>
stringify_make_printer
    ( stringify::v0::output_writer<CharOut>& out
    , const FTuple& ft
    , const stringify::v0::string_with_formatting<CharIn>& ch
    )
{
    return {out, ft, ch};
}

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR_PTR */

