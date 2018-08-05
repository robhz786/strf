#ifndef BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR_HPP_INCLUDED
#define BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR_HPP_INCLUDED

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>
#include <boost/stringify/v0/basic_types.hpp>
#include <boost/stringify/v0/facets_pack.hpp>
#include <boost/stringify/v0/facets/width_calculator.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <class T>
class char_formatting: public stringify::v0::align_formatting<T>
{

    using child_type = T;

public:

    template <typename U>
    using fmt_other = stringify::v0::char_formatting<U>;

    constexpr char_formatting() = default;

    constexpr char_formatting(const char_formatting&) = default;

    template <typename U>
    constexpr char_formatting(const char_formatting<U>& u)
        : stringify::v0::align_formatting<T>(u)
        , m_count(u.count())
    {
    }
    
    ~char_formatting() = default;

    void operator%(int) const = delete;
    
    constexpr child_type&& multi(int count) &&
    {
        m_count = count;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& multi(int count) &
    {
        m_count = count;
        return static_cast<child_type&>(*this);
    }
    constexpr int count() const
    {
        return m_count;
    }

private:

    int m_count = 1;
};


template <typename CharT>
class char_with_formatting
    : public stringify::v0::char_formatting<char_with_formatting<CharT> >
{
public:

    constexpr char_with_formatting(CharT value)
        : m_value(value)
    {
    }

    template <typename U>
    constexpr char_with_formatting(CharT value, const char_formatting<U>& u)
        : stringify::v0::char_formatting<char_with_formatting>(u)
        , m_value(value)
    {
    }

    constexpr char_with_formatting(const char_with_formatting&) = default;

    constexpr CharT value() const
    {
        return m_value;
    }

private:

    CharT m_value = 0;
};

template <typename CharT>
class char32_printer: public printer<CharT>
{
    using input_type = char32_t;
    using writer_type = stringify::v0::output_writer<CharT>;

public:

    template <typename FPack>
    char32_printer
        ( stringify::v0::output_writer<CharT>& out
        , const FPack& fp
        , const stringify::v0::char_with_formatting<char32_t>& input
        ) noexcept
        : char32_printer(out, input, get_width_calculator(fp))
    {
    }

    char32_printer
        ( stringify::v0::output_writer<CharT>& out
        , const stringify::v0::char_with_formatting<char32_t>& input
        , const stringify::v0::width_calculator& wcalc
        ) noexcept;

    virtual ~char32_printer();

    std::size_t necessary_size() const override;

    void write() const override;

    int remaining_width(int w) const override;

private:

    stringify::v0::output_writer<CharT>& m_out;
    stringify::v0::char_with_formatting<char32_t> m_fmt;
    int m_fillcount = 0;

    template <typename FPack>
    static const auto& get_out_encoding(const FPack& fp)
    {
        using category = stringify::v0::encoding_category<CharT>;
        return fp.template get_facet<category, input_type>();
    }

    template <typename FPack>
    static const auto& get_width_calculator(const FPack& fp)
    {
        using category = stringify::v0::width_calculator_category;
        return fp.template get_facet<category, input_type>();
    }

    std::size_t necessary_size(char32_t ch) const
    {
        return m_out.necessary_size(ch);
    }

    void determinate_fill_and_width(const stringify::v0::width_calculator& wcalc)
    {
        int content_width = 0;
        if(m_fmt.width() < 0)
        {
            m_fmt.width(0);
        }
        if (m_fmt.count() > 0)
        {
            content_width = m_fmt.count() * wcalc.width_of(m_fmt.value());
        }
        if (content_width >= m_fmt.width())
        {
            m_fillcount = 0;
            m_fmt.width(content_width);
        }
        else
        {
            m_fillcount = m_fmt.width() - content_width;
        }
    }
};


template <typename CharT>
char32_printer<CharT>::char32_printer
    ( stringify::v0::output_writer<CharT>& out
    , const stringify::v0::char_with_formatting<char32_t>& input
    , const stringify::v0::width_calculator& wcalc
    ) noexcept
    : m_out(out)
    , m_fmt(input)
{
    determinate_fill_and_width(wcalc);
}


template <typename CharT>
char32_printer<CharT>::~char32_printer()
{
}


template <typename CharT>
std::size_t char32_printer<CharT>::necessary_size() const
{
    std::size_t len = 0;
    if (m_fmt.count() > 0)
    {
        len = m_fmt.count() * necessary_size(m_fmt.value());
    }
    if (m_fillcount > 0)
    {
        len += m_fillcount * necessary_size(m_fmt.fill());
    }
    return len;
}


template <typename CharT>
void char32_printer<CharT>::write() const
{
    if (m_fillcount == 0)
    {
        m_out.put32(m_fmt.count(), m_fmt.value());
    }
    else
    {
        switch(m_fmt.alignment())
        {
            case stringify::v0::alignment::left:
            {
                m_out.put32(m_fmt.count(), m_fmt.value());
                m_out.put32(m_fillcount, m_fmt.fill());
                break;
            }
            case stringify::v0::alignment::center:
            {
                auto halfcount = m_fillcount / 2;
                m_out.put32(halfcount, m_fmt.fill());
                m_out.put32(m_fmt.count(), m_fmt.value());
                m_out.put32(m_fillcount - halfcount, m_fmt.fill());
                break;
            }
            default:
            {
                m_out.put32(m_fillcount, m_fmt.fill());
                m_out.put32(m_fmt.count(), m_fmt.value());
            }
        }
    }
}


template <typename CharT>
int char32_printer<CharT>::remaining_width(int w) const
{
    if (w > m_fmt.width())
    {
        return w - m_fmt.width();
    }
    return 0;
}

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char32_printer<wchar_t>;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

template <typename CharT>
class char_printer: public printer<CharT>
{
    using input_type = CharT;
    using writer_type = stringify::v0::output_writer<CharT>;

public:

    template <typename FPack>
    char_printer
        ( stringify::v0::output_writer<CharT>& out
        , const FPack& fp
        , const stringify::v0::char_with_formatting<CharT>& input
        ) noexcept
        : char_printer(out, input, get_width_calculator(fp))
    {
    }

    char_printer
        ( stringify::v0::output_writer<CharT>& out
        , const stringify::v0::char_with_formatting<CharT>& input
        , const stringify::v0::width_calculator& wcalc
        ) noexcept;

    virtual ~char_printer();

    std::size_t necessary_size() const override;

    void write() const override;

    int remaining_width(int w) const override;

private:

    stringify::v0::output_writer<CharT>& m_out;
    stringify::v0::char_with_formatting<CharT> m_fmt;
    int m_fillcount = 0;

    template <typename FPack>
    static const auto& get_out_encoding(const FPack& fp)
    {
        using category = stringify::v0::encoding_category<CharT>;
        return fp.template get_facet<category, input_type>();
    }

    template <typename FPack>
    static const auto& get_width_calculator(const FPack& fp)
    {
        using category = stringify::v0::width_calculator_category;
        return fp.template get_facet<category, input_type>();
    }

    std::size_t necessary_size(char32_t ch) const
    {
        return m_out.necessary_size(ch);
    }

    void determinate_fill_and_width(const stringify::v0::width_calculator& wcalc)
    {
        int content_width = 0;
        if(m_fmt.width() < 0)
        {
            m_fmt.width(0);
        }
        if (m_fmt.count() > 0 )
        {
            char32_t ch32 = m_fmt.value(); // todo: use convertion_to_utf32 facet ?
            content_width = m_fmt.count() * wcalc.width_of(ch32);
        }
        if (content_width >= m_fmt.width())
        {
            m_fillcount = 0;
            m_fmt.width(content_width);
        }
        else
        {
            m_fillcount = m_fmt.width() - content_width;
        }
    }
};


template <typename CharT>
char_printer<CharT>::char_printer
    ( stringify::v0::output_writer<CharT>& out
    , const stringify::v0::char_with_formatting<CharT>& input
    , const stringify::v0::width_calculator& wcalc
    ) noexcept
    : m_out(out)
    , m_fmt(input)
{
    determinate_fill_and_width(wcalc);
}

template <typename CharT>
char_printer<CharT>::~char_printer()
{
}


template <typename CharT>
std::size_t char_printer<CharT>::necessary_size() const
{
    std::size_t len = m_fmt.count();
    if (m_fillcount > 0)
    {
        len += m_fillcount * necessary_size(m_fmt.fill());
    }
    return len;
}


template <typename CharT>
void char_printer<CharT>::write() const
{
    if (m_fillcount == 0)
    {
        m_out.put(m_fmt.count(), m_fmt.value());
    }
    else
    {
        switch(m_fmt.alignment())
        {
            case stringify::v0::alignment::left:
            {
                m_out.put(m_fmt.count(), m_fmt.value());
                m_out.put32(m_fillcount, m_fmt.fill());
                break;
            }
            case stringify::v0::alignment::center:
            {
                auto halfcount = m_fillcount / 2;
                m_out.put32(halfcount, m_fmt.fill());
                m_out.put(m_fmt.count(), m_fmt.value());
                m_out.put32(m_fillcount - halfcount, m_fmt.fill());
                break;
            }
            default:
            {
                m_out.put32(m_fillcount, m_fmt.fill());
                m_out.put(m_fmt.count(), m_fmt.value());
            }
        }
    }
}


template <typename CharT>
int char_printer<CharT>::remaining_width(int w) const
{
    if (w > m_fmt.width())
    {
        return w - m_fmt.width();
    }
    return 0;
}

#if defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_printer<char>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_printer<char16_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_printer<char32_t>;
BOOST_STRINGIFY_EXPLICIT_TEMPLATE class char_printer<wchar_t>;

#endif // defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

// namespace detail {

// template <typename CharOut, typename CharIn> struct char_input_traits_helper;

// template <typename CharT> struct char_input_traits_helper<CharT, CharT>
// {
//     using type = stringify::v0::char_printer<CharT>;
// };
// template <> struct char_input_traits_helper<char, char32_t>
// {
//     using type = stringify::v0::char32_printer<char>;
// };
// template <> struct char_input_traits_helper<wchar_t, char32_t>
// {
//     using type = stringify::v0::char32_printer<wchar_t>;
// };
// template <> struct char_input_traits_helper<char16_t, char32_t>
// {
//     using type = stringify::v0::char32_printer<char16_t>;
// };

// struct char_input_traits
// {
//     template <typename CharOut, typename CharIn>
//     using printer_type
//     = typename stringify::v0::detail::char_input_traits_helper<CharOut, CharIn>::type;

//     template <typename CharOut, typename FPack, typename CharIn>
//     static inline printer_type<CharOut, CharIn> make_printer
//         ( const FPack& fp, CharIn ch )
//     {
//         return {fp, ch};
//     }
//     template <typename CharOut, typename FPack, typename CharIn>
//     static inline printer_type<CharOut, CharIn> make_printer
//         ( const FPack& fp
//         , const stringify::v0::char_with_formatting<CharIn>& ch
//         )
//     {
//         return {fp, ch};
//     }
//     template <typename CharIn>
//     static inline stringify::v0::char_with_formatting<CharIn> fmt(CharIn ch)
//     {
//         return {ch};
//     }
// };

// }

// stringify::v0::detail::char_input_traits stringify_get_input_traits(char);
// stringify::v0::detail::char_input_traits stringify_get_input_traits(wchar_t);
// stringify::v0::detail::char_input_traits stringify_get_input_traits(char16_t);
// stringify::v0::detail::char_input_traits stringify_get_input_traits(char32_t);
// stringify::v0::detail::char_input_traits stringify_get_input_traits( stringify::v0::char_with_formatting<char> );
// stringify::v0::detail::char_input_traits stringify_get_input_traits( stringify::v0::char_with_formatting<wchar_t> );
// stringify::v0::detail::char_input_traits stringify_get_input_traits( stringify::v0::char_with_formatting<char16_t> );
// stringify::v0::detail::char_input_traits stringify_get_input_traits( stringify::v0::char_with_formatting<char32_t> );

template
    < typename CharT
    , typename FPack
    , typename = typename std::enable_if<!std::is_same<CharT, char32_t>::value>::type
    >
inline stringify::v0::char32_printer<CharT>
make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& fp
    , char32_t ch
    )
{
    return {out, fp, ch};
}

template
    < typename CharT
    , typename FPack
    , typename = typename std::enable_if<!std::is_same<CharT, char32_t>::value>::type
    >
inline stringify::v0::char32_printer<CharT>
make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& fp
    , const stringify::v0::char_with_formatting<char32_t>& ch )
{
    return {out, fp, ch};
}

template <typename CharT, typename FPack>
inline stringify::v0::char_printer<CharT>
make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& fp
    , CharT ch )
{
    return {out, fp, ch};
}

template <typename CharT, typename FPack>
inline stringify::v0::char_printer<CharT>
make_printer
    ( stringify::v0::output_writer<CharT>& out
    , const FPack& fp
    , const stringify::v0::char_with_formatting<CharT>& ch )
{
    return {out, fp, ch};
}

inline stringify::v0::char_with_formatting<char> make_fmt(stringify::v0::tag, char ch)
{
    return {ch};
}
inline stringify::v0::char_with_formatting<wchar_t> make_fmt(stringify::v0::tag, wchar_t ch)
{
    return {ch};
}
inline stringify::v0::char_with_formatting<char16_t> make_fmt(stringify::v0::tag, char16_t ch)
{
    return {ch};
}
inline stringify::v0::char_with_formatting<char32_t> make_fmt(stringify::v0::tag, char32_t ch)
{
    return {ch};
}

template <typename> struct is_char: public std::false_type {};
template <> struct is_char<char>: public std::true_type {};
template <> struct is_char<char16_t>: public std::true_type {};
template <> struct is_char<char32_t>: public std::true_type {};
template <> struct is_char<wchar_t>: public std::true_type {};

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif // BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR_HPP_INCLUDED



