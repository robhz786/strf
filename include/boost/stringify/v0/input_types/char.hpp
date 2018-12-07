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

struct char_formatting
{
    template <class T>
    class fn
    {
    public:

        constexpr fn() = default;

        constexpr fn(const fn&) = default;

        template <typename U>
        constexpr fn(const fn<U>& u)
            : m_count(u.count())
        {
        }

        constexpr T&& multi(int count) &&
        {
            m_count = count;
            return static_cast<T&&>(*this);
        }
        constexpr T& multi(int count) &
        {
            m_count = count;
            return static_cast<T&>(*this);
        }
        constexpr int count() const
        {
            return m_count;
        }

    private:

        int m_count = 1;
    };
};

template <typename CharT>
using char_with_format = stringify::v0::value_with_format
    < CharT
    , stringify::v0::char_formatting
    , stringify::v0::alignment_format >;

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
        , const stringify::v0::char_with_format<char32_t>& input
        ) noexcept
        : char32_printer(out, input, get_width_calculator(fp))
    {
    }

    char32_printer
        ( stringify::v0::output_writer<CharT>& out
        , const stringify::v0::char_with_format<char32_t>& input
        , const stringify::v0::width_calculator& wcalc
        ) noexcept;

    virtual ~char32_printer();

    std::size_t necessary_size() const override;

    void write() const override;

    int remaining_width(int w) const override;

private:

    stringify::v0::output_writer<CharT>& m_out;
    stringify::v0::char_with_format<char32_t> m_fmt;
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
    , const stringify::v0::char_with_format<char32_t>& input
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
        , const stringify::v0::char_with_format<CharT>& input
        ) noexcept
        : char_printer(out, input, get_width_calculator(fp))
    {
    }

    char_printer
        ( stringify::v0::output_writer<CharT>& out
        , const stringify::v0::char_with_format<CharT>& input
        , const stringify::v0::width_calculator& wcalc
        ) noexcept;

    virtual ~char_printer();

    std::size_t necessary_size() const override;

    void write() const override;

    int remaining_width(int w) const override;

private:

    stringify::v0::output_writer<CharT>& m_out;
    stringify::v0::char_with_format<CharT> m_fmt;
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
    , const stringify::v0::char_with_format<CharT>& input
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

template
    < typename CharOut
    , typename FPack
    , typename = typename std::enable_if<!std::is_same<CharOut, char32_t>::value>::type
    >
inline stringify::v0::char32_printer<CharOut>
make_printer
    ( stringify::v0::output_writer<CharOut>& out
    , const FPack& fp
    , const stringify::v0::char_with_format<char32_t>& ch )
{
    return {out, fp, ch};
}

template <typename CharOut, typename FPack>
inline stringify::v0::char_printer<CharOut>
make_printer
    ( stringify::v0::output_writer<CharOut>& out
    , const FPack& fp
    , char ch )
{
    static_assert( std::is_same<CharOut, char>::value
                 , "encoding convertion for single char not supported yet" );
    return {out, fp, stringify::v0::char_with_format<char>{ch}};
}

template <typename CharOut, typename FPack>
inline stringify::v0::char_printer<CharOut>
make_printer
    ( stringify::v0::output_writer<CharOut>& out
    , const FPack& fp
    , wchar_t ch )
{
    static_assert( std::is_same<CharOut, wchar_t>::value
                 , "encoding convertion for single char not supported yet" );
    return {out, fp, stringify::v0::char_with_format<wchar_t>{ch}};
}

template <typename CharOut, typename FPack>
inline stringify::v0::char_printer<CharOut>
make_printer
    ( stringify::v0::output_writer<CharOut>& out
    , const FPack& fp
    , char16_t ch )
{
    static_assert( std::is_same<CharOut, char16_t>::value
                 , "encoding convertion for single char not supported yet" );
    return {out, fp, stringify::v0::char_with_format<char16_t>{ch}};
}

template< typename CharOut, typename FPack >
std::conditional_t
    < std::is_same<CharOut, char32_t>::value
    , stringify::v0::char_printer<char32_t>
    , stringify::v0::char32_printer<CharOut> >
make_printer
    ( stringify::v0::output_writer<CharOut>& out
    , const FPack& fp
    , char32_t ch
    )
{
    return {out, fp, stringify::v0::char_with_format<char32_t>{ch}};
}

template <typename CharOut, typename FPack>
inline stringify::v0::char_printer<CharOut>
make_printer
    ( stringify::v0::output_writer<CharOut>& out
    , const FPack& fp
    , const stringify::v0::char_with_format<CharOut>& ch )
{
    return {out, fp, ch};
}

inline auto make_fmt(stringify::v0::tag, char ch)
{
    return stringify::v0::char_with_format<char>{ch};
}
inline auto make_fmt(stringify::v0::tag, wchar_t ch)
{
    return stringify::v0::char_with_format<wchar_t>{ch};
}
inline auto make_fmt(stringify::v0::tag, char16_t ch)
{
    return stringify::v0::char_with_format<char16_t>{ch};
}
inline auto make_fmt(stringify::v0::tag, char32_t ch)
{
    return stringify::v0::char_with_format<char32_t>{ch};
}

template <typename> struct is_char: public std::false_type {};
template <> struct is_char<char>: public std::true_type {};
template <> struct is_char<char16_t>: public std::true_type {};
template <> struct is_char<char32_t>: public std::true_type {};
template <> struct is_char<wchar_t>: public std::true_type {};

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif // BOOST_STRINGIFY_V0_INPUT_TYPES_CHAR_HPP_INCLUDED



