#ifndef BOOST_STRINGIFY_V0_DETAIL_CHAR_FLAGS_HPP
#define BOOST_STRINGIFY_V0_DETAIL_CHAR_FLAGS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/config.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename WidthType, WidthType DefaultWidth, typename StringFmt>
class basic_arg_fmt: public StringFmt
{
public:

    using width_type = WidthType;
    constexpr static WidthType default_width = DefaultWidth;

    constexpr basic_arg_fmt() = default;

    constexpr basic_arg_fmt
        ( const basic_arg_fmt&
        ) = default;

    constexpr basic_arg_fmt
        ( width_type width_
        , char32_t fill_ = U' '
        )
        : m_fill(fill_)
        , m_width(width_)
    {
    }

    constexpr basic_arg_fmt
        ( width_type width_
        , char32_t fill_
        , const char* flags
        )
        : StringFmt(flags)
        , m_fill(fill_)
        , m_width(width_)
    {
    }

    constexpr basic_arg_fmt
        ( width_type width_
        , const char* flags
        )
        : StringFmt(flags)
        , m_width(width_)
    {
    }

    constexpr basic_arg_fmt(const char* flags)
        : StringFmt(flags)
    {
    }

    constexpr char32_t fill() const
    {
        return m_fill;
    }

    constexpr width_type width() const
    {
        return m_width;
    }

    constexpr void fill(char32_t f)
    {
        m_fill = f;
    }

    constexpr void width(width_type w)
    {
        m_width = w;
    }

private:

    char32_t   m_fill  = U' ';
    width_type m_width = default_width;
};


template
    < typename WidthType
    , WidthType DefaultWidth
    , typename CountType
    , CountType DefaultCount
    , typename StringFmt
    >
class basic_arg_fmt_with_count: public StringFmt
{

public:

    using width_type = WidthType;
    using count_type = CountType;
    constexpr static WidthType default_width = DefaultWidth;
    constexpr static CountType default_count = DefaultCount;

    constexpr basic_arg_fmt_with_count() = default;

    constexpr basic_arg_fmt_with_count
        ( const basic_arg_fmt_with_count&
        ) = default;


    constexpr basic_arg_fmt_with_count
        ( width_type width_
        , char32_t fill_ = U' '
        , count_type count_ = default_count
        )
        : m_fill(fill_)
        , m_width(width_)
        , m_count(count_)
    {
    }

    constexpr basic_arg_fmt_with_count
        ( width_type width_
        , char32_t fill_
        , const char* flags
        , count_type count_ = default_count
        )
        : StringFmt(flags)
        , m_fill(fill_)
        , m_width(width_)
        , m_count(count_)
    {
    }

    constexpr basic_arg_fmt_with_count
        ( width_type width_
        , char fill_
        , const char* flags
        , count_type count_ = default_count
        )
        : StringFmt(flags)
        , m_fill(fill_)
        , m_width(width_)
        , m_count(count_)
    {
    }

    constexpr basic_arg_fmt_with_count
        ( width_type width_
        , const char* flags
        , count_type count_ = default_count
        )
        : StringFmt(flags)
        , m_width(width_)
        , m_count(count_)
    {
    }

    constexpr basic_arg_fmt_with_count
        ( const char* flags
        , count_type count_ = default_count
        )
        : StringFmt(flags)
        , m_count(count_)
    {
    }

    constexpr basic_arg_fmt_with_count
        ( width_type width_
        , count_type count_
        )
        : m_width(width_)
        , m_count(count_)
    {
    }

    constexpr char32_t fill() const
    {
        return m_fill;
    }

    constexpr width_type width() const
    {
        return m_width;
    }

    constexpr count_type count() const
    {
        return m_count;
    }

    constexpr void fill(char32_t f)
    {
        m_fill = f;
    }

    constexpr void width(width_type w)
    {
        m_width = w;
    }

    constexpr void count(count_type c)
    {
        m_count = c;
    }

private:

    char32_t   m_fill = U' ';
    width_type m_width = default_width;
    count_type m_count = default_count;
};


enum class basic_alignment {left = '<', right = '>', center = '^'};

class alignment_str
{
public:

    constexpr alignment_str() = default;

    constexpr alignment_str(const alignment_str&) = default;

    alignment_str(const char* it);

    constexpr auto alignment() const
    {
        return m_alignment;
    }

private:

    boost::stringify::v0::basic_alignment m_alignment =
        boost::stringify::v0::basic_alignment::right;
};

#if defined(BOOST_STRINGIFY_SOURCE) || ! defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

BOOST_STRINGIFY_INLINE alignment_str::alignment_str(const char* it)
{
    for(;*it != '\0'; ++it)
    {
        switch(*it)
        {
            case '<':
            case '^':
            case '>':
                m_alignment = static_cast<basic_alignment>(*it);
                break;
        }
    }
}

#endif

enum class int_alignment {left = '<', right = '>', internal = '=' , center = '^'};

class int_format_str
{
public:

    constexpr int_format_str() = default;

    constexpr int_format_str(const int_format_str&) = default;

    int_format_str(const char*);

    constexpr bool showpos() const
    {
        return m_showpos;
    }

    constexpr bool showbase() const
    {
        return m_showbase;
    }

    constexpr bool uppercase() const
    {
        return m_uppercase;
    }

    constexpr auto alignment() const
    {
        return m_alignment;
    }

    constexpr auto base() const
    {
        return m_base;
    }

private:

    unsigned m_base = 10;
    boost::stringify::v0::int_alignment m_alignment =
        boost::stringify::v0::int_alignment::right;
    bool m_showpos = false;
    bool m_showbase = false;
    bool m_uppercase = false;
};


#if defined(BOOST_STRINGIFY_SOURCE) || ! defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)
BOOST_STRINGIFY_INLINE
int_format_str::int_format_str(const char* it)
{
    for(;*it != '\0'; ++it)
    {
        switch(*it)
        {
            case '<':
            case '=':
            case '^':
            case '>':
                m_alignment = static_cast<int_alignment>(*it);
                break;
            case '+':
                m_showpos = true;
                break;
            case '#':
                m_showbase = true;
                break;
            case 'X':
                m_base = 16;
                m_uppercase = true;
                break;
            case 'x':
                m_base = 16;
                break;
            case 'o':
                m_base = 8;
        }
    }
}

#endif // defined(BOOST_STRINGIFY_SOURCE) || ! defined(BOOST_STRINGIFY_NOT_HEADER_ONLY)

using int_fmt = stringify::v0::basic_arg_fmt
    <int, 0, stringify::v0::int_format_str>;

using char_fmt = stringify::v0::basic_arg_fmt_with_count
    <int, 0, int, 1, stringify::v0::alignment_str>;

using string_fmt = stringify::v0::basic_arg_fmt
    <int, 0, stringify::v0::alignment_str>;


inline const stringify::v0::int_fmt& default_int_fmt()
{
    static const int_fmt f {};
    return f;
}
inline const stringify::v0::char_fmt& default_char_fmt()
{
    static const char_fmt f {};
    return f;
}
inline const stringify::v0::string_fmt& default_string_fmt()
{
    static const stringify::v0::string_fmt f {};
    return f;
}


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_DETAIL_CHAR_FLAGS_HPP */

