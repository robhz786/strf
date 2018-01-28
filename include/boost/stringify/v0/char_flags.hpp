#ifndef BOOST_STRINGIFY_V0_DETAIL_CHAR_FLAGS_HPP
#define BOOST_STRINGIFY_V0_DETAIL_CHAR_FLAGS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/config.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <char ...>
class char_flags;

template <>
class char_flags<>
{

public:

    constexpr char_flags(const char*) : m_bits(0)
    {
    }

    constexpr char_flags() : m_bits(0)
    {
    }

    template <char Ch>
    constexpr bool has_char() const
    {
        return false;
    }

    constexpr bool has_char(char) const
    {
        return false;
    }

    constexpr char_flags(const char_flags&) = default;

    char_flags& operator=(const char_flags&) = default;

protected:

    template <char Ch>
    constexpr static int mask()
    {
        return 0;
    }
    
    constexpr static int mask(char)
    {
        return 0;
    }

    int m_bits;
};


template <char Char, char ... OtherChars>
class char_flags<Char, OtherChars ...> : private char_flags<OtherChars ...>
{

    typedef char_flags<OtherChars ...> parent;

    static_assert(sizeof...(OtherChars) <= 8 * sizeof(int), "too many chars");

public:

    constexpr char_flags()
    {
    }

    constexpr char_flags(const char_flags& other) = default;

    char_flags& operator=(const char_flags& other) = default;

    constexpr char_flags(const char* str)
    {
        for (std::size_t i = 0; str[i] != '\0'; ++i)
        {
            this->m_bits |= mask(str[i]);
        }
    }

    constexpr bool has_char(char ch) const
    {
        return 0 != (this->m_bits & mask(ch));
    }

    template <char Ch>
    constexpr bool has_char() const
    {
        return 0 != (this->m_bits & mask<Ch>());
    }
    
protected:

    using parent::m_bits;

    constexpr static int mask(char ch)
    {
        return ch == Char ? this_mask() : parent::mask(ch);
    }

    template <char Ch>
    constexpr static int mask()
    {
        return Ch == Char ? this_mask() : parent::template mask<Ch>();
    }
    
    constexpr static int this_mask()
    {
        return 1 << sizeof...(OtherChars);
    }

    constexpr static bool has_char(const char* str, char ch)
    {
        return *str != 0 && (*str == ch || has_char(str + 1, ch));
    }
};


template
    < typename WidthType
    , WidthType DefaultWidth
    , char ... Flags
    >
class basic_arg_fmt: public stringify::v0::char_flags<Flags ...>
{
public:

    using width_type = WidthType;
    constexpr static WidthType default_width = DefaultWidth;

    using char_flags_type = stringify::v0::char_flags<Flags ...>;

    using char_flags_type::has_char;

    constexpr basic_arg_fmt() = default;

    constexpr basic_arg_fmt
        ( const basic_arg_fmt&
        ) = default;

    constexpr basic_arg_fmt
        ( width_type width_
        , char32_t fill_ = U' '
        , const char* flags = ""
        )
        : char_flags_type(flags)
        , m_fill(fill_)
        , m_width(width_)
    {
    }

    constexpr basic_arg_fmt
        ( char32_t fill_
        , width_type width_
        , const char* flags = ""
        )
        : char_flags_type(flags)
        , m_fill(fill_)
        , m_width(width_)
    {
    }

    constexpr basic_arg_fmt
        ( width_type width_
        , char fill_
        , const char* flags = ""
        )
        : char_flags_type(flags)
        , m_fill(fill_)
        , m_width(width_)
    {
    }

    constexpr basic_arg_fmt
        ( char fill_
        , width_type width_
        , const char* flags = ""
        )
        : char_flags_type(flags)
        , m_fill(fill_)
        , m_width(width_)
    {
    }

    constexpr basic_arg_fmt
        ( width_type width_
        , const char* flags
        )
        : char_flags_type(flags)
        , m_width(width_)
    {
    }

    constexpr basic_arg_fmt(const char* flags)
        : char_flags_type(flags)
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
    , char ... Flags
    >
class basic_arg_fmt_with_count: public stringify::v0::char_flags<Flags ...>
{

public:

    using width_type = WidthType;
    using count_type = CountType;
    constexpr static WidthType default_width = DefaultWidth;
    constexpr static CountType default_count = DefaultCount;

    using char_flags_type = stringify::v0::char_flags<Flags ...>;

    using char_flags_type::has_char;

    constexpr basic_arg_fmt_with_count() = default;

    constexpr basic_arg_fmt_with_count
        ( const basic_arg_fmt_with_count&
        ) = default;

/*
todo: more constructors
rule:
- in order to pass count, one must also pass width or flags ( or both )
- count is aways the last argument
- fill can be char or char32_t
 */


    constexpr basic_arg_fmt_with_count
        ( width_type width_
        , char32_t fill_ = U' '
        , const char* flags = ""
        , count_type count_ = default_count
        )
        : char_flags_type(flags)
        , m_fill(fill_)
        , m_width(width_)
        , m_count(count_)
    {
    }

    constexpr basic_arg_fmt_with_count
        ( width_type width_
        , char fill_
        , const char* flags = ""
        , count_type count_ = default_count
        )
        : char_flags_type(flags)
        , m_fill(fill_)
        , m_width(width_)
        , m_count(count_)
    {
    }

    constexpr basic_arg_fmt_with_count
        ( char32_t fill_
        , width_type width_
        , const char* flags = ""
        , count_type count_ = default_count
        )
        : char_flags_type(flags)
        , m_fill(fill_)
        , m_width(width_)
        , m_count(count_)
    {
    }

    constexpr basic_arg_fmt_with_count
        ( char fill_
        , width_type width_
        , const char* flags = ""
        , count_type count_ = default_count
        )
        : char_flags_type(flags)
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
        : char_flags_type(flags)
        , m_width(width_)
        , m_count(count_)
    {
    }

    constexpr basic_arg_fmt_with_count
        ( const char* flags
        , count_type count_ = default_count
        )
        : char_flags_type(flags)
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





BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_DETAIL_CHAR_FLAGS_HPP */

