#ifndef BOOST_STRINGIFY_V0_ARG_FORMAT_HPP
#define BOOST_STRINGIFY_V0_ARG_FORMAT_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/config.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

enum class alignment {left, right, internal, center};

template <class T = void>
class int_format
{

    using child_type = typename std::conditional
        < std::is_same<T, void>::value
        , int_format<void>
        , T
        > :: type;

public:

    constexpr int_format() = default;

    constexpr int_format(const int_format&) = default;

    ~int_format() = default;

    constexpr child_type&& uphex() &&
    {
        m_base = 16;
        m_uppercase = true;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& uphex() &
    {
        m_base = 16;
        m_uppercase = true;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& hex() &&
    {
        m_base = 16;
        m_uppercase = false;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& hex() &
    {
        m_base = 16;
        m_uppercase = false;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& dec() &&
    {
        m_base = 10;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& dec() &
    {
        m_base = 10;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& oct() &&
    {
        m_base = 8;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& oct() &
    {
        m_base = 8;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& operator+() &&
    {
        m_showpos = true;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& operator+() &
    {
        m_showpos = true;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type&& operator~() &&
    {
        m_showbase = true;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& operator~() &
    {
        m_showbase = true;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& operator<(int width) &&
    {
        m_alignment = stringify::v0::alignment::left;
        m_width = width;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& operator<(int width) &
    {
        m_alignment = stringify::v0::alignment::left;
        m_width = width;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& operator>(int width) &&
    {
        m_alignment = stringify::v0::alignment::right;
        m_width = width;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& operator>(int width) &
    {
        m_alignment = stringify::v0::alignment::right;
        m_width = width;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& operator^(int width) &&
    {
        m_alignment = stringify::v0::alignment::center;
        m_width = width;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& operator^(int width) &
    {
        m_alignment = stringify::v0::alignment::center;
        m_width = width;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& operator%(int width) &&
    {
        m_alignment = stringify::v0::alignment::internal;
        m_width = width;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& operator%(int width) &
    {
        m_alignment = stringify::v0::alignment::internal;
        m_width = width;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& fill(char32_t ch) &&
    {
        m_fill = ch;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& fill(char32_t ch) &
    {
        m_fill = ch;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& width(int w) &&
    {
        m_width = w;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& width(int w) &
    {
        m_width = w;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& uppercase(bool u) &&
    {
        m_uppercase = u;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& uppercase(bool u) &
    {
        m_uppercase = u;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& showbase(bool s) &&
    {
        m_showbase = s;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& showbase(bool s) &
    {
        m_showbase = s;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& showpos(bool s) &&
    {
        m_showpos = s;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& showpos(bool s) &
    {
        m_showpos = s;
        return static_cast<child_type&>(*this);
    }
    constexpr int width() const
    {
        return m_width;
    }
    constexpr stringify::v0::alignment alignment() const
    {
        return m_alignment;
    }
    constexpr char32_t fill() const
    {
        return m_fill;
    }
    constexpr unsigned base() const
    {
        return m_base;
    }
    constexpr bool showbase() const
    {
        return m_showbase;
    }
    constexpr bool showpos() const
    {
        return m_showpos;
    }
    constexpr bool uppercase() const
    {
        return m_uppercase;
    }

private:

    char32_t m_fill = U' ';
    int m_width = 0;
    stringify::v0::alignment m_alignment = stringify::v0::alignment::right;
    unsigned short m_base = 10;
    bool m_showbase = false;
    bool m_showpos = false;
    bool m_uppercase = false;
};



template <class T = void>
class string_format
{

    using child_type = typename std::conditional
        < std::is_same<T, void>::value
        , string_format<void>
        , T
        > :: type;

public:

    constexpr string_format() = default;

    constexpr string_format(const string_format&) = default;

    template <typename U>
    constexpr string_format(const string_format<U>& other)
        : m_fill(other.m_fill)
        , m_width(other.m_width)
        , m_alignment(other.m_alignment)
    {
    }

    ~string_format() = default;

    constexpr child_type&& operator<(int width) &&
    {
        m_alignment = stringify::v0::alignment::left;
        m_width = width;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& operator<(int width) &
    {
        m_alignment = stringify::v0::alignment::left;
        m_width = width;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& operator>(int width) &&
    {
        m_alignment = stringify::v0::alignment::right;
        m_width = width;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& operator>(int width) &
    {
        m_alignment = stringify::v0::alignment::right;
        m_width = width;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& operator^(int width) &&
    {
        m_alignment = stringify::v0::alignment::center;
        m_width = width;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& operator^(int width) &
    {
        m_alignment = stringify::v0::alignment::center;
        m_width = width;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& fill(char32_t ch) &&
    {
        m_fill = ch;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& fill(char32_t ch) &
    {
        m_fill = ch;
        return static_cast<child_type&>(*this);
    }
    constexpr child_type&& width(int w) &&
    {
        m_width = w;
        return static_cast<child_type&&>(*this);
    }
    constexpr child_type& width(int w) &
    {
        m_width = w;
        return static_cast<child_type&>(*this);
    }
    constexpr int width() const
    {
        return m_width;
    }
    constexpr stringify::v0::alignment alignment() const
    {
        return m_alignment;
    }
    constexpr char32_t fill() const
    {
        return m_fill;
    }

private:

    template <typename>
    friend class string_format;

    char32_t m_fill = U' ';
    int m_width = 0;
    stringify::v0::alignment m_alignment = stringify::v0::alignment::right;
};


template <class T = void>
class char_format: public stringify::v0::string_format<T>
{

    using child_type = typename std::conditional
        < std::is_same<T, void>::value
        , char_format<void>
        , T
        > :: type;

public:

    constexpr char_format() = default;

    constexpr char_format(const char_format&) = default;

    template <typename U>
    constexpr char_format(const char_format<U>& other)
        : stringify::v0::string_format<T>(other)
        , m_count(other.m_count)
    {
    }

    ~char_format() = default;

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

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_ARG_FORMAT_HPP

