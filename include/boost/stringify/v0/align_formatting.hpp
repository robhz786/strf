#ifndef BOOST_STRINGIFY_V0_ALIGN_FORMATING_HPP
#define BOOST_STRINGIFY_V0_ALIGN_FORMATING_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/config.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

enum class alignment {left, right, internal, center};


template <class T = void>
class align_formatting
{
    using child_type = typename std::conditional
        < std::is_void<T>::value
        , align_formatting<void>
        , T
        > :: type;

public:

    template <typename U>
    using other = stringify::v0::align_formatting<U>;

    constexpr align_formatting()
    {
        static_assert(std::is_base_of<align_formatting, child_type>::value, "");
    }

    constexpr align_formatting(const align_formatting&) = default;

    template <typename U>
    constexpr align_formatting(const align_formatting<U>& other)
        : m_fill(other.m_fill)
        , m_width(other.m_width)
        , m_alignment(other.m_alignment)
    {
    }

    ~align_formatting() = default;

    template <typename U>
    constexpr child_type& format_as(const align_formatting<U>& other) &
    {
        m_fill = other.m_fill;
        m_width = other.m_width;
        m_alignment = other.m_alignment;
        return static_cast<child_type&>(*this);
    }

    template <typename U>
    constexpr child_type&& format_as(const align_formatting<U>& other) &&
    {
        return static_cast<child_type&&>(format_as(other));
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
    friend class align_formatting;

    char32_t m_fill = U' ';
    int m_width = 0;
    stringify::v0::alignment m_alignment = stringify::v0::alignment::right;
};



BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_ARG_FORMAT_HPP

