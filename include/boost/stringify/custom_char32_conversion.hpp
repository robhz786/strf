#ifndef BOOST_STRINGIFY_CUSTOM_CHAR32_CONVERSION_HPP
#define BOOST_STRINGIFY_CUSTOM_CHAR32_CONVERSION_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/type_traits.hpp>

namespace boost {
namespace stringify {
namespace detail {

class char32_to_utf8_impl
{
public:

    static constexpr std::size_t max_lenght = 4;

    std::size_t length(char32_t ch) const noexcept
    {
        return (ch <     0x80 ? 1 :
                ch <    0x800 ? 2 :
                ch <  0x10000 ? 3 :
                ch < 0x110000 ? 4 :
                /* invalid codepoit */ 0);
    }

    template <typename Output>
    void write(char32_t ch, Output& out) const
    {
        if (ch < 0x80)
        {
            out.put(static_cast<char>(ch));
        }
        else if (ch < 0x800)
        {
            out.put(static_cast<char>(0xC0 | ((ch & 0x7C0) >> 6)));
            out.put(static_cast<char>(0x80 |  (ch &  0x3F)));
        }
        else if (ch <  0x10000)
        {
            out.put(static_cast<char>(0xE0 | ((ch & 0xF000) >> 12)));
            out.put(static_cast<char>(0x80 | ((ch &  0xFC0) >> 6)));
            out.put(static_cast<char>(0x80 |  (ch &   0x3F)));
        }
        else if (ch < 0x110000)
        {
            out.put(static_cast<char>(0xF0 | ((ch & 0x1C0000) >> 18)));
            out.put(static_cast<char>(0x80 | ((ch &  0x3F000) >> 12)));
            out.put(static_cast<char>(0x80 | ((ch &    0xFC0) >> 6)));
            out.put(static_cast<char>(0x80 |  (ch &     0x3F)));
        }
    }

    constexpr char convert_if_length_is_1(char32_t ch) const noexcept
    {
        return ch < 0x80 ? static_cast<char>(ch) : '\0';
    }

};

template <typename CharT>
class char32_to_utf16_impl
{

public:

    static constexpr std::size_t max_lenght = 2;

    std::size_t length(char32_t ch) const noexcept
    {
        if(single_char_range(ch))
        {
            return 1;
        }
        if(two_chars_range(ch))
        {
            return 2;
        }
        return 0;
    }

    template <typename Output>
    void write(char32_t ch, Output& out) const
    {
        if (single_char_range(ch))
        {
            out.put(static_cast<CharT>(ch));
        }
        else if (two_chars_range(ch))
        {
            char32_t sub_codepoint = ch - 0x10000;
            char32_t high_surrogate = 0xD800 + ((sub_codepoint & 0xFFC00) >> 10);
            char32_t low_surrogate  = 0xDC00 +  (sub_codepoint &  0x3FF);
            out.put(static_cast<CharT>(high_surrogate));
            out.put(static_cast<CharT>(low_surrogate));
        }
    }

    constexpr char16_t convert_if_length_is_1(char32_t ch) const noexcept
    {
        return  single_char_range(ch) ? static_cast<char16_t>(ch) : u'\0';
    }


private:

    bool single_char_range(char32_t ch) const
    {
        return ch < 0xd800 || (0xdfff < ch && ch <  0x10000);
    }

    bool two_chars_range(char32_t ch) const
    {
        return 0xffff < ch && ch < 0x110000;
    }
};

template <typename CharT>
class char32_to_utf32_impl
{
public:

    static constexpr std::size_t max_lenght = 4;

    constexpr std::size_t length(char32_t ch) const noexcept
    {
        return 1;
    }

    template <typename Output>
    void write (char32_t ch, Output& out) const
    {
        out.put(static_cast<CharT>(ch));
    }

    constexpr char32_t convert_if_length_is_1(char32_t ch) const noexcept
    {
        return ch;
    }
};


}// namespace detail


template <typename CharT> struct char32_to_str_tag;


template <template <class> class Filter>
class char32_to_utf8
    : public boost::stringify::detail::char32_to_utf8_impl
{
public:
    typedef char32_to_str_tag<char> category;
    template <typename T> using accept_input_type = Filter<T>;
};


template <template <class> class Filter>
class char32_to_utf16
    : public boost::stringify::detail::char32_to_utf16_impl<char16_t>
{
public:
    typedef char32_to_str_tag<char16_t> category;
    template <typename T> using accept_input_type = Filter<T>;
};


template <template <class> class Filter>
class char32_to_utf32
    : public boost::stringify::detail::char32_to_utf32_impl<char32_t>
{
public:
    typedef char32_to_str_tag<char32_t> category;
    template <typename T> using accept_input_type = Filter<T>;
};


template <template <class> class Filter>
class default_char32_to_wstr
    : public boost::stringify::detail::ternary_t
        < (sizeof(wchar_t) == sizeof(char32_t))
        , boost::stringify::detail::char32_to_utf32_impl<wchar_t>
        , boost::stringify::detail::char32_to_utf16_impl<wchar_t>
        >
{
    typedef char32_to_str_tag<wchar_t> category;
    template <typename T> using accept_input_type = Filter<T>;
};


template <> struct char32_to_str_tag<char>
{
    typedef
        boost::stringify::char32_to_utf8<boost::stringify::true_trait>
        default_impl;
};


template <> struct char32_to_str_tag<wchar_t>
{
    typedef
        boost::stringify::default_char32_to_wstr<boost::stringify::true_trait>
        default_impl;
};


template <> struct char32_to_str_tag<char16_t>
{
    typedef
        boost::stringify::char32_to_utf16<boost::stringify::true_trait>
        default_impl;
};


template <> struct char32_to_str_tag<char32_t>
{
    typedef
        boost::stringify::char32_to_utf32<boost::stringify::true_trait>
        default_impl;
};


template <typename CharT, typename InputType, typename Formatting>
decltype(auto) get_char32_writer(const Formatting& fmt) noexcept
{
    typedef  boost::stringify::char32_to_str_tag<CharT> tag;
    return fmt.template get<tag, InputType>();
}

template <typename CharT, typename InputType, typename Formatting>
std::size_t get_char32_length(const Formatting& fmt, char32_t ch) noexcept
{
    typedef  boost::stringify::char32_to_str_tag<CharT> tag;
    return fmt.template get<tag, InputType>().length(ch);
}

template <typename InputType, typename Formatting, typename Output>
void write_char32(const Formatting& fmt, Output& out, char32_t ch)
{
    using tag = boost::stringify::char32_to_str_tag<typename Output::char_type>;
    fmt.template get<tag, InputType>().write(ch, out);
}

} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_CUSTOM_CHAR32_TO_STR_HPP

