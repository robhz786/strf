#ifndef BOOST_STRINGIFY_V0_FACETS_CONVERSION_FROM_UTF32_HPP
#define BOOST_STRINGIFY_V0_FACETS_CONVERSION_FROM_UTF32_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/output_writer.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT> struct conversion_from_utf32_tag;

template <typename CharT> class conversion_from_utf32
{
public:

    virtual ~conversion_from_utf32()
    {
    };

    virtual std::size_t length(char32_t ch) const = 0;

    virtual void write
        ( boost::stringify::v0::output_writer<CharT>& destination
        , std::size_t count
        , char32_t ch
        ) const = 0;

};


class to_utf8: public conversion_from_utf32<char>
{
public:

    using category = boost::stringify::v0::conversion_from_utf32_tag<char>;

    to_utf8() = default;

    ~to_utf8() = default;

    std::size_t length(char32_t ch) const noexcept override;

    void write
        ( boost::stringify::v0::output_writer<char>& destination
        , std::size_t count
        , char32_t ch
        ) const override;

};


class to_utf16: public conversion_from_utf32<char16_t>
{
public:

    using category = boost::stringify::v0::conversion_from_utf32_tag<char16_t>;

    to_utf16() = default;

    ~to_utf16() = default;

    std::size_t length(char32_t ch) const noexcept override;

    void write
        ( boost::stringify::v0::output_writer<char16_t>& destination
        , std::size_t count
        , char32_t ch
        ) const override;
};


class utf32_to_utf32: public conversion_from_utf32<char32_t>
{
public:

    using category = boost::stringify::v0::conversion_from_utf32_tag<char32_t>;

    utf32_to_utf32() = default;

    ~utf32_to_utf32() = default;

    std::size_t length(char32_t) const noexcept override
    {
        return 1;
    }

    void write
        ( boost::stringify::v0::output_writer<char32_t>& destination
        , std::size_t count
        , char32_t ch
        ) const override
    {
        destination.repeat(count, ch);
    }
};


class default_utf32_to_wstr : public conversion_from_utf32<wchar_t>
{
public:

    using category = boost::stringify::v0::conversion_from_utf32_tag<wchar_t>;

    std::size_t length(char32_t) const noexcept override;

    void write
        ( boost::stringify::v0::output_writer<wchar_t>& destination
        , std::size_t count
        , char32_t ch
        ) const override;
};


template <> struct conversion_from_utf32_tag<char>
{
    static const to_utf8& get_default() noexcept;
};

template <> struct conversion_from_utf32_tag<char16_t>
{
    static const to_utf16& get_default() noexcept;
};

template <> struct conversion_from_utf32_tag<char32_t>
{
    static const utf32_to_utf32& get_default() noexcept;
};

template <> struct conversion_from_utf32_tag<wchar_t>
{
    static const default_utf32_to_wstr& get_default() noexcept;
};


#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_INLINE std::size_t to_utf8::length(char32_t ch) const noexcept
{
    return (ch <     0x80 ? 1 :
            ch <    0x800 ? 2 :
            ch <  0x10000 ? 3 :
            ch < 0x110000 ? 4 :
            length(0xFFFD));
}

BOOST_STRINGIFY_INLINE void to_utf8::write
    ( boost::stringify::v0::output_writer<char>& destination
    , std::size_t count
    , char32_t ch
    ) const
{
    if (ch < 0x80)
    {
        if (count == 1)
        {
            destination.put(static_cast<char>(ch));
        }
        else
        {
            destination.repeat(count, static_cast<char>(ch));
        }
    }
    else if (ch < 0x800)
    {
        destination.repeat
            ( count
            , static_cast<char>(0xC0 | ((ch & 0x7C0) >> 6))
            , static_cast<char>(0x80 |  (ch &  0x3F))
            );
    }
    else if (ch <  0x10000)
    {
        destination.repeat
            ( count
            , static_cast<char>(0xE0 | ((ch & 0xF000) >> 12))
            , static_cast<char>(0x80 | ((ch &  0xFC0) >> 6))
            , static_cast<char>(0x80 |  (ch &   0x3F))
            );
    }
    else if (ch < 0x110000)
    {
        destination.repeat
            ( count
            , static_cast<char>(0xF0 | ((ch & 0x1C0000) >> 18))
            , static_cast<char>(0x80 | ((ch &  0x3F000) >> 12))
            , static_cast<char>(0x80 | ((ch &    0xFC0) >> 6))
            , static_cast<char>(0x80 |  (ch &     0x3F))
            );
    }
    else
    {
        write(destination, count, 0xFFFD);
    }
}


namespace detail
{

class to_utf16_impl
{
public:

    static std::size_t length(char32_t ch) noexcept
    {
        if(single_char_range(ch))
        {
            return 1;
        }
        if(two_chars_range(ch))
        {
            return 2;
        }
        return length(0xFFFD);
    }

    template <typename CharT>
    static void write
        ( boost::stringify::v0::output_writer<CharT>& destination
        , std::size_t count
        , char32_t ch
        )
    {
        if (single_char_range(ch))
        {
            if (count == 1)
            {
                destination.put(static_cast<CharT>(ch));
            }
            else
            {
                destination.repeat(count, static_cast<CharT>(ch));
            }
        }
        else if (two_chars_range(ch))
        {
            char32_t sub_codepoint = ch - 0x10000;
            char32_t high_surrogate = 0xD800 + ((sub_codepoint & 0xFFC00) >> 10);
            char32_t low_surrogate  = 0xDC00 +  (sub_codepoint &  0x3FF);
            destination.repeat
                ( count
                , static_cast<CharT>(high_surrogate)
                , static_cast<CharT>(low_surrogate)
                );
        }
        else
        {
            write(destination, count, 0xFFFD);
        }
    }

    static bool single_char_range(char32_t ch) noexcept
    {
        return ch < 0xd800 || (0xdfff < ch && ch < 0x10000);
    }

    static bool two_chars_range(char32_t ch) noexcept
    {
        return 0xffff < ch && ch < 0x110000;
    }
};

} //namespace detail


BOOST_STRINGIFY_INLINE std::size_t to_utf16::length(char32_t ch) const noexcept
{
    return detail::to_utf16_impl::length(ch);
}

BOOST_STRINGIFY_INLINE void to_utf16::write
    ( boost::stringify::v0::output_writer<char16_t>& destination
    , std::size_t count
    , char32_t ch
    ) const
{
    detail::to_utf16_impl::write<char16_t>(destination, count, ch);
}


BOOST_STRINGIFY_INLINE std::size_t default_utf32_to_wstr::length(char32_t ch) const noexcept
{
    if (sizeof(wchar_t) == sizeof(char16_t))
    {
        return detail::to_utf16_impl::length(ch);
    }
    else
    {
        return 1;
    }
}

BOOST_STRINGIFY_INLINE void default_utf32_to_wstr::write
    ( boost::stringify::v0::output_writer<wchar_t>& destination
    , std::size_t count
    , char32_t ch
    ) const
{
    if (sizeof(wchar_t) == sizeof(char16_t))
    {
        detail::to_utf16_impl::write<wchar_t>(destination, count, ch);
    }
    else
    {
        destination.repeat(count, static_cast<wchar_t>(ch));
    }
}


BOOST_STRINGIFY_INLINE
const to_utf8& conversion_from_utf32_tag<char>::get_default() noexcept
{
    const static to_utf8 x{};
    return x;
}

BOOST_STRINGIFY_INLINE
const to_utf16& conversion_from_utf32_tag<char16_t>::get_default() noexcept
{
    const static to_utf16 x{};
    return x;
}

BOOST_STRINGIFY_INLINE
const utf32_to_utf32& conversion_from_utf32_tag<char32_t>::get_default() noexcept
{
    const static utf32_to_utf32 x{};
    return x;
}

BOOST_STRINGIFY_INLINE
const default_utf32_to_wstr&
conversion_from_utf32_tag<wchar_t>::get_default() noexcept
{
    const static default_utf32_to_wstr x{};
    return x;
}

#endif // ! defined(BOOST_STRINGIFY_OMIT_IMPL)

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_FACETS_CONVERSION_FROM_UTF32_HPP

