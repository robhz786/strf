#ifndef BOOST_STRINGIFY_V0_FACETS_CONVERSION_FROM_UTF32_HPP
#define BOOST_STRINGIFY_V0_FACETS_CONVERSION_FROM_UTF32_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/type_traits.hpp>
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
        , char32_t ch
        , std::size_t count = 1
        ) const = 0;
};

template <template <class> class Filter = boost::stringify::v0::true_trait>
class to_utf8: public conversion_from_utf32<char>
{
public:

    using category = boost::stringify::v0::conversion_from_utf32_tag<char>;

    template <typename T> using accept_input_type = Filter<T>;

    to_utf8() = default;

    ~to_utf8() = default;

    std::size_t length(char32_t ch) const noexcept override
    {
        return (ch <     0x80 ? 1 :
                ch <    0x800 ? 2 :
                ch <  0x10000 ? 3 :
                ch < 0x110000 ? 4 :
                length(0xFFFD));
    }

    void write
        ( boost::stringify::v0::output_writer<char>& out
        , char32_t ch
        , std::size_t count
        ) const noexcept override
    {
        if (ch < 0x80)
        {
            out.repeat(static_cast<char>(ch), count);
        }
        else if (ch < 0x800)
        {
            out.repeat
                ( static_cast<char>(0xC0 | ((ch & 0x7C0) >> 6))
                , static_cast<char>(0x80 |  (ch &  0x3F))
                , count);
        }
        else if (ch <  0x10000)
        {
            out.repeat
                ( static_cast<char>(0xE0 | ((ch & 0xF000) >> 12))
                , static_cast<char>(0x80 | ((ch &  0xFC0) >> 6))
                , static_cast<char>(0x80 |  (ch &   0x3F))
                , count
                );
        }
        else if (ch < 0x110000)
        {
            out.repeat
                ( static_cast<char>(0xF0 | ((ch & 0x1C0000) >> 18))
                , static_cast<char>(0x80 | ((ch &  0x3F000) >> 12))
                , static_cast<char>(0x80 | ((ch &    0xFC0) >> 6))
                , static_cast<char>(0x80 |  (ch &     0x3F))
                , count
                );
        }
        else
        {
            write(out, 0xFFFD, count);
        }
    }

};


template
    < template <class> class Filter = boost::stringify::v0::true_trait
    , typename CharT = char16_t
    >
class to_utf16: public conversion_from_utf32<CharT>
{
public:

    using category = boost::stringify::v0::conversion_from_utf32_tag<CharT>;

    template <typename T> using accept_input_type = Filter<T>;

    to_utf16() = default;

    ~to_utf16() = default;

    std::size_t length(char32_t ch) const noexcept override
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

    virtual void write
        ( boost::stringify::v0::output_writer<CharT>& destination
        , char32_t ch
        , std::size_t count
        ) const noexcept override
    {
        if (single_char_range(ch))
        {
            destination.repeat(static_cast<CharT>(ch), count);
        }
        else if (two_chars_range(ch))
        {
            char32_t sub_codepoint = ch - 0x10000;
            char32_t high_surrogate = 0xD800 + ((sub_codepoint & 0xFFC00) >> 10);
            char32_t low_surrogate  = 0xDC00 +  (sub_codepoint &  0x3FF);
            destination.repeat
                ( static_cast<CharT>(high_surrogate)
                , static_cast<CharT>(low_surrogate)
                , count
                );
        }
        else
        {
            write(destination, 0xFFFD, count);
        }
    }

private:

    bool single_char_range(char32_t ch) const noexcept
    {
        return ch < 0xd800 || (0xdfff < ch && ch < 0x10000);
    }

    bool two_chars_range(char32_t ch) const noexcept
    {
        return 0xffff < ch && ch < 0x110000;
    }
};

template
    < template <class> class Filter = boost::stringify::v0::true_trait
    , typename CharT = char32_t
    >
class utf32_to_utf32: public conversion_from_utf32<CharT>
{
public:

    using category = boost::stringify::v0::conversion_from_utf32_tag<CharT>;

    template <typename T> using accept_input_type = Filter<T>;

    utf32_to_utf32() = default;

    ~utf32_to_utf32() = default;

    std::size_t length(char32_t) const noexcept override
    {
        return 1;
    }

    void write
        ( boost::stringify::v0::output_writer<CharT>& destination
        , char32_t ch
        , std::size_t count
        ) const noexcept override
    {
        destination.repeat(ch, count);
    }
};

template <template <class> class Filter = boost::stringify::v0::true_trait>
class utf32_to_wstr
    : public boost::stringify::v0::detail::ternary_t
        < sizeof(wchar_t) == sizeof(char16_t)
        , to_utf16<Filter, wchar_t>
        , utf32_to_utf32<Filter, wchar_t>
        >
{
};


template <> struct conversion_from_utf32_tag<char>
{
    static const auto& get_default() noexcept
    {
        const static boost::stringify::v0::to_utf8<boost::stringify::v0::true_trait> x{};
        return x;
    }
};


template <> struct conversion_from_utf32_tag<char16_t>
{
    static const auto& get_default() noexcept
    {
        const static boost::stringify::v0::to_utf16<boost::stringify::v0::true_trait> x{};
        return x;
    }
};


template <> struct conversion_from_utf32_tag<char32_t>
{
    static const auto& get_default() noexcept
    {
        const static boost::stringify::v0::utf32_to_utf32<boost::stringify::v0::true_trait> x{};
        return x;
    }
};

template <> struct conversion_from_utf32_tag<wchar_t>
{
    static const auto& get_default() noexcept
    {
        const static boost::stringify::v0::utf32_to_wstr<boost::stringify::v0::true_trait> x{};
        return x;
    }
};

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_FACETS_CONVERSION_FROM_UTF32_HPP

