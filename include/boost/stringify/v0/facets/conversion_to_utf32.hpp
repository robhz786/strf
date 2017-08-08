#ifndef BOOST_STRINGIFY_V0_FACETS_CONVERSION_TO_UTF32_HPP
#define BOOST_STRINGIFY_V0_FACETS_CONVERSION_TO_UTF32_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <functional>
#include <boost/assert.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <typename CharT> struct conversion_to_utf32_tag;


// class char32_accumulator
// {
// public:

//     virtual ~char32_accumulator() = default;

//     virtual bool operator()(char32_t) = 0;
// };


template <typename CharT> class conversion_to_utf32
{
public:

    virtual ~conversion_to_utf32() = default;

    virtual void convert
        ( std::function<bool(char32_t)> accumulator
        , const CharT* str
        , std::size_t str_len
        ) const = 0;
};


class from_utf8: public conversion_to_utf32<char>
{
public:

    using category = boost::stringify::v0::conversion_to_utf32_tag<char>;

    virtual ~from_utf8() = default;

    virtual void convert
        ( std::function<bool(char32_t)> //accumulator
        , const char* //str
        , std::size_t str_len
        ) const override
    {
        // TODO
        (void)str_len;
        BOOST_ASSERT_MSG(str_len == 0, "from_utf8 not implemented yet");
    }
};


class from_utf16: public conversion_to_utf32<char16_t>
{
public:

    using category = boost::stringify::v0::conversion_to_utf32_tag<char16_t>;

    virtual ~from_utf16() = default;

    virtual void convert
        ( std::function<bool(char32_t)> //accumulator
        , const char16_t* //str
        , std::size_t str_len
        ) const override
    {
        // TODO
        (void)str_len;
        BOOST_ASSERT_MSG(str_len == 0, "from_utf16 not implemented yet");
    }
};


class from_utf32: public conversion_to_utf32<char32_t>
{
public:

    using category = boost::stringify::v0::conversion_to_utf32_tag<char32_t>;

    virtual ~from_utf32() = default;

    virtual void convert
        ( std::function<bool(char32_t)> //accumulator
        , const char32_t* //str
        , std::size_t str_len
        ) const override
    {
        (void)str_len;
        BOOST_ASSERT_MSG(str_len == 0, "from_utf32::convert should not be used");
    }
};


class from_wstr: public conversion_to_utf32<wchar_t>
{
public:

    virtual ~from_wstr() = default;

    virtual void convert
        ( std::function<bool(char32_t)> //accumulator
        , const wchar_t* //str
        , std::size_t str_len
        ) const override
    {
        // TODO
        (void)str_len;
        BOOST_ASSERT_MSG(str_len == 0, "from_wstr not implemented yet");
    }
};


template <> struct conversion_to_utf32_tag<char>
{
    static const conversion_to_utf32<char>& get_default() noexcept
    {
        const static boost::stringify::v0::from_utf8 x{};
        return x;
    }
};


template <> struct conversion_to_utf32_tag<char16_t>
{
    static const conversion_to_utf32<char16_t>& get_default() noexcept
    {
        const static boost::stringify::v0::from_utf16 x{};
        return x;
    }
};


template <> struct conversion_to_utf32_tag<char32_t>
{
    static const conversion_to_utf32<char32_t>& get_default() noexcept
    {
        const static boost::stringify::v0::from_utf32 x{};
        return x;
    }
};


template <> struct conversion_to_utf32_tag<wchar_t>
{
    static const conversion_to_utf32<wchar_t>& get_default() noexcept
    {
        const static boost::stringify::v0::from_wstr x{};
        return x;
    }
};


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_FACETS_CONVERSION_TO_UTF32_HPP

