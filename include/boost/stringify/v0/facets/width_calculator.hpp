#ifndef BOOST_STRINGIFY_V0_FACETS_WIDTH_CALCULATOR_HPP
#define BOOST_STRINGIFY_V0_FACETS_WIDTH_CALCULATOR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/facets/conversion_to_utf32.hpp>
#include <boost/stringify/v0/ftuple.hpp>
#include <boost/assert.hpp>
#include <string>
#include <limits>
#include <algorithm>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

struct width_calculator_tag;

class width_calculator
{

public:

    typedef boost::stringify::v0::width_calculator_tag category;

    virtual ~width_calculator() = default;

    virtual int width_of(char32_t) const = 0;

    virtual int remaining_width
        ( int width
        , const char32_t* str
        , std::size_t str_len
        ) const;

    virtual int remaining_width
        ( int width
        , const char32_t* str
        , std::size_t str_len
        , const conversion_to_utf32<char32_t>&
        ) const;

    virtual int remaining_width
        ( int width
        , const char* str
        , std::size_t str_len
        , const conversion_to_utf32<char>& conv
        ) const;
    
    virtual int remaining_width
        ( int width
        , const char16_t* str
        , std::size_t str_len
        , const conversion_to_utf32<char16_t>& conv
        ) const;

    virtual int remaining_width
        ( int width
        , const wchar_t* str
        , std::size_t str_len
        , const conversion_to_utf32<wchar_t>& conv
        ) const;
};


class simplest_width_calculator: public width_calculator
{
public:

    int width_of(char32_t) const override;

    int remaining_width
        ( int width
        , const char32_t* str
        , std::size_t str_len
        ) const override;

    int remaining_width
        ( int width
        , const char* str
        , std::size_t str_len
        , const conversion_to_utf32<char>&
        ) const override;

    int remaining_width
        ( int width
        , const char16_t* str
        , std::size_t str_len
        , const conversion_to_utf32<char16_t>&
        ) const override;

    int remaining_width
        ( int width
        , const wchar_t* str
        , std::size_t str_len
        , const conversion_to_utf32<wchar_t>&
        ) const override;
};


struct width_calculator_tag
{
    static const simplest_width_calculator& get_default();
};

#if ! defined(BOOST_STRINGIFY_OMIT_IMPL)

namespace detail{

struct width_decrementer
{
    int width = 0;
    const width_calculator& wcalc;

    bool operator()(char32_t ch)
    {
        width -= wcalc.width_of(ch);
        return width > 0;
    }
};
} // namespace detail

BOOST_STRINGIFY_INLINE int width_calculator::remaining_width
    ( int width
    , const char32_t* str
    , std::size_t str_len
    ) const
{
    const char32_t* end = str + str_len;
    for(const char32_t* it = str; it != end && width > 0; ++it)
    {
        width -= width_of(*it);
    }

    return (std::max)(0, width);
}

BOOST_STRINGIFY_INLINE int width_calculator::remaining_width
    ( int width
    , const char32_t* str
    , std::size_t str_len
    , const conversion_to_utf32<char32_t>&
    ) const
{
    return remaining_width(width, str, str_len);
}

BOOST_STRINGIFY_INLINE int width_calculator::remaining_width
    ( int width
    , const char* str
    , std::size_t str_len
    , const conversion_to_utf32<char>& conv
    ) const
{
    detail::width_decrementer decrementer = {width, *this};
    conv.convert(decrementer, str, str_len);
    return (std::max)(0, decrementer.width);
}

BOOST_STRINGIFY_INLINE int width_calculator::remaining_width
    ( int width
    , const char16_t* str
    , std::size_t str_len
    , const conversion_to_utf32<char16_t>& conv
    ) const
{
    detail::width_decrementer decrementer = {width, *this};
    conv.convert(decrementer, str, str_len);
    return (std::max)(0, decrementer.width);
}

BOOST_STRINGIFY_INLINE int width_calculator::remaining_width
    ( int width
    , const wchar_t* str
    , std::size_t str_len
    , const conversion_to_utf32<wchar_t>& conv
    ) const
{
    detail::width_decrementer decrementer = {width, *this};
    conv.convert(decrementer, str, str_len);
    return (std::max)(0, decrementer.width);
}

namespace detail{

template <typename CharT>
int simplest_remaining_width_impl(int width, const CharT*, std::size_t len)
{
    if (len > (std::size_t)(width))
    {
        return 0;
    }
    return width - static_cast<int>(len);
}

} // namespace detail

BOOST_STRINGIFY_INLINE int simplest_width_calculator::width_of(char32_t) const
{
    return 1;
}

BOOST_STRINGIFY_INLINE int simplest_width_calculator::remaining_width
    ( int width
    , const char32_t* str
    , std::size_t str_len
    ) const
{
    return detail::simplest_remaining_width_impl(width, str, str_len);
}

BOOST_STRINGIFY_INLINE int simplest_width_calculator::remaining_width
    ( int width
    , const char* str
    , std::size_t str_len
    , const conversion_to_utf32<char>&
    ) const
{
    return detail::simplest_remaining_width_impl(width, str, str_len);
}

BOOST_STRINGIFY_INLINE int simplest_width_calculator::remaining_width
    ( int width
    , const char16_t* str
    , std::size_t str_len
    , const conversion_to_utf32<char16_t>&
    ) const
{
    return detail::simplest_remaining_width_impl(width, str, str_len);
}

BOOST_STRINGIFY_INLINE int simplest_width_calculator::remaining_width
    ( int width
    , const wchar_t* str
    , std::size_t str_len
    , const conversion_to_utf32<wchar_t>&
    ) const
{
    return detail::simplest_remaining_width_impl(width, str, str_len);
}

BOOST_STRINGIFY_INLINE
const simplest_width_calculator& width_calculator_tag::get_default()
{
    static simplest_width_calculator x {};
    return x;
}

#endif // ! defined(BOOST_STRINGIFY_OMIT_IMPL)


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_FACETS_WIDTH_CALCULATOR_HPP

