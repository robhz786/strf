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

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

struct width_calculator_tag;

class width_calculator
{

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

public:

    typedef boost::stringify::v0::width_calculator_tag category;

    virtual ~width_calculator() = default;

    virtual int width_of(char32_t) const = 0;

    virtual int remaining_width
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
        return std::max(0, width);
    }

    virtual int remaining_width
        ( int width
        , const char32_t* str
        , std::size_t str_len
        , const conversion_to_utf32<char32_t>&
        ) const
    {
        return remaining_width(width, str, str_len);
    }

    virtual int remaining_width
        ( int width
        , const char* str
        , std::size_t str_len
        , const conversion_to_utf32<char>& conv
        ) const
    {
        width_decrementer decrementer = {width, *this};
        conv.convert(decrementer, str, str_len);
        return std::max(0, decrementer.width);
    }

    virtual int remaining_width
        ( int width
        , const char16_t* str
        , std::size_t str_len
        , const conversion_to_utf32<char16_t>& conv
        ) const
    {
        width_decrementer decrementer = {width, *this};
        conv.convert(decrementer, str, str_len);
        return std::max(0, decrementer.width);
    }

    virtual int remaining_width
        ( int width
        , const wchar_t* str
        , std::size_t str_len
        , const conversion_to_utf32<wchar_t>& conv
        ) const
    {
        width_decrementer decrementer = {width, *this};
        conv.convert(decrementer, str, str_len);
        return std::max(0, decrementer.width);
    }

};


class simplest_width_calculator: public width_calculator
{
public:

    int width_of(char32_t) const override
    {
        return 1;
    }

    template <typename CharT>
    int width_of(const CharT*, std::size_t len) const
    {
        BOOST_ASSERT(len < (std::size_t) std::numeric_limits<int>::max ());
        return static_cast<int>(len);
    }

   virtual int remaining_width
        ( int width
        , const char32_t* str
        , std::size_t str_len
        ) const override
    {
        return remaining_width_impl(width, str, str_len);
    }

    virtual int remaining_width
        ( int width
        , const char* str
        , std::size_t str_len
        , const conversion_to_utf32<char>&
        ) const override
    {
        return remaining_width_impl(width, str, str_len);
    }

    virtual int remaining_width
        ( int width
        , const char16_t* str
        , std::size_t str_len
        , const conversion_to_utf32<char16_t>&
        ) const override
    {
        return remaining_width_impl(width, str, str_len);
    }

    virtual int remaining_width
        ( int width
        , const wchar_t* str
        , std::size_t str_len
        , const conversion_to_utf32<wchar_t>&
        ) const override
    {
        return remaining_width_impl(width, str, str_len);
    }

private:

    template <typename CharT>
    int remaining_width_impl
        (int total_width, const CharT*, std::size_t str_len) const
    {
        if (str_len > (std::size_t)(total_width))
        {
            return 0;
        }
        return total_width - static_cast<int>(str_len);
    }
};


struct width_calculator_tag
{
    static const auto& get_default()
    {
        static simplest_width_calculator x {};
        return x;
    }
};


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_FACETS_WIDTH_CALCULATOR_HPP

