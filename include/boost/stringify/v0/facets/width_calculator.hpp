#ifndef BOOST_STRINGIFY_V0_FACETS_WIDTH_CALCULATOR_HPP
#define BOOST_STRINGIFY_V0_FACETS_WIDTH_CALCULATOR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/constrained_facet.hpp>
#include <boost/stringify/v0/ftuple.hpp>
#include <boost/assert.hpp>
#include <string>
#include <limits>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

struct width_calculator_tag;

class simplest_width_calculator
{
public:
    typedef boost::stringify::v0::width_calculator_tag category;

    template <typename CharT>
    constexpr int width_of(CharT) const
    {
        return 1;
    }

    template <typename CharT>
    constexpr int width_of(const CharT*, std::size_t len) const
    {
        BOOST_ASSERT(len < (std::size_t) std::numeric_limits<int>::max ());
        return static_cast<int>(len);
    }

    template <typename CharT>
    int width_of(const CharT* str) const
    {
        return width_of(str, std::char_traits<CharT>::length(str));
    }

    template <typename CharT>
    int remaining_width
    (int total_width, const CharT*, std::size_t str_len) const
    {
        if (str_len > (std::size_t)(total_width))
        {
            return 0;
        }
        return total_width - static_cast<int>(str_len);
    }
};

constexpr simplest_width_calculator default_width_calculator {};

struct width_calculator_tag
{
    constexpr static const auto& get_default()
    {
        return boost::stringify::v0::default_width_calculator;        
    }
};

template <typename InputType, typename FTuple>
const auto& get_width_calculator(const FTuple& fmt)
{
    return boost::stringify::v0::get_facet
        <boost::stringify::v0::width_calculator_tag, InputType>(fmt);
}


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_FACETS_WIDTH_CALCULATOR_HPP

