#ifndef BOOST_STRINGIFY_V1_FMT_WIDTH_CALCULATOR_HPP
#define BOOST_STRINGIFY_V1_FMT_WIDTH_CALCULATOR_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v1/type_traits.hpp>
#include <boost/stringify/v1/ftuple.hpp>
#include <boost/assert.hpp>
#include <string>
#include <limits>

namespace boost {
namespace stringify {
inline namespace v1 {

struct width_calculator_tag;

template <template <class> class Filter=boost::stringify::v1::true_trait>
class simplest_width_calculator
{
public:
    typedef boost::stringify::v1::width_calculator_tag category;

    template <typename T> using accept_input_type = Filter<T>;
  
    template <typename CharT>
    constexpr int width_of(CharT ch) const
    {
        return 1;
    }

    template <typename CharT>
    constexpr int width_of(const CharT* str, std::size_t len) const
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
    (int total_width, const CharT* str, std::size_t str_len) const
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
    typedef
    boost::stringify::v1::simplest_width_calculator<boost::stringify::v1::true_trait>
    default_impl;
};

template <typename InputType, typename FTuple>
decltype(auto) get_width_calculator(const FTuple& fmt)
{
    return boost::stringify::v1::get_facet
        <boost::stringify::v1::width_calculator_tag, InputType>(fmt);
}


} // inline namespace v1
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_V1_FMT_WIDTH_CALCULATOR_HPP

