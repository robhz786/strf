#ifndef BOOST_STRINGIFY_V0_FACETS_FILL_HPP
#define BOOST_STRINGIFY_V0_FACETS_FILL_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/type_traits.hpp>

namespace boost {
namespace stringify {
inline namespace v0 {

struct fill_tag;

template
    < char32_t FillChar = U' '
    , template <class> class Filter = boost::stringify::v0::true_trait
    >
struct fill_impl_t
{
    typedef boost::stringify::v0::fill_tag category;
    template <typename T> using accept_input_type = Filter<T>;

    constexpr char32_t fill_char() const
    {
        return FillChar;
    }
};


template <template <class> class Filter = boost::stringify::v0::true_trait>
class fill_impl
{
public:

    constexpr fill_impl(char32_t ch)
        : m_fillchar(ch)
    {
    }

    typedef boost::stringify::v0::fill_tag category;
    template <typename T> using accept_input_type = Filter<T>;

    constexpr char32_t fill_char() const
    {
        return m_fillchar;
    }

private:

    char32_t m_fillchar;
};


constexpr
boost::stringify::v0::fill_impl_t<U' ', boost::stringify::v0::true_trait>
default_fill {};

struct fill_tag
{
    constexpr static const auto& get_default() noexcept
    {
        return boost::stringify::v0::default_fill;
    }
};

auto fill(char32_t fillChar)
{
    return fill_impl<boost::stringify::v0::true_trait>(fillChar);
}

template <template <class> class Filter>
auto fill_if(char32_t fillChar)
{
    return fill_impl<Filter>(fillChar);
}


template <char32_t fillChar>
auto fill_t = fill_impl_t<fillChar, boost::stringify::v0::true_trait>();

template <char32_t Char, template <class> class Filter>
auto fill_t_if = fill_impl_t<Char, Filter>();


} // inline namespace v0
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_V0_FACETS_FILL_HPP

