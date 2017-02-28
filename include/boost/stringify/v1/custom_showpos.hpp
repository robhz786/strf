#ifndef BOOST_STRINGIFY_V1_FMT_SHOWPOS_HPP
#define BOOST_STRINGIFY_V1_FMT_SHOWPOS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v1/type_traits.hpp>

namespace boost {
namespace stringify {
inline namespace v1 {

struct showpos_tag;

template <template <class> class Filter = boost::stringify::v1::true_trait>
struct showpos_impl
{
    template <typename T> using accept_input_type = Filter<T>;

    typedef boost::stringify::v1::showpos_tag category;
    
    constexpr showpos_impl(bool show) : m_show(show)
    {
    }
    
    constexpr bool value() const
    {
        return m_show;
    }
    
    bool m_show;
};


template <template <class> class Filter>
struct showpos_true_impl
{
    template <typename T> using accept_input_type = Filter<T>;

    typedef boost::stringify::v1::showpos_tag category;

    constexpr bool value() const { return true; }

    constexpr boost::stringify::v1::showpos_impl<Filter> operator()(bool s) const
    {
        return boost::stringify::v1::showpos_impl<Filter>(s);
    }

    constexpr boost::stringify::v1::showpos_true_impl<Filter> operator()() const
    {
        return *this;
    }
    
};


template <template <class> class Filter>
struct showpos_false_impl
{
    template <typename T> using accept_input_type = Filter<T>;

    typedef boost::stringify::v1::showpos_tag category;

    constexpr bool value() const { return false; }

    constexpr boost::stringify::v1::showpos_false_impl<Filter> operator()() const
    {
        return *this;
    }
};


constexpr auto showpos
= boost::stringify::v1::showpos_true_impl<boost::stringify::v1::true_trait>();

constexpr auto noshowpos
= boost::stringify::v1::showpos_false_impl<boost::stringify::v1::true_trait>();

template <template <class> class F>
boost::stringify::v1::showpos_true_impl<F> showpos_if
= boost::stringify::v1::showpos_true_impl<F>();

template <template <class> class F>
boost::stringify::v1::showpos_false_impl<F> noshowpos_if
= boost::stringify::v1::showpos_false_impl<F>();


struct showpos_tag
{
    typedef
        boost::stringify::v1::showpos_false_impl<boost::stringify::v1::true_trait>
        default_impl;
};


} // inline namespace v1
} // namespace stringify
} // namespace boost


#endif  /* BOOST_STRINGIFY_V1_FMT_SHOWPOS_HPP */

