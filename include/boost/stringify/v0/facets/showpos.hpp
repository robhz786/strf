#ifndef BOOST_STRINGIFY_V0_FACETS_SHOWPOS_HPP
#define BOOST_STRINGIFY_V0_FACETS_SHOWPOS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/constrained_facet.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

struct showpos_tag;

// struct showpos_impl
// {
//     typedef boost::stringify::v0::showpos_tag category;
    
//     constexpr showpos_impl(bool show) : m_show(show)
//     {
//     }
    
//     constexpr bool value() const
//     {
//         return m_show;
//     }
    
//     bool m_show;
// };


struct showpos_true_impl
{
    typedef boost::stringify::v0::showpos_tag category;

    constexpr bool value() const { return true; }

    // constexpr boost::stringify::v0::showpos_impl<Filter> operator()(bool s) const
    // {
    //     return boost::stringify::v0::showpos_impl<Filter>(s);
    // }

    // constexpr boost::stringify::v0::showpos_true_impl<Filter> operator()() const
    // {
    //     return *this;
    // }
};


struct showpos_false_impl
{
    typedef boost::stringify::v0::showpos_tag category;

    constexpr bool value() const { return false; }

    // constexpr boost::stringify::v0::showpos_false_impl<Filter> operator()() const
    // {
    //     return *this;
    // }
};


constexpr auto showpos
= boost::stringify::v0::showpos_true_impl();

constexpr auto noshowpos
= boost::stringify::v0::showpos_false_impl();

template <template <class> class F>
constrained_facet<F, boost::stringify::v0::showpos_true_impl> showpos_if {};

template <template <class> class F>
constrained_facet<F, boost::stringify::v0::showpos_false_impl> noshowpos_if {};

struct showpos_tag
{
    constexpr static const auto& get_default() noexcept
    {
        return boost::stringify::v0::noshowpos;
    }
};


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  /* BOOST_STRINGIFY_V0_FACETS_SHOWPOS_HPP */

