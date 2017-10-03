#ifndef BOOST_STRINGIFY_V0_FACETS_SHOWBASE_HPP
#define BOOST_STRINGIFY_V0_FACETS_SHOWBASE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

struct showbase_tag;

template <bool ShowBase>
struct showbase_impl_t
{
    using category = boost::stringify::v0::showbase_tag;

    constexpr bool value() const
    {
        return ShowBase;
    }
};

constexpr boost::stringify::v0::showbase_impl_t<true> showbase {};

constexpr boost::stringify::v0::showbase_impl_t<false> noshowbase {};


template <template <class> class F>
constrained_facet<F, showbase_impl_t<true>> showbase_if {}; 

template <template <class> class F>
constrained_facet<F, showbase_impl_t<false>> noshowbase_if {};


struct showbase_tag
{
    constexpr static const auto& get_default() noexcept
    {
        return boost::stringify::v0::noshowbase;
    }
};

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_FACETS_SHOWBASE_HPP

