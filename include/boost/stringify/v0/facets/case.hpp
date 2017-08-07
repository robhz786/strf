#ifndef BOOST_STRINGIFY_V0_FACETS_CASE_HPP
#define BOOST_STRINGIFY_V0_FACETS_CASE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

struct case_tag;

template <bool Uppercase>
struct case_impl_t
{
    using category = boost::stringify::v0::case_tag;

    constexpr bool uppercase() const
    {
        return Uppercase;
    }
};


constexpr boost::stringify::v0::case_impl_t<true> uppercase {};

constexpr boost::stringify::v0::case_impl_t<false> lowercase {};


template <template <class> class F>
constrained_facet<F, case_impl_t<true>> upppercase_if {};

template <template <class> class F>
constrained_facet<F, case_impl_t<false>> lowercase_if {};


struct case_tag
{
    constexpr static const auto& get_default() noexcept
    {
        return boost::stringify::v0::lowercase;
    }
};


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_FACETS_CASE_HPP

