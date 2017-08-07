#ifndef BOOST_STRINGIFY_V0_FACETS_INTBASE_HPP
#define BOOST_STRINGIFY_V0_FACETS_INTBASE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

struct intbase_tag;

template <int Base>
struct intbase_impl_t
{
    using category = boost::stringify::v0::intbase_tag;

    constexpr int value() const
    {
        return Base;
    }
};

constexpr boost::stringify::v0::intbase_impl_t<8> oct {};

constexpr boost::stringify::v0::intbase_impl_t<10> dec {};

constexpr boost::stringify::v0::intbase_impl_t<16> hex {};

template <template <class> class F>
constrained_facet<F, intbase_impl_t<8> > oct_if {};

template <template <class> class F>
constrained_facet<F, intbase_impl_t<10> > dec_if {};

template <template <class> class F>
constrained_facet<F, intbase_impl_t<16> > hex_if {};

struct intbase_tag
{
    constexpr static const auto& get_default() noexcept
    {
        return boost::stringify::v0::dec;
    }
};

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_FACETS_INTBASE_HPP

