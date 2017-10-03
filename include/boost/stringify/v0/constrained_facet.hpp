#ifndef BOOST_STRINGIFY_V0_CONSTRAINED_FACET_HPP
#define BOOST_STRINGIFY_V0_CONSTRAINED_FACET_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/config.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <template <class> class Filter, typename Facet>
class constrained_facet
{
public:

    template <typename InputType>
    using matches = Filter<InputType>;

    using category = typename Facet::category;

    constrained_facet() = default;

    constrained_facet(const constrained_facet&) = default;
    
    constrained_facet(constrained_facet&&) = default;
    
    constrained_facet(const Facet& f)
        : facet(f)
    {
    }

    Facet facet;
};


template <template <class> class Filter, typename Facet>
auto constrain(const Facet& facet)
{
    return constrained_facet<Filter, Facet>(facet);
}


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_CONSTRAINED_FACET_HPP

