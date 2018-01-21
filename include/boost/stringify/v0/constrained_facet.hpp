#ifndef BOOST_STRINGIFY_V0_CONSTRAINED_FACET_HPP
#define BOOST_STRINGIFY_V0_CONSTRAINED_FACET_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/config.hpp>
#include <functional>

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
        : m_facet(f)
    {
    }

    constexpr Facet& get_facet()
    {
        return m_facet;
    }

    constexpr const Facet& get_facet() const
    {
        return m_facet;
    }

private:

    Facet m_facet;
};

template <template <class> class Filter, typename Facet>
class constrained_facet<Filter, std::reference_wrapper<Facet>>
{
public:

    template <typename InputType>
    using matches = Filter<InputType>;

    using category = typename Facet::category;

    constrained_facet() = default;

    constrained_facet(const constrained_facet&) = default;

    constrained_facet(constrained_facet&&) = default;

    constrained_facet(std::reference_wrapper<Facet> f)
        : m_ref_facet(f)
    {
    }

    constexpr const Facet& get_facet() const
    {
        return m_ref_facet.get();
    }

private:

    std::reference_wrapper<const Facet> m_ref_facet;
};

template
    < template <class> class FilterA
    , template <class> class FilterB
    , typename Facet
    >
class constrained_facet
    < FilterA
    , boost::stringify::v0::constrained_facet<FilterB, Facet>
    >
{
    using other_constrained_facet
        = boost::stringify::v0::constrained_facet<FilterB, Facet>;

public:

    template <typename InputType>
    struct matches
        : public std::integral_constant
            < bool
            , FilterA<InputType>::value && FilterB<InputType>::value
            >
    {
    };

    using category = typename other_constrained_facet::category;

    constrained_facet() = default;

    constrained_facet(const constrained_facet&) = default;

    constrained_facet(constrained_facet&&) = default;

    constrained_facet(const other_constrained_facet& cf)
        : m_cfacet(cf)
    {
    }

    constexpr const Facet& get_facet() const
    {
        return m_cfacet.get_facet();
    }

private:

    other_constrained_facet m_cfacet;
};


template <template <class> class Filter, typename Facet>
auto constrain(const Facet& facet)
{
    return constrained_facet<Filter, Facet>(facet);
}


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_CONSTRAINED_FACET_HPP

