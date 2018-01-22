#ifndef BOOST_STRINGIFY_V0_CONSTRAINED_FACET_HPP
#define BOOST_STRINGIFY_V0_CONSTRAINED_FACET_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/config.hpp>
#include <functional>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

template <template <class> class Filter, typename Facet>
class constrained_facet;

namespace detail
{

template <template <class> class ParentFilter, typename F>
struct constrained_facet_helper
{
    constexpr static const auto& get(const F& f)
    {
        return f;
    }

    template <typename InputType>
    using matches = ParentFilter<InputType>;

    using category = typename F::category;
};


template <template <class> class ParentFilter, typename F>
struct constrained_facet_helper<ParentFilter, std::reference_wrapper<F>>
{
    using helper
        = stringify::v0::detail::constrained_facet_helper<ParentFilter, F>;

    constexpr static const auto& get(std::reference_wrapper<F> f)
    {
        return helper::get(f.get());
    }

    template <typename InputType>
    using matches = typename helper::template matches<InputType>;

    using category = typename helper::category;
};


template
    < template <class> class ParentFilter
    , template <class> class Filter
    , typename F
    >
struct constrained_facet_helper
    < ParentFilter
    , stringify::v0::constrained_facet<Filter, F>
    >
{
    using constrained_f = stringify::v0::constrained_facet<Filter, F>;
    using helper
        = stringify::v0::detail::constrained_facet_helper<ParentFilter, F>;

    constexpr static const auto& get(const constrained_f& f)
    {
        return helper::get(f.m_facet);
    }

    template <typename T>
    using matches = std::integral_constant
        < bool
        , Filter<T>::value && helper::template matches<T>::value
        >;

    using category = typename helper::category;
};


} // namespace detail


template <template <class> class Filter, typename Facet>
class constrained_facet
{
    using helper =
        stringify::v0::detail::constrained_facet_helper<Filter, Facet>;

public:

    template <typename InputType>
    using matches = typename helper::template matches<InputType>;

    using category = typename helper::category;

    constrained_facet() = default;

    constrained_facet(const constrained_facet&) = default;

    constrained_facet(constrained_facet&&) = default;

    constrained_facet(const Facet& f)
        : m_facet(f)
    {
    }

    constexpr const auto& get_facet() const
    {
        return helper::get(m_facet);
    }

private:

    template <template <class> class, class>
    friend struct stringify::v0::detail::constrained_facet_helper;

    Facet m_facet;
};

template <template <class> class Filter, typename Facet>
auto constrain(const Facet& facet)
{
    return constrained_facet<Filter, Facet>(facet);
}


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_CONSTRAINED_FACET_HPP

