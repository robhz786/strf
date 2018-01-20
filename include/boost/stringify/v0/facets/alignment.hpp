#ifndef BOOST_STRINGIFY_V0_FACETS_ALIGNMENT_HPP
#define BOOST_STRINGIFY_V0_FACETS_ALIGNMENT_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/config.hpp>
#include <boost/stringify/v0/constrained_facet.hpp>

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN

enum class alignment{left, right, internal, center};

struct alignment_tag;

struct align_impl
{
    typedef boost::stringify::v0::alignment_tag category;

    constexpr align_impl(boost::stringify::v0::alignment align)
        : m_align(align)
    {
    }

    constexpr boost::stringify::v0::alignment value() const
    {
        return m_align;
    }

private:

    const boost::stringify::v0::alignment m_align;
};


template <boost::stringify::v0::alignment Align>
struct align_impl_t
{
    typedef boost::stringify::v0::alignment_tag category;

    constexpr boost::stringify::v0::alignment value() const
    {
        return Align;
    }
};


constexpr auto align(boost::stringify::v0::alignment a)
{
    return boost::stringify::v0::align_impl{a};
}


template < template <class> class F>
constexpr auto align_if(boost::stringify::v0::alignment a)
{
    return constrained_facet<F, align_impl>{a};
}


constexpr align_impl_t <alignment::left> left {};
constexpr align_impl_t <alignment::right> right {};
constexpr align_impl_t <alignment::internal> internal {};
constexpr align_impl_t <alignment::center> center {};

template <template <class> class F>
constrained_facet<F, align_impl_t<alignment::left> > left_if {};

template <template <class> class F>
constrained_facet<F, align_impl_t<alignment::right> > right_if {};

template <template <class> class F>
constrained_facet<F, align_impl_t<alignment::internal> > internal_if {};

template <template <class> class F>
constrained_facet<F, align_impl_t<alignment::center> > center_if {};

struct alignment_tag
{
    constexpr static const auto& get_default() noexcept
    {
        return boost::stringify::v0::right;
    }
};


BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_FACETS_ALIGNMENT_HPP

