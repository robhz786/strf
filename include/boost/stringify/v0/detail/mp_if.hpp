#ifndef BOOST_STRINGIFY_V0_DETAIL_MP_IF_HPP
#define BOOST_STRINGIFY_V0_DETAIL_MP_IF_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/v0/config.hpp>

// TODO: use mp11::mp_if instead when available

BOOST_STRINGIFY_V0_NAMESPACE_BEGIN
namespace detail {


template <bool Condition, typename ThenType, typename ElseType>
struct mp_if_impl
{
    typedef ThenType type;
};


template <typename ThenType, typename ElseType>
struct mp_if_impl<false, ThenType, ElseType>
{
    typedef ElseType type;
};


template <bool Condition, typename ThenType, typename ElseType>
using mp_if = typename mp_if_impl<Condition, ThenType, ElseType>::type;


} // namespace detail

BOOST_STRINGIFY_V0_NAMESPACE_END

#endif  // BOOST_STRINGIFY_V0_TYPE_TRAITS_HPP

