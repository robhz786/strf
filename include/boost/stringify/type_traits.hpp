#ifndef BOOST_STRINGIFY_TYPE_TRAITS_HPP
#define BOOST_STRINGIFY_TYPE_TRAITS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

namespace boost {
namespace stringify {
namespace detail {

template <bool Condition, typename ThenType, typename ElseType>
struct ternary_trait
{
    typedef ThenType type;
};

template <typename ThenType, typename ElseType>
struct ternary_trait<false, ThenType, ElseType>
{
    typedef ElseType type;
};

template <bool Condition, typename ThenType, typename ElseType>
using ternary_t = typename ternary_trait<Condition, ThenType, ElseType>::type;

} // namespace detail

template <typename = void> struct accept_any_type : public std::true_type
{
};

} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_TYPE_TRAITS_HPP

