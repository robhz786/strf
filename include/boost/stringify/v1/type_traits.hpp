#ifndef BOOST_STRINGIFY_V1_TYPE_TRAITS_HPP
#define BOOST_STRINGIFY_V1_TYPE_TRAITS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>

namespace boost {
namespace stringify {
inline namespace v1 {
namespace detail {

// ternary_trait

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


// ternary_t

template <bool Condition, typename ThenType, typename ElseType>
using ternary_t = typename ternary_trait<Condition, ThenType, ElseType>::type;


// has_arg_format_type

template <typename Stringifier, typename = typename Stringifier::arg_format_type>
auto has_arg_format_type_helper(const Stringifier*) -> std::true_type;

template <typename Stringifier>
auto has_arg_format_type_helper(...)  ->  std::false_type;

template <typename Stringifier>
using has_arg_format_type
= decltype(boost::stringify::v1::detail::has_arg_format_type_helper<Stringifier>
           ((Stringifier*)0));


} // namespace detail


template <typename> struct true_trait : public std::true_type
{
};


} // inline namespace v1
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_V1_TYPE_TRAITS_HPP

