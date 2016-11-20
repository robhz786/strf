#ifndef BOOST_STRINGIFY_TYPE_TRAITS_HPP
#define BOOST_STRINGIFY_TYPE_TRAITS_HPP

namespace boost {
namespace stringify {

template <typename = void> struct accept_any_type : public std::true_type
{
};

namespace detail
{

// TODO propose ternary_trait to Boost.Traits
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
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_TYPE_TRAITS_HPP

