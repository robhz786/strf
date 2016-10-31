#ifndef BOOST_STRINGIFY_TYPE_TRAITS_HPP
#define BOOST_STRINGIFY_TYPE_TRAITS_HPP

namespace boost {
namespace stringify {

template <typename = void> struct accept_any_type : public std::true_type
{
};

namespace detail
{

// TODO propose if_else_type to Boost.Traits
template <bool Condition, typename ThenType, typename ElseType>
struct if_else_type
{
    typedef ThenType type;
};

template <typename ThenType, typename ElseType>
struct if_else_type<false, ThenType, ElseType>
{
    typedef ElseType type;
};

} // namespace detail
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_TYPE_TRAITS_HPP

