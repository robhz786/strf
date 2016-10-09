#ifndef BOOST_STRINGIFY_TYPE_TRAITS_HPP
#define BOOST_STRINGIFY_TYPE_TRAITS_HPP

namespace boost {
namespace stringify {

template <typename = void> struct accept_any_type : public std::true_type
{
};


} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_TYPE_TRAITS_HPP

