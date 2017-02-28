#ifndef BOOST_STRINGIFY_V1_CUSTOM_CASE_HPP
#define BOOST_STRINGIFY_V1_CUSTOM_CASE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

namespace boost {
namespace stringify {
inline namespace v1 {

struct case_tag;

template <bool Uppercase, template <class> class Filter>
struct case_impl_t
{
    using category = boost::stringify::v1::case_tag;

    template <typename T> using accept_input_type = Filter<T>;
    
    constexpr bool uppercase() const
    {
        return Uppercase;
    }
};

constexpr auto uppercase =
    boost::stringify::v1::case_impl_t<true, boost::stringify::v1::true_trait>();

constexpr auto  lowercase =
    boost::stringify::v1::case_impl_t<false, boost::stringify::v1::true_trait>();

template <template <class> class F>
constexpr auto uppercase_if = boost::stringify::v1::case_impl_t<true, F>();

template <template <class> class F>
constexpr auto lowercase_if = boost::stringify::v1::case_impl_t<false, F>();


struct case_tag
{
    using default_impl =
        boost::stringify::v1::case_impl_t<false, boost::stringify::v1::true_trait>;
};


} // inline namespace v1
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_V1_CUSTOM_CASE_HPP

