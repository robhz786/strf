#ifndef BOOST_STRINGIFY_V0_FACETS_SHOWBASE_HPP
#define BOOST_STRINGIFY_V0_FACETS_SHOWBASE_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

namespace boost {
namespace stringify {
inline namespace v0 {

struct showbase_tag;

template <bool ShowBase, template <class> class Filter>
struct showbase_impl_t
{
    using category = boost::stringify::v0::showbase_tag;

    template <typename T> using accept_input_type = Filter<T>;
    
    constexpr bool value() const
    {
        return ShowBase;
    }
};

constexpr auto showbase =
    boost::stringify::v0::showbase_impl_t<true, boost::stringify::v0::true_trait>();

constexpr auto noshowbase =
    boost::stringify::v0::showbase_impl_t<false, boost::stringify::v0::true_trait>();

template <template <class> class F>
auto showbase_if = boost::stringify::v0::showbase_impl_t<true, F>();

template <template <class> class F>
auto noshowbase_if = boost::stringify::v0::showbase_impl_t<false, F>();


struct showbase_tag
{
    constexpr static const auto& get_default() noexcept
    {
        return boost::stringify::v0::noshowbase;
    }
};


} // inline namespace v0
} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_V0_FACETS_SHOWBASE_HPP

