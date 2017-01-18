#ifndef BOOST_STRINGIFY_FMT_SHOWPOS_HPP
#define BOOST_STRINGIFY_FMT_SHOWPOS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/type_traits.hpp>

namespace boost {
namespace stringify {

struct showpos_tag;

template <template <class> class Filter = boost::stringify::true_trait>
struct showpos_impl
{
    template <typename T> using accept_input_type = Filter<T>;

    typedef boost::stringify::showpos_tag category;
    
    constexpr showpos_impl(bool show) : m_show(show)
    {
    }
    
    constexpr bool show() const
    {
        return m_show;
    }
    
    bool m_show;
};


template <template <class> class Filter>
struct showpos_true_impl
{
    template <typename T> using accept_input_type = Filter<T>;

    typedef boost::stringify::showpos_tag category;

    constexpr bool show() const { return true; }

    constexpr boost::stringify::showpos_impl<Filter> operator()(bool s) const
    {
        return boost::stringify::showpos_impl<Filter>(s);
    }

    constexpr boost::stringify::showpos_true_impl<Filter> operator()() const
    {
        return *this;
    }
    
};


template <template <class> class Filter>
struct showpos_false_impl
{
    template <typename T> using accept_input_type = Filter<T>;

    typedef boost::stringify::showpos_tag category;

    constexpr bool show() const { return false; }

    constexpr boost::stringify::showpos_false_impl<Filter> operator()() const
    {
        return *this;
    }
};


constexpr auto showpos
= boost::stringify::showpos_true_impl<boost::stringify::true_trait>();

constexpr auto noshowpos
= boost::stringify::showpos_false_impl<boost::stringify::true_trait>();

template <template <class> class F>
boost::stringify::showpos_true_impl<F> showpos_if
= boost::stringify::showpos_true_impl<F>();

template <template <class> class F>
boost::stringify::showpos_false_impl<F> noshowpos_if
= boost::stringify::showpos_false_impl<F>();


struct showpos_tag
{
    typedef
        boost::stringify::showpos_false_impl<boost::stringify::true_trait>
        default_impl;
};


template <typename InputType, typename Formatting>
bool get_showpos(const Formatting& fmt) noexcept
{
    return fmt.template get<boost::stringify::showpos_tag, InputType>().show();
}


} // namespace stringify
} // namespace boost


#endif  /* BOOST_STRINGIFY_FMT_SHOWPOS_HPP */

