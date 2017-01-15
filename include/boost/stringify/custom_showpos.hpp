#ifndef BOOST_STRINGIFY_FMT_SHOWPOS_HPP
#define BOOST_STRINGIFY_FMT_SHOWPOS_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/stringify/type_traits.hpp>

namespace boost {
namespace stringify {

struct showpos_tag;

template <bool ShowIt, template <class> class Filter>
struct fimpl_static_showpos
{
    template <typename T> using accept_input_type = Filter<T>;

    typedef boost::stringify::showpos_tag category;

    constexpr bool show() const { return ShowIt; }
};

template <template <class> class Filter = boost::stringify::accept_any_type>
constexpr auto noshowpos = fimpl_static_showpos<false, Filter>();

template <template <class> class Filter = boost::stringify::accept_any_type>
constexpr auto showpos = fimpl_static_showpos<true, Filter>();

template <template <class> class Filter = boost::stringify::accept_any_type>
struct fimpl_dyn_showpos
{
    template <typename T> using accept_input_type = Filter<T>;

    typedef boost::stringify::showpos_tag category;
    
    fimpl_dyn_showpos(bool show) : m_show(show)
    {
    }
    
    constexpr bool show() const
    {
        return m_show;
    }
    
    bool m_show;
};

struct showpos_tag
{
    typedef
        boost::stringify::fimpl_static_showpos
        <false, boost::stringify::accept_any_type>
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

