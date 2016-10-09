#ifndef BOOST_STRINGIFY_FMT_SHOWPOS_HPP
#define BOOST_STRINGIFY_FMT_SHOWPOS_HPP

#include <boost/stringify/type_traits.hpp>

namespace boost {
namespace stringify {

struct ftype_showpos;

template <bool ShowIt, template <class> class Filter>
struct fimpl_static_showpos
{
    template <typename T> using accept_input_type = Filter<T>;

    typedef boost::stringify::ftype_showpos fmt_type;

    constexpr bool show() const { return ShowIt; }
};

template <template <class> class Filter = boost::stringify::accept_any_type>
constexpr fimpl_static_showpos<false, Filter> noshowpos
  = fimpl_static_showpos<false, Filter>();

template <template <class> class Filter = boost::stringify::accept_any_type>
constexpr fimpl_static_showpos<true, Filter> showpos
  = fimpl_static_showpos<true, Filter>();

template <template <class> class Filter = boost::stringify::accept_any_type>
struct fimpl_dyn_showpos
{
    template <typename T> using accept_input_type = Filter<T>;

    typedef boost::stringify::ftype_showpos fmt_type;
    
    fimpl_dyn_showpos(bool show) : m_show(show)
    {
    }
    
    constexpr bool show() const
    {
        return m_show;
    }
    
    bool m_show;
};

struct ftype_showpos
{
    typedef
        fimpl_static_showpos<false, boost::stringify::accept_any_type>
        default_impl;
};


} // namespace stringify
} // namespace boost


#endif  /* BOOST_STRINGIFY_FMT_SHOWPOS_HPP */

