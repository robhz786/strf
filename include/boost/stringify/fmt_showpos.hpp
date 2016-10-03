#ifndef BOOST_STRINGIFY_FMT_SHOWPOS_HPP
#define BOOST_STRINGIFY_FMT_SHOWPOS_HPP

#include <type_traits>

namespace boost {
namespace stringify {

template <typename T>
 struct is_anytype: public std::true_type
 {
 };


struct ftype_showpos;

template <bool ShowIt, template <class> class Filter = std::is_object>
struct static_showpos
{
    template <typename T> using accept_input_type = Filter<T>;

    typedef boost::stringify::ftype_showpos fmt_type;

    constexpr bool show() const { return ShowIt; }
};

template <template <class> class Filter = std::is_object>
decltype(auto) noshowpos()
{
    return boost::stringify::static_showpos<false, Filter>();
}

template <template <class> class Filter = std::is_object>
decltype(auto) showpos()
{
    return boost::stringify::static_showpos<true, Filter>();
}

template <template <class> class Filter = std::is_object>
struct dyn_showpos
{
    template <typename T> using accept_input_type = Filter<T>;

    typedef boost::stringify::ftype_showpos fmt_type;
    
    dyn_showpos(bool show) : m_show(show)
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
    typedef static_showpos<false> default_impl;
};


} // namespace stringify
} // namespace boost


#endif  /* BOOST_STRINGIFY_FMT_SHOWPOS_HPP */

