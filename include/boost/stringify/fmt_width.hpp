#ifndef BOOST_STRINGIFY_FMT_WIDTH_HPP
#define BOOST_STRINGIFY_FMT_WIDTH_HPP

#include <boost/stringify/type_traits.hpp>
#include <boost/stringify/width_t.hpp>

namespace boost {
namespace stringify {

struct ftype_width;
template <typename charT> struct ftype_filler;

template <template <class> class Filter>
struct fimpl_width
{
    typedef boost::stringify::ftype_width fmt_type;
    template <typename T> using accept_input_type = Filter<T>;    

    constexpr fimpl_width(boost::stringify::width_t v = 0)
        : m_value(v)
    {
    }

    constexpr fimpl_width(const fimpl_width& w) = default;

    constexpr boost::stringify::width_t width() const noexcept
    {
        return m_value;
    }
    
private:
    
    boost::stringify::width_t m_value;
};

template <template <class> class Filter>
struct fimpl_default_width
{
    typedef boost::stringify::ftype_width fmt_type;
    template <typename T> using accept_input_type = Filter<T>;    

    constexpr boost::stringify::width_t width() const noexcept
    {
        return 0;
    }
};

struct ftype_width
{
    typedef
        boost::stringify::fimpl_default_width<boost::stringify::accept_any_type>
        default_impl;
};

template <template <class> class Filter = boost::stringify::accept_any_type>
constexpr auto width(boost::stringify::width_t w) noexcept
{
    return boost::stringify::fimpl_width<Filter>(w);
}

} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_FMT_WIDTH_HPP

