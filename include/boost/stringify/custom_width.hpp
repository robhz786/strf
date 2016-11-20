#ifndef BOOST_STRINGIFY_FMT_WIDTH_HPP
#define BOOST_STRINGIFY_FMT_WIDTH_HPP

#include <boost/stringify/type_traits.hpp>
#include <boost/stringify/width_t.hpp>

namespace boost {
namespace stringify {

struct width_tag;
template <typename CharT> struct filler_tag;

template <template <class> class Filter>
struct fimpl_width
{
    typedef boost::stringify::width_tag category;
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
    typedef boost::stringify::width_tag category;
    template <typename T> using accept_input_type = Filter<T>;    

    constexpr boost::stringify::width_t width() const noexcept
    {
        return 0;
    }
};

struct width_tag
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

template <typename InputType, typename Formatting>
boost::stringify::width_t get_width(const Formatting& fmt) noexcept
{
    return fmt.template get<boost::stringify::width_tag, InputType>().width();
}

} // namespace stringify
} // namespace boost

#endif  // BOOST_STRINGIFY_FMT_WIDTH_HPP

