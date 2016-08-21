#ifndef BOOST_STRINGIFY_DETAIL_STR_WRITE_REF_HPP
#define BOOST_STRINGIFY_DETAIL_STR_WRITE_REF_HPP

#include <boost/stringify/input_base.hpp>

namespace boost {
namespace stringify {
namespace detail {

template <typename charT, typename Formating>
struct input_base_ref
{
    input_base_ref
        ( const boost::stringify::input_base<charT, Formating>& w
        )
        noexcept
        : writer(w)
    {
    }

    template <class T>
    using input_base_of
    = decltype(argf<charT, Formating>(std::declval<const T>()));
           
    template <typename T, typename ... ExtraArgs>
    input_base_ref
        ( const T& value
        , input_base_of<T> && wt = input_base_of<T>()
        )
        noexcept
        : writer(wt)
    {
        wt.set(value);
    }

    /*   
    template <typename T, typename ... ExtraArgs>
    input_base_ref
        ( const T& value
        , const typename input_base_of<T>::local_formating& fmt
        , input_base_of<T> && wt = input_base_of<T, const char*>()
        ) noexcept
        : writer(wt)
    {
        wt.set(value, fmt);
    }
    */
    
    const boost::stringify::input_base<charT, Formating>& writer;
};


} // namespace detail
} // namespace stringify
} // namespace boost

#endif  /* BOOST_STRINGIFY_DETAIL_STR_WRITE_REF_HPP */

