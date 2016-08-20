#ifndef BOOST_STRINGIFY_DETAIL_STR_WRITE_REF_HPP
#define BOOST_STRINGIFY_DETAIL_STR_WRITE_REF_HPP

#include <boost/stringify/str_writer.hpp>

namespace boost {
namespace stringify {
namespace detail {

template <typename charT, typename Formating>
struct str_writer_ref
{
    str_writer_ref
        ( const boost::stringify::str_writer<charT, Formating>& w
        )
        noexcept
        : writer(w)
    {
    }

    template <class T>
    using str_writer_of
    = decltype(argf<charT, Formating>(std::declval<const T>()));
           
    template <typename T, typename ... ExtraArgs>
    str_writer_ref
        ( const T& value
        , str_writer_of<T> && wt = str_writer_of<T>()
        )
        noexcept
        : writer(wt)
    {
        wt.set(value);
    }

    /*   
    template <typename T, typename ... ExtraArgs>
    str_writer_ref
        ( const T& value
        , const typename str_writer_of<T>::local_formating& fmt
        , str_writer_of<T> && wt = str_writer_of<T, const char*>()
        ) noexcept
        : writer(wt)
    {
        wt.set(value, fmt);
    }
    */
    
    const boost::stringify::str_writer<charT, Formating>& writer;
};


} // namespace detail
} // namespace stringify
} // namespace boost

#endif  /* BOOST_STRINGIFY_DETAIL_STR_WRITE_REF_HPP */

