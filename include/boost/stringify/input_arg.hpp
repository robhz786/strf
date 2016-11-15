#ifndef BOOST_STRINGIFY_INPUT_ARG_HPP
#define BOOST_STRINGIFY_INPUT_ARG_HPP

#include <boost/stringify/input_base.hpp>

namespace boost {
namespace stringify {

template <typename charT, typename Output, typename Formating>
class input_arg
{
public:

    template <class T>
    using input_base_of
    = decltype(argf<charT, Output, Formating>(std::declval<const T>()));

    template <typename T>
    input_arg
        ( const T& value
        , input_base_of<T> && wt = input_base_of<T>() // will be input_base_of<T>(value)
                                                      // after P0145R2 is supported
        )
        noexcept
        : writer(wt)
    {
        wt.set(value); // will be removed after compilers add support to P0145R2
    }
   
    template <typename T>
    input_arg
        ( const T& arg
        , const typename input_base_of<T>::local_formatting& arg_format
        , input_base_of<T> && wt = input_base_of<T>() // will be input_base_of<T>(arg, arg_format)
        ) noexcept
        : writer(wt)
    {
        wt.set(arg, arg_format);  // will be removed after compilers add support to P0145R2
    }

    const boost::stringify::input_base<charT, Output, Formating>& writer;
};


} // namespace stringify
} // namespace boost

#endif  /* BOOST_STRINGIFY_DETAIL_STR_WRITE_REF_HPP */

