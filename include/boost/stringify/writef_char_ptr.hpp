#ifndef BOOST_STRINGIFY_WRITEF_CHAR_PTR_HPP
#define BOOST_STRINGIFY_WRITEF_CHAR_PTR_HPP

#include <string>
//#include <boost/assert.hpp>
#include <boost/stringify/listf.hpp>
#include <boost/stringify/formater_tuple.hpp>

namespace boost
{
namespace stringify
{
/*
template <typename charT, typename ... Formaters, typename ... Args>
auto writef_impl
    ( charT* output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const Args& ... args
    )
    -> typename std::char_traits<charT>::char_type*;
           

template <typename charT, typename ... Formaters, typename Arg>
auto writef_impl
    ( charT* output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const Arg& arg
    )
    -> typename std::char_traits<charT>::char_type*
{
    return arg.write(output, fmt);
}


template <typename charT, typename ... Formaters, typename ... Args, typename Arg>
auto writef_impl
    ( charT* output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const Arg& arg
    , const Args& ... args
    )
    -> typename std::char_traits<charT>::char_type*
{
    return boost::stringify::writef_impl(arg.write(output, fmt), fmt, args...);
}
*/
/*
template
    < typename charT
    , int N
    , typename ... Formaters
    , typename Arg
    , typename ... Args
    >
auto writef_impl
    ( charT (&)[N] output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const Arg& arg
    , const Args& ... args
    )
    -> typename std::char_traits<charT>::char_type*
{
    //BOOST_ASSERT(N > boost::stringify::length(fmt, arg, args...));
    return arg.write
        ( boost::stringify::write_impl((charT*)output, fmt, args...)
        , fmt
        );
}
*/

template <typename charT, typename ... Formaters>
auto writef_il
    ( charT* output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::listf
        < charT
        , typename std::decay<decltype(fmt)>::type
        > & arg_list
    )
    -> typename std::char_traits<charT>::char_type*
{
    return arg_list.write(output, fmt);
}

namespace detail{
char     char_type_of(char*);
char16_t char_type_of(char16_t*);
char32_t char_type_of(char32_t*);
wchar_t  char_type_of(wchar_t*);
void     char_type_of(...);
}


template <typename charT, typename ... Formaters>
auto writef
    ( charT* output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < decltype(boost::stringify::detail::char_type_of(output))
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    )
    -> typename std::char_traits<charT>::char_type*
{
    return arg1.write(output, fmt);
}


template <typename charT, typename ... Formaters>
auto writef
    ( charT* output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < decltype(boost::stringify::detail::char_type_of(output))
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    )
    -> typename std::char_traits<charT>::char_type* 
{
    return arg2.write(arg1.write(output, fmt), fmt);
}


template <typename charT, typename ... Formaters>
auto writef
    ( charT* output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < decltype(boost::stringify::detail::char_type_of(output))
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    )
    -> typename std::char_traits<charT>::char_type*
{
    return arg3.write(arg2.write(arg1.write(output, fmt), fmt), fmt);
}


template <typename charT, typename ... Formaters>
auto writef
    ( charT* output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < decltype(boost::stringify::detail::char_type_of(output))
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    , decltype(arg1) arg4
    )
    -> typename std::char_traits<charT>::char_type*
{
    return boost::stringify::writef
        ( boost::stringify::writef(output, fmt, arg1, arg2)
        , fmt, arg3, arg4
        );
}

   
template <typename charT, typename ... Formaters>
auto writef
    ( charT* output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < decltype(boost::stringify::detail::char_type_of(output))
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    , decltype(arg1) arg4
    , decltype(arg1) arg5      
    )
    -> typename std::char_traits<charT>::char_type*
{
    return boost::stringify::writef
        ( boost::stringify::writef(output, fmt, arg1, arg2)
        , fmt, arg3, arg4, arg5
        );
}


template <typename charT, typename ... Formaters>
auto writef
    ( charT* output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < decltype(boost::stringify::detail::char_type_of(output))
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    , decltype(arg1) arg4
    , decltype(arg1) arg5
    , decltype(arg1) arg6
    )
    -> typename std::char_traits<charT>::char_type*
{
    return boost::stringify::writef
        ( boost::stringify::writef(fmt, arg1, arg2)
        , fmt, arg3, arg4, arg5, arg6
        );
}


} // namespace stringify
} // namespace boost    

#endif  /* BOOST_STRINGIFY_WRITEF_CHAR_PTR_HPP */

