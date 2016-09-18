#ifndef BOOST_STRINGIFY_LENGTHF_HPP
#define BOOST_STRINGIFY_LENGTHF_HPP

#include <boost/stringify/formater_tuple.hpp>
#include <boost/stringify/listf.hpp>

namespace boost {
namespace stringify {


template <typename ... Formaters>
std::size_t lengthf_il
    ( const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::listf
        < char
        , std::char_traits<char>
        , typename std::decay<decltype(fmt)>::type
        > & arg_list
    )
{
    return arg_list.length(fmt);
}


template <typename ... Formaters>
std::size_t lengthf
    ( const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < char
        , std::char_traits<char>
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    )
{
    return arg1.length(fmt);
}


template <typename ... Formaters>
std::size_t lengthf
    ( const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < char
        , std::char_traits<char>
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    )
{
    return arg1.length(fmt) + arg2.length(fmt);
}


template <typename ... Formaters>
std::size_t lengthf
    ( const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < char
        , std::char_traits<char>
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    )
{
    return arg1.length(fmt) + arg2.length(fmt) + arg3.length(fmt);
}


template <typename ... Formaters>
std::size_t lengthf
    ( const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < char
        , std::char_traits<char>
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    , decltype(arg1) arg4
    )
{
    return
        arg1.length(fmt) + arg2.length(fmt) + arg3.length(fmt) +
        arg4.length(fmt);
}


template <typename ... Formaters>
std::size_t lengthf
    ( const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < char
        , std::char_traits<char>
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    , decltype(arg1) arg4
    , decltype(arg1) arg5
    )
{
    return 
        arg1.length(fmt) + arg2.length(fmt) + arg3.length(fmt) +
        arg4.length(fmt) + arg5.length(fmt);
}


template <typename ... Formaters>
std::size_t lengthf
    ( const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < char
        , std::char_traits<char>
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    , decltype(arg1) arg4
    , decltype(arg1) arg5
    , decltype(arg1) arg6
    )
{
    return
        arg1.length(fmt) + arg2.length(fmt) + arg3.length(fmt) +
        arg4.length(fmt) + arg5.length(fmt) + arg6.length(fmt);
}


template <typename ... Formaters>
std::size_t lengthf
    ( const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < char
        , std::char_traits<char>
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    , decltype(arg1) arg4
    , decltype(arg1) arg5
    , decltype(arg1) arg6
    , decltype(arg1) arg7
    )
{
    return
        arg1.length(fmt) + arg2.length(fmt) + arg3.length(fmt) +
        arg4.length(fmt) + arg5.length(fmt) + arg6.length(fmt) +
        arg7.length(fmt);
}


template <typename ... Formaters>
std::size_t lengthf
    ( const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < char
        , std::char_traits<char>
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    , decltype(arg1) arg4
    , decltype(arg1) arg5
    , decltype(arg1) arg6
    , decltype(arg1) arg7
    , decltype(arg1) arg8
    )
{
    return
        arg1.length(fmt) + arg2.length(fmt) + arg3.length(fmt) +
        arg4.length(fmt) + arg5.length(fmt) + arg6.length(fmt) +
        arg7.length(fmt) + arg8.length(fmt);
}


template <typename ... Formaters>
std::size_t lengthf
    ( const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < char
        , std::char_traits<char>
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    , decltype(arg1) arg4
    , decltype(arg1) arg5
    , decltype(arg1) arg6
    , decltype(arg1) arg7
    , decltype(arg1) arg8
    , decltype(arg1) arg9
    )
{
    return
        arg1.length(fmt) + arg2.length(fmt) + arg3.length(fmt) +
        arg4.length(fmt) + arg5.length(fmt) + arg6.length(fmt) +
        arg7.length(fmt) + arg8.length(fmt) + arg9.length(fmt);
}


template <typename ... Formaters>
std::size_t lengthf
    ( const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < char
        , std::char_traits<char>
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    , decltype(arg1) arg4
    , decltype(arg1) arg5
    , decltype(arg1) arg6
    , decltype(arg1) arg7
    , decltype(arg1) arg8
    , decltype(arg1) arg9
    , decltype(arg1) arg10
    )
{
    return
        arg1.length(fmt) + arg2.length(fmt) + arg3.length(fmt) +
        arg4.length(fmt) + arg5.length(fmt) + arg6.length(fmt) +
        arg7.length(fmt) + arg8.length(fmt) + arg9.length(fmt) +
        arg10.length(fmt);
}


} //namespace stringify
} //namespace boost

#endif  /* BOOST_STRINGIFY_LENGTHF_HPP */

