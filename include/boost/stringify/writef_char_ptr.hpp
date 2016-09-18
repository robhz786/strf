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


template
    < typename charT
    , typename charTraits
    , typename ... Formaters
    >
charT* basic_writef_il
    ( charT* output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::listf
        < charT
        , charTraits
        , typename std::decay<decltype(fmt)>::type
        > & arg_list
    )
{
    return arg_list.write(output, fmt);
}


template
    < typename charT
    , typename charTraits
    , typename ... Formaters
    >
charT* basic_writef
    ( charT* output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < charT
        , charTraits
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    )
{
    return arg1.write(output, fmt);
}


} // namespace stringify
} // namespace boost    

#endif  /* BOOST_STRINGIFY_WRITEF_CHAR_PTR_HPP */

