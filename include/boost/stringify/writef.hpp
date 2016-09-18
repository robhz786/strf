#ifndef BOOST_STRINGIFY_WRITEF_HPP
#define BOOST_STRINGIFY_WRITEF_HPP

#include <boost/stringify/writef_char_ptr.hpp>

namespace boost {
namespace stringify {

template <typename Output, typename ... Formaters>
auto writef_il
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::listf
        < char
        , std::char_traits<char>
        , typename std::decay<decltype(fmt)>::type
        > & arg_list
    )
    -> decltype
        ( basic_writef_il<char, std::char_traits<char>, Formaters...>
            (output, fmt, arg_list)
        )
{
    return basic_writef_il<char, std::char_traits<char>, Formaters...>
        (output, fmt, arg_list);
}


template <typename Output, typename ... Formaters>
auto writef16_il
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::listf
        < char16_t
        , std::char_traits<char16_t>
        , typename std::decay<decltype(fmt)>::type
        > & arg_list
    )
    -> decltype
        ( basic_writef_il<char16_t, std::char_traits<char16_t>, Formaters...>
            (output, fmt, arg_list)
        )
{
    return basic_writef_il<char16_t, std::char_traits<char16_t>, Formaters...>
        (output, fmt, arg_list);
}


template <typename Output, typename ... Formaters>
auto writef32_il
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::listf
        < char32_t
        , std::char_traits<char32_t>
        , typename std::decay<decltype(fmt)>::type
        > & arg_list
    )
    -> decltype
        ( basic_writef_il<char32_t, std::char_traits<char32_t>, Formaters...>
            (output, fmt, arg_list)
        )
{
    return basic_writef_il<char32_t, std::char_traits<char32_t>, Formaters...>
        (output, fmt, arg_list);
}


template <typename Output, typename ... Formaters>
auto wwritef_il
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::listf
        < wchar_t
        , std::char_traits<wchar_t>
        , typename std::decay<decltype(fmt)>::type
        > & arg_list
    )
    -> decltype
        ( basic_writef_il<wchar_t, std::char_traits<wchar_t>, Formaters...>
            (output, fmt, arg_list)
        )
{
    return basic_writef_il<wchar_t, std::char_traits<wchar_t>, Formaters...>
        (output, fmt, arg_list);
}


template
    < typename charType
    , typename charTraits
    , typename Output
    , typename ... Formaters
    >
auto basic_writef
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < charType
        , charTraits
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    )
    -> decltype(basic_writef<charType, charTraits, Formaters ...>(output, fmt, arg1))
{
    return
        basic_writef<charType, charTraits, Formaters ...>
        ( basic_writef<charType, charTraits, Formaters ...>(output, fmt, arg1)
        , fmt, arg2);
}

template
    < typename charType
    , typename charTraits
    , typename Output
    , typename ... Formaters
    >
auto basic_writef
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < charType
        , charTraits
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    )
    -> decltype(basic_writef<charType, charTraits, Formaters ...>(output, fmt, arg1))
{
    return
        basic_writef<charType, charTraits, Formaters ...>
        ( basic_writef<charType, charTraits, Formaters ...>
        ( basic_writef<charType, charTraits, Formaters ...>(output, fmt, arg1)
        , fmt, arg2)
        , fmt, arg3);
}


template
    < typename charType
    , typename charTraits
    , typename Output
    , typename ... Formaters
    >
auto basic_writef
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < charType
        , charTraits
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    , decltype(arg1) arg4
    )
    -> decltype(basic_writef<charType, charTraits, Formaters ...>(output, fmt, arg1))
{
    return
        basic_writef<charType, charTraits, Formaters ...>
        ( basic_writef<charType, charTraits, Formaters ...>
        ( basic_writef<charType, charTraits, Formaters ...>
        ( basic_writef<charType, charTraits, Formaters ...>(output, fmt, arg1)
        , fmt, arg2)
        , fmt, arg3)
        , fmt, arg4);
}


template
    < typename charType
    , typename charTraits
    , typename Output
    , typename ... Formaters
    >
auto basic_writef
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < charType
        , charTraits
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    , decltype(arg1) arg4
    , decltype(arg1) arg5
    )
    -> decltype(basic_writef<charType, charTraits, Formaters ...>(output, fmt, arg1))
{
    return
        basic_writef<charType, charTraits, Formaters ...>
        ( basic_writef<charType, charTraits, Formaters ...>
        ( basic_writef<charType, charTraits, Formaters ...>
        ( basic_writef<charType, charTraits, Formaters ...>
        ( basic_writef<charType, charTraits, Formaters ...>
        ( output, fmt, arg1)
        , fmt, arg2)
        , fmt, arg3)
        , fmt, arg4)
        , fmt, arg5);
}

template <typename Output, typename ... Formaters>
auto writef
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < char
        , std::char_traits<char>
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    )
    -> decltype
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
            (output, fmt, arg1)
        )
{
    return basic_writef<char, std::char_traits<char>, Formaters ...>(output, fmt, arg1);
}

template <typename Output, typename ... Formaters>
auto writef
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < char
        , std::char_traits<char>
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    )
   -> decltype
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
            (output, fmt, arg1)
        )
{
    return
        basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( output, fmt, arg1)
        , fmt, arg2);
}

template <typename Output, typename ... Formaters>
auto writef
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < char
        , std::char_traits<char>
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    )
   -> decltype
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
            (output, fmt, arg1)
        )
{
    return
        basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( output, fmt, arg1)
        , fmt, arg2)
        , fmt, arg3);
}

template <typename Output, typename ... Formaters>
auto writef
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::input_base_ref
        < char
        , std::char_traits<char>
        , typename std::decay<decltype(fmt)>::type
        > & arg1
    , decltype(arg1) arg2
    , decltype(arg1) arg3
    , decltype(arg1) arg4
    )
   -> decltype
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
            (output, fmt, arg1)
        )
{
    return
        basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( output, fmt, arg1)
        , fmt, arg2)
        , fmt, arg3)
        , fmt, arg4);
}


template <typename Output, typename ... Formaters>
auto writef
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
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
   -> decltype
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
            (output, fmt, arg1)
        )
{
    return
        basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( output, fmt, arg1)
        , fmt, arg2)
        , fmt, arg3)
        , fmt, arg4)
        , fmt, arg5);
}

template <typename Output, typename ... Formaters>
auto writef
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
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
   -> decltype
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
            (output, fmt, arg1)
        )
{
    return
        basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( output, fmt, arg1)
        , fmt, arg2)
        , fmt, arg3)
        , fmt, arg4)
        , fmt, arg5)
        , fmt, arg6);
}
    
template <typename Output, typename ... Formaters>
auto writef
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
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
   -> decltype
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
            (output, fmt, arg1)
        )
{
    return
        basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( output, fmt, arg1)
        , fmt, arg2)
        , fmt, arg3)
        , fmt, arg4)
        , fmt, arg5)
        , fmt, arg6)
        , fmt, arg7);
}

template <typename Output, typename ... Formaters>
auto writef
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
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
   -> decltype
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
            (output, fmt, arg1)
        )
{
    return
        basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( output, fmt, arg1)
        , fmt, arg2)
        , fmt, arg3)
        , fmt, arg4)
        , fmt, arg5)
        , fmt, arg6)
        , fmt, arg7)
        , fmt, arg8);
}

template <typename Output, typename ... Formaters>
auto writef
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
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
   -> decltype
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
            (output, fmt, arg1)
        )
{
    return
        basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( output, fmt, arg1)
        , fmt, arg2)
        , fmt, arg3)
        , fmt, arg4)
        , fmt, arg5)
        , fmt, arg6)
        , fmt, arg7)
        , fmt, arg8)
        , fmt, arg9);
}


template <typename Output, typename ... Formaters>
auto writef
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
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
   -> decltype
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
            (output, fmt, arg1)
        )
{
    return
        basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( basic_writef<char, std::char_traits<char>, Formaters ...>
        ( output, fmt, arg1)
        , fmt, arg2)
        , fmt, arg3)
        , fmt, arg4)
        , fmt, arg5)
        , fmt, arg6)
        , fmt, arg7)
        , fmt, arg8)
        , fmt, arg9)
        , fmt, arg10);
}

} // namespace stringify
} // namespace boost
    
#endif  /* BOOST_STRINGIFY_WRITEF_HPP */

