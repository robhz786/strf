#ifndef BOOST_STRINGIFY_OUTPUT_TYPES_HPP_INCLUDED
#define BOOST_STRINGIFY_OUTPUT_TYPES_HPP_INCLUDED

#include <initializer_list>
#include <string>
#include <ostream>
#include <boost/stringify/input_base.hpp>
#include <boost/stringify/formater_tuple.hpp>
#include <boost/stringify/listf.hpp>

namespace boost
{
namespace stringify 
{

//
// charT*
//

template<typename charT, typename Formating>
void basic_write
    ( charT* output
    , const Formating& fmt
    , const boost::stringify::input_base<charT, Formating>& writer
    )
{
    writer.write(output, fmt);
}

template <typename ... Formaters>
char* swrite
    ( char* output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::detail::input_base_ref
        < char
        , typename std::decay<decltype(fmt)>::type
        > & awr1
    )
{
    return awr1.writer.write(output, fmt);
}


template <typename ... Formaters>
char* swrite
    ( char* output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::detail::input_base_ref
        < char
        , typename std::decay<decltype(fmt)>::type
        > & awr1
    , const boost::stringify::detail::input_base_ref
        < char
        , typename std::decay<decltype(fmt)>::type
        > & lastawr  
    )
{
    return lastawr.writer.write
        ( boost::stringify::swrite(output, fmt, awr1)
        , fmt
        );
}

template <typename ... Formaters>
char* swrite
    ( char* output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::detail::input_base_ref
        < char
        , typename std::decay<decltype(fmt)>::type
        > & awr1
    , const boost::stringify::detail::input_base_ref
        < char
        , typename std::decay<decltype(fmt)>::type
        > & awr2
    , const boost::stringify::detail::input_base_ref
        < char
        , typename std::decay<decltype(fmt)>::type
        > & lastawr
    )      
{
    return lastawr.writer.write
        ( boost::stringify::swrite(output, fmt, awr1, awr2)
        , fmt
        );
}

template <typename ... Formaters>
char* swrite
    ( char* output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::detail::input_base_ref
        < char
        , typename std::decay<decltype(fmt)>::type
        > & awr1
    , const boost::stringify::detail::input_base_ref
        < char
        , typename std::decay<decltype(fmt)>::type
        > & awr2
    , const boost::stringify::detail::input_base_ref
        < char
        , typename std::decay<decltype(fmt)>::type
        > & awr3
    , const boost::stringify::detail::input_base_ref
        < char
        , typename std::decay<decltype(fmt)>::type
        > & lastawr
    )
{
    return lastawr.writer.write
        ( boost::stringify::swrite(output, fmt, awr1, awr2, awr3)
        , fmt
        );
}


//
// std::basic_string
//

template<typename charT, typename traits, typename Allocator, typename Formarting>
void basic_assign
    ( std::basic_string<charT, traits, Allocator>& str
    , const Formarting& fmt
    , const boost::stringify::input_base<charT, Formarting>& input
    )
{
    str.assign(input.minimal_length(fmt), charT());
    charT* begin = &*str.begin();
    charT* end = input.write_without_termination_char(begin, fmt);

    // apply char_traits if necessary
    std::size_t length = end - begin;
    BOOST_ASSERT(length <= input.minimal_length(fmt));
    if( ! std::is_same<traits, std::char_traits<charT> >::value)
        traits::move(begin, begin, length);

    str.resize(length);
}


template<typename charT, typename traits, typename Allocator, typename Formarting>
void basic_append
    ( std::basic_string<charT, traits, Allocator>& str
    , const Formarting& fmt
    , const boost::stringify::input_base<charT, Formarting>& input
    )
{
    // allocate memorry
    std::size_t initial_length = str.length();
    str.append(input.minimal_length(fmt), charT());
    charT* append_begin = & str[initial_length];

    // write
    charT* append_end = input.write_without_termination_char(append_begin, fmt);

    // apply char_traits if necessary
    std::size_t append_length = append_end - append_begin;
    BOOST_ASSERT(append_length <= input.minimal_length(fmt));
    if( ! std::is_same<traits, std::char_traits<charT> >::value)
        traits::move(append_begin, append_begin, append_length);

    // set correct size ( current size might be greater )
    str.resize(initial_length + append_length);
}

//
// std::ostream
//


namespace detail
{
template <typename charT, typename traits>
class std_ostream_adapter: public boost::stringify::simple_ostream<charT>
{
public:
    std_ostream_adapter(std::basic_ostream<charT, traits>& _out)
        : out(_out)
    {
    }

    virtual bool good() noexcept
    {
        return out.good();
    }

    virtual void put(charT c) noexcept
    {
        out.put(c);
    }

    virtual void write(const charT* str, std::size_t len) noexcept
    {
        out.write(str, len);
    }

private:
    std::basic_ostream<charT, traits>& out;
};
} //namespace detail


template<typename charT, typename traits, typename Formarting>
void basic_write
    ( std::basic_ostream<charT, traits>& output
    , const Formarting& fmt
    , const boost::stringify::input_base<charT, Formarting>& input
    )
{
    boost::stringify::detail::std_ostream_adapter<charT, traits>
        adapted_output(output);
    
    input.write(adapted_output, fmt);
}


template <typename Output, typename ... Formaters>
void write
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::listf
        < char
        , typename std::decay<decltype(fmt)>::type
        > & input 
    )
{
    boost::stringify::basic_write<char>(output, fmt, input);
}

      
template <typename Output, typename ... Formaters>
void write16
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::listf
        < char16_t
        , typename std::decay<decltype(fmt)>::type
        > & input
    )
{
    boost::stringify::basic_write<char16_t>(output, fmt, input);
}

      
template <typename Output, typename ... Formaters>
void write32
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::listf
        < char32_t
        , typename std::decay<decltype(fmt)>::type
        > & input
    )
{
    boost::stringify::basic_write<char32_t>(output, fmt, input);
}

      
template <typename Output, typename ... Formaters>
void wwrite
    ( Output&& output
    , const boost::stringify::formater_tuple<Formaters ...>& fmt
    , const boost::stringify::listf
        < wchar_t
        , typename std::decay<decltype(fmt)>::type
        > & input
    )
{
    boost::stringify::basic_write<wchar_t>(output, fmt, input);
}

} //namespace stringify
} //namespace boost

#endif
