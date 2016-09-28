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

namespace detail
{

template
    < typename charT
    , typename charTraits
    >
class char_ptr_writer
{
public:

    typedef charT* return_type;
    typedef charT* output_type;

    template <typename Fmt, typename InputIterator>
    static return_type write_range
        ( charT* out
        , const Fmt& fmt
        , InputIterator begin
        , InputIterator end
        )
    {
        for(auto it = begin; it != end; ++it)
        {
            out = it->write_without_termination_char(out, fmt);
        }
        charTraits::assign(*out, charT());
        return out;
    }

    template <typename Fmt>
    static return_type write_va(charT* output, const Fmt& fmt)
    {
        return output;
    }

    template <typename Fmt, typename Arg1>
    static return_type write_va(charT* output, const Fmt& fmt, const Arg1& arg1)
    {
        return arg1.write(output, fmt);
    }
   
    
    template <typename Fmt, typename Arg1, typename ... Args>
    static return_type write_va
        ( charT* output
        , const Fmt& fmt
        , const Arg1& arg1
        , const Args& ... args)
    {
        return write_va(arg1.write(output, fmt), fmt, args...); 
    }

};

template
    < typename charT
    , typename charTraits  
    , typename Formating
    , typename Impl  
    >
class final_writer
{
public:
    typedef 
        boost::stringify::input_base_ref<charT, charTraits, Formating>
        arg_type;

    typedef
        typename Impl::return_type
        return_type;

    typedef
        typename Impl::output_type
        output_type;

    
    template <typename ... FmtArgs>
    final_writer(output_type output, const FmtArgs& ... fmtargs)
        : m_output(output)
        , m_fmt(fmtargs ...)
    {
    }

    return_type operator[](const std::initializer_list<arg_type>& lst) const
    {
        return Impl::write_range(m_output, m_fmt, lst.begin(), lst.end());
    }
    
    return_type operator()() const
    {
        return Impl::write_va(m_output, m_fmt);
    }

    return_type operator()(const arg_type& a1) const
    {
        return Impl::write_va(m_output, m_fmt, a1);
    }
  
    return_type operator()
        ( const arg_type& a1
        , const arg_type& a2
        ) const
    {
        return Impl::write_va(m_output, m_fmt, a1, a2);
    }
    
    return_type operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        ) const
    {
        return Impl::write_va(m_output, m_fmt, a1, a2, a3);
    }

    return_type operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        , const arg_type& a4
        ) const
    {
        return Impl::write_va(m_output, m_fmt, a1, a2, a3, a4);
    }

    return_type operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        , const arg_type& a4
        , const arg_type& a5
        ) const
    {
        return Impl::write_va(m_output, m_fmt, a1, a2, a3, a4, a5);
    }

    return_type operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        , const arg_type& a4
        , const arg_type& a5
        , const arg_type& a6
        ) const
    {
        return Impl::write_va(m_output, m_fmt, a1, a2, a3, a4, a5, a6);
    }

    return_type operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        , const arg_type& a4
        , const arg_type& a5
        , const arg_type& a6
        , const arg_type& a7
        ) const
    {
        return Impl::write_va
            (m_output, m_fmt, a1, a2, a3, a4, a5, a6, a7);
    }

    return_type operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        , const arg_type& a4
        , const arg_type& a5
        , const arg_type& a6
        , const arg_type& a7
        , const arg_type& a8
        ) const
    {
        return Impl::write_va
            (m_output, m_fmt, a1, a2, a3, a4, a5, a6, a7, a8);
    }

    return_type operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        , const arg_type& a4
        , const arg_type& a5
        , const arg_type& a6
        , const arg_type& a7
        , const arg_type& a8
        , const arg_type& a9
        ) const
    {
        return Impl::write_va
            (m_output, m_fmt, a1, a2, a3, a4, a5, a6, a7, a8, a9);
    }

    return_type operator()
        ( const arg_type& a1
        , const arg_type& a2
        , const arg_type& a3
        , const arg_type& a4
        , const arg_type& a5
        , const arg_type& a6
        , const arg_type& a7
        , const arg_type& a8
        , const arg_type& a9
        , const arg_type& a10
        ) const
    {
        return Impl::write_va
            (m_output, m_fmt, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
    }

private:
    output_type m_output;
    Formating m_fmt;
};


template
    < typename charT
    , typename charTraits 
    , typename Impl  
    >
class fmt_receiver
{
public:
    typedef
        typename Impl::return_type
        return_type;

    typedef
        typename Impl::output_type
        output_type;

    typedef 
        boost::stringify::input_base_ref
            < charT
            , charTraits
            , boost::stringify::formater_tuple<>
            >
        default_arg_type;

    template <typename ... Formaters>
        using final_writer_type
        = boost::stringify::detail::final_writer
            < charT
            , charTraits
            , boost::stringify::formater_tuple<Formaters...>
            , Impl 
            >;

    fmt_receiver(output_type output)
        : m_output(output)
    {
    }

    template <typename ... Formaters>
    final_writer_type<Formaters ...> operator() (Formaters ... formaters) const
    {
        return final_writer_type<Formaters ...>(m_output, formaters ...);
    }
  
    return_type operator[](std::initializer_list<default_arg_type> lst) const
    {
        return final_writer_type<>(m_output)[lst];
    }
    
private:
    output_type m_output;
};

} // namespace detail




template<typename charT, typename charTraits = std::char_traits<charT> >
inline auto writef(charT* output)
    -> boost::stringify::detail::fmt_receiver
        < charT
        , charTraits
        , boost::stringify::detail::char_ptr_writer<charT, charTraits>
        >
{
    return boost::stringify::detail::fmt_receiver
        < charT
        , charTraits
        , boost::stringify::detail::char_ptr_writer<charT, charTraits>
        >
        (output);
}
    







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

